# ============================================================
# common_utils.py
# 공통 함수 및 상수 정의
#
# OS: Windows 11 / Python 3.11.9
# ============================================================

import numpy as np
import pandas as pd
import os, glob, re, random
from nptdms import TdmsFile, TdmsWriter, ChannelObject
from scipy.stats import kurtosis
from scipy.fft import fft, fftfreq
from xgboost import XGBClassifier, XGBRegressor

# ── 경로 설정 ─────────────────────────────────────────────────
TRAIN_PATH = '/data/Train'
TEST_PATH  = '/data/Test'
OUTPUT_PATH = '/data/output'

# ── 전역 상수 ─────────────────────────────────────────────────
FS          = 25600          # 샘플링 주파수 (Hz)
WIN         = FS             # 1초 단위 윈도우 크기 (25600 샘플)
SEED        = 42
CH_NAMES    = ['ch1', 'ch2', 'ch3', 'ch4']
FAULT_TYPES = ['BPFI', 'BPFO', 'BSF', 'Cage']
RPM_MAP     = {0: 750.0, 1: 950.0}
 
# 베어링 고장 주파수 (1000 RPM 기준, Hz)
FAULT_FREQ = {'BPFI': 140, 'BPFO': 93, 'BSF': 78, 'Cage': 6.7}
FAULT_BW   = {'BPFI': 8,   'BPFO': 6,  'BSF': 6,  'Cage': 1.5}
 
# 샤프트 주파수 피처 (RPM 이진 분류용)
SHAFT_FEATS = [
    'peak_freq', 'peak_rpm', 'peak_mag', 'confidence',
    'low_energy', 'high_energy', 'energy_ratio', 'shaft_energy_ratio',
]
 
# cutoff 설정
TRAIN_CUTOFFS = [0.1, 0.3, 0.4, 0.6, 0.7, 0.9]
VALID_CUTOFFS = [0.2, 0.5, 0.8]
EXCLUDE_COLS  = ['epoch', 'window', 'RUL', 'life_pct', 'ttf',
                 'rpm', 'train_id', 'test_id']
 
# XGBoost 하이퍼파라미터
XGB_PARAMS = dict(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8,    colsample_bytree=0.8,
    random_state=SEED, verbosity=0,
)
 
# 앙상블 비율: BEST_ALPHA × log(RUL) + (1-BEST_ALPHA) × TTF
BEST_ALPHA = 0.5

# ============================================================
# 1. 유틸리티 함수
# ============================================================
 
def fix_seed(seed=SEED):
    """재현성을 위한 랜덤 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
 
 
def calc_score(act_rul, pred_rul):
    """
    채점 함수
    Er = 100 × (ActRUL - PredRUL) / ActRUL

    Er <= 0 (과대 예측): exp(-ln(0.5) × Er/30)
      Er이 음수 → -ln(0.5)×(음수)/30 → 음수 → 1보다 작음
      패널티 강함 (분모 30)

    Er > 0 (과소 예측): exp(+ln(0.5) × Er/50)
      ln(0.5) 음수 → 양수×음수/50 → 음수 → 1보다 작음
      패널티 약함 (분모 50)
    """
    if act_rul == 0:
        return 1.0 if pred_rul == 0 else 0.0
    er = 100 * (act_rul - pred_rul) / act_rul
    if er <= 0:
        return float(np.exp(-np.log(0.5) * er / 30))
    else:
        return float(np.exp(+np.log(0.5) * er / 50))
 
def calc_sample_weights(cutoff_values):
    """
    말기 cutoff에 높은 가중치 부여 (co^2)
    0.9 → 높음 / 0.1 → 낮음
    """
    co = np.array(cutoff_values, dtype=np.float32)
    w  = co ** 2
    return (w / w.mean()).astype(np.float32)

# ============================================================
# 2. RUL 라벨링 (Step 1)
# ============================================================
 
def get_train_num(file_name):
    """파일명에서 Train 번호 추출"""
    return int(re.findall(r'\d+', file_name)[0])
 
 
def label_trainset(base_path=TRAIN_PATH, fs=FS):
    """
    Train TDMS + Operation CSV → TTF/RUL 라벨링
    출력: _labeled.tdms, _labeled.csv
 
    TTF 확정 로직:
      1) 운전 데이터 기반: 온도/토크 고장 조건 확인
      2) 진동 데이터 기반: 마지막 파일 수집 길이 확인
      3) actual 우선, 없으면 expect 중 큰 값
    """
    op_files = sorted(glob.glob(
        os.path.join(base_path, 'Train*_Operation.csv')))
 
    for op_path in op_files:
        if '_labeled' in op_path:
            continue
        t_num = get_train_num(os.path.basename(op_path))
        print(f'\n>>> Train {t_num} 라벨링')
 
        df_op   = pd.read_csv(op_path, encoding='cp949')
        vib_dir = os.path.join(base_path, f'Train{t_num}_Vibration')
        vib_files = sorted(
            [f for f in glob.glob(os.path.join(vib_dir, '*.tdms'))
             if '_labeled' not in f],
            key=lambda x: int(os.path.basename(x).split('.')[0])
        )
 
        # 1) 운전 데이터 기반 TTF
        cond_temp   = ((df_op['  TC SP Front[℃]'] >= 200) |
                       (df_op['  TC SP Rear[℃]']  >= 200))
        cond_torque = (df_op['  Torque[Nm]'] <= -20)
        cond_fail   = cond_temp | cond_torque
        last_op_time = df_op['Time[sec]'].iloc[-1]
 
        actual_op = None
        expect_op = None
        if cond_fail.any():
            actual_op = df_op.loc[cond_fail.idxmax(), 'Time[sec]']
        else:
            expect_op = last_op_time + 1
 
        # 2) 진동 데이터 기반 TTF
        last_vib_path = vib_files[-1]
        last_vib_num  = int(
            os.path.basename(last_vib_path).split('.')[0])
        with TdmsFile.read(last_vib_path) as tdms:
            data_len = len(tdms.groups()[0].channels()[0])
 
        actual_vib = None
        expect_vib = None
        if data_len < fs * 60:
            actual_vib = (last_vib_num - 1) * 600 + (data_len / fs)
        else:
            expect_vib = (last_vib_num - 1) * 600 + 61
 
        # 3) 최종 TTF 확정
        if actual_op or actual_vib:
            ttf = min(filter(None, [actual_op, actual_vib]))
        else:
            ttf = max(filter(None, [expect_op, expect_vib]))
            print('  [예측 고장 시점 사용]')
 
        # 4) Operation CSV 저장
        df_op['ttf'] = ttf
        df_op['RUL'] = (ttf - df_op['Time[sec]']).clip(lower=0)
        df_op.to_csv(op_path.replace('.csv', '_labeled.csv'),
                     index=False, encoding='utf-8')
 
        # 5) TDMS 라벨링
        for fpath in vib_files:
            f_num = int(os.path.basename(fpath).split('.')[0])
            with TdmsFile.read(fpath) as tdms:
                group      = tdms.groups()[0]
                group_name = group.name
                n_samples  = len(group.channels()[0])
                cur_time   = ((f_num - 1) * 600 +
                              np.arange(n_samples) / fs)
                vib_RUL = (ttf - cur_time).clip(min=0)
                vib_ttf = np.full(n_samples, ttf)
 
                new_fpath = fpath.replace('.tdms', '_labeled.tdms')
                with TdmsWriter(new_fpath) as writer:
                    channels = [
                        ChannelObject(group_name, ch.name, ch[:])
                        for ch in group.channels()
                    ]
                    channels.append(
                        ChannelObject(group_name, 'ttf', vib_ttf))
                    channels.append(
                        ChannelObject(group_name, 'RUL', vib_RUL))
                    writer.write_segment(channels)
 
        print(f'  TTF: {ttf:.0f}s | 저장 완료')

# ============================================================
# 3. RPM 예측 모델 (Step 2-1)
# ============================================================
 
def extract_shaft_features(sig, fs=FS):
    """
    CH1 신호 → 샤프트 주파수(10.8~16.8Hz) 기반 피처
    700RPM ≈ 11.67Hz / 950RPM ≈ 15.83Hz
    """
    N     = len(sig)
    freqs = fftfreq(N, 1/fs)[:N//2]
    mag   = np.abs(fft(sig))[:N//2]
 
    total_energy = np.sum(mag**2) + 1e-10
    shaft_mask   = (freqs >= 10.8) & (freqs <= 16.8)
    shaft_mag    = mag[shaft_mask]
    shaft_freq   = freqs[shaft_mask]
    peak_idx     = np.argmax(shaft_mag)
 
    low_mask  = (freqs >= 10.8) & (freqs < 13.8)
    high_mask = (freqs >= 13.8) & (freqs <= 16.8)
 
    return {
        'peak_freq':          float(shaft_freq[peak_idx]),
        'peak_rpm':           float(shaft_freq[peak_idx] * 60),
        'peak_mag':           float(shaft_mag[peak_idx]),
        'confidence':         float(shaft_mag[peak_idx] /
                                    (shaft_mag.mean() + 1e-10)),
        'low_energy':         float(np.sum(mag[low_mask]**2)),
        'high_energy':        float(np.sum(mag[high_mask]**2)),
        'energy_ratio':       float(np.sum(mag[low_mask]**2) /
                                    (np.sum(mag[high_mask]**2) + 1e-10)),
        'shaft_energy_ratio': float(np.sum(shaft_mag**2) / total_energy),
    }

def load_rpm_series(train_num):
    """Operation CSV → RPM 시계열 로드"""
    path = os.path.join(TRAIN_PATH,
                        f'Train{train_num}_Operation.csv')
    df   = pd.read_csv(path, encoding='cp949')
    df.columns = [c.strip() for c in df.columns]
    return df.set_index(df.columns[0])[df.columns[2]]
 
 
def epoch_rpm(rpm_series, f_num):
    """epoch(파일 번호) 기준 평균 RPM 계산"""
    t0  = (f_num - 1) * 600
    t1  = t0 + 60
    sub = rpm_series[(rpm_series.index >= t0) &
                     (rpm_series.index <= t1)]
    return float(sub.mean()) if len(sub) > 0 else 850.0
 
 
def train_rpm_classifier():
    """
    Train1~4 샤프트 피처로 XGBoost 이진 분류 모델 학습
    0: 750RPM / 1: 950RPM
    """
    print('\n=== RPM 이진 분류 모델 학습 ===')
    all_rows = []
    for t in [1, 2, 3, 4]:
        vib_dir    = os.path.join(TRAIN_PATH, f'Train{t}_Vibration')
        files      = sorted(
            glob.glob(os.path.join(vib_dir, '*_labeled.tdms')),
            key=lambda x: int(os.path.basename(x).split('_')[0])
        )
        rpm_series = load_rpm_series(t)
        rows = []
        for fpath in files:
            f_num      = int(os.path.basename(fpath).split('_')[0])
            rpm_actual = epoch_rpm(rpm_series, f_num)
            rpm_label  = 0 if rpm_actual < 825 else 1
            with TdmsFile.read(fpath) as tdms:
                ch1 = tdms.groups()[0].channels()[0][:].astype(
                    np.float32)
            for w in range(len(ch1) // WIN):
                feat = extract_shaft_features(ch1[w*WIN:(w+1)*WIN])
                feat.update({'rpm_label': rpm_label})
                rows.append(feat)
        print(f'  Train {t}: {len(rows)}행')
        all_rows.extend(rows)
 
    df_all = pd.DataFrame(all_rows)
    fix_seed()
    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, verbosity=0)
    model.fit(df_all[SHAFT_FEATS].values,
              df_all['rpm_label'].values)
    print('  RPM 모델 학습 완료')
    return model

def predict_rpm(sig, model):
    """CH1 1초 신호 → RPM 예측 (750 or 950)"""
    feat = extract_shaft_features(sig)
    X    = np.array([[feat[f] for f in SHAFT_FEATS]])
    return RPM_MAP[int(model.predict(X)[0])]

# ============================================================
# 4. 피처 엔지니어링 (Step 2-2)
# ============================================================
 
def time_features(sig):
    """
    1초 진동 신호 → 시간 도메인 피처
    rms, kurtosis, peak2peak, crest_factor,
    shape_factor, impulse_factor
    """
    rms      = np.sqrt(np.mean(sig**2))
    peak     = np.max(np.abs(sig))
    mean_abs = np.mean(np.abs(sig)) + 1e-10
    return {
        'rms':            np.float32(rms),
        'kurtosis':       np.float32(kurtosis(sig, fisher=True)),
        'peak2peak':      np.float32(np.ptp(sig)),
        'crest_factor':   np.float32(peak / (rms + 1e-10)),
        'shape_factor':   np.float32(rms / mean_abs),
        'impulse_factor': np.float32(peak / mean_abs),
    }
 
 
def freq_features(sig, rpm_pred):
    """
    1초 진동 신호 → 주파수 도메인 피처
    spectral_centroid, spectral_entropy,
    fault_BPFI/BPFO/BSF/Cage (RPM 보정 고장 주파수 대역 에너지)
    """
    N     = len(sig)
    freqs = fftfreq(N, 1/FS)[:N//2]
    mag   = np.abs(fft(sig))[:N//2]
    power = mag**2
    total = np.sum(power) + 1e-10
    scale = rpm_pred / 1000  # RPM 보정 계수
 
    result = {
        'spectral_centroid': np.float32(
            np.sum(freqs * power) / total),
    }
    np_norm = power / total
    np_norm = np_norm[np_norm > 0]
    result['spectral_entropy'] = np.float32(
        -np.sum(np_norm * np.log2(np_norm)))
 
    for name, fc in FAULT_FREQ.items():
        bw   = FAULT_BW[name]
        band = ((freqs >= fc*scale - bw) &
                (freqs <= fc*scale + bw))
        result[f'fault_{name}'] = np.float32(
            np.sum(power[band]) / band.sum()
            if band.sum() > 0 else 0.0)
    return result
 
 
def add_derived_features(df):
    """
    DataFrame 전체 기준 파생 피처 생성
    1) RMS 정규화 (rpm_pred 보정)
    2) 전후방 열화 지표 (front_deg, rear_deg)
    3) 열화 위치 비율 (deg_location)
    4) 수직/축방향 결함 에너지 (vertical/axial_fail_energy)
    5) 결함 방향 비율 (fail_direction)
    6) 초기 결함 강도 (initial_fault_intensity, 초반 2시간 상수)
    7) 불필요 컬럼 제거
    """
 
    def norm_col(col):
        s = df[col] if isinstance(col, str) else col
        return s / (s.abs().max() + 1e-10)
 
    # 1. RMS 정규화
    for ch in CH_NAMES:
        df[f'{ch}_rms_norm'] = (
            df[f'{ch}_rms'] / (df['rpm_pred'] / 1000)
        ).astype(np.float32)
 
    # 2. 전후방 열화 지표
    front_fault = ([f'ch1_fault_{f}' for f in FAULT_TYPES] +
                   [f'ch2_fault_{f}' for f in FAULT_TYPES])
    rear_fault  = ([f'ch3_fault_{f}' for f in FAULT_TYPES] +
                   [f'ch4_fault_{f}' for f in FAULT_TYPES])
 
    front_fre = df[front_fault].sum(axis=1)
    rear_fre  = df[rear_fault].sum(axis=1)
 
    df['front_deg'] = ((
        norm_col('ch1_peak2peak') + norm_col('ch2_kurtosis') +
        norm_col('ch2_peak2peak') + norm_col(front_fre)
    ) / 4).astype(np.float32)
 
    df['rear_deg'] = ((
        norm_col('ch3_peak2peak') + norm_col('ch4_kurtosis') +
        norm_col('ch4_peak2peak') + norm_col(rear_fre)
    ) / 4).astype(np.float32)
 
    # 3. 열화 위치 비율
    df['deg_location'] = (
        (df['front_deg'] - df['rear_deg']) /
        (df['front_deg'] + df['rear_deg'] + 1e-10)
    ).astype(np.float32)
 
    # 4. 수직/축방향 결함 에너지
    vertical = ([f'ch1_fault_{f}' for f in FAULT_TYPES] +
                [f'ch3_fault_{f}' for f in FAULT_TYPES])
    axial    = ([f'ch2_fault_{f}' for f in FAULT_TYPES] +
                [f'ch4_fault_{f}' for f in FAULT_TYPES])
    df['vertical_fail_energy'] = (
        df[vertical].sum(axis=1).astype(np.float32))
    df['axial_fail_energy'] = (
        df[axial].sum(axis=1).astype(np.float32))
 
    # 5. 결함 방향 비율
    df['fail_direction'] = (
        (df['vertical_fail_energy'] - df['axial_fail_energy']) /
        (df['vertical_fail_energy'] + df['axial_fail_energy'] + 1e-10)
    ).astype(np.float32)
 
    # 6. 초기 결함 강도 (초반 2시간 기준 상수값)
    # 2시간 = 700/950 RPM 각 1사이클 포함 → 베어링 고유 초기 상태 반영
    intensity_feats = []
    for ch in CH_NAMES:
        intensity_feats += [
            f'{ch}_kurtosis',     f'{ch}_crest_factor',
            f'{ch}_shape_factor', f'{ch}_impulse_factor',
        ]

    max_t = df['t_abs'].max()
    if max_t >= 7200:
        early = df[df['t_abs'] <= 7200]  # 초반 2시간
    else:
        early = df  # 전체 데이터 (2시간 미만인 경우)

    intensity = (early[intensity_feats].mean().mean()
                if len(early) > 0 else 0.0)
    df['initial_fault_intensity'] = np.float32(intensity)
 
    # 7. 불필요 컬럼 제거
    drop_cols = (
        [f'{ch}_rms'            for ch in CH_NAMES] +
        [f'{ch}_kurtosis'       for ch in CH_NAMES] +
        [f'{ch}_crest_factor'   for ch in CH_NAMES] +
        [f'{ch}_shape_factor'   for ch in CH_NAMES] +
        [f'{ch}_impulse_factor' for ch in CH_NAMES]
    )
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df
 
 
def extract_summary_rows(tdms_path, f_num, rpm_model, is_test=False):
    """
    TDMS 파일 → 1초 단위 피처 행 리스트 반환
    t_abs = (epoch-1)*600 + window + 1 (1, 2, 3, ...)
    is_test=True: ttf, RUL, life_pct 제외
    """
    with TdmsFile.read(tdms_path) as tdms:
        ch_data = {ch.name: ch[:].astype(np.float32)
                   for ch in tdms.groups()[0].channels()}
 
    signals   = [ch_data[f'CH{i+1}'] for i in range(4)]
    n_windows = len(signals[0]) // WIN
 
    if not is_test:
        ttf     = float(ch_data['ttf'][0])
        rul_arr = ch_data['RUL']
    else:
        ttf = rul_arr = None
 
    rows = []
    for w in range(n_windows):
        start    = w * WIN
        end      = start + WIN
        mid      = (start + end) // 2
        t_abs    = (f_num - 1) * 600 + w + 1  # float64: 타임스탬프 정밀도
        rpm_pred = predict_rpm(signals[0][start:end], rpm_model)
 
        row = {
            'epoch':    f_num,
            'window':   w,
            't_abs':    t_abs,
            'rpm_pred': np.float32(rpm_pred),
        }
        if not is_test:
            row['ttf']      = ttf                 # float64
            row['RUL']      = float(rul_arr[mid]) # float64
            row['life_pct'] = float(t_abs / ttf * 100)
 
        for i, ch in enumerate(CH_NAMES):
            sig = signals[i][start:end]
            for k, v in time_features(sig).items():
                row[f'{ch}_{k}'] = v
            for k, v in freq_features(sig, rpm_pred).items():
                row[f'{ch}_{k}'] = v
        rows.append(row)
    return rows
 
 
def build_train_summary(train_num, rpm_model):
    """Train TDMS → 피처 CSV 생성 및 저장"""
    vib_dir = os.path.join(TRAIN_PATH, f'Train{train_num}_Vibration')
    files   = sorted(
        glob.glob(os.path.join(vib_dir, '*_labeled.tdms')),
        key=lambda x: int(os.path.basename(x).split('_')[0])
    )
    all_rows = []
    for fpath in files:
        f_num = int(os.path.basename(fpath).split('_')[0])
        all_rows.extend(
            extract_summary_rows(fpath, f_num, rpm_model, is_test=False))
 
    df = pd.DataFrame(all_rows)
    df = add_derived_features(df)
 
    # 타임스탬프 제외 float32 변환
    time_cols  = ['t_abs', 'ttf', 'RUL', 'life_pct']
    float_cols = [c for c in df.select_dtypes('float64').columns
                  if c not in time_cols]
    df[float_cols] = df[float_cols].astype(np.float32)
 
    save_path = os.path.join(
        TRAIN_PATH,
        f'Train{train_num}_Vibration_featured_summary.csv')
    df.to_csv(save_path, index=False, encoding='utf-8')
    print(f'  Train {train_num}: {len(df)}행 | '
          f'intensity={df["initial_fault_intensity"].iloc[0]:.4f} | '
          f'저장 완료')
    return df
 
 
def build_test_summary(test_num, rpm_model):
    """Test TDMS → 피처 CSV 생성 및 저장"""
    vib_dir = os.path.join(TEST_PATH, f'Test{test_num}')
    files   = sorted(
        glob.glob(os.path.join(vib_dir, '*.tdms')),
        key=lambda x: int(os.path.basename(x).split('.')[0])
    )
    all_rows = []
    for fpath in files:
        f_num = int(os.path.basename(fpath).split('.')[0])
        all_rows.extend(
            extract_summary_rows(fpath, f_num, rpm_model, is_test=True))
 
    df = pd.DataFrame(all_rows)
    df = add_derived_features(df)
 
    float_cols = [c for c in df.select_dtypes('float64').columns
                  if c != 't_abs']
    df[float_cols] = df[float_cols].astype(np.float32)
 
    save_path = os.path.join(
        TEST_PATH,
        f'Test{test_num}_Vibration_featured_summary.csv')
    df.to_csv(save_path, index=False, encoding='utf-8')
    print(f'  Test {test_num}: {len(df)}행 | '
          f'intensity={df["initial_fault_intensity"].iloc[0]:.4f} | '
          f'저장 완료')
    return df

# ============================================================
# 5. 데이터 로드 (Step 3)
# ============================================================
 
def load_train_data():
    """Train1~4 피처 CSV 로드 + float32 변환"""
    dfs = []
    for t in [1, 2, 3, 4]:
        path = os.path.join(
            TRAIN_PATH,
            f'Train{t}_Vibration_featured_summary.csv')
        df   = pd.read_csv(path, encoding='utf-8')
        df['train_id'] = t
        time_cols  = ['t_abs', 'ttf', 'RUL', 'life_pct']
        float_cols = [c for c in df.select_dtypes('float64').columns
                      if c not in time_cols]
        df[float_cols] = df[float_cols].astype(np.float32)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)
 
 
def load_test_data():
    """Test1~6 피처 CSV 로드 + float32 변환"""
    test_folders = glob.glob(os.path.join(TEST_PATH, 'Test*'))
    test_nums    = sorted([
        int(re.search(r'Test(\d+)', os.path.basename(f)).group(1))
        for f in test_folders
        if os.path.isdir(f) and
           re.search(r'Test(\d+)', os.path.basename(f))
    ])
    dfs = []
    for t in test_nums:
        path = os.path.join(
            TEST_PATH,
            f'Test{t}_Vibration_featured_summary.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, encoding='utf-8')
        df['test_id'] = t
        float_cols = [c for c in df.select_dtypes('float64').columns
                      if c != 't_abs']
        df[float_cols] = df[float_cols].astype(np.float32)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else None

# ============================================================
# 6. 모델링 유틸리티 (Step 3)
# ============================================================
 
def add_trend_features(df, windows=[5, 10], group_col='train_id'):
    """
    slope, rolling mean 트렌드 피처 추가
    windows: 이동 윈도우 크기 리스트
    """
    sort_cols = ([group_col, 'epoch', 'window']
                 if 'window' in df.columns
                 else [group_col, 'epoch'])
    df = df.sort_values(sort_cols).reset_index(drop=True)
 
    exclude = ['epoch', 'window', 't_abs', 'rpm_pred',
               'ttf', 'RUL', 'life_pct', 'rpm',
               'train_id', 'test_id']
    feat_cols = [c for c in df.columns if c not in exclude]
 
    new_cols = {}
    for col in feat_cols:
        for w in windows:
            new_cols[f'{col}_rmean{w}'] = (
                df.groupby(group_col)[col]
                .transform(lambda x, w=w:
                           x.rolling(w, min_periods=1).mean())
            )
            new_cols[f'{col}_slope{w}'] = (
                df.groupby(group_col)[col]
                .transform(lambda x, w=w: x.diff(w) / w)
                .fillna(0)
            )
    df = pd.concat(
        [df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df
 
 
def make_cutoff_samples(df, cutoffs):
    """
    cutoff 시점별 마지막 행 피처 + TTF/RUL 샘플 생성
    Train × cutoff = 학습 샘플
    """
    samples = []
    for tid in df['train_id'].unique():
        df_t = df[df['train_id'] == tid].copy()
        ttf  = float(df_t['ttf'].iloc[0])
        for co in cutoffs:
            df_cut = df_t[df_t['t_abs'] <= ttf * co]
            if len(df_cut) == 0:
                continue
            last_row = df_cut.loc[df_cut['t_abs'].idxmax()]
            rul      = float(ttf - last_row['t_abs'])
            samples.append({
                'train_id': tid, 'cutoff': co,
                'ttf': ttf,      'RUL':    rul,
                't_abs': float(last_row['t_abs']),
                **{c: last_row[c] for c in df_cut.columns
                   if c not in EXCLUDE_COLS + ['train_id']}
            })
    return pd.DataFrame(samples)
 
 
def train_rul_models(df_feat):
    """
    두 모델 학습:
      model_ttf: TTF 직접 예측 (기존 Model G)
      model_log: log1p(RUL) 예측 (타겟 변환, Train3 개선)
    최종 예측: BEST_ALPHA×log + (1-BEST_ALPHA)×TTF
    """
    df_all    = make_cutoff_samples(
        df_feat, TRAIN_CUTOFFS + VALID_CUTOFFS)
    feat_cols = [c for c in df_all.columns
                 if c not in ['train_id', 'cutoff', 'ttf',
                               'RUL', 't_abs']]
    X = df_all[feat_cols].values.astype(np.float32)
    w = calc_sample_weights(df_all['cutoff'].values)
 
    fix_seed()
    model_ttf = XGBRegressor(**XGB_PARAMS)
    model_ttf.fit(X,
                  df_all['ttf'].values.astype(np.float32),
                  sample_weight=w)
 
    fix_seed()
    model_log = XGBRegressor(**XGB_PARAMS)
    model_log.fit(X,
                  np.log1p(df_all['RUL'].values).astype(np.float32),
                  sample_weight=w)
 
    return model_ttf, model_log, feat_cols
