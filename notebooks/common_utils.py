import numpy as np
import pandas as pd
import os
import glob
import random
import torch
import matplotlib.pyplot as plt

TRAIN_PATH = '../data/Train'
TEST_PATH  = '../data/Test'
META_COLS  = ['epoch', 'window', 't_abs', 'rpm_pred',
              'ttf', 'RUL', 'life_pct', 'rpm']
SEED       = 42


# ── 시드 고정 ─────────────────────────────────────────────────
def fix_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ── 스코어 함수 ───────────────────────────────────────────────
def calc_score(act_rul, pred_rul):
    er = 100 * (act_rul - pred_rul) / (act_rul + 1e-10)
    if er <= 0:
        return float(np.exp(-np.log(0.5) * er / 20))
    else:
        return float(np.exp(np.log(0.5) * er / 50))

calc_score_vec = np.vectorize(calc_score)


# ── 데이터 로드 ───────────────────────────────────────────────
def load_train_data():
    dfs = []
    for t in [1, 2, 3, 4]:
        path = os.path.join(TRAIN_PATH,
               f'Train{t}_Vibration_featured_summary.csv')
        df = pd.read_csv(path)
        df['train_id'] = t

        # 시간 관련 컬럼은 float64 유지, 나머지 피처는 float32
        time_cols  = ['t_abs', 'ttf', 'RUL', 'life_pct']
        float_cols = [c for c in df.select_dtypes('float64').columns
                      if c not in time_cols]
        df[float_cols] = df[float_cols].astype(np.float32)

        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_test_data():
    test_folders = glob.glob(os.path.join(TEST_PATH, 'Test*'))
    test_nums    = sorted([
        int(os.path.basename(f).replace('Test', ''))
        for f in test_folders if os.path.isdir(f)
    ])
    dfs = []
    for t in test_nums:
        path = os.path.join(
            TEST_PATH,
            f'Test{t}_Vibration_featured_summary.csv'
        )
        if not os.path.exists(path):
            print(f'Test {t}: CSV 없음 → 건너뜀')
            continue
        df = pd.read_csv(path)
        df['test_id'] = t

        # 시간 관련 컬럼은 float64 유지, 나머지 피처는 float32
        time_cols  = ['t_abs']  # Test는 ttf, RUL, life_pct 없음
        float_cols = [c for c in df.select_dtypes('float64').columns
                      if c not in time_cols]
        df[float_cols] = df[float_cols].astype(np.float32)

        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else None


# ── 피처 컬럼 선택 ────────────────────────────────────────────
def get_feat_cols(df):
    exclude = META_COLS + ['train_id', 'test_id']
    return [c for c in df.columns if c not in exclude]


# ── 트렌드 피처 추가 ──────────────────────────────────────────
def add_trend_features(df, windows=[5, 10], group_col='train_id'):
    sort_cols = [group_col, 'epoch', 'window'] \
                if 'window' in df.columns \
                else [group_col, 'epoch']
    df = df.sort_values(sort_cols).reset_index(drop=True)

    feat_cols = get_feat_cols(df)

    # 트렌드 피처 미리 계산 후 concat으로 한번에 추가
    new_cols = {}
    for col in feat_cols:
        for w in windows:
            new_cols[f'{col}_rmean{w}'] = (
                df.groupby(group_col)[col]
                .transform(lambda x: x.rolling(w, min_periods=1).mean())
            )
            new_cols[f'{col}_slope{w}'] = (
                df.groupby(group_col)[col]
                .transform(lambda x: x.diff(w) / w)
                .fillna(0)
            )

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


# ── epoch 단위 집계 ───────────────────────────────────────────
def aggregate_epoch(df):
    feat_cols = get_feat_cols(df)
    agg_dict  = {c: 'median' for c in feat_cols}
    agg_dict['t_abs'] = 'max'

    if 'RUL' in df.columns:
        agg_dict['RUL'] = 'min'
    if 'life_pct' in df.columns:
        agg_dict['life_pct'] = 'max'
    if 'ttf' in df.columns:
        agg_dict['ttf'] = 'first'   # ← 추가: ttf는 상수값이라 first로

    group_cols = ['train_id', 'epoch'] \
                 if 'train_id' in df.columns \
                 else ['test_id', 'epoch']

    return df.groupby(group_cols).agg(agg_dict).reset_index()


# ── LOO 결과 시각화 ───────────────────────────────────────────
def plot_loo_results(loo_results, title, save_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=13)

    ids    = [r['val_id']   for r in loo_results]
    acts   = [r['act_rul']  for r in loo_results]
    preds  = [r['pred_rul'] for r in loo_results]
    scores = [r['score']    for r in loo_results]
    x      = np.arange(len(ids))

    ax = axes[0]
    ax.bar(x - 0.2, [a/3600 for a in acts],  0.4,
           label='실제 RUL', color='steelblue', alpha=0.8)
    ax.bar(x + 0.2, [p/3600 for p in preds], 0.4,
           label='예측 RUL', color='tomato',    alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Train{i}' for i in ids])
    ax.set_ylabel('RUL [시간]')
    ax.set_title('실제 vs 예측 RUL')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.35)

    ax = axes[1]
    ax.bar(x, scores, color='steelblue', alpha=0.8)
    ax.axhline(np.mean(scores), color='red', linestyle='--',
               label=f'평균={np.mean(scores):.4f}')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Train{i}' for i in ids])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Validation Score')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.35)

    plt.tight_layout()
    os.makedirs('../output', exist_ok=True)
    plt.savefig(f'../output/{save_name}.png', bbox_inches='tight')
    plt.show()
    print(f'평균 Score: {np.mean(scores):.4f}')
