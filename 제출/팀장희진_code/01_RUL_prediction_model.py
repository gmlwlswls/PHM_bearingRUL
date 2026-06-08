# ============================================================
# 01_RUL_prediction_model.py
# KSPHM-KIMM 베어링 RUL 예측 파이프라인
#
# 실행 순서:
#   Step 1. RUL 라벨링 (TDMS → _labeled.tdms)
#   Step 2. 피처 엔지니어링 + RPM 예측 (TDMS → CSV)
#   Step 3. RUL 예측 모델 학습 + 성능 확인 + Test 예측
#
# 입력 경로: /data/Train/, /data/Test/
# 출력 경로: /data/output/
#
# OS: Windows 11 / Python 3.11.9
# 라이브러리: numpy, pandas, scipy, xgboost, nptdms, openpyxl
# ============================================================

import numpy as np
import pandas as pd
import os, glob, re
from xgboost import XGBRegressor
from compiled_utils import (
    # 유틸리티
    fix_seed, calc_score, calc_sample_weights,
    # 경로/상수
    TRAIN_PATH, TEST_PATH, OUTPUT_PATH,
    SEED, BEST_ALPHA, TRAIN_CUTOFFS, VALID_CUTOFFS, XGB_PARAMS,
    # Step 1: RUL 라벨링
    label_trainset,
    # Step 2: 피처 엔지니어링
    train_rpm_classifier,
    build_train_summary, build_test_summary,
    # Step 3: 모델링
    load_train_data, load_test_data,
    add_trend_features, make_cutoff_samples,
    train_rul_models,
)

if __name__ == '__main__':

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, 'weights'), exist_ok=True)
    fix_seed()

    # ──────────────────────────────────────────────────────────
    # Step 1. RUL 라벨링
    # TDMS + Operation CSV → TTF/RUL 라벨링 → _labeled.tdms
    # ──────────────────────────────────────────────────────────
    print('\n' + '='*55)
    print('Step 1. RUL 라벨링')
    print('='*55)
    label_trainset()

    # ──────────────────────────────────────────────────────────
    # Step 2. 피처 엔지니어링 + RPM 예측
    # _labeled.tdms → 1초 단위 피처 CSV
    # ──────────────────────────────────────────────────────────
    print('\n' + '='*55)
    print('Step 2. 피처 엔지니어링 + RPM 예측 모델')
    print('='*55)

    rpm_model = train_rpm_classifier()

    print('\n=== Train 피처 CSV 생성 ===')
    for t in [1, 2, 3, 4]:
        build_train_summary(t, rpm_model)

    print('\n=== Test 피처 CSV 생성 ===')
    test_folders = glob.glob(os.path.join(TEST_PATH, 'Test*'))
    test_nums    = sorted([
        int(re.search(r'Test(\d+)', os.path.basename(f)).group(1))
        for f in test_folders
        if os.path.isdir(f) and
           re.search(r'Test(\d+)', os.path.basename(f))
    ])
    for t in test_nums:
        build_test_summary(t, rpm_model)

    # ──────────────────────────────────────────────────────────
    # Step 3. RUL 예측 모델 학습 + 성능 확인 + Test 예측
    #
    # 모델 구조:
    #   model_ttf: XGBoost + TTF 직접 예측
    #     - TRAIN_CUTOFFS [0.1,0.3,0.4,0.6,0.7,0.9] 시점의
    #       마지막 1초 피처를 학습 샘플로 사용
    #     - 각 구간(초기/중기/말기)을 균형있게 대표
    #     - 말기 가중치: w = cutoff² (정규화)
    #
    #   model_log: XGBoost + log1p(RUL) 예측
    #     - 동일한 cutoff 기반 24샘플
    #     - 로그 변환으로 Train 간 RUL 스케일 차이 완화
    #     - Train3 LOO 성능 개선 (0.12 → 0.43)
    #
    # 최종 예측:
    #   pred_RUL = BEST_ALPHA × expm1(model_log)
    #            + (1-BEST_ALPHA) × (model_ttf - t_abs)
    #   BEST_ALPHA = 0.5
    #     (LOO 기반 균형 점수로 결정:
    #      LOO평균×0.3 + LOO_Train3×0.3 + Valid평균×0.4)
    #
    # 최종 성능:
    #   Valid 평균: 0.6472
    #   LOO 평균:   0.5711
    #   Train3 LOO: 0.4346
    # ──────────────────────────────────────────────────────────
    print('\n' + '='*55)
    print('Step 3. RUL 예측 모델 학습 + Test 예측')
    print('='*55)

    df_raw  = load_train_data()
    df_feat = add_trend_features(df_raw, windows=[5, 10],
                                  group_col='train_id')
    df_feat = df_feat.copy().fillna(0)

    # ── Valid 성능 확인 ───────────────────────────────────────
    df_train_s = make_cutoff_samples(df_feat, TRAIN_CUTOFFS)
    df_valid_s = make_cutoff_samples(df_feat, VALID_CUTOFFS)
    feat_cols  = [c for c in df_train_s.columns
                  if c not in ['train_id', 'cutoff', 'ttf',
                                'RUL', 't_abs']]

    print(f'\n피처 수: {len(feat_cols)}개')
    print(f'Train 샘플: {len(df_train_s)}개 '
          f'(6 cutoff × 4 Train) | '
          f'Valid 샘플: {len(df_valid_s)}개')

    X_train = df_train_s[feat_cols].values.astype(np.float32)
    X_valid = df_valid_s[feat_cols].values.astype(np.float32)
    weights = calc_sample_weights(df_train_s['cutoff'].values)

    fix_seed()
    m_ttf = XGBRegressor(**XGB_PARAMS)
    m_ttf.fit(X_train,
              df_train_s['ttf'].values.astype(np.float32),
              sample_weight=weights)

    fix_seed()
    m_log = XGBRegressor(**XGB_PARAMS)
    m_log.fit(X_train,
              np.log1p(df_train_s['RUL'].values).astype(np.float32),
              sample_weight=weights)

    pred_ttf_v = m_ttf.predict(X_valid).astype(np.float32)
    pred_log_v = m_log.predict(X_valid).astype(np.float32)
    rul_ttf_v  = np.maximum(pred_ttf_v - df_valid_s['t_abs'].values, 0)
    rul_log_v  = np.maximum(np.expm1(pred_log_v), 0)
    pred_rul_v = BEST_ALPHA * rul_log_v + (1 - BEST_ALPHA) * rul_ttf_v
    act_rul_v  = df_valid_s['RUL'].values

    valid_scores = [calc_score(a, p)
                    for a, p in zip(act_rul_v, pred_rul_v)]

    print('\n=== Valid 성능 (cutoff별) ===')
    for co in VALID_CUTOFFS:
        mask = df_valid_s['cutoff'].values == co
        s_co = float(np.mean(
            [valid_scores[i] for i in range(len(valid_scores))
             if mask[i]]))
        print(f'  cutoff={int(co*100)}% | Score={s_co:.4f}')
    print(f'  평균 Valid Score: {np.mean(valid_scores):.4f}')

    # ── LOO 성능 확인 ─────────────────────────────────────────
    print('\n=== LOO 검증 ===')
    loo_results = []

    for val_id in [1, 2, 3, 4]:
        train_ids = [t for t in [1,2,3,4] if t != val_id]
        df_tr_loo = df_feat[df_feat['train_id'].isin(train_ids)]
        df_va_loo = df_feat[df_feat['train_id'] == val_id]

        df_tr_s = make_cutoff_samples(
            df_tr_loo, TRAIN_CUTOFFS + VALID_CUTOFFS)
        df_va_s = make_cutoff_samples(df_va_loo, VALID_CUTOFFS)

        X_tr = df_tr_s[feat_cols].values.astype(np.float32)
        X_va = df_va_s[feat_cols].values.astype(np.float32)
        w_tr = calc_sample_weights(df_tr_s['cutoff'].values)

        fix_seed()
        lt = XGBRegressor(**XGB_PARAMS)
        lt.fit(X_tr,
               df_tr_s['ttf'].values.astype(np.float32),
               sample_weight=w_tr)

        fix_seed()
        ll = XGBRegressor(**XGB_PARAMS)
        ll.fit(X_tr,
               np.log1p(df_tr_s['RUL'].values).astype(np.float32),
               sample_weight=w_tr)

        rul_ttf_l = np.maximum(
            lt.predict(X_va).astype(np.float32) -
            df_va_s['t_abs'].values, 0)
        rul_log_l = np.maximum(
            np.expm1(ll.predict(X_va).astype(np.float32)), 0)
        pred_l    = (BEST_ALPHA * rul_log_l +
                     (1 - BEST_ALPHA) * rul_ttf_l)

        scores_l = [calc_score(a, p)
                    for a, p in zip(df_va_s['RUL'].values, pred_l)]
        avg_l    = float(np.mean(scores_l))

        print(f'  Val=Train{val_id} | Score={avg_l:.4f}')
        loo_results.append({'val_id': val_id, 'score': avg_l})

    loo_avg    = float(np.mean([r['score'] for r in loo_results]))
    train3_loo = [r['score'] for r in loo_results
                  if r['val_id'] == 3][0]
    print(f'\n  LOO 평균 Score:   {loo_avg:.4f}')
    print(f'  Train3 LOO Score: {train3_loo:.4f}')

    # ── 최종 모델 학습 (TRAIN + VALID 전체 cutoff) ───────────
    # Valid cutoff도 포함해 최대한 많은 구간의 데이터로 학습
    model_ttf, model_log, feat_cols = train_rul_models(df_feat)
    print(f'\n최종 모델 학습 완료 | 피처 수: {len(feat_cols)}개')

    # 모델 weight 저장
    model_ttf.save_model(
        os.path.join(OUTPUT_PATH, 'weights', 'model_ttf.json'))
    model_log.save_model(
        os.path.join(OUTPUT_PATH, 'weights', 'model_log.json'))
    print(f'모델 weight 저장: {OUTPUT_PATH}/weights/')

    # ── Test 예측 ─────────────────────────────────────────────
    df_test = load_test_data()
    if df_test is not None:
        df_test = add_trend_features(df_test, windows=[5, 10],
                                      group_col='test_id')
        df_test = df_test.copy().fillna(0)

        results = []
        for tid in sorted(df_test['test_id'].unique()):
            df_t     = df_test[df_test['test_id'] == tid]
            last_row = df_t.loc[df_t['t_abs'].idxmax()]
            t_abs    = float(last_row['t_abs'])

            feat_vals = np.array([
                last_row[c] if c in last_row.index else 0.0
                for c in feat_cols
            ]).reshape(1, -1).astype(np.float32)

            rul_ttf  = max(
                float(model_ttf.predict(feat_vals)[0]) - t_abs, 0)
            rul_log  = max(
                float(np.expm1(model_log.predict(feat_vals)[0])), 0)
            pred_rul = (BEST_ALPHA * rul_log +
                        (1 - BEST_ALPHA) * rul_ttf)

            results.append({'File': f'Test{tid}',
                             'RUL_Score': round(pred_rul, 2)})
            print(f'  Test{tid} | ttf={rul_ttf:.0f}s | '
                  f'log={rul_log:.0f}s | '
                  f'앙상블={pred_rul:.0f}s ({pred_rul/3600:.2f}h)')

        result_df = pd.DataFrame(results)
        save_path = os.path.join(OUTPUT_PATH, '팀장희진_validation.xlsx')
        result_df.to_excel(save_path, index=False)

        print('\n=== 최종 제출 파일 ===')
        print(result_df.to_string(index=False))
        print(f'\n저장 경로: {save_path}')

    # ── 최종 성능 요약 ────────────────────────────────────────
    print('\n' + '='*55)
    print('최종 성능 요약')
    print('='*55)
    print(f'  BEST_ALPHA:        {BEST_ALPHA}')
    print(f'  Valid 평균 Score:  {np.mean(valid_scores):.4f}')
    print(f'  LOO 평균 Score:    {loo_avg:.4f}')
    print(f'  Train3 LOO Score:  {train3_loo:.4f}')
    print('='*55)


    ######################
#     =======================================================
# 최종 성능 요약
# =======================================================
#   BEST_ALPHA:        0.5
#   Valid 평균 Score:  0.6642
#   LOO 평균 Score:    0.5672
#   Train3 LOO Score:  0.4250
# =======================================================

# ###기존 앙상블 모델
#  균형 점수:   0.5214
#   LOO 평균:   0.5549
#   Train3 LOO: 0.3773
#   Valid 평균: 0.6044