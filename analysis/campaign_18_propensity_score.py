"""캠페인 18번 성향점수(p_score) 추정 — 로지스틱 회귀.

analysis/campaign_18_ps_variables.md 에서 확정한 6개 발행 전 변수로
"캠페인 18을 받을 확률(성향점수)"을 로지스틱 회귀로 추정한다.
아직 pipeline/ 파일로 묶지 않고, 이 스크립트 안에서 전처리→모델→저장을 순서대로 실행하며
각 단계의 행·열 수를 출력한다.

★ 아래 PREPROCESSING RULES 는 나중에 pipeline/prepare_data.py·estimate_effect.py 로
  그대로 옮길 수 있도록 이 스크립트의 유일한 "규칙 출처"로 유지한다. 규칙을 바꿀 때는
  여기 한 곳만 고치면 된다.

성향점수는 구매확률이나 캠페인 효과가 아니라 "캠페인 수신확률"이다(CLAUDE.md 분석 설계
규칙 4). 입력은 발행 전 변수 6개뿐이며 결과변수·캠페인 기간·캠페인 이후 정보는 넣지 않는다.

실행:
    .venv/bin/python analysis/campaign_18_propensity_score.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from campaign_18_groups import TARGET_CAMPAIGN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_FILE = PROJECT_ROOT / "outputs" / f"campaign_{TARGET_CAMPAIGN}" / "analysis_data.csv"

# =====================================================================
# PREPROCESSING RULES — pipeline 이관 시 이 블록을 그대로 옮긴다.
# =====================================================================
# 1) 성향점수 모델 입력변수 (campaign_18_ps_variables.md 확정 목록, 6개)
PS_FEATURES = [
    "recency", "pre_baskets", "pre_sales",
    "pre_target_baskets", "pre_coupon_redemptions", "pre_campaign_count",
]

# 2) 로그변환 규칙: 표본 왜도(skewness) > 1 인 변수에 log1p 적용.
#    (recency 7.04, pre_baskets 2.62, pre_sales 3.69, pre_target_baskets 2.01,
#     pre_coupon_redemptions 5.16 — 적용 / pre_campaign_count 0.84 — 미적용)
LOG1P_SKEW_THRESHOLD = 1.0

# 3) 표준화 규칙: 로그변환 후 z-score 표준화(평균 0, 표준편차 1).
#    평균·표준편차는 이 845가구 표본 전체로 계산한다(성향점수는 held-out 예측이 아니라
#    표본 내 캠페인 수신확률 추정이므로 train/test 분리가 필요 없다).
STANDARDIZE = True

# 4) 모델: 상수항 포함 로지스틱 회귀(statsmodels Logit), 정규화 없음.
MODEL_TYPE = "logit_no_regularization_with_intercept"
# =====================================================================


def banner(step: str, title: str) -> None:
    print("\n" + "=" * 92)
    print(f"[{step}] {title}")
    print("=" * 92)


def main() -> None:
    # ------------------------------------------------------------------
    banner("STEP 1", "분석표 불러오기 · 입력변수 선택")
    # ------------------------------------------------------------------
    df = pd.read_csv(ANALYSIS_FILE)
    print(f"  입력 파일: {ANALYSIS_FILE.relative_to(PROJECT_ROOT)}")
    print(f"  원본 분석표: {df.shape[0]}행 × {df.shape[1]}열")

    model_df = df[["household_key", "group", "treatment"] + PS_FEATURES].copy()
    print(f"  PS 입력 부분표: {model_df.shape[0]}행 × {model_df.shape[1]}열 "
          f"(household_key, group, treatment + 입력변수 {len(PS_FEATURES)}개)")

    n_missing = int(model_df[PS_FEATURES].isna().sum().sum())
    print(f"  입력변수 결측치 총합: {n_missing}건 (0이어야 함)")
    assert n_missing == 0, "결측치가 있으면 로지스틱 회귀 전에 처리 규칙을 먼저 정해야 한다."

    # ------------------------------------------------------------------
    banner("STEP 2", "전처리 1/2 — 로그변환 (왜도 기준)")
    # ------------------------------------------------------------------
    skew = model_df[PS_FEATURES].skew()
    log_cols = skew[skew.abs() > LOG1P_SKEW_THRESHOLD].index.tolist()
    print(f"  변수별 왜도:\n{skew.round(2).to_string()}")
    print(f"\n  |왜도| > {LOG1P_SKEW_THRESHOLD} 라서 log1p 적용: {log_cols}")
    print(f"  그대로 사용: {[c for c in PS_FEATURES if c not in log_cols]}")

    X = model_df[PS_FEATURES].copy()
    for col in log_cols:
        X[col] = np.log1p(X[col])
    X = X.rename(columns={c: f"log1p_{c}" for c in log_cols})
    print(f"\n  로그변환 후: {X.shape[0]}행 × {X.shape[1]}열 (열 이름: {list(X.columns)})")

    # ------------------------------------------------------------------
    banner("STEP 3", "전처리 2/2 — 표준화(z-score)")
    # ------------------------------------------------------------------
    means = X.mean()
    stds = X.std(ddof=0)
    X_std = (X - means) / stds
    print(f"  표준화 전 평균/표준편차:\n{pd.DataFrame({'평균': means, '표준편차': stds}).round(3).to_string()}")
    print(f"\n  표준화 후: {X_std.shape[0]}행 × {X_std.shape[1]}열")
    print(f"  표준화 후 평균(≈0)/표준편차(≈1) 확인:\n"
          f"{pd.DataFrame({'평균': X_std.mean(), '표준편차': X_std.std(ddof=0)}).round(3).to_string()}")

    # ------------------------------------------------------------------
    banner("STEP 4", "로지스틱 회귀 학습 — treatment ~ 발행 전 특성")
    # ------------------------------------------------------------------
    X_design = sm.add_constant(X_std)
    y = model_df["treatment"]
    print(f"  설계행렬(X): {X_design.shape[0]}행 × {X_design.shape[1]}열 (상수항 포함)")
    print(f"  목표변수(y): treatment, 1={int(y.sum())} / 0={int((1 - y).sum())}")

    logit_model = sm.Logit(y, X_design)
    result = logit_model.fit(disp=0)
    print()
    print(result.summary())

    print("\n  계수 부호·유의성 점검 (다중공선성으로 인한 이상 여부):")
    coef_table = pd.DataFrame({
        "계수": result.params.round(3),
        "p값": result.pvalues.round(3),
    })
    print(coef_table.to_string())

    # ------------------------------------------------------------------
    banner("STEP 5", "성향점수(p_score) 계산")
    # ------------------------------------------------------------------
    model_df["p_score"] = result.predict(X_design)
    print(f"  p_score 계산 완료: {model_df.shape[0]}행 (household_key당 1개)")
    print(f"  p_score 범위: [{model_df['p_score'].min():.4f}, {model_df['p_score'].max():.4f}]")
    n_out_of_range = int(((model_df["p_score"] < 0) | (model_df["p_score"] > 1)).sum())
    print(f"  [0,1] 범위를 벗어난 값: {n_out_of_range}건 (0이어야 함)")

    # ------------------------------------------------------------------
    banner("STEP 6", "집단별 p_score 분포")
    # ------------------------------------------------------------------
    dist = model_df.groupby("group")["p_score"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    pd.set_option("display.width", 200)
    print(dist.round(4).to_string())

    t_scores = model_df.loc[model_df["treatment"] == 1, "p_score"]
    c_scores = model_df.loc[model_df["treatment"] == 0, "p_score"]
    overlap_lo = max(t_scores.min(), c_scores.min())
    overlap_hi = min(t_scores.max(), c_scores.max())
    n_t_in = int(((t_scores >= overlap_lo) & (t_scores <= overlap_hi)).sum())
    n_c_in = int(((c_scores >= overlap_lo) & (c_scores <= overlap_hi)).sum())
    print(f"\n  공통지지영역(두 집단 p_score 범위의 교집합): [{overlap_lo:.4f}, {overlap_hi:.4f}]")
    print(f"    영역 안 처치 가구: {n_t_in}/{len(t_scores)} ({n_t_in/len(t_scores):.1%})")
    print(f"    영역 안 대조 가구: {n_c_in}/{len(c_scores)} ({n_c_in/len(c_scores):.1%})")
    print("  ※ 매칭·공통지지영역 트리밍은 이 단계에서 하지 않는다. 다음 단계에서 별도로 진행한다.")

    # ------------------------------------------------------------------
    banner("STEP 7", "분석표에 p_score 반영 · 저장")
    # ------------------------------------------------------------------
    before_cols = set(df.columns)
    df = df.merge(model_df[["household_key", "p_score"]], on="household_key", how="left")
    added_cols = set(df.columns) - before_cols
    print(f"  저장 전 분석표: {df.shape[0]}행 × {len(before_cols)}열")
    print(f"  추가된 열: {sorted(added_cols)}")
    print(f"  저장 후 분석표: {df.shape[0]}행 × {df.shape[1]}열")
    assert df["p_score"].isna().sum() == 0, "p_score 결측 발생 — merge 키 불일치 의심"

    df.to_csv(ANALYSIS_FILE, index=False)
    print(f"\n  저장 경로: {ANALYSIS_FILE.relative_to(PROJECT_ROOT)} (덮어씀, p_score 열 추가)")
    print("  이 스크립트는 파이프라인 파일이 아니다 — 처리 규칙은 파일 상단 PREPROCESSING RULES")
    print("  블록에 남겨 pipeline/prepare_data.py, pipeline/estimate_effect.py 작성 시 재사용한다.")


if __name__ == "__main__":
    main()
