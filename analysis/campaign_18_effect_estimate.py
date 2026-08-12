"""캠페인 18번 효과 추정 — 매칭 전 단순차이 vs 매칭 후 차이 (95% CI 포함).

caliper 0.2×SD(logit) 매칭 결과(analysis/campaign_18_matching.py 결과 중 채택안)를
확정해 matched_data.csv 로 저장하고, 이를 이용해 주요·보조 결과의 처치-대조 차이를
매칭 전(단순 비교)과 매칭 후(1:1 짝 비교)로 나란히 계산한다.

주요 결과(캠페인 대상 상품, CLAUDE.md 분석 설계 규칙 5): target_purchase, target_sales, target_quantity
보조 결과(전체 상품)                                : any_purchase, total_sales, baskets

신뢰구간 계산 규칙
    - 매칭 전(단순 비교, 845가구, 독립표본) : Welch t-검정 기반 CI
      SE = sqrt(var_t/n_t + var_c/n_c), df = Welch-Satterthwaite, CI = diff ± t(df)*SE
    - 매칭 후(228쌍, 대응표본)              : 쌍별 차이(처치-대조)의 평균에 대한 대응표본 CI
      SE = sd(차이)/sqrt(n_pairs), df = n_pairs-1, CI = mean(차이) ± t(df)*SE
    - 두 CI 계산 방식이 다른 이유: 매칭 후에는 짝이 지어져 있어 표본이 독립이 아니다.
      대응표본으로 계산해야 짝짓기로 줄어든 분산을 올바르게 반영한다.

이 단계의 결과는 매칭 추정치이며, 관찰되지 않은 교란요인을 통제했다는 보장은 없다
(CLAUDE.md 품질과 해석 규칙 2·6). "효과"라는 표현은 매칭 기반 추정치라는 전제 하에서만 쓴다.

산출:
    outputs/campaign_18/matched_data.csv  (228쌍 = 456행)
    outputs/campaign_18/results.json      (효과 추정치 + 품질 진단)

실행:
    .venv/bin/python analysis/campaign_18_effect_estimate.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from campaign_18_build_analysis_table import PRIMARY_COLUMNS, SECONDARY_COLUMNS
from campaign_18_groups import TARGET_CAMPAIGN
from campaign_18_matching import CALIPER_MULTIPLIERS, nearest_neighbor_match
from campaign_18_propensity_score import PS_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / f"campaign_{TARGET_CAMPAIGN}"
ANALYSIS_FILE = OUTPUT_DIR / "analysis_data.csv"
MATCHED_FILE = OUTPUT_DIR / "matched_data.csv"
RESULTS_FILE = OUTPUT_DIR / "results.json"

CHOSEN_CALIPER_MULTIPLIER = 0.2  # 지난 단계 비교표에서 채택: 최대|SMD| 최솟값이면서 표본 손실도 중간

LABELS = {
    "target_purchase": "대상 상품 구매율", "target_sales": "대상 상품 구매금액", "target_quantity": "대상 상품 구매수량",
    "any_purchase": "전체 구매율", "total_sales": "전체 구매금액", "baskets": "장바구니 수",
}


def banner(step: str, title: str) -> None:
    print("\n" + "=" * 92)
    print(f"[{step}] {title}")
    print("=" * 92)


def welch_ci(vals_t: pd.Series, vals_c: pd.Series, conf: float = 0.95):
    n_t, n_c = len(vals_t), len(vals_c)
    m_t, m_c = vals_t.mean(), vals_c.mean()
    v_t, v_c = vals_t.var(ddof=1), vals_c.var(ddof=1)
    diff = m_t - m_c
    se = np.sqrt(v_t / n_t + v_c / n_c)
    df = (v_t / n_t + v_c / n_c) ** 2 / ((v_t / n_t) ** 2 / (n_t - 1) + (v_c / n_c) ** 2 / (n_c - 1))
    crit = stats.t.ppf(1 - (1 - conf) / 2, df)
    return diff, se, (diff - crit * se, diff + crit * se), df


def paired_ci(diffs: pd.Series, conf: float = 0.95):
    n = len(diffs)
    mean_d = diffs.mean()
    se = diffs.std(ddof=1) / np.sqrt(n)
    df = n - 1
    crit = stats.t.ppf(1 - (1 - conf) / 2, df)
    return mean_d, se, (mean_d - crit * se, mean_d + crit * se), df


def main() -> None:
    # ------------------------------------------------------------------
    banner("STEP 1", "caliper 0.2×SD 매칭 재현 및 확정")
    # ------------------------------------------------------------------
    assert CHOSEN_CALIPER_MULTIPLIER in CALIPER_MULTIPLIERS, "campaign_18_matching.py 의 후보 목록과 다르다"

    df = pd.read_csv(ANALYSIS_FILE)
    print(f"  입력: {ANALYSIS_FILE.relative_to(PROJECT_ROOT)}  shape={df.shape}")

    t_all, c_all = df[df["treatment"] == 1], df[df["treatment"] == 0]
    lo = max(t_all["p_score"].min(), c_all["p_score"].min())
    hi = min(t_all["p_score"].max(), c_all["p_score"].max())
    pool = df[(df["p_score"] >= lo) & (df["p_score"] <= hi)].copy()
    pool["logit_p"] = np.log(pool["p_score"] / (1 - pool["p_score"]))
    logit_sd = pool["logit_p"].std(ddof=1)
    caliper = CHOSEN_CALIPER_MULTIPLIER * logit_sd

    pairs, unmatched_t = nearest_neighbor_match(pool, caliper)
    print(f"  공통지지영역 [{lo:.4f}, {hi:.4f}] 내 {len(pool)}가구에서 caliper={caliper:.4f}(logit)로 매칭")
    print(f"  매칭된 쌍 수: {len(pairs)}  (처치 매칭률 {len(pairs)/len(t_all):.1%}, 510가구 기준)")

    # ------------------------------------------------------------------
    banner("STEP 2", "matched_data.csv 저장")
    # ------------------------------------------------------------------
    match_id_map = {}
    for i, (t_hh, c_hh) in enumerate(pairs, start=1):
        match_id_map[t_hh] = i
        match_id_map[c_hh] = i

    matched_hh = list(match_id_map.keys())
    matched_df = df[df["household_key"].isin(matched_hh)].copy()
    matched_df["match_id"] = matched_df["household_key"].map(match_id_map)
    matched_df = matched_df.sort_values(["match_id", "treatment"], ascending=[True, False]).reset_index(drop=True)

    print(f"  matched_data.csv 행/열: {matched_df.shape[0]}행 × {matched_df.shape[1]}열 "
          f"({len(pairs)}쌍 × 2 = {len(pairs)*2}행 확인: {matched_df.shape[0] == len(pairs)*2})")
    assert matched_df["household_key"].duplicated().sum() == 0, "매칭 결과에 가구 중복"
    assert matched_df.groupby("match_id").size().eq(2).all(), "쌍이 아닌 match_id 존재"
    assert matched_df.groupby("match_id")["treatment"].sum().eq(1).all(), "쌍 안에 처치 1명이 아님"

    matched_df.to_csv(MATCHED_FILE, index=False)
    print(f"  저장: {MATCHED_FILE.relative_to(PROJECT_ROOT)}")

    # ------------------------------------------------------------------
    banner("STEP 3", "매칭 전 단순 비교 (845가구, 독립표본 Welch CI)")
    # ------------------------------------------------------------------
    unadjusted_rows = []
    for col in PRIMARY_COLUMNS + SECONDARY_COLUMNS:
        diff, se, ci, dof = welch_ci(t_all[col], c_all[col])
        unadjusted_rows.append({
            "변수": LABELS[col], "열": col,
            "처치평균": round(t_all[col].mean(), 3), "대조평균": round(c_all[col].mean(), 3),
            "차이": round(diff, 3), "SE": round(se, 4), "df": round(dof, 1),
            "95%CI_하한": round(ci[0], 3), "95%CI_상한": round(ci[1], 3),
            "0포함여부": "포함" if ci[0] <= 0 <= ci[1] else "미포함",
        })
    unadjusted = pd.DataFrame(unadjusted_rows)
    pd.set_option("display.width", 220)
    print(unadjusted.drop(columns=["열"]).to_string(index=False))

    # ------------------------------------------------------------------
    banner("STEP 4", "매칭 후 차이 (228쌍, 대응표본 CI)")
    # ------------------------------------------------------------------
    wide = matched_df.pivot(index="match_id", columns="treatment", values=PRIMARY_COLUMNS + SECONDARY_COLUMNS)
    print(f"  대응표본 재구성: {wide.shape[0]}쌍 × {wide.shape[1]}열 (변수 6개 × treatment 0/1)")

    matched_rows = []
    for col in PRIMARY_COLUMNS + SECONDARY_COLUMNS:
        vals_t, vals_c = wide[(col, 1)], wide[(col, 0)]
        pair_diff = vals_t - vals_c
        mean_d, se, ci, dof = paired_ci(pair_diff)
        matched_rows.append({
            "변수": LABELS[col], "열": col,
            "처치평균": round(vals_t.mean(), 3), "대조평균": round(vals_c.mean(), 3),
            "차이": round(mean_d, 3), "SE": round(se, 4), "df": int(dof),
            "95%CI_하한": round(ci[0], 3), "95%CI_상한": round(ci[1], 3),
            "0포함여부": "포함" if ci[0] <= 0 <= ci[1] else "미포함",
        })
    matched_result = pd.DataFrame(matched_rows)
    print(matched_result.drop(columns=["열"]).to_string(index=False))

    # ------------------------------------------------------------------
    banner("STEP 5", "매칭 전후 나란히 비교")
    # ------------------------------------------------------------------
    combo = unadjusted[["변수", "열", "차이", "95%CI_하한", "95%CI_상한", "0포함여부"]].merge(
        matched_result[["열", "차이", "95%CI_하한", "95%CI_상한", "0포함여부"]],
        on="열", suffixes=("_매칭전", "_매칭후"),
    )
    is_primary = combo["열"].isin(PRIMARY_COLUMNS)
    combo.insert(0, "구분", np.where(is_primary, "주요(대상상품)", "보조(전체)"))
    print(combo.drop(columns=["열"]).to_string(index=False))

    # ------------------------------------------------------------------
    banner("STEP 6", "results.json 저장")
    # ------------------------------------------------------------------
    # 캠페인 기간은 campaign_18_groups.build_groups() 의 meta 에서 가져온다(재조회, 하드코딩 금지)
    from campaign_18_groups import build_groups
    _, meta = build_groups()

    results = {
        "campaign": TARGET_CAMPAIGN,
        "campaign_period_day": {"start": meta["start_day"], "end": meta["end_day"]},
        "unit_of_observation": "household_key",
    }
    results["sample"] = {
        "total_households": int(len(df)),
        "treated_total": int(len(t_all)), "control_total": int(len(c_all)),
        "common_support": {"lo": round(lo, 4), "hi": round(hi, 4), "n_pool": int(len(pool))},
        "matching": {
            "method": "1:1 nearest neighbor on logit(p_score), without replacement",
            "caliper_multiplier": CHOSEN_CALIPER_MULTIPLIER,
            "caliper_logit_sd": round(logit_sd, 4),
            "caliper_value_logit": round(caliper, 4),
            "n_pairs": int(len(pairs)),
            "treated_match_rate_of_510": round(len(pairs) / len(t_all), 4),
        },
    }
    pooled_sd = np.sqrt((t_all[PS_FEATURES].var(ddof=1) + c_all[PS_FEATURES].var(ddof=1)) / 2)
    smd_before = (t_all[PS_FEATURES].mean() - c_all[PS_FEATURES].mean()) / pooled_sd
    wide_ps = matched_df.pivot(index="match_id", columns="treatment", values=PS_FEATURES)
    smd_after = (wide_ps.xs(1, axis=1, level=1).mean() - wide_ps.xs(0, axis=1, level=1).mean()) / pooled_sd
    results["balance"] = {
        "ps_features": PS_FEATURES,
        "max_abs_smd_before": round(smd_before.abs().max(), 4),
        "max_abs_smd_after": round(smd_after.abs().max(), 4),
        "smd_denominator": "845가구 전체 pooled SD 고정(매칭 전후 동일)",
    }
    results["effects"] = {
        "note": "매칭 기반 추정치. 관찰되지 않은 교란요인 통제는 보장하지 않음. "
                "매칭전=845가구 독립표본 비교, 매칭후=228쌍 대응표본 비교. 금액·수량 단위는 원본(SALES_VALUE/QUANTITY) 그대로.",
        "primary_target_product": {},
        "secondary_overall": {},
    }
    for _, row in combo.iterrows():
        bucket = "primary_target_product" if row["열"] in PRIMARY_COLUMNS else "secondary_overall"
        results["effects"][bucket][row["열"]] = {
            "label": row["변수"],
            "unadjusted_diff": row["차이_매칭전"],
            "unadjusted_ci95": [row["95%CI_하한_매칭전"], row["95%CI_상한_매칭전"]],
            "matched_diff": row["차이_매칭후"],
            "matched_ci95": [row["95%CI_하한_매칭후"], row["95%CI_상한_매칭후"]],
        }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  저장: {RESULTS_FILE.relative_to(PROJECT_ROOT)}")

    # ------------------------------------------------------------------
    banner("STEP 7", "해석 시 유의사항")
    # ------------------------------------------------------------------
    print("  - 매칭 표본(228쌍)은 처치 510가구 중 44.7%만 대표한다 — 고활동 상위 처치가구는")
    print("    빠져 있다(campaign_18_matching.py 단계에서 확인). 효과는 이 하위집단에 한정된다.")
    print("  - recency 는 매칭 후에도 SMD 0.082로 완전히 균형을 이루지 못했다.")
    print("  - 관찰되지 않은 교란요인(가격민감도, 경쟁매장 이용 등)은 통제되지 않았다.")
    print("  - 위 수치는 매칭 추정치이며, 확정된 인과효과로 보고하지 않는다.")


if __name__ == "__main__":
    main()
