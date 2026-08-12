"""캠페인 18번 최근접이웃(1:1, 비복원) 매칭 — caliper 0.1 / 0.2 / 0.3.

analysis/campaign_18_common_support.py 가 정의한 공통지지영역 [0.0941, 0.9808] 안의
818가구(처치 502 + 대조 316)만을 매칭 후보로 쓴다. 영역 밖 27가구(처치 8, 대조 19)는
애초에 후보에서 빠진다.

매칭 규칙
    - 거리 척도: logit(p_score) = log(p/(1-p))  (caliper 표준 관행 — 원점수가 아니라 로짓 척도)
    - caliper  = {0.1, 0.2, 0.3} × 공통지지영역 내 818가구 logit(p_score) 표준편차
    - 방식     : 1:1 최근접이웃, 대조 가구 비복원(재사용 금지)
    - 순서     : 처치 가구를 household_key 오름차순으로 처리(재현 가능한 결정적 순서).
                 동률 거리면 household_key가 작은 대조 가구를 선택한다.
    - caliper 안에 후보가 없으면 그 처치 가구는 미매칭으로 남긴다.

SMD 비교 규칙
    - 매칭 전/후 SMD 모두 같은 분모(845가구 전체 기준 pooled SD, 지난 균형표와 동일)를 쓴다.
      분모를 고정해야 매칭 후 SMD 변화가 "평균차이 축소"만을 반영하고 표본이 줄어들며
      분산이 변하는 효과와 섞이지 않는다(Stuart 2010 권장 방식).
    - 비교 대상 변수는 campaign_18_ps_variables.md 에서 확정한 성향점수 입력 6개.

산출: 터미널 출력만(파일 생성 없음). 매칭 결과 저장(matched_data.csv)은 caliper를
      확정한 다음 단계에서 별도로 진행한다.

실행:
    .venv/bin/python analysis/campaign_18_matching.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from campaign_18_groups import TARGET_CAMPAIGN
from campaign_18_propensity_score import PS_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_FILE = PROJECT_ROOT / "outputs" / f"campaign_{TARGET_CAMPAIGN}" / "analysis_data.csv"

CALIPER_MULTIPLIERS = [0.1, 0.2, 0.3]


def banner(step: str, title: str) -> None:
    print("\n" + "=" * 92)
    print(f"[{step}] {title}")
    print("=" * 92)


def compute_smd(df: pd.DataFrame, treat_col: str, cols: list[str], pooled_sd: pd.Series) -> pd.Series:
    """고정된 pooled_sd(매칭 전 845가구 기준)로 SMD를 계산한다."""
    t = df[df[treat_col] == 1]
    c = df[df[treat_col] == 0]
    diff = t[cols].mean() - c[cols].mean()
    return diff / pooled_sd[cols]


def nearest_neighbor_match(pool: pd.DataFrame, caliper: float) -> tuple[list[tuple[int, int]], list[int]]:
    """household_key 오름차순 처치 순서로 1:1 비복원 최근접이웃 매칭을 수행한다.

    반환: (매칭쌍 household_key 리스트[(처치, 대조), ...], 미매칭 처치 household_key 리스트)
    """
    treated = pool[pool["treatment"] == 1].sort_values("household_key")
    control = pool[pool["treatment"] == 0].sort_values("household_key")

    control_available = control.set_index("household_key")["logit_p"].to_dict()
    pairs: list[tuple[int, int]] = []
    unmatched: list[int] = []

    for _, trow in treated.iterrows():
        t_hh, t_logit = int(trow["household_key"]), trow["logit_p"]
        if not control_available:
            unmatched.append(t_hh)
            continue

        # 거리 계산 후 (거리, household_key)로 정렬해 동률을 결정적으로 처리
        candidates = sorted(
            ((abs(t_logit - c_logit), c_hh) for c_hh, c_logit in control_available.items())
        )
        best_dist, best_hh = candidates[0]

        if best_dist <= caliper:
            pairs.append((t_hh, best_hh))
            del control_available[best_hh]
        else:
            unmatched.append(t_hh)

    return pairs, unmatched


def main() -> None:
    # ------------------------------------------------------------------
    banner("STEP 1", "공통지지영역 내 매칭 후보 구성")
    # ------------------------------------------------------------------
    df = pd.read_csv(ANALYSIS_FILE)
    print(f"  입력: {ANALYSIS_FILE.relative_to(PROJECT_ROOT)}  shape={df.shape}")

    # 매칭 전(845가구 전체) 기준 pooled SD — SMD 분모를 매칭 전후 동일하게 고정
    t_all = df[df["treatment"] == 1]
    c_all = df[df["treatment"] == 0]
    pooled_sd = np.sqrt((t_all[PS_FEATURES].var(ddof=1) + c_all[PS_FEATURES].var(ddof=1)) / 2)

    t_min, t_max = t_all["p_score"].min(), t_all["p_score"].max()
    c_min, c_max = c_all["p_score"].min(), c_all["p_score"].max()
    lo, hi = max(t_min, c_min), min(t_max, c_max)
    print(f"  공통지지영역: [{lo:.4f}, {hi:.4f}] (campaign_18_common_support.py 와 동일 정의)")

    pool = df[(df["p_score"] >= lo) & (df["p_score"] <= hi)].copy()
    pool["logit_p"] = np.log(pool["p_score"] / (1 - pool["p_score"]))
    n_t_pool = int((pool["treatment"] == 1).sum())
    n_c_pool = int((pool["treatment"] == 0).sum())
    print(f"  매칭 후보: {len(pool)}가구 (처치 {n_t_pool} / 대조 {n_c_pool})  "
          f"— 영역 밖 처치 {len(t_all)-n_t_pool}, 대조 {len(c_all)-n_c_pool}는 후보에서 제외")

    logit_sd = pool["logit_p"].std(ddof=1)
    print(f"\n  logit(p_score) 표준편차(공통지지영역 {len(pool)}가구 기준): {logit_sd:.4f}")
    for m in CALIPER_MULTIPLIERS:
        print(f"    caliper {m} × SD = {m * logit_sd:.4f} (logit 척도)")

    # ------------------------------------------------------------------
    banner("STEP 2", "caliper별 매칭 실행")
    # ------------------------------------------------------------------
    rows = []
    matched_sets = {}

    for m in CALIPER_MULTIPLIERS:
        caliper = m * logit_sd
        pairs, unmatched_t = nearest_neighbor_match(pool, caliper)
        matched_t_hh = [p[0] for p in pairs]
        matched_c_hh = [p[1] for p in pairs]
        matched_sets[m] = (matched_t_hh, matched_c_hh)

        n_pairs = len(pairs)
        match_rate_of_510 = n_pairs / len(t_all)
        match_rate_of_pool = n_pairs / n_t_pool
        n_unmatched_treated_total = len(t_all) - n_pairs  # 영역 밖 8 + 영역 안인데 caliper 밖
        n_unused_control_total = len(c_all) - n_pairs

        matched_df = df[df["household_key"].isin(matched_t_hh + matched_c_hh)].copy()
        smd_after = compute_smd(matched_df, "treatment", PS_FEATURES, pooled_sd)
        max_abs_smd = smd_after.abs().max()
        worst_var = smd_after.abs().idxmax()

        print(f"\n  --- caliper = {m} × SD = {caliper:.4f} ---")
        print(f"    매칭된 쌍 수            : {n_pairs}")
        print(f"    처치 매칭률(전체 510 기준): {match_rate_of_510:.1%}")
        print(f"    처치 매칭률(공통지지 {n_t_pool} 기준): {match_rate_of_pool:.1%}")
        print(f"    미매칭 처치 가구(전체 기준): {n_unmatched_treated_total} "
              f"(영역 밖 {len(t_all)-n_t_pool}건 포함)")
        print(f"    미사용 대조 가구(전체 기준): {n_unused_control_total} "
              f"(영역 밖 {len(c_all)-n_c_pool}건 포함)")
        print(f"    매칭 후 SMD (변수별):\n{smd_after.round(3).to_string()}")
        print(f"    매칭 후 최대 |SMD|      : {max_abs_smd:.3f}  ({worst_var})")

        rows.append({
            "caliper": f"{m} × SD",
            "caliper_logit": round(caliper, 4),
            "매칭쌍수": n_pairs,
            "처치매칭률(510기준)": round(match_rate_of_510 * 100, 1),
            "처치매칭률(공통지지기준)": round(match_rate_of_pool * 100, 1),
            "미매칭처치": n_unmatched_treated_total,
            "미사용대조": n_unused_control_total,
            "매칭후최대|SMD|": round(max_abs_smd, 3),
            "최대SMD변수": worst_var,
        })

    # ------------------------------------------------------------------
    banner("STEP 3", "caliper별 비교표")
    # ------------------------------------------------------------------
    compare = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(compare.to_string(index=False))

    print()
    print("  매칭 전(845가구) 최대 |SMD| 참고값:")
    smd_before = compute_smd(df, "treatment", PS_FEATURES, pooled_sd)
    print(f"    {smd_before.abs().max():.3f}  ({smd_before.abs().idxmax()})")

    # ------------------------------------------------------------------
    banner("STEP 4", "요약")
    # ------------------------------------------------------------------
    print("  caliper가 좁을수록(0.1×SD) 매칭 품질(SMD)은 좋아지지만 매칭쌍 수는 줄어들고,")
    print("  caliper가 넓을수록(0.3×SD) 쌍 수는 늘지만 균형이 나빠질 수 있다 — 위 표에서 대조 확인.")
    print("  이 단계에서는 caliper를 확정하지 않았다. matched_data.csv 저장은 다음 단계에서 진행한다.")


if __name__ == "__main__":
    main()
