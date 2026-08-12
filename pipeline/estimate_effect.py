"""성향점수 → 공통영역 → 매칭 → 균형·효과 추정 — analysis_data.csv 를 입력으로 받는다.

실행 순서:
    7. 성향점수 추정 (로지스틱 회귀)
    8. 공통지지영역 확인            ← 게이트: 공통영역 부족
    9. 매칭 (caliper 0.1/0.2/0.3 자동 선택)  ← 게이트: 매칭 표본 부족
    10. 균형 확인                   ← 게이트: 잔여 불균형 (중단하지 않고 상태만 표시)
    11. 효과 추정 (매칭 전/후, 95% CI)

게이트를 하나라도 못 넘으면 그 지점에서 멈추고 상태(status)와 이유(reason)를 반환한다.
잔여 불균형 게이트만 예외 — CLAUDE.md 규칙대로 "효과를 확정하지 않고 상태와 이유를
반환"하되, 이미 계산된 수치는 참고용으로 남긴다(중단이 아니라 경고).

CLI:
    .venv/bin/python pipeline/estimate_effect.py --campaign_id 18
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import rules

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def banner(step: str, title: str) -> None:
    print("\n" + "=" * 92)
    print(f"[{step}] {title}")
    print("=" * 92)


def estimate(campaign_id: int, verbose: bool = True) -> dict:
    out_dir = PROJECT_ROOT / "outputs" / f"campaign_{campaign_id}"
    analysis_file = out_dir / "analysis_data.csv"
    if not analysis_file.exists():
        return {"status": "중단", "reason": f"{analysis_file} 없음 — prepare_data.py 를 먼저 실행하라"}

    df = pd.read_csv(analysis_file)
    if verbose:
        banner("STEP 7", "성향점수 추정 (로지스틱 회귀)")
        print(f"  입력: {analysis_file.relative_to(PROJECT_ROOT)}  shape={df.shape}")

    zero_var = rules.check_variable_variance(df, rules.PS_FEATURES)
    if zero_var:
        print(f"  ⚠ 분산 0 변수 제외하고 학습: {zero_var}")
    feature_cols = [c for c in rules.PS_FEATURES if c not in zero_var]

    p_score, ps_info = rules.fit_propensity_score(df, feature_cols)
    df["p_score"] = p_score
    df.to_csv(analysis_file, index=False)  # p_score를 analysis_data.csv에 영구 저장 (대시보드 재사용)
    if verbose:
        print(f"  로그변환 적용 변수: {ps_info['log_cols']}")
        print(f"  p_score 범위: [{df['p_score'].min():.4f}, {df['p_score'].max():.4f}]")
        print(f"  유의(p<0.05) 변수: {[v for v in ps_info['pvalues'].index if ps_info['pvalues'][v] < 0.05 and v != 'const']}")

    # ------------------------------------------------------------------
    banner("STEP 8", "공통지지영역 확인")
    # ------------------------------------------------------------------
    t_all, c_all = df[df["treatment"] == 1], df[df["treatment"] == 0]
    lo, hi = rules.common_support(df)
    pool = df[(df["p_score"] >= lo) & (df["p_score"] <= hi)].copy()
    pool["logit_p"] = np.log(pool["p_score"] / (1 - pool["p_score"]))
    n_t_pool = int((pool["treatment"] == 1).sum())
    n_c_pool = int((pool["treatment"] == 0).sum())
    print(f"  공통지지영역: [{lo:.4f}, {hi:.4f}]  ({'정상' if hi > lo else '퇴화(구간 없음)'})")
    print(f"  영역 내 처치 {n_t_pool} / 대조 {n_c_pool}  (영역 밖 처치 {len(t_all)-n_t_pool}, 대조 {len(c_all)-n_c_pool})")

    support_ok = hi > lo and n_t_pool >= rules.MIN_SUPPORT_SIZE and n_c_pool >= rules.MIN_SUPPORT_SIZE
    if not support_ok:
        reasons = []
        if hi <= lo:
            reasons.append("공통지지영역이 퇴화(하한>=상한)")
        if n_t_pool < rules.MIN_SUPPORT_SIZE:
            reasons.append(f"영역 내 처치 {n_t_pool}명 < {rules.MIN_SUPPORT_SIZE}")
        if n_c_pool < rules.MIN_SUPPORT_SIZE:
            reasons.append(f"영역 내 대조 {n_c_pool}명 < {rules.MIN_SUPPORT_SIZE}")
        reason = "공통영역 부족: " + ", ".join(reasons)
        print(f"\n  [게이트: 공통영역 부족] 미통과 — {reason}")
        return {
            "status": "중단", "reason": reason, "campaign_id": campaign_id, "stage_reached": "공통지지영역 확인",
            "sample": {
                "treated_total": int(len(t_all)), "control_total": int(len(c_all)),
                "common_support": {"lo": round(lo, 4), "hi": round(hi, 4),
                                    "n_treated_in_support": n_t_pool, "n_control_in_support": n_c_pool},
            },
        }
    print(f"  [게이트: 공통영역 부족] 통과")

    # ------------------------------------------------------------------
    banner("STEP 9", "매칭 (caliper 0.1/0.2/0.3 자동 선택)")
    # ------------------------------------------------------------------
    pooled_sd = np.sqrt((t_all[feature_cols].var(ddof=1) + c_all[feature_cols].var(ddof=1)) / 2)
    selection = rules.select_caliper(pool, feature_cols, pooled_sd)
    chosen = selection["chosen"]

    print(f"  logit(p_score) 표준편차: {selection['logit_sd']:.4f}")
    for cand in selection["all_candidates"]:
        mark = " ← 선택" if cand is chosen else ""
        max_smd_str = f"{cand['max_smd']:.3f}" if np.isfinite(cand["max_smd"]) else "매칭 0건"
        print(f"    {cand['multiplier']}×SD (caliper={cand['caliper']:.4f}): "
              f"{len(cand['pairs'])}쌍, 최대|SMD|={max_smd_str}{mark}")

    pairs = chosen["pairs"]
    n_pairs = len(pairs)
    print(f"\n  채택: {chosen['multiplier']}×SD → {n_pairs}쌍 "
          f"(처치 매칭률 {n_pairs/len(t_all):.1%}, {len(t_all)}가구 기준)")

    matched_ok = n_pairs >= rules.MIN_MATCHED_PAIRS
    if not matched_ok:
        reason = f"매칭 표본 부족: {n_pairs}쌍 < {rules.MIN_MATCHED_PAIRS}"
        print(f"\n  [게이트: 매칭 표본 부족] 미통과 — {reason}")
        return {
            "status": "중단", "reason": reason, "campaign_id": campaign_id, "stage_reached": "매칭",
            "sample": {
                "treated_total": int(len(t_all)), "control_total": int(len(c_all)),
                "common_support": {"lo": round(lo, 4), "hi": round(hi, 4),
                                    "n_treated_in_support": n_t_pool, "n_control_in_support": n_c_pool},
                "matching": {"caliper_multiplier": chosen["multiplier"], "n_pairs": n_pairs,
                             "treated_match_rate": round(n_pairs / len(t_all), 4)},
            },
        }
    print(f"  [게이트: 매칭 표본 부족] 통과")

    match_id_map = {}
    for i, (t_hh, c_hh) in enumerate(pairs, start=1):
        match_id_map[t_hh] = i
        match_id_map[c_hh] = i
    matched_df = df[df["household_key"].isin(match_id_map.keys())].copy()
    matched_df["match_id"] = matched_df["household_key"].map(match_id_map)
    matched_df = matched_df.sort_values(["match_id", "treatment"], ascending=[True, False]).reset_index(drop=True)

    matched_file = out_dir / "matched_data.csv"
    matched_df.to_csv(matched_file, index=False)
    print(f"  저장: {matched_file.relative_to(PROJECT_ROOT)}  ({matched_df.shape[0]}행 × {matched_df.shape[1]}열)")

    # ------------------------------------------------------------------
    banner("STEP 10", "균형 확인 (잔여 불균형 게이트)")
    # ------------------------------------------------------------------
    smd_before = rules.compute_smd(df, feature_cols, pooled_sd)
    smd_after = rules.compute_smd(matched_df, feature_cols, pooled_sd)
    max_smd_before, max_smd_after = smd_before.abs().max(), smd_after.abs().max()
    print(f"  매칭 전 최대|SMD|: {max_smd_before:.3f}  ({smd_before.abs().idxmax()})")
    print(f"  매칭 후 최대|SMD|: {max_smd_after:.3f}  ({smd_after.abs().idxmax()})")

    balance_ok = max_smd_after < rules.SMD_BALANCE_THRESHOLD
    balance_reason = None
    if not balance_ok:
        balance_reason = f"잔여 불균형: 매칭 후 최대|SMD|={max_smd_after:.3f} >= {rules.SMD_BALANCE_THRESHOLD}"
        print(f"\n  [게이트: 잔여 불균형] 미통과(경고) — {balance_reason}")
        print("  → 중단하지 않고 계속 계산하되, 아래 효과는 '확정'이 아니라 '참고용'으로 표시한다.")
    else:
        print(f"  [게이트: 잔여 불균형] 통과")

    # ------------------------------------------------------------------
    banner("STEP 11", "효과 추정 (매칭 전/후, 95% CI)")
    # ------------------------------------------------------------------
    effects = {"primary_target_product": {}, "secondary_overall": {}}
    outcome_report_rows = []
    for col in rules.PRIMARY_COLUMNS + rules.SECONDARY_COLUMNS:
        u_diff, _, u_ci, _ = rules.welch_ci(t_all[col], c_all[col])

        wide = matched_df.pivot(index="match_id", columns="treatment", values=col)
        pair_diff = wide[1] - wide[0]
        m_diff, _, m_ci, _ = rules.paired_ci(pair_diff)

        bucket = "primary_target_product" if col in rules.PRIMARY_COLUMNS else "secondary_overall"
        effects[bucket][col] = {
            "unadjusted_diff": round(u_diff, 4), "unadjusted_ci95": [round(u_ci[0], 4), round(u_ci[1], 4)],
            "matched_diff": round(m_diff, 4), "matched_ci95": [round(m_ci[0], 4), round(m_ci[1], 4)],
        }
        outcome_report_rows.append((col, u_diff, u_ci, m_diff, m_ci))

    pd.set_option("display.width", 200)
    report = pd.DataFrame(
        [{"변수": c, "매칭전차이": round(ud, 3), "매칭전CI": f"[{uc[0]:.3f},{uc[1]:.3f}]",
          "매칭후차이": round(md, 3), "매칭후CI": f"[{mc[0]:.3f},{mc[1]:.3f}]"}
         for c, ud, uc, md, mc in outcome_report_rows]
    )
    print(report.to_string(index=False))

    status = "완료" if balance_ok else "완료(잔여 불균형 경고)"
    run_config = rules.load_run_config(campaign_id)

    results = {
        "campaign": campaign_id,
        "pre_days": run_config["pre_days"] if run_config else None,
        "status": status,
        "reason": balance_reason,
        "sample": {
            "treated_total": int(len(t_all)), "control_total": int(len(c_all)),
            "common_support": {"lo": round(lo, 4), "hi": round(hi, 4),
                                "n_treated_in_support": n_t_pool, "n_control_in_support": n_c_pool},
            "matching": {
                "method": "1:1 nearest neighbor on logit(p_score), without replacement",
                "caliper_multiplier": chosen["multiplier"], "caliper_value_logit": round(chosen["caliper"], 4),
                "n_pairs": n_pairs, "treated_match_rate": round(n_pairs / len(t_all), 4),
            },
        },
        "balance": {
            "ps_features": feature_cols,
            "smd_before": {k: round(v, 4) for k, v in smd_before.to_dict().items()},
            "smd_after": {k: round(v, 4) for k, v in smd_after.to_dict().items()},
            "max_abs_smd_before": round(max_smd_before, 4), "max_abs_smd_after": round(max_smd_after, 4),
            "threshold": rules.SMD_BALANCE_THRESHOLD, "passed": bool(balance_ok),
        },
        "effects": effects,
    }

    results_file = out_dir / "results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {results_file.relative_to(PROJECT_ROOT)}")
    print(f"최종 상태: {status}" + (f" — {balance_reason}" if balance_reason else ""))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="성향점수·매칭·효과 추정")
    parser.add_argument("--campaign_id", type=int, required=True)
    args = parser.parse_args()
    result = estimate(args.campaign_id)
    if result.get("status") == "중단":
        print(f"\n중단: {result['reason']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
