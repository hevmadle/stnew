"""캠페인 입력형 분석표 생성 — campaign_id, pre_days 를 받아 analysis_data.csv 를 만든다.

실행 순서(요청하신 순서 그대로):
    1. 캠페인 정보조회
    2. 집단 구성 (처치/대조)
    3. 사전 변수 계산 및 생성
    4. 결측치 처리
    5. 인코딩
    6. 결과변수 계산 (analysis_data.csv 에 포함, 성향점수 입력에는 쓰지 않음)

규칙은 전부 pipeline/rules.py 에서 가져온다 — 이 파일은 순서 제어와 로그 출력만 한다.

CLI:
    .venv/bin/python pipeline/prepare_data.py --campaign_id 18 --pre_days 586

반환(다른 스크립트에서 import 할 때): prepare(campaign_id, pre_days) -> (df, meta, gate_result)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import rules

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def banner(step: str, title: str) -> None:
    print("\n" + "=" * 92)
    print(f"[{step}] {title}")
    print("=" * 92)


def check_group_size_gate(meta: dict) -> dict:
    n_t, n_c = len(meta["treated"]), len(meta["control"])
    ok = n_t >= rules.MIN_GROUP_SIZE and n_c >= rules.MIN_GROUP_SIZE
    reason = None
    if not ok:
        parts = []
        if n_t < rules.MIN_GROUP_SIZE:
            parts.append(f"처치 {n_t}명 < {rules.MIN_GROUP_SIZE}")
        if n_c < rules.MIN_GROUP_SIZE:
            parts.append(f"대조 {n_c}명 < {rules.MIN_GROUP_SIZE}")
        reason = "표본 부족: " + ", ".join(parts)
    return {"gate": "표본 부족", "passed": ok, "reason": reason, "n_treated": n_t, "n_control": n_c}


def prepare(campaign_id: int, pre_days: int, verbose: bool = True) -> tuple[pd.DataFrame | None, dict, dict]:
    if verbose:
        banner("STEP 1", "캠페인 정보조회")
    info = rules.get_campaign_info(campaign_id)
    if verbose:
        print(f"  캠페인 {info['campaign_id']} ({info['description']}): "
              f"DAY {info['start_day']}~{info['end_day']} ({info['duration']}일)")

    if verbose:
        banner("STEP 2", "집단 구성 (처치/대조)")
    assignment, meta = rules.build_groups(campaign_id)
    n_t, n_c = len(meta["treated"]), len(meta["control"])
    print(f"  전체 모집단(거래 이력 보유): {len(meta['all_households']):,}가구")
    print(f"  겹치는 캠페인 {len(meta['overlapping_ids'])}개: {meta['overlapping_ids']}")
    print(f"  처치 {n_t}가구 / 대조 {n_c}가구 / 제외 {len(meta['excluded'])}가구")
    print(f"  분석표(assignment): {assignment.shape[0]}행 × {assignment.shape[1]}열")

    gate = check_group_size_gate(meta)
    if verbose:
        print(f"\n  [게이트: 표본 부족] {'통과' if gate['passed'] else '미통과'}"
              + (f" — {gate['reason']}" if not gate["passed"] else ""))
    if not gate["passed"]:
        return None, {**info, **{k: meta[k] for k in ["overlapping_ids", "treated", "control", "excluded"]}}, gate

    if verbose:
        banner("STEP 3", f"사전 변수 계산 및 생성 (pre_days={pre_days})")
    df, diag = rules.build_pre_features(assignment, meta, pre_days)
    print(f"  발행 전 관찰 구간: DAY {diag['pre_start']}~{diag['pre_end']} ({diag['window_len']}일)")
    print(f"  거래 없어 recency를 창 길이로 채운 가구: {int(df[rules.RECENCY_NO_PURCHASE_FLAG].sum())}가구")
    print(f"  발행 전 변수 계산 후: {df.shape[0]}행 × {df.shape[1]}열")

    if verbose:
        banner("STEP 4", "결측치 처리")
    n_missing = int(df[rules.ALL_PRE_FEATURE_COLUMNS].isna().sum().sum())
    print(f"  결측치 처리 규칙: 구매집계 열은 거래 없음=0, recency는 관찰창 길이로 대체(플래그 보존)")
    print(f"  처리 후 남은 결측치: {n_missing}건 (0이어야 함)")
    assert n_missing == 0, "결측치 처리 규칙 적용 후에도 결측치가 남았다."

    if verbose:
        banner("STEP 5", "인코딩")
    zero_var = rules.check_variable_variance(df, rules.PS_FEATURES)
    encoded_df, feature_cols_after = rules.encode_categoricals(df, rules.PS_FEATURES)
    cat_found = len(feature_cols_after) != len(rules.PS_FEATURES) or list(feature_cols_after) != rules.PS_FEATURES
    print(f"  확정 성향점수 변수 {len(rules.PS_FEATURES)}개 중 범주형: "
          f"{0 if not cat_found else len(feature_cols_after) - len(rules.PS_FEATURES)}개"
          f" — {'인코딩 불필요(전부 연속형)' if not cat_found else '원-핫 인코딩 적용'}")
    if zero_var:
        print(f"  ⚠ 분산 0인 변수(이 캠페인에서 변별력 없음): {zero_var}")
    df = encoded_df

    if verbose:
        banner("STEP 6", "결과변수 계산")
    df = rules.add_outcome_variables(df, meta, diag["tx_scope"], diag["target_products"])
    print(f"  주요 결과: {rules.PRIMARY_COLUMNS} / 보조 결과: {rules.SECONDARY_COLUMNS}")
    print(f"  최종 analysis_data: {df.shape[0]}행 × {df.shape[1]}열")

    meta_out = {
        **info,
        "pre_days": pre_days,
        "overlapping_ids": meta["overlapping_ids"],
        "n_treated": n_t, "n_control": n_c, "n_excluded": len(meta["excluded"]),
        "pre_start": diag["pre_start"], "pre_end": diag["pre_end"],
        "zero_variance_features": zero_var,
    }
    return df, meta_out, gate


def main() -> None:
    parser = argparse.ArgumentParser(description="캠페인 분석표(analysis_data.csv) 생성")
    parser.add_argument("--campaign_id", type=int, required=True)
    parser.add_argument("--pre_days", type=int, required=True, help="발행 전 관찰 기간(일)")
    args = parser.parse_args()

    df, meta, gate = prepare(args.campaign_id, args.pre_days)

    if df is None:
        print(f"\n중단: {gate['reason']}")
        raise SystemExit(1)

    out_dir = PROJECT_ROOT / "outputs" / f"campaign_{args.campaign_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "analysis_data.csv"
    df.to_csv(out_path, index=False)
    rules.save_run_config(args.campaign_id, args.pre_days)
    print(f"\n저장: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
