"""캠페인 분석 파이프라인 진입점 — campaign_id, pre_days 만 입력하면 끝까지 실행한다.

순서: 캠페인 정보조회 → 집단 구성 → 사전 변수 계산 → 결측치 처리 → 인코딩 →
      결과변수 계산 → 성향점수 → 공통영역 확인 → 매칭 → 균형 확인 → 효과 추정

게이트(표본 부족 / 공통영역 부족 / 매칭 표본 부족)를 통과하지 못하면 그 지점에서
멈추고 상태와 이유를 반환한다. 잔여 불균형은 중단하지 않고 상태에 경고로 남긴다.
규칙은 전부 pipeline/rules.py 에 고정되어 있고, 이 파일은 두 단계(prepare_data,
estimate_effect)를 순서대로 호출만 한다 — 캠페인 18에서 검증한 규칙을 바꾸지 않는다.

CLI:
    .venv/bin/python pipeline/run_pipeline.py --campaign_id 18 --pre_days 586
"""

from __future__ import annotations

import argparse
from pathlib import Path

import rules
from prepare_data import prepare
from estimate_effect import estimate

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run(campaign_id: int, pre_days: int) -> dict:
    print("#" * 92)
    print(f"# 캠페인 {campaign_id} 파이프라인 실행 (pre_days={pre_days})")
    print("#" * 92)

    df, meta, gate = prepare(campaign_id, pre_days)
    if df is None:
        status = {"status": "중단", "stage": "prepare_data", "reason": gate["reason"], "campaign_id": campaign_id}
        print(f"\n### 파이프라인 중단 (prepare_data 단계) ###\n사유: {gate['reason']}")
        return status

    out_dir = PROJECT_ROOT / "outputs" / f"campaign_{campaign_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "analysis_data.csv", index=False)
    rules.save_run_config(campaign_id, pre_days)
    print(f"\n저장: {(out_dir / 'analysis_data.csv').relative_to(PROJECT_ROOT)}")

    result = estimate(campaign_id)
    if result.get("status") == "중단":
        result["stage"] = "estimate_effect"
        print(f"\n### 파이프라인 중단 (estimate_effect 단계) ###\n사유: {result['reason']}")
        return result

    print("\n" + "#" * 92)
    print(f"# 캠페인 {campaign_id} 파이프라인 완료 — 최종 상태: {result['status']}")
    print("#" * 92)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="캠페인 분석 파이프라인 (prepare_data + estimate_effect)")
    parser.add_argument("--campaign_id", type=int, required=True)
    parser.add_argument("--pre_days", type=int, required=True, help="발행 전 관찰 기간(일)")
    args = parser.parse_args()
    run(args.campaign_id, args.pre_days)


if __name__ == "__main__":
    main()
