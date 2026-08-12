"""캠페인 효과 → 수익성 변환 계산.

데이터에서 추정한 값(대상 상품 고객당 증분매출과 95% 신뢰구간, 매칭 품질 상태)과
사용자가 입력하는 수익성 가정(매출총이익률, 쿠폰 할인비용, 예상 사용률, 발행 수, 운영비)을
분명히 분리한다 — 어떤 숫자가 관찰된 사실/모델 추정치이고 어떤 숫자가 사용자 가정인지
DataEstimate / ProfitAssumptions 두 클래스로 나눠 섞이지 않게 한다(CLAUDE.md 품질과
해석 규칙 6).

계산식
    총 증분매출        = 발행수 × 고객당 증분매출(데이터)
    총 증분 매출총이익  = 총 증분매출 × 매출총이익률(가정)
    총 쿠폰비용        = 발행수 × 예상사용률(가정) × 쿠폰당 할인비용(가정)
    총 비용            = 총 쿠폰비용 + 운영비(가정)
    순증분이익         = 총 증분 매출총이익 − 총 비용
    ROI               = 순증분이익 / 총비용
    손익분기 최대 쿠폰당 할인비용 = (총 증분 매출총이익 − 운영비) / (발행수 × 예상사용률)

95% CI가 있는 값(증분매출)은 점추정치뿐 아니라 CI 하한·상한으로도 같은 계산을 반복해
순증분이익·ROI의 불확실성 범위를 함께 보여준다.

효과 추정이 불확실(매칭 품질 게이트 미통과, CI가 0 포함 또는 전부 음수)하거나 점추정치가
음수이면 warnings 리스트에 경고를 담아 반환한다 — 어떤 시나리오도 "최적"이라고 확정하지
않는다(CLAUDE.md 품질과 해석 규칙 7).

CLI(터미널 확인용):
    .venv/bin/python app/profitability.py --campaign_id 18 --n_issued 510 \\
        --gross_margin_rate 0.3 --redemption_rate 0.4 --coupon_cost 2.0 --operating_cost 50000
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =========================================================================
# 데이터에서 추정한 값 — 사용자가 손댈 수 없는 사실/모델 추정치
# =========================================================================
@dataclass
class DataEstimate:
    campaign_id: int
    status: str                              # results.json 의 매칭 품질 상태
    outcome: str                              # 어떤 결과변수를 썼는지 (기본 target_sales)
    incremental_revenue_per_customer: float   # 매칭 후 차이 (household당, 원본 SALES_VALUE 단위)
    incremental_revenue_ci95: tuple[float, float]
    n_matched_pairs: int
    n_treated_total: int


# =========================================================================
# 사용자가 입력하는 수익성 가정 — 데이터에서 나온 값이 아니다
# =========================================================================
@dataclass
class ProfitAssumptions:
    n_issued: int                       # 발행 수
    gross_margin_rate: float            # 매출총이익률 (0~1)
    redemption_rate: float              # 예상 사용률 (0~1)
    coupon_cost_per_redemption: float   # 쿠폰당 할인비용
    operating_cost: float = 0.0         # 운영비(고정비)


@dataclass
class ProfitResult:
    data: DataEstimate
    assumptions: ProfitAssumptions
    total_incremental_revenue: float
    total_incremental_revenue_ci95: tuple[float, float]
    total_gross_profit: float
    total_gross_profit_ci95: tuple[float, float]
    total_coupon_cost: float
    total_cost: float
    net_incremental_profit: float
    net_incremental_profit_ci95: tuple[float, float]
    roi: float | None
    roi_ci95: tuple[float | None, float | None]
    breakeven_max_coupon_cost: float
    warnings: list[str] = field(default_factory=list)


def load_data_estimate(campaign_id: int, outcome: str = "target_sales") -> DataEstimate:
    """pipeline/estimate_effect.py 가 저장한 results.json 에서 데이터 추정치만 읽는다.

    이 함수는 계산을 하지 않는다 — 사실(추정치)을 그대로 옮겨올 뿐이다.
    """
    results_path = PROJECT_ROOT / "outputs" / f"campaign_{campaign_id}" / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"{results_path} 없음 — 먼저 pipeline/run_pipeline.py 를 실행하라.")
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    effect = results.get("effects", {}).get("primary_target_product", {}).get(outcome)
    if effect is None:
        raise KeyError(f"results.json에 effects.primary_target_product.{outcome} 이 없다.")

    return DataEstimate(
        campaign_id=campaign_id,
        status=results.get("status", "?"),
        outcome=outcome,
        incremental_revenue_per_customer=effect["matched_diff"],
        incremental_revenue_ci95=tuple(effect["matched_ci95"]),
        n_matched_pairs=results.get("sample", {}).get("matching", {}).get("n_pairs", 0),
        n_treated_total=results.get("sample", {}).get("treated_total", 0),
    )


def compute_profitability(data: DataEstimate, assumptions: ProfitAssumptions) -> ProfitResult:
    warnings: list[str] = []

    # --- 입력값 검증 --------------------------------------------------------
    if not (0 <= assumptions.gross_margin_rate <= 1):
        raise ValueError(f"gross_margin_rate는 0~1 사이여야 한다: {assumptions.gross_margin_rate}")
    if not (0 <= assumptions.redemption_rate <= 1):
        raise ValueError(f"redemption_rate는 0~1 사이여야 한다: {assumptions.redemption_rate}")
    if assumptions.n_issued <= 0:
        raise ValueError(f"n_issued는 양수여야 한다: {assumptions.n_issued}")
    if assumptions.coupon_cost_per_redemption < 0 or assumptions.operating_cost < 0:
        raise ValueError("coupon_cost_per_redemption·operating_cost는 음수일 수 없다.")

    # --- 효과 추정 신뢰도 경고 ------------------------------------------------
    if data.status != "완료":
        warnings.append(
            f"효과 추정이 불확실합니다 — 매칭 품질 상태: '{data.status}'. "
            "공통영역·매칭·잔여 불균형 게이트를 완전히 통과하지 못했으니 아래 수치는 참고용입니다."
        )

    lo, hi = data.incremental_revenue_ci95
    if data.incremental_revenue_per_customer <= 0:
        warnings.append(
            f"데이터 기반 고객당 증분매출 추정치가 0 이하입니다({data.incremental_revenue_per_customer:.2f}). "
            "이 캠페인이 대상 상품 매출을 늘렸다는 근거가 없어 아래 수익성 계산은 실질적 의미가 없을 수 있습니다."
        )
    elif lo <= 0 <= hi:
        warnings.append(
            f"증분매출 95% 신뢰구간 [{lo:.2f}, {hi:.2f}]이 0을 포함합니다 — 통계적으로 유의한 효과가 "
            "아닙니다. 점추정치로 계산한 수익성 수치를 확정된 값으로 보지 마세요."
        )
    if hi < 0:
        warnings.append(
            f"증분매출 95% 신뢰구간 [{lo:.2f}, {hi:.2f}]이 전부 음수입니다 — "
            "이 매칭 조건에서는 오히려 감소했을 가능성이 있습니다."
        )

    if data.n_treated_total > 0:
        ratio = assumptions.n_issued / data.n_treated_total
        if ratio >= 3 or ratio <= 1 / 3:
            warnings.append(
                f"발행 수({assumptions.n_issued:,})가 이번 분석의 처치 표본({data.n_treated_total:,}가구)과 "
                f"{ratio:.1f}배 차이 납니다. 효과 추정치는 이 표본 규모·구성에서 관찰된 것이라, 크게 다른 "
                "발행 규모로 확대 적용하면 실제 효과가 달라질 수 있습니다."
            )

    # --- 계산 ---------------------------------------------------------------
    def revenue_to_gross(rev_per_customer: float) -> tuple[float, float]:
        total_rev = assumptions.n_issued * rev_per_customer
        total_gross = total_rev * assumptions.gross_margin_rate
        return total_rev, total_gross

    total_rev, total_gross = revenue_to_gross(data.incremental_revenue_per_customer)
    total_rev_lo, total_gross_lo = revenue_to_gross(lo)
    total_rev_hi, total_gross_hi = revenue_to_gross(hi)

    total_coupon_cost = assumptions.n_issued * assumptions.redemption_rate * assumptions.coupon_cost_per_redemption
    total_cost = total_coupon_cost + assumptions.operating_cost

    net_profit = total_gross - total_cost
    net_profit_lo = total_gross_lo - total_cost
    net_profit_hi = total_gross_hi - total_cost

    roi = net_profit / total_cost if total_cost > 0 else None
    roi_lo = net_profit_lo / total_cost if total_cost > 0 else None
    roi_hi = net_profit_hi / total_cost if total_cost > 0 else None

    denom = assumptions.n_issued * assumptions.redemption_rate
    if denom > 0:
        breakeven_max_coupon_cost = (total_gross - assumptions.operating_cost) / denom
    else:
        breakeven_max_coupon_cost = float("nan")
        warnings.append("예상 사용률 또는 발행 수가 0이라 손익분기 할인비용을 계산할 수 없습니다.")

    if net_profit < 0:
        be_txt = f"{breakeven_max_coupon_cost:,.2f}" if denom > 0 else "계산 불가"
        warnings.append(
            f"입력한 가정 기준 순증분이익이 음수입니다({net_profit:,.0f}). 손익분기를 맞추려면 쿠폰당 "
            f"할인비용을 {be_txt} 이하로 낮추거나 사용률·발행수·마진 가정을 재검토하세요."
        )

    return ProfitResult(
        data=data, assumptions=assumptions,
        total_incremental_revenue=total_rev, total_incremental_revenue_ci95=(total_rev_lo, total_rev_hi),
        total_gross_profit=total_gross, total_gross_profit_ci95=(total_gross_lo, total_gross_hi),
        total_coupon_cost=total_coupon_cost, total_cost=total_cost,
        net_incremental_profit=net_profit, net_incremental_profit_ci95=(net_profit_lo, net_profit_hi),
        roi=roi, roi_ci95=(roi_lo, roi_hi),
        breakeven_max_coupon_cost=breakeven_max_coupon_cost,
        warnings=warnings,
    )


def print_report(result: ProfitResult) -> None:
    d, a = result.data, result.assumptions
    print("=" * 92)
    print(f"캠페인 {d.campaign_id} 수익성 계산 — 결과변수: {d.outcome}  (매칭 상태: {d.status})")
    print("=" * 92)
    print("\n[데이터에서 추정한 값 — 사실/모델 추정치]")
    print(f"  고객당 증분매출(매칭 후)      : {d.incremental_revenue_per_customer:,.2f}  "
          f"95% CI [{d.incremental_revenue_ci95[0]:,.2f}, {d.incremental_revenue_ci95[1]:,.2f}]")
    print(f"  매칭쌍 수 / 처치 전체         : {d.n_matched_pairs:,} / {d.n_treated_total:,}")

    print("\n[사용자 입력 가정]")
    print(f"  발행 수                       : {a.n_issued:,}")
    print(f"  매출총이익률                  : {a.gross_margin_rate:.1%}")
    print(f"  예상 사용률                   : {a.redemption_rate:.1%}")
    print(f"  쿠폰당 할인비용                : {a.coupon_cost_per_redemption:,.2f}")
    print(f"  운영비                        : {a.operating_cost:,.2f}")

    print("\n[계산 결과] (점추정 / 95% CI 하한 / 95% CI 상한)")
    r = result
    print(f"  총 증분매출                   : {r.total_incremental_revenue:,.0f}  "
          f"[{r.total_incremental_revenue_ci95[0]:,.0f}, {r.total_incremental_revenue_ci95[1]:,.0f}]")
    print(f"  총 증분 매출총이익             : {r.total_gross_profit:,.0f}  "
          f"[{r.total_gross_profit_ci95[0]:,.0f}, {r.total_gross_profit_ci95[1]:,.0f}]")
    print(f"  총 쿠폰비용                   : {r.total_coupon_cost:,.0f}")
    print(f"  총 비용(쿠폰비용+운영비)        : {r.total_cost:,.0f}")
    print(f"  순증분이익                    : {r.net_incremental_profit:,.0f}  "
          f"[{r.net_incremental_profit_ci95[0]:,.0f}, {r.net_incremental_profit_ci95[1]:,.0f}]")
    roi_txt = f"{r.roi:.1%}" if r.roi is not None else "계산 불가(총비용 0)"
    print(f"  ROI                          : {roi_txt}")
    print(f"  손익분기 최대 쿠폰당 할인비용    : {r.breakeven_max_coupon_cost:,.2f}")

    if r.warnings:
        print(f"\n[주의 {len(r.warnings)}건]")
        for w in r.warnings:
            print(f"  ⚠ {w}")
    else:
        print("\n[주의사항 없음]")


# =========================================================================
# 쿠폰 후보안 비교 — 같은 캠페인 효과(DataEstimate)를 공유하고,
# 후보마다 할인액·예상사용률·발행수·운영비만 다르게 넣어 나란히 비교한다.
# =========================================================================
@dataclass
class Candidate:
    label: str                          # 후보 이름 (표의 행 식별자)
    coupon_cost_per_redemption: float   # 할인액
    redemption_rate: float              # 예상 사용률 (0~1)
    n_issued: int                       # 발행 수
    operating_cost: float = 0.0         # 운영비


def compare_candidates(
    data: DataEstimate,
    gross_margin_rate: float,
    candidates: list[Candidate],
) -> tuple[pd.DataFrame, str | None, list[ProfitResult]]:
    """같은 데이터 추정치(data)·매출총이익률로 후보들을 계산해 비교표를 만든다.

    반환:
        비교표(DataFrame), 추천 후보 라벨(없으면 None), 후보별 ProfitResult 리스트
        — "추천"은 입력한 후보군 안에서 손익분기(순증분이익>=0)를 만족하는 후보 중
        순증분이익이 가장 큰 것이다. 이 목록 밖의 다른 조합까지 포함한 전역 최적이
        아니다(CLAUDE.md 품질과 해석 규칙 7).
    """
    if not candidates:
        raise ValueError("후보가 1개 이상 있어야 한다.")

    rows = []
    results = []
    for c in candidates:
        assumptions = ProfitAssumptions(
            n_issued=c.n_issued,
            gross_margin_rate=gross_margin_rate,
            redemption_rate=c.redemption_rate,
            coupon_cost_per_redemption=c.coupon_cost_per_redemption,
            operating_cost=c.operating_cost,
        )
        r = compute_profitability(data, assumptions)
        results.append(r)
        rows.append({
            "후보": c.label,
            "할인액": c.coupon_cost_per_redemption,
            "예상사용률": c.redemption_rate,
            "발행수": c.n_issued,
            "운영비": c.operating_cost,
            "예상증분매출": round(r.total_incremental_revenue, 0),
            "총비용": round(r.total_cost, 0),
            "증분이익": round(r.net_incremental_profit, 0),
            "증분이익95%CI": f"[{r.net_incremental_profit_ci95[0]:,.0f}, {r.net_incremental_profit_ci95[1]:,.0f}]",
            "ROI": round(r.roi, 4) if r.roi is not None else None,
            "손익분기충족": bool(r.net_incremental_profit >= 0),
            "주의건수": len(r.warnings),
        })

    df = pd.DataFrame(rows)

    breakeven_ok = df["손익분기충족"]
    if breakeven_ok.any():
        best_idx = df.loc[breakeven_ok, "증분이익"].idxmax()
        recommended_label = df.loc[best_idx, "후보"]
    else:
        recommended_label = None

    df["추천"] = df["후보"].map(lambda x: "⭐ 추천" if x == recommended_label else "")

    return df, recommended_label, results


def print_comparison(df: pd.DataFrame, recommended_label: str | None, data: DataEstimate) -> None:
    print("=" * 92)
    print(f"캠페인 {data.campaign_id} 쿠폰 후보안 비교 — {data.outcome} 기준 (매칭 상태: {data.status})")
    print("=" * 92)
    print(f"고객당 증분매출(매칭 후, 공통 적용): {data.incremental_revenue_per_customer:,.2f} "
          f"95% CI [{data.incremental_revenue_ci95[0]:,.2f}, {data.incremental_revenue_ci95[1]:,.2f}]")
    print()
    print(df.to_string(index=False))
    print()
    if recommended_label:
        print(f"추천: '{recommended_label}' — 입력한 후보군 중 손익분기(순증분이익≥0)를 만족하면서 "
              "증분이익이 가장 큰 안이다. 이 후보군 밖의 다른 조합까지 포함한 전역 최적은 아니다.")
    else:
        print("추천 없음 — 입력한 후보군 중 손익분기(순증분이익≥0)를 만족하는 안이 없다.")


def main() -> None:
    parser = argparse.ArgumentParser(description="캠페인 효과 → 수익성 계산")
    parser.add_argument("--campaign_id", type=int, required=True)
    parser.add_argument("--outcome", default="target_sales", choices=["target_sales"])
    parser.add_argument("--n_issued", type=int, required=True, help="발행 수")
    parser.add_argument("--gross_margin_rate", type=float, required=True, help="매출총이익률 (0~1)")
    parser.add_argument("--redemption_rate", type=float, required=True, help="예상 사용률 (0~1)")
    parser.add_argument("--coupon_cost", type=float, required=True, help="쿠폰당 할인비용")
    parser.add_argument("--operating_cost", type=float, default=0.0, help="운영비(고정비)")
    args = parser.parse_args()

    data = load_data_estimate(args.campaign_id, args.outcome)
    assumptions = ProfitAssumptions(
        n_issued=args.n_issued,
        gross_margin_rate=args.gross_margin_rate,
        redemption_rate=args.redemption_rate,
        coupon_cost_per_redemption=args.coupon_cost,
        operating_cost=args.operating_cost,
    )
    result = compute_profitability(data, assumptions)
    print_report(result)


if __name__ == "__main__":
    main()
