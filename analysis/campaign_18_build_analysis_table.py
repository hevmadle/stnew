"""캠페인 18번 analysis_data.csv 생성 — 발행 전 특성 + 캠페인 기간 결과변수.

analysis/campaign_18_pre_features.py 의 build_pre_features() 로 만든 845가구(처치 510 / 대조 335)
발행 전 특성표에, 캠페인 기간(DAY 587~642) 결과변수를 붙여 최종 분석표를 만든다.

주요 결과(캠페인 대상 상품, CLAUDE.md 분석 설계 규칙 5):
    target_purchase  : 대상 상품 구매 여부 (0/1)
    target_sales      : 대상 상품 구매금액 합계
    target_quantity   : 대상 상품 구매수량 합계

보조 결과(전체 상품):
    any_purchase : 아무 상품이나 구매했는지 여부 (0/1)
    total_sales   : 전체 구매금액 합계
    baskets       : 장바구니 수 (BASKET_ID 고유 개수)

캠페인 18은 56일로 표준 33일 캠페인보다 길어, CLAUDE.md 분석 설계 규칙 6에 따라
첫 30일(DAY 587~616) 기준 결과와 일평균(실제기간/56)도 함께 만든다. 실제 기간 결과가
기본값이며 이 두 세트는 향후 다른 캠페인과 비교할 때 쓰는 보조 열이다.

거래가 없는 가구도 분석표에 남기고 모든 결과 열을 0으로 채운다(왼쪽 조인 + fillna(0)).

읽는 원본(읽기 전용, 수정 없음):
    campaign_18_pre_features.build_pre_features() 가 읽는 파일 전부 +
    data/raw/transaction_data.csv (캠페인 기간 결과 계산용, pre_features 단계에서 이미 로드한
    845가구분 tx_scope 를 재사용해 파일을 다시 읽지 않는다)

출력: outputs/campaign_18/analysis_data.csv 1개 (CLAUDE.md 파일 관리 규칙 2·3)

실행:
    .venv/bin/python analysis/campaign_18_build_analysis_table.py
"""

from pathlib import Path

import pandas as pd

from campaign_18_groups import TARGET_CAMPAIGN
from campaign_18_pre_features import PRE_FEATURE_COLUMNS, build_pre_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / f"campaign_{TARGET_CAMPAIGN}"
OUTPUT_FILE = OUTPUT_DIR / "analysis_data.csv"

FIRST_N_DAYS = 30

PRIMARY_COLUMNS = ["target_purchase", "target_sales", "target_quantity"]
SECONDARY_COLUMNS = ["any_purchase", "total_sales", "baskets"]


def banner(step: str, title: str) -> None:
    print("\n" + "=" * 92)
    print(f"[{step}] {title}")
    print("=" * 92)


def aggregate_outcomes(period_tx: pd.DataFrame, target_products: set, households: pd.DataFrame,
                        suffix: str = "") -> pd.DataFrame:
    """period_tx(특정 DAY 구간의 거래)로부터 주요·보조 결과변수를 만들어 households 에 왼쪽 조인한다.

    거래가 없는 가구도 households 의 모든 행을 유지하고 결과는 0으로 채운다.
    """
    target_tx = period_tx[period_tx["PRODUCT_ID"].isin(target_products)]

    primary = target_tx.groupby("household_key").agg(
        target_sales=("SALES_VALUE", "sum"),
        target_quantity=("QUANTITY", "sum"),
    )
    primary["target_purchase"] = 1  # groupby 결과에 존재한다는 것 자체가 구매 발생을 의미

    secondary = period_tx.groupby("household_key").agg(
        total_sales=("SALES_VALUE", "sum"),
        baskets=("BASKET_ID", "nunique"),
    )
    secondary["any_purchase"] = 1

    out = households[["household_key"]].merge(primary, on="household_key", how="left")
    out = out.merge(secondary, on="household_key", how="left")

    for col in ["target_sales", "target_quantity", "total_sales", "baskets"]:
        out[col] = out[col].fillna(0)
    for col in ["target_purchase", "any_purchase"]:
        out[col] = out[col].fillna(0).astype(int)

    if suffix:
        out = out.rename(columns={c: f"{c}{suffix}" for c in out.columns if c != "household_key"})
    return out


def main() -> None:
    # ------------------------------------------------------------------
    banner("STEP 1", "발행 전 특성표 불러오기 (재사용)")
    # ------------------------------------------------------------------
    analysis, meta, diag = build_pre_features(verbose=False)
    start_day, end_day = diag["start_day"], diag["end_day"]
    target_products = diag["target_products"]
    tx_scope = diag["tx_scope"]  # 845가구의 전 기간 거래 (transaction_data.csv 를 다시 읽지 않는다)

    duration = end_day - start_day + 1
    first_end = min(end_day, start_day + FIRST_N_DAYS - 1)

    print(f"  캠페인 {TARGET_CAMPAIGN} 기간   : DAY {start_day} ~ {end_day} ({duration}일)")
    print(f"  분석 대상 가구    : {len(analysis):,}가구 (처치 {int(analysis['treatment'].sum()):,} / "
          f"대조 {int((1 - analysis['treatment']).sum()):,})")
    print(f"  발행 전 특성 열({len(PRE_FEATURE_COLUMNS)}개): {PRE_FEATURE_COLUMNS}")
    print(f"  대상 상품 수      : {len(target_products):,}개 (coupon.csv → PRODUCT_ID, CAMPAIGN=={TARGET_CAMPAIGN})")

    # ------------------------------------------------------------------
    banner("STEP 2", "캠페인 기간 거래 슬라이스")
    # ------------------------------------------------------------------
    period_tx = tx_scope[(tx_scope["DAY"] >= start_day) & (tx_scope["DAY"] <= end_day)].copy()
    first30_tx = tx_scope[(tx_scope["DAY"] >= start_day) & (tx_scope["DAY"] <= first_end)].copy()

    print(f"  transaction_data.csv → 845가구 전체 기간 행 수 : {len(tx_scope):,}행")
    print(f"  실제 캠페인 기간(DAY {start_day}~{end_day})    : {len(period_tx):,}행")
    if len(period_tx):
        print(f"    → 실제 관찰된 DAY 범위: {period_tx['DAY'].min()} ~ {period_tx['DAY'].max()}")
    print(f"  첫 {FIRST_N_DAYS}일(DAY {start_day}~{first_end})       : {len(first30_tx):,}행")

    n_hh_with_period_tx = period_tx["household_key"].nunique()
    n_hh_no_tx = len(analysis) - n_hh_with_period_tx
    print(f"\n  캠페인 기간 중 거래가 있는 가구: {n_hh_with_period_tx:,}가구")
    print(f"  캠페인 기간 중 거래가 없는 가구: {n_hh_no_tx:,}가구 → 결과 열을 0으로 채워 분석표에 유지한다")

    # ------------------------------------------------------------------
    banner("STEP 3", "주요 결과(대상 상품) · 보조 결과(전체 상품) 계산")
    # ------------------------------------------------------------------
    full = aggregate_outcomes(period_tx, target_products, analysis[["household_key"]])
    first30 = aggregate_outcomes(first30_tx, target_products, analysis[["household_key"]], suffix="_first30")

    result = analysis.merge(full, on="household_key", how="left").merge(first30, on="household_key", how="left")

    for col in ["target_sales", "total_sales"]:
        result[f"{col}_per_day"] = (result[col] / duration).round(4)

    print("  주요 결과(대상 상품, 실제 기간):")
    for col in PRIMARY_COLUMNS:
        print(f"    {col:16s} 결측 {int(result[col].isna().sum())}건 / 0인 가구 {int((result[col]==0).sum()):,} / "
              f"합계 또는 발생 {result[col].sum() if col!='target_purchase' else int(result[col].sum())}")
    print("  보조 결과(전체 상품, 실제 기간):")
    for col in SECONDARY_COLUMNS:
        print(f"    {col:16s} 결측 {int(result[col].isna().sum())}건 / 0인 가구 {int((result[col]==0).sum()):,}")

    # ------------------------------------------------------------------
    banner("STEP 4", "결측/0 처리 검증 — 거래 없는 가구가 빠지지 않았는지")
    # ------------------------------------------------------------------
    outcome_cols = PRIMARY_COLUMNS + SECONDARY_COLUMNS
    n_missing_total = int(result[outcome_cols].isna().sum().sum())
    print(f"  분석표 행 수 : {len(result):,} (== 845가구? {len(result) == 845})")
    print(f"  가구 중복 여부: {result['household_key'].duplicated().any()}")
    print(f"  주요·보조 결과 열 결측치 총합: {n_missing_total}건 (0이어야 함)")

    zero_tx_hh = set(analysis["household_key"]) - set(period_tx["household_key"].unique())
    zero_rows = result[result["household_key"].isin(zero_tx_hh)]
    all_zero = (zero_rows[outcome_cols] == 0).all(axis=None)
    print(f"  캠페인 기간 거래가 없던 {len(zero_tx_hh):,}가구가 분석표에 남아있는가: "
          f"{zero_tx_hh <= set(result['household_key'])}")
    print(f"  그 가구들의 주요·보조 결과가 전부 0인가: {all_zero}")

    # ------------------------------------------------------------------
    banner("STEP 5", "결과 요약 — 집단별 평균")
    # ------------------------------------------------------------------
    pd.set_option("display.width", 220)
    summary = result.groupby("group")[
        PRIMARY_COLUMNS + SECONDARY_COLUMNS + [f"{c}_first30" for c in PRIMARY_COLUMNS + SECONDARY_COLUMNS]
        + ["target_sales_per_day", "total_sales_per_day"]
    ].mean().T
    summary["차이(처치-대조)"] = summary.iloc[:, 0] - summary.iloc[:, 1]
    print(summary.round(3).to_string())
    print()
    print("  ※ 이 표는 단순 평균 차이이며, 매칭·성향점수 보정 전이므로 인과효과로 해석하지 않는다"
          " (CLAUDE.md 품질과 해석 규칙 2).")

    # ------------------------------------------------------------------
    banner("STEP 6", "저장")
    # ------------------------------------------------------------------
    column_order = (
        ["household_key", "group", "treatment"]
        + PRE_FEATURE_COLUMNS
        + PRIMARY_COLUMNS
        + SECONDARY_COLUMNS
        + [f"{c}_first30" for c in PRIMARY_COLUMNS + SECONDARY_COLUMNS]
        + ["target_sales_per_day", "total_sales_per_day"]
    )
    result = result[column_order]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print(f"  저장 경로: {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")
    print(f"  shape    : {result.shape}")
    print(f"  캠페인   : {TARGET_CAMPAIGN} / 관측 단위: household_key / 기간: DAY {start_day}~{end_day}")
    print(f"  발행 전 관찰: DAY {diag['pre_start']}~{diag['pre_end']} / 결과 관찰: DAY {start_day}~{end_day}"
          f" (첫 {FIRST_N_DAYS}일: DAY {start_day}~{first_end})")
    print(f"  단위     : 금액=SALES_VALUE 원본 단위(데이터셋 표기 그대로), 수량=QUANTITY 원본 단위")
    print("  재현     : .venv/bin/python analysis/campaign_18_build_analysis_table.py")


if __name__ == "__main__":
    main()
