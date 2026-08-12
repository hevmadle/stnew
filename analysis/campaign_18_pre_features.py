"""캠페인 18번 발행 전(pre-treatment) 구매 특성 생성.

analysis/campaign_18_groups.py 의 build_groups() 로 확정한 처치 510 + 대조 335 = 845가구를
대상으로, 캠페인 시작일(DAY 587) '이전' 정보만 사용해 성향점수 입력 후보 변수를 만든다.

생성 변수
    recency               : 캠페인 시작일 기준 마지막 구매 이후 경과 일수
    pre_baskets           : 사전 기간 장바구니 수 (BASKET_ID 고유 개수)
    pre_sales             : 사전 기간 구매금액 합계
    pre_quantity          : 사전 기간 구매수량 합계
    pre_active_days       : 사전 기간 구매가 있었던 일수
    pre_target_purchase   : 캠페인 대상 상품 구매 여부 (0/1)
    pre_target_baskets    : 캠페인 대상 상품을 담은 장바구니 수
    pre_target_sales      : 캠페인 대상 상품 구매금액
    pre_target_quantity   : 캠페인 대상 상품 구매수량
    pre_coupon_redemptions: 사전 기간 쿠폰 사용 횟수
    pre_campaign_count    : 사전 기간에 종료된 캠페인 수신 횟수

build_pre_features() 는 이 모듈의 핵심 함수이며, 결과변수를 붙이는
analysis/campaign_18_build_analysis_table.py 에서도 그대로 재사용한다(로직 중복 방지).

CLAUDE.md 분석 설계 규칙 3·4 준수:
    - 모든 변수는 DAY < START_DAY 인 기록만 사용한다.
    - 캠페인 기간의 구매·쿠폰 사용은 성향점수 입력에 포함하지 않는다.
    - 캠페인 18 종료 후 시작하는 캠페인(23·24·25)은 사후 정보이므로 pre_campaign_count에서 제외한다.
    이 스크립트를 직접 실행하면 STEP 7에서 변수별 실제 관찰 구간을 출력해 사후 정보 혼입 여부를 확인한다.

읽는 원본(읽기 전용, 수정 없음):
    data/raw/campaign_desc.csv    : START_DAY, END_DAY
    data/raw/campaign_table.csv   : 캠페인 수신 이력
    data/raw/coupon.csv           : 캠페인 18 대상 상품 PRODUCT_ID
    data/raw/coupon_redempt.csv   : household_key, DAY  → 쿠폰 사용
    data/raw/transaction_data.csv : household_key, BASKET_ID, DAY, PRODUCT_ID, QUANTITY, SALES_VALUE

이 스크립트를 직접 실행하면 결과를 터미널에만 출력한다(파일 생성 없음).

실행:
    .venv/bin/python analysis/campaign_18_pre_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from campaign_18_groups import GROUP_CONTROL, GROUP_TREATED, TARGET_CAMPAIGN, build_groups

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# 발행 전 관찰 구간 길이(일). None이면 데이터 시작부터 캠페인 시작 전날까지 전체를 사용한다.
# 예: 30 으로 두면 DAY (START_DAY-30) ~ (START_DAY-1) 만 본다.
PRE_WINDOW_DAYS: int | None = None

PRE_FEATURE_COLUMNS = [
    "recency", "pre_baskets", "pre_sales", "pre_quantity", "pre_active_days",
    "pre_target_purchase", "pre_target_baskets", "pre_target_sales", "pre_target_quantity",
    "pre_coupon_redemptions", "pre_campaign_count",
]


def banner(step: str, title: str) -> None:
    print("\n" + "=" * 92)
    print(f"[{step}] {title}")
    print("=" * 92)


def build_pre_features(pre_window_days: int | None = PRE_WINDOW_DAYS, verbose: bool = True):
    """처치+대조 845가구에 발행 전 특성 11개를 붙여 반환한다.

    반환값:
        analysis : household_key, group, treatment + PRE_FEATURE_COLUMNS 를 담은 DataFrame
        meta     : build_groups() 가 반환한 메타 정보(캠페인 기간, 집합 등)
        diag     : STEP7/8 검증에 쓰는 중간 산출물(사전 거래 슬라이스, 관찰 구간 등)
    """
    assignment, meta = build_groups()
    start_day, end_day = meta["start_day"], meta["end_day"]

    analysis = assignment[assignment["group"].isin([GROUP_TREATED, GROUP_CONTROL])].copy()
    analysis = analysis[["household_key", "group", "treatment"]].reset_index(drop=True)
    hh_in_scope = set(analysis["household_key"])

    pre_end = start_day - 1
    pre_start = 1 if pre_window_days is None else max(1, start_day - pre_window_days)

    if verbose:
        banner("STEP 1", "분석 대상 가구와 발행 전 관찰 구간 정의")
        print(f"  캠페인 {TARGET_CAMPAIGN} 기간   : DAY {start_day} ~ {end_day}")
        print(f"  발행 전 관찰 구간 : DAY {pre_start} ~ {pre_end}  "
              f"({'데이터 시작부터 전체' if pre_window_days is None else f'{pre_window_days}일'})")
        print(f"  분석 대상 가구    : {len(analysis):,}가구 "
              f"(처치 {int(analysis['treatment'].sum()):,} / 대조 {int((1 - analysis['treatment']).sum()):,})")

    # ------------------------------------------------------------------
    # 거래 데이터 적재와 사전 기간 필터
    # ------------------------------------------------------------------
    tx = pd.read_csv(
        RAW_DIR / "transaction_data.csv",
        usecols=["household_key", "BASKET_ID", "DAY", "PRODUCT_ID", "QUANTITY", "SALES_VALUE"],
    )
    tx_scope = tx[tx["household_key"].isin(hh_in_scope)]
    pre_tx = tx_scope[(tx_scope["DAY"] >= pre_start) & (tx_scope["DAY"] <= pre_end)].copy()

    if verbose:
        banner("STEP 2", "거래 데이터 적재와 사전 기간 필터")
        print(f"  transaction_data.csv 전체            : {len(tx):,}행 (DAY {tx['DAY'].min()} ~ {tx['DAY'].max()})")
        print(f"  분석 대상 845가구로 한정             : {len(tx_scope):,}행")
        print(f"  발행 전 구간(DAY {pre_start}~{pre_end})으로 한정 : {len(pre_tx):,}행")
        if len(pre_tx):
            print(f"    → 실제 관찰된 DAY 범위: {pre_tx['DAY'].min()} ~ {pre_tx['DAY'].max()}")
        print()
        print("  데이터 품질 점검(사전 기간 거래):")
        n_nonpos_qty = int((pre_tx["QUANTITY"] <= 0).sum())
        print(f"    QUANTITY <= 0 인 행 : {n_nonpos_qty:,}행 "
              f"({n_nonpos_qty / len(pre_tx):.2%}) — 반품·조정 가능성, 원본 그대로 합산한다")
        print(f"    QUANTITY 최댓값     : {int(pre_tx['QUANTITY'].max()):,}")
        print(f"    SALES_VALUE < 0     : {int((pre_tx['SALES_VALUE'] < 0).sum()):,}행")
        print(f"    SALES_VALUE 최댓값  : {pre_tx['SALES_VALUE'].max():,.2f}")

    # ------------------------------------------------------------------
    # 기본 구매 특성 — recency · 장바구니 · 금액 · 수량
    # ------------------------------------------------------------------
    basic = pre_tx.groupby("household_key").agg(
        last_purchase_day=("DAY", "max"),
        pre_active_days=("DAY", "nunique"),
        pre_baskets=("BASKET_ID", "nunique"),
        pre_sales=("SALES_VALUE", "sum"),
        pre_quantity=("QUANTITY", "sum"),
    )
    basic["recency"] = start_day - basic["last_purchase_day"]
    basic = basic.drop(columns=["last_purchase_day"])

    analysis = analysis.merge(basic, on="household_key", how="left")

    n_no_pre_tx = int(analysis["pre_baskets"].isna().sum())
    for col in ["pre_baskets", "pre_sales", "pre_quantity", "pre_active_days"]:
        analysis[col] = analysis[col].fillna(0)

    if verbose:
        banner("STEP 3", "기본 구매 특성 — recency · 장바구니 · 금액 · 수량")
        print(f"  사전 기간 거래가 있는 가구 : {len(analysis) - n_no_pre_tx:,}가구")
        print(f"  사전 기간 거래가 없는 가구 : {n_no_pre_tx:,}가구")
        if n_no_pre_tx:
            print("    → 이 가구들은 recency 를 계산할 수 없다(결측 유지). 구매 집계는 0으로 채운다.")
        print()
        print(f"  recency 정의: 캠페인 시작일(DAY {start_day}) - 마지막 구매일")
        print(f"    최솟값 {analysis['recency'].min():.0f}일 / 최댓값 {analysis['recency'].max():.0f}일 / "
              f"중앙값 {analysis['recency'].median():.0f}일")

    # ------------------------------------------------------------------
    # 캠페인 대상 상품의 발행 전 구매 이력
    # ------------------------------------------------------------------
    coupon = pd.read_csv(RAW_DIR / "coupon.csv")
    target_products = set(coupon.loc[coupon["CAMPAIGN"] == TARGET_CAMPAIGN, "PRODUCT_ID"].unique())
    pre_target_tx = pre_tx[pre_tx["PRODUCT_ID"].isin(target_products)]

    target_agg = pre_target_tx.groupby("household_key").agg(
        pre_target_baskets=("BASKET_ID", "nunique"),
        pre_target_sales=("SALES_VALUE", "sum"),
        pre_target_quantity=("QUANTITY", "sum"),
    )
    analysis = analysis.merge(target_agg, on="household_key", how="left")
    for col in ["pre_target_baskets", "pre_target_sales", "pre_target_quantity"]:
        analysis[col] = analysis[col].fillna(0)
    analysis["pre_target_purchase"] = (analysis["pre_target_baskets"] > 0).astype(int)

    if verbose:
        banner("STEP 4", "캠페인 18 대상 상품의 발행 전 구매 이력")
        print(f"  캠페인 {TARGET_CAMPAIGN} 대상 상품 : {len(target_products):,}개 "
              f"(coupon.csv → PRODUCT_ID, CAMPAIGN=={TARGET_CAMPAIGN})")
        print(f"  사전 기간 중 대상 상품 거래행 : {len(pre_target_tx):,}행 "
              f"(전체 사전 거래의 {len(pre_target_tx) / len(pre_tx):.1%})")
        print()
        print(f"  대상 상품을 한 번이라도 산 가구: {int(analysis['pre_target_purchase'].sum()):,}가구 "
              f"({analysis['pre_target_purchase'].mean():.1%})")

    # ------------------------------------------------------------------
    # 발행 전 쿠폰 사용 횟수
    # ------------------------------------------------------------------
    redempt = pd.read_csv(RAW_DIR / "coupon_redempt.csv")
    pre_redempt = redempt[
        redempt["household_key"].isin(hh_in_scope)
        & (redempt["DAY"] >= pre_start)
        & (redempt["DAY"] <= pre_end)
    ]
    redempt_agg = pre_redempt.groupby("household_key").size().rename("pre_coupon_redemptions")
    analysis = analysis.merge(redempt_agg, on="household_key", how="left")
    analysis["pre_coupon_redemptions"] = analysis["pre_coupon_redemptions"].fillna(0).astype(int)

    if verbose:
        banner("STEP 5", "발행 전 쿠폰 사용 횟수")
        print(f"  coupon_redempt.csv 전체 : {len(redempt):,}행 (DAY {redempt['DAY'].min()} ~ {redempt['DAY'].max()})")
        print(f"  845가구 · 발행 전 구간으로 한정 : {len(pre_redempt):,}행")
        if len(pre_redempt):
            print(f"    → 실제 관찰된 DAY 범위: {pre_redempt['DAY'].min()} ~ {pre_redempt['DAY'].max()}")
        print()
        print(f"  쿠폰 사용 이력이 있는 가구: {int((analysis['pre_coupon_redemptions'] > 0).sum()):,}가구 "
              f"({(analysis['pre_coupon_redemptions'] > 0).mean():.1%})")

    # ------------------------------------------------------------------
    # 발행 전 캠페인 수신 횟수
    # ------------------------------------------------------------------
    campaign_desc = pd.read_csv(RAW_DIR / "campaign_desc.csv")
    campaign_table = meta["campaign_table"]

    pre_campaigns = sorted(campaign_desc.loc[campaign_desc["END_DAY"] < start_day, "CAMPAIGN"].tolist())
    post_campaigns = sorted(campaign_desc.loc[campaign_desc["START_DAY"] > end_day, "CAMPAIGN"].tolist())

    scope_rows = campaign_table[campaign_table["household_key"].isin(hh_in_scope)]
    pre_camp_rows = scope_rows[scope_rows["CAMPAIGN"].isin(pre_campaigns)]
    camp_agg = pre_camp_rows.groupby("household_key").size().rename("pre_campaign_count")
    analysis = analysis.merge(camp_agg, on="household_key", how="left")
    analysis["pre_campaign_count"] = analysis["pre_campaign_count"].fillna(0).astype(int)

    if verbose:
        banner("STEP 6", "발행 전 캠페인 수신 횟수")
        print(f"  사전 캠페인({len(pre_campaigns)}개, END_DAY < {start_day}) : {pre_campaigns}")
        print(f"  사후 캠페인({len(post_campaigns)}개, START_DAY > {end_day}) : {post_campaigns} → 제외")
        received = set(scope_rows["CAMPAIGN"].unique())
        unexpected = received - set(pre_campaigns) - set(post_campaigns) - {TARGET_CAMPAIGN}
        print(f"  845가구가 받은 캠페인이 사전·사후·18번으로만 구성되는가: {not unexpected} "
              f"{'' if not unexpected else f'(예외 {sorted(unexpected)})'}")
        print()
        print(f"  사전 캠페인 수신 이력이 있는 가구: {int((analysis['pre_campaign_count'] > 0).sum()):,}가구 "
              f"({(analysis['pre_campaign_count'] > 0).mean():.1%})")

    diag = {
        "start_day": start_day, "end_day": end_day,
        "pre_start": pre_start, "pre_end": pre_end,
        "pre_tx": pre_tx, "pre_target_tx": pre_target_tx, "pre_redempt": pre_redempt,
        "pre_campaigns": pre_campaigns, "post_campaigns": post_campaigns,
        "campaign_desc": campaign_desc, "target_products": target_products,
        "tx_scope": tx_scope,  # 845가구의 전체 기간(모든 DAY) 거래 — 결과변수 계산 단계에서 재사용
    }
    return analysis, meta, diag


def main() -> None:
    analysis, meta, diag = build_pre_features(verbose=True)
    start_day, end_day = diag["start_day"], diag["end_day"]
    pre_end = diag["pre_end"]
    pre_tx, pre_target_tx, pre_redempt = diag["pre_tx"], diag["pre_target_tx"], diag["pre_redempt"]
    pre_campaigns, post_campaigns = diag["pre_campaigns"], diag["post_campaigns"]
    campaign_desc = diag["campaign_desc"]

    print()
    print("  ※ 발행 전 구간에 캠페인 13(DAY 504~551)이 포함된다. 처치·대조 양쪽 모두 수신 가능한")
    print("    비겹침 캠페인이며, 노출 횟수는 pre_campaign_count 로 통제 대상에 넣는다.")

    # ------------------------------------------------------------------
    banner("STEP 7", "변수별 관찰 구간 검증 — 캠페인 시작일 이후 정보 혼입 여부")
    # ------------------------------------------------------------------
    def day_range(df: pd.DataFrame, col: str = "DAY") -> tuple:
        if len(df) == 0:
            return (None, None)
        return (int(df[col].min()), int(df[col].max()))

    tx_range = day_range(pre_tx)
    tgt_range = day_range(pre_target_tx)
    red_range = day_range(pre_redempt)
    camp_end_max = (
        int(campaign_desc.loc[campaign_desc["CAMPAIGN"].isin(pre_campaigns), "END_DAY"].max())
        if pre_campaigns else None
    )
    camp_start_min = (
        int(campaign_desc.loc[campaign_desc["CAMPAIGN"].isin(pre_campaigns), "START_DAY"].min())
        if pre_campaigns else None
    )

    rows = [
        ("recency", "transaction_data.csv", "DAY", *tx_range),
        ("pre_baskets", "transaction_data.csv", "BASKET_ID", *tx_range),
        ("pre_sales", "transaction_data.csv", "SALES_VALUE", *tx_range),
        ("pre_quantity", "transaction_data.csv", "QUANTITY", *tx_range),
        ("pre_active_days", "transaction_data.csv", "DAY", *tx_range),
        ("pre_target_purchase", "transaction_data.csv × coupon.csv", "PRODUCT_ID", *tgt_range),
        ("pre_target_baskets", "transaction_data.csv × coupon.csv", "BASKET_ID", *tgt_range),
        ("pre_target_sales", "transaction_data.csv × coupon.csv", "SALES_VALUE", *tgt_range),
        ("pre_target_quantity", "transaction_data.csv × coupon.csv", "QUANTITY", *tgt_range),
        ("pre_coupon_redemptions", "coupon_redempt.csv", "DAY", *red_range),
        ("pre_campaign_count", "campaign_desc.csv × campaign_table.csv", "END_DAY", camp_start_min, camp_end_max),
    ]

    check = pd.DataFrame(rows, columns=["변수", "원본 파일", "기준 열", "관찰 시작일", "관찰 종료일"])
    check["허용 상한"] = pre_end
    check["시작일 이전만 사용"] = check["관찰 종료일"].map(
        lambda v: "확인" if v is not None and v <= pre_end else ("데이터 없음" if v is None else "위반")
    )
    print(check.to_string(index=False))

    leak = check[check["시작일 이전만 사용"] == "위반"]
    print()
    print(f"  캠페인 시작일(DAY {start_day}) 이후 정보 혼입: {'없음' if len(leak) == 0 else f'{len(leak)}건 발견'}")
    print()
    print("  추가 확인:")
    print(f"    사전 거래에 DAY >= {start_day} 인 행: {int((pre_tx['DAY'] >= start_day).sum()):,}행")
    print(f"    사전 쿠폰사용에 DAY >= {start_day} 인 행: {int((pre_redempt['DAY'] >= start_day).sum()):,}행")
    print(f"    pre_campaign_count 에 사후 캠페인({post_campaigns}) 포함 여부: "
          f"{bool(set(pre_campaigns) & set(post_campaigns))}")

    # ------------------------------------------------------------------
    banner("STEP 8", "발행 전 특성 요약 — 집단별 평균")
    # ------------------------------------------------------------------
    summary = analysis.groupby("group")[PRE_FEATURE_COLUMNS].mean().T
    summary["차이(처치-대조)"] = summary[GROUP_TREATED] - summary[GROUP_CONTROL]

    t = analysis[analysis["treatment"] == 1]
    c = analysis[analysis["treatment"] == 0]
    smd = []
    for col in PRE_FEATURE_COLUMNS:
        pooled = np.sqrt((t[col].var(ddof=1) + c[col].var(ddof=1)) / 2)
        smd.append((t[col].mean() - c[col].mean()) / pooled if pooled > 0 else np.nan)
    summary["표준화평균차"] = smd

    pd.set_option("display.width", 200)
    print(summary.round(3).to_string())
    print()
    imbalanced = summary[summary["표준화평균차"].abs() > 0.1]
    print("  표준화평균차 |SMD| > 0.1 인 변수(매칭 전 불균형): "
          + (", ".join(imbalanced.index) if len(imbalanced) else "없음"))
    print()
    print(f"  분석표 shape: {analysis.shape} — 행 {len(analysis):,} = 처치 {int(analysis['treatment'].sum()):,} "
          f"+ 대조 {int((1 - analysis['treatment']).sum()):,}")
    print(f"  다음 단계: 캠페인 기간(DAY {start_day}~{end_day}) 결과변수를 붙여 analysis_data.csv 를 만든다.")
    print("  이 단계에서는 파일을 만들지 않는다.")


if __name__ == "__main__":
    main()
