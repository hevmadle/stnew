"""캠페인 18번 프로파일 — 기간 · 수신가구 · 대상상품 · 겹치는 캠페인 · 처치/대조 집단 정의.

이 스크립트가 읽는 원본 테이블(읽기 전용, 원본 수정 없음):
    data/raw/campaign_desc.csv   : 캠페인 정의 (기간)
    data/raw/campaign_table.csv  : 캠페인 수신 가구
    data/raw/coupon.csv          : 캠페인 쿠폰 대상 상품
    data/raw/transaction_data.csv: household_key 열만 사용 — 전체 가구 모집단(2,500) 확인용

각 단계마다 "어느 파일의 어느 열에서 나온 값인지"를 함께 출력한다.
캠페인 번호·기간·대상상품은 코드에 고정하지 않고 원본 테이블에서 조회한다(CLAUDE.md 데이터 보존 규칙 3).

집단 정의(사용자 지정):
    처치군 = 캠페인 18을 받았고, 18 기간과 겹치는 캠페인은 받지 않은 가구
    대조군 = 캠페인 18을 받지 않았고, 18 기간과 겹치는 캠페인도 받지 않았으나,
             겹치지 않는 시기의 다른 캠페인은 받은 적 있는 가구
    → 캠페인을 한 번도 받은 적 없는 가구는 대조군에 포함하지 않는다(STEP 6 참고).

주의: 이 단계는 집단 구성까지이며, 성향점수 매칭이나 효과 추정은 수행하지 않는다.

실행:
    .venv/bin/python analysis/campaign_18_profile.py
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

TARGET_CAMPAIGN = 18  # 분석 처치 대상 캠페인 (사용자 지정)


def banner(step: str, title: str) -> None:
    print("\n" + "=" * 88)
    print(f"[{step}] {title}")
    print("=" * 88)


def source_note(text: str) -> None:
    """이 단계의 값이 어느 원본 테이블·열에서 나왔는지 표시."""
    print(f"  └ 출처: {text}")


def main() -> None:
    # ------------------------------------------------------------------
    # STEP 0. 원본 3개 테이블 로드
    # ------------------------------------------------------------------
    banner("STEP 0", "원본 테이블 로드")

    campaign_desc = pd.read_csv(RAW_DIR / "campaign_desc.csv")
    campaign_table = pd.read_csv(RAW_DIR / "campaign_table.csv")
    coupon = pd.read_csv(RAW_DIR / "coupon.csv")

    for name, df in [
        ("campaign_desc.csv", campaign_desc),
        ("campaign_table.csv", campaign_table),
        ("coupon.csv", coupon),
    ]:
        print(f"  {name:22s} shape={df.shape}  열={list(df.columns)}")

    # 전체 가구 모집단 확인용 — household_key 열만 읽는다(260만 행 전체 로드 회피)
    tx_households = set(
        pd.read_csv(RAW_DIR / "transaction_data.csv", usecols=["household_key"])["household_key"].unique()
    )
    print(f"  {'transaction_data.csv':22s} household_key 열만 사용 → 고유 가구 {len(tx_households):,}")

    # ------------------------------------------------------------------
    # STEP 1. 캠페인 18의 유형과 기간 — campaign_desc.csv
    # ------------------------------------------------------------------
    banner("STEP 1", f"캠페인 {TARGET_CAMPAIGN}의 유형과 기간")

    desc_row = campaign_desc.loc[campaign_desc["CAMPAIGN"] == TARGET_CAMPAIGN]
    if desc_row.empty:
        raise SystemExit(f"campaign_desc.csv에 CAMPAIGN={TARGET_CAMPAIGN}이 없다. 중단한다.")
    if len(desc_row) > 1:
        raise SystemExit(f"campaign_desc.csv에 CAMPAIGN={TARGET_CAMPAIGN}이 여러 행이다. 중단한다.")

    camp_type = desc_row["DESCRIPTION"].iloc[0]
    start_day = int(desc_row["START_DAY"].iloc[0])
    end_day = int(desc_row["END_DAY"].iloc[0])
    duration = end_day - start_day + 1

    print(f"  원본 행 그대로:\n{desc_row.to_string(index=False)}")
    print()
    print(f"  캠페인 유형   : {camp_type}")
    source_note("campaign_desc.csv → DESCRIPTION 열")
    print(f"  시작일        : DAY {start_day}")
    source_note("campaign_desc.csv → START_DAY 열")
    print(f"  종료일        : DAY {end_day}")
    source_note("campaign_desc.csv → END_DAY 열")
    print(f"  기간          : {duration}일 (END_DAY - START_DAY + 1로 계산, 양 끝 포함)")
    source_note("campaign_desc.csv의 START_DAY·END_DAY에서 파생 계산")
    print()
    print("  ※ DAY는 달력 날짜가 아니라 데이터셋 기준 상대 일자다(DATA_DICTIONARY.md).")

    # ------------------------------------------------------------------
    # STEP 2. 캠페인 18의 수신 가구 — campaign_table.csv
    # ------------------------------------------------------------------
    banner("STEP 2", f"캠페인 {TARGET_CAMPAIGN}의 수신 가구")

    ct_18 = campaign_table.loc[campaign_table["CAMPAIGN"] == TARGET_CAMPAIGN]
    recipients = set(ct_18["household_key"])

    print(f"  수신 행 수         : {len(ct_18):,}행")
    source_note("campaign_table.csv → CAMPAIGN == 18인 행")
    print(f"  고유 수신 가구 수  : {len(recipients):,}가구")
    source_note("campaign_table.csv → household_key 열의 고유값 개수")
    print(f"  행 수 == 가구 수 ? : {len(ct_18) == len(recipients)} (중복 수신 행 없음 확인)")
    print(f"  campaign_table의 DESCRIPTION 값: {sorted(ct_18['DESCRIPTION'].unique())}")
    source_note("campaign_table.csv → DESCRIPTION 열 (campaign_desc의 유형과 일치하는지 대조용)")
    print(f"  campaign_desc의 유형과 일치? : {sorted(ct_18['DESCRIPTION'].unique()) == [camp_type]}")
    print()
    print(f"  household_key 범위 : {min(recipients)} ~ {max(recipients)}")
    print(f"  수신 가구 예시(정렬 상위 10개): {sorted(recipients)[:10]}")

    # 전체 모집단 대비 비중
    universe = set(campaign_table["household_key"])
    print()
    print(f"  campaign_table 전체 고유 가구 수: {len(universe):,}가구")
    print(f"  → 캠페인 {TARGET_CAMPAIGN}은 그중 {len(recipients)/len(universe):.1%}에 발송됨")
    source_note("campaign_table.csv → household_key 전체 고유값 대비 비율")

    # ------------------------------------------------------------------
    # STEP 3. 캠페인 18의 쿠폰과 대상 상품 — coupon.csv
    # ------------------------------------------------------------------
    banner("STEP 3", f"캠페인 {TARGET_CAMPAIGN}의 쿠폰과 대상 상품")

    coupon_18 = coupon.loc[coupon["CAMPAIGN"] == TARGET_CAMPAIGN]
    coupon_upcs = set(coupon_18["COUPON_UPC"])
    target_products = set(coupon_18["PRODUCT_ID"])

    print(f"  원본 행 수            : {len(coupon_18):,}행")
    source_note("coupon.csv → CAMPAIGN == 18인 행")
    print(f"  고유 쿠폰 수(COUPON_UPC): {len(coupon_upcs):,}개")
    source_note("coupon.csv → COUPON_UPC 열의 고유값 개수")
    print(f"  고유 대상 상품 수      : {len(target_products):,}개")
    source_note("coupon.csv → PRODUCT_ID 열의 고유값 개수")

    dup_rows = coupon_18.duplicated(subset=["COUPON_UPC", "PRODUCT_ID"]).sum()
    print(f"  (COUPON_UPC, PRODUCT_ID) 완전중복 행: {dup_rows:,}행")
    source_note("coupon.csv 내 중복 점검 — 대상 상품 집합을 만들 때 중복 제거 필요 여부 확인용")

    print()
    print("  쿠폰별 대상 상품 수 분포:")
    per_coupon = coupon_18.groupby("COUPON_UPC")["PRODUCT_ID"].nunique().sort_values(ascending=False)
    print(f"    쿠폰 수={len(per_coupon)}, 최소={per_coupon.min()}, 중앙값={int(per_coupon.median())}, "
          f"최대={per_coupon.max()}, 합계(중복포함)={per_coupon.sum():,}")
    source_note("coupon.csv → COUPON_UPC별 PRODUCT_ID 고유 개수")
    print()
    print("    상위 5개 쿠폰(대상 상품 수 기준):")
    for upc, n in per_coupon.head(5).items():
        print(f"      COUPON_UPC={upc}  대상상품 {n:,}개")

    print()
    print(f"  대상 상품 PRODUCT_ID 예시(정렬 상위 10개): {sorted(target_products)[:10]}")
    print()
    print("  ※ 상품의 부서·브랜드·카테고리 설명은 product.csv에 있으나 이번 단계 입력 파일이 아니어서 조회하지 않았다.")

    # ------------------------------------------------------------------
    # STEP 4. 캠페인 18과 기간이 겹치는 다른 캠페인 — campaign_desc.csv 자기 비교
    # ------------------------------------------------------------------
    banner("STEP 4", f"캠페인 {TARGET_CAMPAIGN}과 기간이 겹치는 다른 캠페인")

    print(f"  겹침 판정식: (다른 캠페인 START_DAY <= {end_day}) AND ({start_day} <= 다른 캠페인 END_DAY)")
    source_note("campaign_desc.csv → START_DAY·END_DAY 열끼리 비교 (자기 자신 제외)")

    others = campaign_desc.loc[campaign_desc["CAMPAIGN"] != TARGET_CAMPAIGN].copy()
    overlap_mask = (others["START_DAY"] <= end_day) & (start_day <= others["END_DAY"])
    overlapping = others.loc[overlap_mask].copy()

    # 겹치는 일수도 함께 계산해 얼마나 심하게 겹치는지 보여준다
    overlapping["겹치는_시작"] = overlapping["START_DAY"].clip(lower=start_day)
    overlapping["겹치는_종료"] = overlapping["END_DAY"].clip(upper=end_day)
    overlapping["겹치는_일수"] = overlapping["겹치는_종료"] - overlapping["겹치는_시작"] + 1
    overlapping["캠페인18기간_대비"] = (overlapping["겹치는_일수"] / duration).map(lambda x: f"{x:.0%}")

    # 겹치는 캠페인별 수신 가구 수도 campaign_table에서 조회
    overlapping["수신가구수"] = overlapping["CAMPAIGN"].map(
        campaign_table.groupby("CAMPAIGN")["household_key"].nunique()
    )

    print()
    print(f"  겹치는 캠페인 수: {len(overlapping)}개 → {sorted(overlapping['CAMPAIGN'].tolist())}")
    print()
    print(
        overlapping.sort_values("CAMPAIGN")[
            [
                "CAMPAIGN",
                "DESCRIPTION",
                "START_DAY",
                "END_DAY",
                "겹치는_시작",
                "겹치는_종료",
                "겹치는_일수",
                "캠페인18기간_대비",
                "수신가구수",
            ]
        ].to_string(index=False)
    )
    source_note(
        "CAMPAIGN·DESCRIPTION·START_DAY·END_DAY = campaign_desc.csv / "
        "수신가구수 = campaign_table.csv의 household_key 고유 개수 / 겹치는 일수는 파생 계산"
    )

    # ------------------------------------------------------------------
    # STEP 5. 전체 가구 2,500을 겹침 기준으로 분해
    # ------------------------------------------------------------------
    banner("STEP 5", "전체 가구를 캠페인 18 기준으로 분해")

    households_by_campaign = campaign_table.groupby("CAMPAIGN")["household_key"].apply(set).to_dict()

    overlapping_households: set[int] = set()
    for oc in overlapping["CAMPAIGN"]:
        overlapping_households |= households_by_campaign.get(int(oc), set())

    treated = recipients - overlapping_households  # 18 단독 수신
    treated_excluded = recipients & overlapping_households  # 18을 받았지만 겹치는 캠페인도 받음
    control = universe - recipients - overlapping_households  # 비겹침 캠페인만 받음
    control_excluded = (universe - recipients) & overlapping_households  # 18 미수신이나 겹치는 캠페인 수신
    never_campaigned = tx_households - universe  # 캠페인을 한 번도 받은 적 없음

    print(f"  겹치는 캠페인({len(overlapping)}개)을 하나라도 받은 가구: {len(overlapping_households):,}가구")
    source_note("campaign_table.csv → 겹치는 CAMPAIGN들의 household_key 합집합")
    print()

    decomposition = pd.DataFrame(
        [
            ("A. 처치군", "18 수신, 겹치는 캠페인 미수신", len(treated), "처치군으로 사용"),
            ("B. 제외", "18 수신, 겹치는 캠페인도 수신", len(treated_excluded), "동시 노출 — 제외"),
            ("C. 대조군", "18 미수신, 겹치는 캠페인 미수신, 비겹침 캠페인은 수신", len(control), "대조군으로 사용"),
            ("D. 제외", "18 미수신, 겹치는 캠페인 수신", len(control_excluded), "동시 노출 — 제외"),
            ("E. 제외", "캠페인을 한 번도 받은 적 없음", len(never_campaigned), "STEP 6 사유로 제외"),
        ],
        columns=["구분", "정의", "가구수", "처리"],
    )
    print(decomposition.to_string(index=False))
    source_note(
        "A~D = campaign_table.csv의 household_key 집합 연산 / "
        "E = transaction_data.csv의 household_key 중 campaign_table.csv에 없는 가구"
    )

    total = sum([len(treated), len(treated_excluded), len(control), len(control_excluded), len(never_campaigned)])
    print()
    print(f"  합계 검증: {total:,} == transaction_data 고유 가구 {len(tx_households):,} ? "
          f"{total == len(tx_households)}")
    print(f"  campaign_table 소계 검증: A+B+C+D = "
          f"{len(treated)+len(treated_excluded)+len(control)+len(control_excluded):,} "
          f"== {len(universe):,} ? "
          f"{len(treated)+len(treated_excluded)+len(control)+len(control_excluded) == len(universe)}")

    # ------------------------------------------------------------------
    # STEP 6. 대조군 확정과 근거 검증
    # ------------------------------------------------------------------
    banner("STEP 6", "대조군 확정 — 비겹침 캠페인 수신 가구(C)")

    print("  대조군 정의(사용자 지정): 캠페인은 받은 적 있으나, 캠페인 18 기간과 겹치는 시기의")
    print("  캠페인은 하나도 받지 않은 가구. 캠페인 무경험 가구(E)는 대조군에서 제외한다.")
    print()
    print("  [검증 1] 대조군이 실제로 받은 캠페인 목록")
    control_rows = campaign_table[campaign_table["household_key"].isin(control)]
    received = sorted(control_rows["CAMPAIGN"].unique().tolist())
    overlapping_ids = set(overlapping["CAMPAIGN"].tolist())
    print(f"    받은 캠페인: {received}")
    print(f"    겹치는 8개({sorted(overlapping_ids)}) 중 포함된 것: "
          f"{sorted(set(received) & overlapping_ids) or '없음'}")
    print(f"    캠페인 18 포함 여부: {TARGET_CAMPAIGN in received}")
    source_note("campaign_table.csv → 대조군 household_key에 해당하는 CAMPAIGN 열의 고유값")

    print()
    print("  [검증 2] 대조군의 가구당 수신 캠페인 수 분포")
    per_hh = control_rows.groupby("household_key").size().value_counts().sort_index()
    for n_camp, n_hh in per_hh.items():
        print(f"    {n_camp}개 수신: {n_hh:,}가구")
    print(f"    → 대조군 전원이 최소 1개 캠페인 수신: {per_hh.index.min() >= 1}")
    source_note("campaign_table.csv → 대조군 household_key별 행 수")

    print()
    print("  [검증 3] 대조군이 받은 캠페인이 18 기간에서 얼마나 떨어져 있나")
    recv_desc = campaign_desc[campaign_desc["CAMPAIGN"].isin(received)]
    before = recv_desc[recv_desc["END_DAY"] < start_day]
    after = recv_desc[recv_desc["START_DAY"] > end_day]
    print(f"    18 시작({start_day}) 이전에 끝난 캠페인: {sorted(before['CAMPAIGN'].tolist())}")
    print(f"      → 가장 늦게 끝난 캠페인 {int(before.loc[before['END_DAY'].idxmax(), 'CAMPAIGN'])}번이 "
          f"END_DAY {int(before['END_DAY'].max())}, 18 시작까지 {start_day - int(before['END_DAY'].max())}일 간격")
    print(f"    18 종료({end_day}) 이후에 시작한 캠페인: {sorted(after['CAMPAIGN'].tolist())}")
    print(f"      → 가장 빨리 시작한 캠페인 {int(after.loc[after['START_DAY'].idxmin(), 'CAMPAIGN'])}번이 "
          f"START_DAY {int(after['START_DAY'].min())}, 18 종료 후 {int(after['START_DAY'].min()) - end_day}일 간격")
    source_note("campaign_desc.csv → START_DAY·END_DAY 를 캠페인 18 기간과 비교")

    ratio = len(control) / len(treated) if treated else float("nan")
    print()
    print(f"  확정: 처치군 {len(treated):,}가구 / 대조군 {len(control):,}가구 (대조/처치 비율 {ratio:.2f})")

    # ------------------------------------------------------------------
    # STEP 7. 요약 + 값-출처 대조표
    # ------------------------------------------------------------------
    banner("STEP 7", f"캠페인 {TARGET_CAMPAIGN} 요약과 값별 출처")

    summary = [
        ("캠페인 번호", TARGET_CAMPAIGN, "campaign_desc.csv", "CAMPAIGN"),
        ("캠페인 유형", camp_type, "campaign_desc.csv", "DESCRIPTION"),
        ("시작일(DAY)", start_day, "campaign_desc.csv", "START_DAY"),
        ("종료일(DAY)", end_day, "campaign_desc.csv", "END_DAY"),
        ("기간(일)", duration, "campaign_desc.csv", "END_DAY - START_DAY + 1 (파생)"),
        ("수신 가구 수", f"{len(recipients):,}", "campaign_table.csv", "household_key (CAMPAIGN==18)"),
        ("고유 쿠폰 수", f"{len(coupon_upcs):,}", "coupon.csv", "COUPON_UPC (CAMPAIGN==18)"),
        ("대상 상품 수", f"{len(target_products):,}", "coupon.csv", "PRODUCT_ID (CAMPAIGN==18)"),
        ("겹치는 캠페인 수", len(overlapping), "campaign_desc.csv", "START_DAY·END_DAY 비교 (파생)"),
        ("처치군 가구", f"{len(treated):,}", "campaign_table.csv", "household_key 집합 차집합 (파생)"),
        ("대조군 가구", f"{len(control):,}", "campaign_table.csv", "household_key 집합 차집합 (파생)"),
        ("대조군서 제외: 캠페인 무경험", f"{len(never_campaigned):,}", "transaction_data.csv",
         "household_key 중 campaign_table.csv 미등장 (파생)"),
    ]
    summary_df = pd.DataFrame(summary, columns=["항목", "값", "원본 파일", "원본 열 / 계산식"])
    print(summary_df.to_string(index=False))

    print()
    print("  분석 대상 표본: 처치군 + 대조군 = "
          f"{len(treated):,} + {len(control):,} = {len(treated) + len(control):,}가구")
    print(f"  (transaction_data 전체 {len(tx_households):,}가구의 "
          f"{(len(treated) + len(control)) / len(tx_households):.1%})")

    print()
    print("  다음 단계에 필요하지만 이번 입력 파일에는 없는 정보:")
    print("    - 대상 상품의 카테고리·브랜드          → product.csv (PRODUCT_ID로 연결)")
    print("    - 발행 전 구매 특성, 캠페인 기간 결과  → transaction_data.csv (household_key·PRODUCT_ID·DAY로 연결)")
    print("    - 가구 인구통계                        → hh_demographic.csv (household_key로 연결)")
    print("    - 쿠폰 실제 사용 여부                  → coupon_redempt.csv (household_key·CAMPAIGN·COUPON_UPC로 연결)")


if __name__ == "__main__":
    main()
