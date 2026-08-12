"""data/raw/ 원본 CSV 7개의 실제 구조를 검증하는 점검 스크립트.

- 원본 파일을 수정하지 않고 읽기 전용으로만 사용한다 (CLAUDE.md 데이터 보존 규칙).
- 출력은 터미널로만 내보내고 별도 파일을 생성하지 않는다.
- 이 스크립트의 출력 결과가 analysis/data_structure_overview.md 표와 서술의 근거가 된다.

실행:
    .venv/bin/python analysis/inspect_raw_data.py
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

FILES = [
    "campaign_desc.csv",
    "campaign_table.csv",
    "coupon.csv",
    "coupon_redempt.csv",
    "hh_demographic.csv",
    "product.csv",
    "transaction_data.csv",
]


def profile_file(name: str) -> pd.DataFrame:
    path = RAW_DIR / name
    df = pd.read_csv(path)

    print(f"\n{'=' * 70}")
    print(f"[{name}]  path={path}")
    print(f"shape = {df.shape}  (rows={df.shape[0]:,}, cols={df.shape[1]})")
    print("-- dtypes --")
    print(df.dtypes)
    print("-- null counts (0이 아닌 열만) --")
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    print(nulls if len(nulls) else "없음")
    print("-- head(2) --")
    print(df.head(2).to_string())

    return df


def main() -> None:
    frames: dict[str, pd.DataFrame] = {}
    for name in FILES:
        frames[name] = profile_file(name)

    print(f"\n{'=' * 70}")
    print("[키 유일성 / 중복 점검]")

    campaign_desc = frames["campaign_desc.csv"]
    print(
        "campaign_desc: CAMPAIGN 유일?",
        campaign_desc["CAMPAIGN"].is_unique,
        "| 캠페인 수:",
        campaign_desc["CAMPAIGN"].nunique(),
    )
    print("campaign_desc.DESCRIPTION 분포:")
    print(campaign_desc["DESCRIPTION"].value_counts())

    campaign_table = frames["campaign_table.csv"]
    dup_ct = campaign_table.duplicated(subset=["household_key", "CAMPAIGN"]).sum()
    print(
        "campaign_table: (household_key, CAMPAIGN) 중복행 수:",
        dup_ct,
        "| 고유 가구 수:",
        campaign_table["household_key"].nunique(),
        "| 고유 캠페인 수:",
        campaign_table["CAMPAIGN"].nunique(),
    )

    coupon = frames["coupon.csv"]
    dup_cp = coupon.duplicated(subset=["CAMPAIGN", "COUPON_UPC", "PRODUCT_ID"]).sum()
    print(
        "coupon: (CAMPAIGN, COUPON_UPC, PRODUCT_ID) 중복행 수:",
        dup_cp,
        "| 고유 COUPON_UPC 수:",
        coupon["COUPON_UPC"].nunique(),
        "| 고유 PRODUCT_ID 수:",
        coupon["PRODUCT_ID"].nunique(),
    )

    coupon_redempt = frames["coupon_redempt.csv"]
    print(
        "coupon_redempt: 고유 household_key 수:",
        coupon_redempt["household_key"].nunique(),
        "| 고유 CAMPAIGN 수:",
        coupon_redempt["CAMPAIGN"].nunique(),
    )

    hh_demographic = frames["hh_demographic.csv"]
    dup_hh = hh_demographic["household_key"].duplicated().sum()
    print(
        "hh_demographic: household_key 유일?",
        hh_demographic["household_key"].is_unique,
        "(중복",
        dup_hh,
        "건) | 가구 수:",
        hh_demographic["household_key"].nunique(),
    )

    product = frames["product.csv"]
    dup_pd = product["PRODUCT_ID"].duplicated().sum()
    print(
        "product: PRODUCT_ID 유일?",
        product["PRODUCT_ID"].is_unique,
        "(중복",
        dup_pd,
        "건) | 상품 수:",
        product["PRODUCT_ID"].nunique(),
    )

    transaction_data = frames["transaction_data.csv"]
    print(
        "transaction_data: 고유 household_key 수:",
        transaction_data["household_key"].nunique(),
        "| 고유 BASKET_ID 수:",
        transaction_data["BASKET_ID"].nunique(),
        "| 고유 PRODUCT_ID 수:",
        transaction_data["PRODUCT_ID"].nunique(),
    )

    print(f"\n{'=' * 70}")
    print("[household_key 커버리지 비교]")
    hh_ids = set(hh_demographic["household_key"])
    ct_ids = set(campaign_table["household_key"])
    tx_ids = set(transaction_data["household_key"])
    cr_ids = set(coupon_redempt["household_key"])
    print("hh_demographic 가구 수:", len(hh_ids))
    print("campaign_table 가구 수:", len(ct_ids), "| hh_demographic과 교집합:", len(hh_ids & ct_ids))
    print("transaction_data 가구 수:", len(tx_ids), "| hh_demographic과 교집합:", len(hh_ids & tx_ids))
    print("coupon_redempt 가구 수:", len(cr_ids), "| hh_demographic과 교집합:", len(hh_ids & cr_ids))
    print("campaign_table 중 hh_demographic에 없는 가구 수:", len(ct_ids - hh_ids))
    print("transaction_data 중 hh_demographic에 없는 가구 수:", len(tx_ids - hh_ids))

    print(f"\n{'=' * 70}")
    print("[CAMPAIGN 커버리지 비교]")
    cd_camp = set(campaign_desc["CAMPAIGN"])
    ct_camp = set(campaign_table["CAMPAIGN"])
    cp_camp = set(coupon["CAMPAIGN"])
    cr_camp = set(coupon_redempt["CAMPAIGN"])
    print("campaign_desc 캠페인 수:", len(cd_camp))
    print("campaign_table 캠페인 수:", len(ct_camp), "| campaign_desc과 차집합(설명 없는 캠페인):", ct_camp - cd_camp)
    print("coupon 캠페인 수:", len(cp_camp), "| campaign_desc과 차집합:", cp_camp - cd_camp)
    print("coupon_redempt 캠페인 수:", len(cr_camp), "| campaign_desc과 차집합:", cr_camp - cd_camp)

    print(f"\n{'=' * 70}")
    print("[coupon <-> coupon_redempt 연결 점검 (CAMPAIGN + COUPON_UPC 복합키)]")
    coupon_keys = set(zip(coupon["CAMPAIGN"], coupon["COUPON_UPC"]))
    redempt_keys = set(zip(coupon_redempt["CAMPAIGN"], coupon_redempt["COUPON_UPC"]))
    print("coupon 고유 (CAMPAIGN,COUPON_UPC) 조합 수:", len(coupon_keys))
    print("coupon_redempt 고유 (CAMPAIGN,COUPON_UPC) 조합 수:", len(redempt_keys))
    print("coupon_redempt 중 coupon에 없는 조합 수:", len(redempt_keys - coupon_keys))

    print(f"\n{'=' * 70}")
    print("[transaction_data <-> product 연결 점검]")
    tx_products = set(transaction_data["PRODUCT_ID"])
    pd_products = set(product["PRODUCT_ID"])
    print("transaction_data 중 product에 없는 PRODUCT_ID 수:", len(tx_products - pd_products))

    print(f"\n{'=' * 70}")
    print("[DAY / WEEK_NO 범위 (상대 일자 확인)]")
    print("campaign_desc START_DAY/END_DAY 범위:", campaign_desc["START_DAY"].min(), "~", campaign_desc["END_DAY"].max())
    print("transaction_data DAY 범위:", transaction_data["DAY"].min(), "~", transaction_data["DAY"].max())
    print("transaction_data WEEK_NO 범위:", transaction_data["WEEK_NO"].min(), "~", transaction_data["WEEK_NO"].max())
    print("coupon_redempt DAY 범위:", coupon_redempt["DAY"].min(), "~", coupon_redempt["DAY"].max())

    print(f"\n{'=' * 70}")
    print("[COUPON_DISC / COUPON_MATCH_DISC 부호 점검]")
    print("COUPON_DISC 요약:")
    print(transaction_data["COUPON_DISC"].describe())
    print("COUPON_DISC != 0 인 행 수:", (transaction_data["COUPON_DISC"] != 0).sum())
    print("COUPON_MATCH_DISC != 0 인 행 수:", (transaction_data["COUPON_MATCH_DISC"] != 0).sum())
    print("QUANTITY <= 0 인 행 수:", (transaction_data["QUANTITY"] <= 0).sum())
    print("SALES_VALUE < 0 인 행 수:", (transaction_data["SALES_VALUE"] < 0).sum())


if __name__ == "__main__":
    main()
