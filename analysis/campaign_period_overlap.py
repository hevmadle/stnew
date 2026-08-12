"""캠페인별 기간, 수신 가구 수, 기간이 겹치는 다른 캠페인, 처치/대조 후보 가구 수를 계산한다.

- 입력: data/raw/campaign_desc.csv (캠페인 30개, 기간), data/raw/campaign_table.csv (캠페인×수신가구 7,208행)
- 처치 후보 = 선택 캠페인을 받았고, 같은 기간에 겹치는 다른 캠페인은 받지 않은 가구 (CLAUDE.md 분석 설계 규칙 1)
- 대조 후보 = 선택 캠페인을 받지 않았고, 같은 기간에 겹치는 다른 캠페인도 받지 않은 가구 (CLAUDE.md 분석 설계 규칙 2)
- 모집단(대조 후보의 분모)은 이 두 파일만으로 정의 가능한 범위, 즉 campaign_table.csv에
  한 번이라도 등장하는 가구 전체(1,584명)로 한정한다. transaction_data.csv 등 다른 파일에는
  있지만 campaign_table.csv에는 전혀 등장하지 않는 가구(캠페인을 한 번도 받은 적 없는 가구)는
  이번 계산에 포함하지 않았다 — 이 스크립트가 요청받은 두 파일만 사용하기 때문이며, 아래
  출력에도 이 제한을 명시한다.
- 표본/대조군 부족 판정 기준(휴리스틱, CLAUDE.md에 고정값 없음): 처치 후보 < 30명 이거나
  대조 후보 < 30명 이거나 대조/처치 비율 < 1 이면 화면에 경고로 표시한다.

출력은 터미널로만 내보내고 별도 파일을 생성하지 않는다.

실행:
    .venv/bin/python analysis/campaign_period_overlap.py
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

MIN_GROUP_SIZE = 30  # 처치/대조 후보 수가 이 값 미만이면 "표본 부족"으로 표시
MIN_CONTROL_RATIO = 1.0  # 대조/처치 비율이 이 값 미만이면 "대조군 부족"으로 표시


def main() -> None:
    campaign_desc = pd.read_csv(RAW_DIR / "campaign_desc.csv")
    campaign_table = pd.read_csv(RAW_DIR / "campaign_table.csv")

    n_campaigns = campaign_desc["CAMPAIGN"].nunique()
    universe = set(campaign_table["household_key"].unique())
    print(f"campaign_desc.csv 캠페인 수: {n_campaigns}")
    print(f"campaign_table.csv 고유 가구 수(모집단으로 사용): {len(universe):,}")
    print(
        "※ 이 모집단은 campaign_table.csv에 한 번이라도 등장하는 가구만 포함한다. "
        "campaign_table.csv에 전혀 등장하지 않는 가구(어떤 캠페인도 받은 적 없는 가구)는 "
        "이 두 파일만으로는 식별할 수 없어 대조 후보 계산에서 제외됐다.\n"
    )

    # 캠페인별 수신 가구 집합
    households_by_campaign: dict[int, set[int]] = {
        camp: set(sub["household_key"])
        for camp, sub in campaign_table.groupby("CAMPAIGN")
    }

    rows = []
    for _, camp_row in campaign_desc.sort_values("CAMPAIGN").iterrows():
        camp = int(camp_row["CAMPAIGN"])
        desc = camp_row["DESCRIPTION"]
        start, end = int(camp_row["START_DAY"]), int(camp_row["END_DAY"])
        duration = end - start + 1

        recipients = households_by_campaign.get(camp, set())

        # 기간이 겹치는 다른 캠페인 찾기: start1<=end2 and start2<=end1
        other = campaign_desc[campaign_desc["CAMPAIGN"] != camp]
        overlap_mask = (other["START_DAY"] <= end) & (start <= other["END_DAY"])
        overlapping = other[overlap_mask]
        overlapping_camp_ids = overlapping["CAMPAIGN"].tolist()

        overlapping_households: set[int] = set()
        for oc in overlapping_camp_ids:
            overlapping_households |= households_by_campaign.get(int(oc), set())

        treated = recipients - overlapping_households
        control = universe - recipients - overlapping_households

        n_treated = len(treated)
        n_control = len(control)
        ratio = (n_control / n_treated) if n_treated > 0 else float("nan")

        flags = []
        if n_treated < MIN_GROUP_SIZE:
            flags.append(f"처치 표본 부족(<{MIN_GROUP_SIZE})")
        if n_control < MIN_GROUP_SIZE:
            flags.append(f"대조 표본 부족(<{MIN_GROUP_SIZE})")
        elif n_treated > 0 and ratio < MIN_CONTROL_RATIO:
            flags.append(f"대조/처치 비율 낮음({ratio:.2f} < {MIN_CONTROL_RATIO})")
        if n_treated == 0:
            flags.append("처치 후보 0명")

        rows.append(
            {
                "CAMPAIGN": camp,
                "TYPE": desc,
                "START_DAY": start,
                "END_DAY": end,
                "기간(일)": duration,
                "수신가구수(전체)": len(recipients),
                "겹치는캠페인수": len(overlapping_camp_ids),
                "겹치는캠페인목록": ",".join(str(c) for c in sorted(overlapping_camp_ids)) or "-",
                "처치후보": n_treated,
                "대조후보": n_control,
                "대조/처치비율": round(ratio, 2) if n_treated > 0 else None,
                "경고": "; ".join(flags) if flags else "",
            }
        )

    result = pd.DataFrame(rows)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", None)

    print("=" * 100)
    print("[캠페인별 기간 · 수신가구 · 겹치는 캠페인 · 처치/대조 후보]")
    print("=" * 100)
    print(
        result[
            [
                "CAMPAIGN",
                "TYPE",
                "START_DAY",
                "END_DAY",
                "기간(일)",
                "수신가구수(전체)",
                "겹치는캠페인수",
                "처치후보",
                "대조후보",
                "대조/처치비율",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 100)
    print("[겹치는 캠페인 상세 목록]")
    print("=" * 100)
    print(result[["CAMPAIGN", "TYPE", "겹치는캠페인목록"]].to_string(index=False))

    flagged = result[result["경고"] != ""]
    print("\n" + "=" * 100)
    print(f"[표본/대조군 부족 경고 대상: {len(flagged)}개 캠페인]")
    print("=" * 100)
    if len(flagged):
        print(
            flagged[
                ["CAMPAIGN", "TYPE", "처치후보", "대조후보", "대조/처치비율", "경고"]
            ].to_string(index=False)
        )
    else:
        print("경고 대상 없음.")


if __name__ == "__main__":
    main()
