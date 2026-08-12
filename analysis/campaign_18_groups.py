"""캠페인 18번 분석집단 구성 — 처치 / 대조 / 제외.

analysis/campaign_18_profile.py 에서 확인한 사실을 이어받아, 전체 가구를 세 집단으로 나눈다.

    1. 처치집단 : 캠페인 18을 받았고, 18 기간과 겹치는 다른 캠페인은 받지 않은 가구
    2. 대조집단 : 캠페인 18을 받지 않았고, 18 기간과 겹치는 다른 캠페인도 받지 않은 가구
    3. 제외     : 18 기간과 겹치는 다른 캠페인에 노출된 가구

전체 가구 모집단은 transaction_data.csv 의 household_key(2,500가구)로 잡는다.
이 중 캠페인을 한 번도 받은 적 없는 가구(916)는 위 2번 정의를 문자 그대로는 충족하지만,
직전 단계에서 대조집단에 포함하지 않기로 정했으므로 제외집단에 별도 사유로 분류한다.
이 결정은 INCLUDE_NEVER_CAMPAIGNED 로 한 줄에서 뒤집을 수 있게 두었다.

읽는 원본(읽기 전용, 수정 없음):
    data/raw/campaign_desc.csv    : CAMPAIGN, START_DAY, END_DAY  → 기간과 겹침 판정
    data/raw/campaign_table.csv   : CAMPAIGN, household_key       → 수신 여부
    data/raw/transaction_data.csv : household_key 열만            → 전체 가구 모집단

결과는 터미널에만 출력한다. 분석표(analysis_data.csv)는 발행 전 특성과 결과변수가 붙는
다음 단계에서 만든다(CLAUDE.md 파일 관리 규칙 1·2).

실행:
    .venv/bin/python analysis/campaign_18_groups.py
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

TARGET_CAMPAIGN = 18

# 캠페인을 한 번도 받은 적 없는 가구(916)를 대조집단에 넣을지 여부.
# False = 대조집단은 "캠페인 경험은 있으나 18 기간과 겹치는 캠페인은 받지 않은 가구"로 한정한다.
INCLUDE_NEVER_CAMPAIGNED = False

GROUP_TREATED = "1. 처치집단"
GROUP_CONTROL = "2. 대조집단"
GROUP_EXCLUDED = "3. 제외"


def banner(step: str, title: str) -> None:
    print("\n" + "=" * 92)
    print(f"[{step}] {title}")
    print("=" * 92)


def build_groups() -> tuple[pd.DataFrame, dict]:
    """전체 가구에 집단 라벨을 붙인 DataFrame과 계산에 쓴 중간값을 돌려준다."""
    campaign_desc = pd.read_csv(RAW_DIR / "campaign_desc.csv")
    campaign_table = pd.read_csv(RAW_DIR / "campaign_table.csv")
    all_households = set(
        pd.read_csv(RAW_DIR / "transaction_data.csv", usecols=["household_key"])["household_key"].unique()
    )

    # --- 캠페인 18의 기간을 원본에서 조회 (코드에 고정하지 않는다) ---
    desc_row = campaign_desc.loc[campaign_desc["CAMPAIGN"] == TARGET_CAMPAIGN]
    if len(desc_row) != 1:
        raise SystemExit(f"campaign_desc.csv에서 CAMPAIGN={TARGET_CAMPAIGN} 행을 하나로 특정할 수 없다.")
    start_day = int(desc_row["START_DAY"].iloc[0])
    end_day = int(desc_row["END_DAY"].iloc[0])

    # --- 기간이 겹치는 다른 캠페인 ---
    others = campaign_desc.loc[campaign_desc["CAMPAIGN"] != TARGET_CAMPAIGN]
    overlapping_ids = sorted(
        others.loc[(others["START_DAY"] <= end_day) & (start_day <= others["END_DAY"]), "CAMPAIGN"].tolist()
    )

    # --- 가구 집합 ---
    by_campaign = campaign_table.groupby("CAMPAIGN")["household_key"].apply(set).to_dict()
    campaigned = set(campaign_table["household_key"])
    recipients = by_campaign.get(TARGET_CAMPAIGN, set())
    overlapping_households: set[int] = set()
    for cid in overlapping_ids:
        overlapping_households |= by_campaign.get(int(cid), set())

    never_campaigned = all_households - campaigned

    # --- 집단 배정 ---
    treated = recipients - overlapping_households
    excluded_treated_side = recipients & overlapping_households
    excluded_control_side = (campaigned - recipients) & overlapping_households
    control_campaigned = campaigned - recipients - overlapping_households

    if INCLUDE_NEVER_CAMPAIGNED:
        control = control_campaigned | never_campaigned
        excluded_never = set()
    else:
        control = control_campaigned
        excluded_never = never_campaigned

    excluded = excluded_treated_side | excluded_control_side | excluded_never

    # --- 라벨 DataFrame ---
    assignment = pd.DataFrame({"household_key": sorted(all_households)})
    label = pd.Series(GROUP_EXCLUDED, index=assignment.index)
    label[assignment["household_key"].isin(treated).values] = GROUP_TREATED
    label[assignment["household_key"].isin(control).values] = GROUP_CONTROL
    assignment["group"] = label.values
    assignment["treatment"] = (assignment["group"] == GROUP_TREATED).astype(int)

    def reason(hh: int) -> str:
        if hh in treated:
            return "캠페인 18 단독 수신"
        if hh in control:
            return "18 미수신, 겹치는 캠페인도 미수신"
        if hh in excluded_treated_side:
            return "제외: 18 수신 + 겹치는 캠페인도 수신"
        if hh in excluded_control_side:
            return "제외: 18 미수신 + 겹치는 캠페인 수신"
        return "제외: 캠페인 수신 이력 없음"

    assignment["reason"] = assignment["household_key"].map(reason)

    meta = {
        "start_day": start_day,
        "end_day": end_day,
        "overlapping_ids": overlapping_ids,
        "all_households": all_households,
        "campaigned": campaigned,
        "recipients": recipients,
        "overlapping_households": overlapping_households,
        "treated": treated,
        "control": control,
        "excluded": excluded,
        "excluded_treated_side": excluded_treated_side,
        "excluded_control_side": excluded_control_side,
        "excluded_never": excluded_never,
        "never_campaigned": never_campaigned,
        "campaign_table": campaign_table,
    }
    return assignment, meta


def main() -> None:
    assignment, m = build_groups()

    n_all = len(m["all_households"])
    treated, control, excluded = m["treated"], m["control"], m["excluded"]

    # ------------------------------------------------------------------
    banner("STEP 1", f"집단 구성 기준 — 캠페인 {TARGET_CAMPAIGN}")
    # ------------------------------------------------------------------
    print(f"  캠페인 기간          : DAY {m['start_day']} ~ {m['end_day']}  (campaign_desc.csv → START_DAY, END_DAY)")
    print(f"  겹치는 캠페인 {len(m['overlapping_ids'])}개    : {m['overlapping_ids']}")
    print(f"  전체 가구 모집단     : {n_all:,}가구  (transaction_data.csv → household_key 고유값)")
    print(f"  캠페인 수신 이력 가구: {len(m['campaigned']):,}가구  (campaign_table.csv → household_key 고유값)")
    print()
    print(f"  캠페인 무경험 가구를 대조집단에 포함: {INCLUDE_NEVER_CAMPAIGNED}")

    # ------------------------------------------------------------------
    banner("STEP 2", "집단별 가구 수")
    # ------------------------------------------------------------------
    table = pd.DataFrame(
        [
            (GROUP_TREATED, "캠페인 18을 받고, 겹치는 캠페인은 받지 않음", len(treated)),
            (GROUP_CONTROL, "캠페인 18을 받지 않고, 겹치는 캠페인도 받지 않음", len(control)),
            (GROUP_EXCLUDED, "겹치는 캠페인에 노출됨 (+ 캠페인 무경험)", len(excluded)),
        ],
        columns=["집단", "정의", "가구수"],
    )
    table["비율"] = (table["가구수"] / n_all).map(lambda x: f"{x:.1%}")
    table.loc[len(table)] = ["합계", "", table["가구수"].sum(), "100.0%"]
    print(table.to_string(index=False))

    print()
    print("  제외집단 내역:")
    detail = pd.DataFrame(
        [
            ("18 수신 + 겹치는 캠페인도 수신", len(m["excluded_treated_side"]), "동시 노출로 처치효과 분리 불가"),
            ("18 미수신 + 겹치는 캠페인 수신", len(m["excluded_control_side"]), "동시 노출로 대조군 자격 상실"),
            ("캠페인 수신 이력 없음", len(m["excluded_never"]), "직전 단계 결정에 따라 대조집단에서 제외"),
        ],
        columns=["제외 사유", "가구수", "설명"],
    )
    detail.loc[len(detail)] = ["소계", detail["가구수"].sum(), ""]
    print(detail.to_string(index=False))

    # ------------------------------------------------------------------
    banner("STEP 3", "집단 배타성 검증")
    # ------------------------------------------------------------------
    checks: list[tuple[str, bool, str]] = []

    overlap_tc = treated & control
    checks.append(
        ("처치집단 ∩ 대조집단 == 공집합", len(overlap_tc) == 0, f"교집합 {len(overlap_tc)}가구")
    )
    checks.append(
        ("처치집단 ∩ 제외집단 == 공집합", len(treated & excluded) == 0, f"교집합 {len(treated & excluded)}가구")
    )
    checks.append(
        ("대조집단 ∩ 제외집단 == 공집합", len(control & excluded) == 0, f"교집합 {len(control & excluded)}가구")
    )
    checks.append(
        (
            "세 집단 합집합 == 전체 가구",
            (treated | control | excluded) == m["all_households"],
            f"{len(treated | control | excluded):,} vs {n_all:,}",
        )
    )
    checks.append(
        (
            "세 집단 가구수 합 == 전체 가구수",
            len(treated) + len(control) + len(excluded) == n_all,
            f"{len(treated) + len(control) + len(excluded):,} vs {n_all:,}",
        )
    )

    # 라벨 DataFrame 쪽에서도 같은 결론이 나오는지 독립 확인
    counts = assignment["group"].value_counts()
    checks.append(
        (
            "라벨 DataFrame 행 수 == 전체 가구수",
            len(assignment) == n_all,
            f"{len(assignment):,}행",
        )
    )
    checks.append(
        (
            "라벨 DataFrame에 household_key 중복 없음",
            not assignment["household_key"].duplicated().any(),
            f"중복 {int(assignment['household_key'].duplicated().sum())}건",
        )
    )
    checks.append(
        (
            "라벨 집계 == 집합 연산 결과",
            counts.get(GROUP_TREATED, 0) == len(treated)
            and counts.get(GROUP_CONTROL, 0) == len(control)
            and counts.get(GROUP_EXCLUDED, 0) == len(excluded),
            f"처치 {counts.get(GROUP_TREATED, 0)} / 대조 {counts.get(GROUP_CONTROL, 0)} / 제외 {counts.get(GROUP_EXCLUDED, 0)}",
        )
    )

    for name, ok, note in checks:
        print(f"  [{'통과' if ok else '실패'}] {name:38s} — {note}")

    # ------------------------------------------------------------------
    banner("STEP 4", "집단 내용 검증 — 정의대로 구성됐는가")
    # ------------------------------------------------------------------
    ct = m["campaign_table"]
    recipients, ovl_hh = m["recipients"], m["overlapping_households"]

    content_checks = [
        ("처치집단 전원이 캠페인 18 수신자", treated <= recipients, f"미수신 {len(treated - recipients)}가구"),
        ("처치집단 중 겹치는 캠페인 수신자 0", len(treated & ovl_hh) == 0, f"{len(treated & ovl_hh)}가구"),
        ("대조집단 중 캠페인 18 수신자 0", len(control & recipients) == 0, f"{len(control & recipients)}가구"),
        ("대조집단 중 겹치는 캠페인 수신자 0", len(control & ovl_hh) == 0, f"{len(control & ovl_hh)}가구"),
        (
            "처치·대조 전원이 거래데이터에 존재",
            (treated | control) <= m["all_households"],
            f"누락 {len((treated | control) - m['all_households'])}가구",
        ),
    ]
    for name, ok, note in content_checks:
        print(f"  [{'통과' if ok else '실패'}] {name:38s} — {note}")

    print()
    print("  집단별로 실제 수신한 캠페인 목록:")
    for gname, hhset in [(GROUP_TREATED, treated), (GROUP_CONTROL, control)]:
        got = sorted(ct.loc[ct["household_key"].isin(hhset), "CAMPAIGN"].unique().tolist())
        conflict = sorted(set(got) & set(m["overlapping_ids"]))
        print(f"    {gname}: {got}")
        print(f"      겹치는 캠페인 포함: {conflict or '없음'}")

    all_ok = all(ok for _, ok, _ in checks) and all(ok for _, ok, _ in content_checks)

    # ------------------------------------------------------------------
    banner("STEP 5", "분석집단 확정")
    # ------------------------------------------------------------------
    print(assignment["group"].value_counts().sort_index().to_string())
    print()
    print("  라벨 DataFrame 예시:")
    sample = pd.concat(
        [
            assignment[assignment["group"] == GROUP_TREATED].head(2),
            assignment[assignment["group"] == GROUP_CONTROL].head(2),
            assignment[assignment["group"] == GROUP_EXCLUDED].head(2),
        ]
    )
    print(sample.to_string(index=False))

    n_analysis = len(treated) + len(control)
    ratio = len(control) / len(treated) if treated else float("nan")
    print()
    print(f"  분석 표본  : {n_analysis:,}가구 (처치 {len(treated):,} + 대조 {len(control):,})")
    print(f"  대조/처치  : {ratio:.2f}")
    print(f"  검증 결과  : {'모든 검증 통과' if all_ok else '검증 실패 항목 있음 — 확인 필요'}")
    print()
    print("  다음 단계: 발행 전 특성(DAY < "
          f"{m['start_day']})과 캠페인 기간 결과변수(DAY {m['start_day']}~{m['end_day']})를")
    print("  transaction_data.csv에서 계산해 analysis_data.csv를 만든다. 이 단계에서는 파일을 만들지 않는다.")


if __name__ == "__main__":
    main()
