"""캠페인 18번 처치·대조 단순 비교 — 보정 전 관찰된 차이.

검증된 outputs/campaign_18/analysis_data.csv 를 그대로 읽어(재계산하지 않음) 처치집단과
대조집단의 결과변수 평균을 단순 비교한다. 성향점수 매칭이나 다른 보정을 전혀 하지 않은
값이므로, 이 단계의 숫자는 "관찰된 차이"이며 인과효과가 아니다(CLAUDE.md 품질과 해석 규칙 2).

입력: outputs/campaign_18/analysis_data.csv (analysis/campaign_18_build_analysis_table.py 산출물)
출력: 터미널만 (파일 생성 없음, CLAUDE.md 파일 관리 규칙 1)

실행:
    .venv/bin/python analysis/campaign_18_simple_comparison.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from campaign_18_groups import TARGET_CAMPAIGN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "outputs" / f"campaign_{TARGET_CAMPAIGN}" / "analysis_data.csv"

PRIMARY_COLUMNS = ["target_purchase", "target_sales", "target_quantity"]
SECONDARY_COLUMNS = ["any_purchase", "total_sales", "baskets"]

LABELS = {
    "target_purchase": "대상 상품 구매 여부",
    "target_sales": "대상 상품 구매금액",
    "target_quantity": "대상 상품 구매수량",
    "any_purchase": "전체 상품 구매 여부",
    "total_sales": "전체 구매금액",
    "baskets": "장바구니 수",
}


def banner(step: str, title: str) -> None:
    print("\n" + "=" * 92)
    print(f"[{step}] {title}")
    print("=" * 92)


def compare(df: pd.DataFrame, cols: list[str], period_label: str) -> pd.DataFrame:
    t = df[df["treatment"] == 1]
    c = df[df["treatment"] == 0]
    rows = []
    for col in cols:
        mean_t, mean_c = t[col].mean(), c[col].mean()
        diff = mean_t - mean_c
        rel = (diff / mean_c * 100) if mean_c != 0 else np.nan
        rows.append(
            {
                "구분": period_label,
                "변수": LABELS.get(col, col),
                "열": col,
                "처치평균": round(mean_t, 3),
                "대조평균": round(mean_c, 3),
                "차이": round(diff, 3),
                "상대차이(%)": round(rel, 1) if pd.notna(rel) else None,
                "처치n": len(t),
                "대조n": len(c),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    # ------------------------------------------------------------------
    banner("STEP 1", "검증된 분석표 불러오기")
    # ------------------------------------------------------------------
    if not INPUT_FILE.exists():
        raise SystemExit(
            f"{INPUT_FILE} 가 없다. 먼저 .venv/bin/python analysis/campaign_18_build_analysis_table.py 를 실행하라."
        )

    df = pd.read_csv(INPUT_FILE)
    print(f"  입력 파일: {INPUT_FILE.relative_to(PROJECT_ROOT)}")
    print(f"  shape    : {df.shape}")
    print(f"  가구 수  : {len(df):,} (처치 {int((df['treatment']==1).sum()):,} / "
          f"대조 {int((df['treatment']==0).sum()):,})")
    assert df["household_key"].duplicated().sum() == 0, "household_key 중복 발견"
    assert df["treatment"].isin([0, 1]).all(), "treatment 값이 0/1 이 아님"

    # ------------------------------------------------------------------
    banner("STEP 2", "주요 결과 — 캠페인 18 대상 상품 (실제 기간 DAY 587~642)")
    # ------------------------------------------------------------------
    primary = compare(df, PRIMARY_COLUMNS, "실제 기간")
    pd.set_option("display.width", 200)
    print(primary.drop(columns=["구분"]).to_string(index=False))

    # ------------------------------------------------------------------
    banner("STEP 3", "보조 결과 — 전체 상품 (실제 기간)")
    # ------------------------------------------------------------------
    secondary = compare(df, SECONDARY_COLUMNS, "실제 기간")
    print(secondary.drop(columns=["구분"]).to_string(index=False))

    # ------------------------------------------------------------------
    banner("STEP 4", "참고 — 첫 30일 기준 · 일평균 (캠페인 18은 56일, 표준 33일보다 김)")
    # ------------------------------------------------------------------
    first30_cols = [f"{c}_first30" for c in PRIMARY_COLUMNS + SECONDARY_COLUMNS]
    if all(c in df.columns for c in first30_cols):
        first30 = compare(df, first30_cols, "첫 30일")
        first30["열"] = first30["열"].str.replace("_first30", "")
        first30["변수"] = first30["열"].map(lambda c: LABELS.get(c, c))
        print("  [첫 30일]")
        print(first30.drop(columns=["구분"]).to_string(index=False))

    per_day_cols = [c for c in ["target_sales_per_day", "total_sales_per_day"] if c in df.columns]
    if per_day_cols:
        per_day = compare(df, per_day_cols, "일평균")
        print("\n  [일평균]")
        print(per_day.drop(columns=["구분"]).to_string(index=False))

    # ------------------------------------------------------------------
    banner("STEP 5", "요약")
    # ------------------------------------------------------------------
    print("  처치집단이 대조집단보다 모든 주요·보조 결과에서 높게 관찰됨:")
    combined = pd.concat([primary, secondary]).drop(columns=["구분", "처치n", "대조n"])
    print(combined.to_string(index=False))
    print()
    print("  ⚠ 이 수치는 처치·대조 배정 시점의 발행 전 특성 차이(SMD 최대 0.854, 이전 단계 확인)를")
    print("    전혀 보정하지 않은 단순 평균 차이다. 처치집단이 원래 더 활발한 구매층이었을 가능성이")
    print("    커서, 위 차이 중 얼마가 캠페인 효과이고 얼마가 원래 차이인지 이 단계에서는 구분할 수 없다.")
    print("    → 인과효과로 표현하지 않는다(CLAUDE.md 품질과 해석 규칙 2). 다음 단계는 성향점수 매칭이다.")


if __name__ == "__main__":
    main()
