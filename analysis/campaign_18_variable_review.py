"""캠페인 18번 성향점수 모델 입력변수 검토.

검증된 outputs/campaign_18/analysis_data.csv 의 발행 전 특성 11개와, 아직 결합하지 않은
data/raw/hh_demographic.csv 의 인구통계 7개를 후보로 놓고 포함/제외를 결정하기 위한
근거(상관관계, 결측 패턴, 데이터 품질)를 계산한다.

이 스크립트는 결정 자체가 아니라 결정의 근거를 만든다. 최종 채택 목록과 사유는
analysis/campaign_18_ps_variables.md 에 정리한다(CLAUDE.md 파일 관리 규칙 4: 재현 가능하게 남김).

읽는 원본(읽기 전용):
    outputs/campaign_18/analysis_data.csv (이미 검증됨, 재계산하지 않고 그대로 읽음)
    data/raw/hh_demographic.csv

실행:
    .venv/bin/python analysis/campaign_18_variable_review.py
"""

from pathlib import Path

import pandas as pd

from campaign_18_groups import TARGET_CAMPAIGN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ANALYSIS_FILE = PROJECT_ROOT / "outputs" / f"campaign_{TARGET_CAMPAIGN}" / "analysis_data.csv"

BEHAVIOR_CANDIDATES = [
    "recency", "pre_baskets", "pre_sales", "pre_active_days",
    "pre_quantity", "pre_target_purchase", "pre_target_baskets",
    "pre_target_sales", "pre_target_quantity",
    "pre_coupon_redemptions", "pre_campaign_count",
]


def banner(step: str, title: str) -> None:
    print("\n" + "=" * 92)
    print(f"[{step}] {title}")
    print("=" * 92)


def main() -> None:
    banner("STEP 1", "발행 전 구매행동 변수 — 결측·분산·데이터 품질")
    df = pd.read_csv(ANALYSIS_FILE)
    print(f"  입력: {ANALYSIS_FILE.relative_to(PROJECT_ROOT)}  shape={df.shape}")

    quality = pd.DataFrame(
        {
            "결측치": df[BEHAVIOR_CANDIDATES].isna().sum(),
            "분산": df[BEHAVIOR_CANDIDATES].var().round(3),
        }
    )
    print(quality.to_string())

    n_extreme_q = int((df["pre_quantity"] > 10000).sum())
    n_extreme_tq = int((df["pre_target_quantity"] > 10000).sum())
    print(f"\n  pre_quantity > 10,000 인 가구: {n_extreme_q}/{len(df)} ({n_extreme_q/len(df):.0%}) "
          f"— QUANTITY 열의 무게단위 혼입 의심 (이전 단계에서 확인)")
    print(f"  pre_target_quantity > 10,000 인 가구: {n_extreme_tq}/{len(df)} ({n_extreme_tq/len(df):.0%}) "
          f"— 같은 QUANTITY 열에서 파생돼 동일 오염 가능성")

    banner("STEP 2", "발행 전 구매행동 변수 간 상관관계 (다중공선성 점검)")
    corr_cols = ["recency", "pre_baskets", "pre_sales", "pre_active_days",
                 "pre_target_baskets", "pre_target_sales",
                 "pre_coupon_redemptions", "pre_campaign_count"]
    corr = df[corr_cols].corr().round(2)
    pd.set_option("display.width", 200)
    print(corr.to_string())

    pairs = []
    for i, a in enumerate(corr_cols):
        for b in corr_cols[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) >= 0.8:
                pairs.append((a, b, r))
    print("\n  |r| >= 0.8 인 변수쌍(중복정보 의심):")
    for a, b, r in sorted(pairs, key=lambda x: -abs(x[2])):
        print(f"    {a} ~ {b} : r={r}")

    banner("STEP 3", "인구통계(hh_demographic.csv) 커버리지 — 845가구 기준")
    hh_demo = pd.read_csv(RAW_DIR / "hh_demographic.csv")
    hh_845 = set(df["household_key"])
    hh_demo_set = set(hh_demo["household_key"])
    t_hh = set(df.loc[df["treatment"] == 1, "household_key"])
    c_hh = set(df.loc[df["treatment"] == 0, "household_key"])

    cov_all = len(hh_845 & hh_demo_set)
    cov_t = len(t_hh & hh_demo_set)
    cov_c = len(c_hh & hh_demo_set)
    print(f"  845가구 중 hh_demographic 존재: {cov_all} ({cov_all/len(hh_845):.1%})")
    print(f"  처치 {len(t_hh)}가구 중 존재   : {cov_t} ({cov_t/len(t_hh):.1%})")
    print(f"  대조 {len(c_hh)}가구 중 존재   : {cov_c} ({cov_c/len(c_hh):.1%})")
    print(f"  커버리지 차이(처치-대조)      : {cov_t/len(t_hh) - cov_c/len(c_hh):+.1%}p")
    print("  → 커버리지가 처치/대조 간에 크게 다르면, 인구통계로 완전사례분석을 할 때")
    print("    표본이 무작위로 줄지 않고 처치집단 쪽에 유리하게 치우친다(선택 편향 위험).")

    print("\n  hh_demographic.csv 열별 값(파일 내 결측은 없고, 'Unknown'류 범주로 흡수돼 있음):")
    for col in hh_demo.columns:
        if col == "household_key":
            continue
        vals = sorted(hh_demo[col].unique().tolist())
        unknown_like = [v for v in vals if "unknown" in str(v).lower()]
        print(f"    {col}: {len(vals)}개 범주" + (f" (그 중 결측성 범주: {unknown_like})" if unknown_like else ""))

    banner("STEP 4", "결론 요약 (근거만 — 최종 채택 목록은 campaign_18_ps_variables.md)")
    print("  1) pre_active_days 는 pre_baskets 와 r=0.97 → 사실상 같은 정보, 하나만 쓴다.")
    print("  2) pre_quantity·pre_target_quantity 는 QUANTITY 단위 혼입으로 절반 가까운 가구가 오염 → 제외.")
    print("  3) pre_target_purchase 는 분산 0(양쪽 100%) → 제외.")
    print(f"  4) hh_demographic 커버리지 {cov_all/len(hh_845):.0%}, 처치·대조 간 격차 "
          f"{cov_t/len(t_hh) - cov_c/len(c_hh):+.0%}p → 비무작위 결측 의심, 기본 모델에서 제외.")


if __name__ == "__main__":
    main()
