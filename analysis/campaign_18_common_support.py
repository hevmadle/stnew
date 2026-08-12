"""캠페인 18번 성향점수 공통지지영역(common support) 분석.

analysis/campaign_18_propensity_score.py 가 계산해 outputs/campaign_18/analysis_data.csv 에
저장한 p_score 를 그대로 읽어(재계산하지 않음), 처치·대조 분포를 겹쳐 그리고 공통지지영역을
계산한다. 영역 밖 가구가 어떤 발행 전 특성을 가졌는지도 비교한다.

공통지지영역 정의: [max(처치 p_score 최솟값, 대조 p_score 최솟값),
                    min(처치 p_score 최댓값, 대조 p_score 최댓값)]
이 범위 밖의 가구는 반대 집단에 비교 가능한 짝이 존재할 가능성이 낮은 가구다.

이 단계에서는 매칭이나 트리밍(제거)을 하지 않는다 — 영역 안/밖을 확인만 한다.
다음 단계(매칭)에서 트리밍 여부를 결정한다.

산출:
    outputs/campaign_18/common_support.png  (분포 겹침 그래프, 진단용 이미지)

실행:
    .venv/bin/python analysis/campaign_18_common_support.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from campaign_18_groups import TARGET_CAMPAIGN
from campaign_18_propensity_score import PS_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / f"campaign_{TARGET_CAMPAIGN}"
ANALYSIS_FILE = OUTPUT_DIR / "analysis_data.csv"

COLOR_TREATED = "#2a78d6"
COLOR_CONTROL = "#eb6834"
COLOR_OUTSIDE = "#898781"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

LABELS = {
    "recency": "최근 구매 이후 경과일",
    "pre_baskets": "장바구니 수",
    "pre_sales": "구매금액",
    "pre_target_baskets": "대상 상품 장바구니 수",
    "pre_coupon_redemptions": "쿠폰 사용 횟수",
    "pre_campaign_count": "캠페인 수신 횟수",
}

plt.rcParams.update({
    "font.family": "AppleGothic",
    "axes.unicode_minus": False,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "grid.color": GRIDLINE,
})


def banner(step: str, title: str) -> None:
    print("\n" + "=" * 92)
    print(f"[{step}] {title}")
    print("=" * 92)


def main() -> None:
    # ------------------------------------------------------------------
    banner("STEP 1", "p_score 불러오기")
    # ------------------------------------------------------------------
    if "p_score" not in pd.read_csv(ANALYSIS_FILE, nrows=0).columns:
        raise SystemExit("p_score 열이 없다. analysis/campaign_18_propensity_score.py 를 먼저 실행하라.")

    df = pd.read_csv(ANALYSIS_FILE)
    t = df[df["treatment"] == 1]
    c = df[df["treatment"] == 0]
    print(f"  입력: {ANALYSIS_FILE.relative_to(PROJECT_ROOT)}  shape={df.shape}")
    print(f"  처치 {len(t)}가구 / 대조 {len(c)}가구")

    # ------------------------------------------------------------------
    banner("STEP 2", "공통지지영역 계산")
    # ------------------------------------------------------------------
    t_min, t_max = t["p_score"].min(), t["p_score"].max()
    c_min, c_max = c["p_score"].min(), c["p_score"].max()
    lo, hi = max(t_min, c_min), min(t_max, c_max)

    print(f"  처치 p_score 범위: [{t_min:.4f}, {t_max:.4f}]")
    print(f"  대조 p_score 범위: [{c_min:.4f}, {c_max:.4f}]")
    print(f"  공통지지영역     : [{lo:.4f}, {hi:.4f}]  "
          f"(하한 = {'처치' if t_min>c_min else '대조'} 최솟값, 상한 = {'처치' if t_max<c_max else '대조'} 최댓값)")

    df["in_support"] = (df["p_score"] >= lo) & (df["p_score"] <= hi)

    # ------------------------------------------------------------------
    banner("STEP 3", "집단별 영역 안/밖 가구 수")
    # ------------------------------------------------------------------
    tab = df.groupby(["group", "in_support"]).size().unstack(fill_value=0)
    tab = tab.rename(columns={True: "영역 안", False: "영역 밖"})[["영역 안", "영역 밖"]]
    tab["합계"] = tab.sum(axis=1)
    tab["영역 밖 비율"] = (tab["영역 밖"] / tab["합계"]).map(lambda x: f"{x:.1%}")
    print(tab.to_string())

    n_t_out = int((t["p_score"] < lo).sum() + (t["p_score"] > hi).sum())
    n_c_out = int((c["p_score"] < lo).sum() + (c["p_score"] > hi).sum())
    n_t_out_hi = int((t["p_score"] > hi).sum())  # 처치는 하한 쪽엔 거의 없을 것 — 대조 최솟값이 보통 더 작음
    n_c_out_lo = int((c["p_score"] < lo).sum())
    print(f"\n  처치 중 영역 밖: {n_t_out}가구 (상한 초과 {n_t_out_hi}, 하한 미만 {n_t_out - n_t_out_hi})")
    print(f"  대조 중 영역 밖: {n_c_out}가구 (하한 미만 {n_c_out_lo}, 상한 초과 {n_c_out - n_c_out_lo})")
    print(f"  전체 영역 밖    : {n_t_out + n_c_out}가구 / {len(df)}가구 ({(n_t_out+n_c_out)/len(df):.1%})")

    # ------------------------------------------------------------------
    banner("STEP 4", "영역 밖 가구의 발행 전 특성 비교")
    # ------------------------------------------------------------------
    print("  처치집단: 영역 안 vs 영역 밖")
    t_compare = t.groupby(df.loc[t.index, "in_support"])[PS_FEATURES + ["p_score"]].mean()
    t_compare.index = t_compare.index.map({True: "영역 안", False: "영역 밖"})
    print(t_compare.round(2).to_string())

    print("\n  대조집단: 영역 안 vs 영역 밖")
    c_compare = c.groupby(df.loc[c.index, "in_support"])[PS_FEATURES + ["p_score"]].mean()
    c_compare.index = c_compare.index.map({True: "영역 안", False: "영역 밖"})
    print(c_compare.round(2).to_string())

    out_t = df[(df["treatment"] == 1) & (~df["in_support"])]
    out_c = df[(df["treatment"] == 0) & (~df["in_support"])]
    print(f"\n  영역 밖 처치 가구 household_key: {sorted(out_t['household_key'].tolist())}")
    print(f"  영역 밖 대조 가구 household_key(상위10개만): {sorted(out_c['household_key'].tolist())[:10]}"
          f"{' ...' if len(out_c) > 10 else ''}")

    # ------------------------------------------------------------------
    banner("STEP 5", "그래프 — p_score 분포 겹침 + 공통지지영역")
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    bins = np.linspace(0, 1, 41)

    ax.axvspan(0, lo, color=COLOR_OUTSIDE, alpha=0.12, zorder=0)
    ax.axvspan(hi, 1, color=COLOR_OUTSIDE, alpha=0.12, zorder=0)

    ax.hist(t["p_score"], bins=bins, color=COLOR_TREATED, alpha=0.55, density=True,
            label=f"처치집단 (n={len(t)})", zorder=3)
    ax.hist(c["p_score"], bins=bins, color=COLOR_CONTROL, alpha=0.55, density=True,
            label=f"대조집단 (n={len(c)})", zorder=2)

    ax.axvline(lo, color=INK_SECONDARY, linewidth=1.3, linestyle="--", zorder=4)
    ax.axvline(hi, color=INK_SECONDARY, linewidth=1.3, linestyle="--", zorder=4)
    ymax = ax.get_ylim()[1]
    ax.text(lo, ymax * 0.97, f" 공통지지영역 하한\n {lo:.3f}", fontsize=8, color=INK_SECONDARY,
            ha="left", va="top")
    ax.text(hi, ymax * 0.97, f"공통지지영역 상한 \n{hi:.3f} ", fontsize=8, color=INK_SECONDARY,
            ha="right", va="top")

    ax.set_xlabel("성향점수 (p_score) — 캠페인 18 수신확률 추정값")
    ax.set_ylabel("밀도")
    ax.set_title(f"캠페인 {TARGET_CAMPAIGN} 성향점수 분포와 공통지지영역\n"
                 f"회색 음영 = 영역 밖 (처치 {n_t_out}가구, 대조 {n_c_out}가구)",
                 fontsize=11, color=INK_PRIMARY, loc="left")
    ax.legend(frameon=False, fontsize=10)
    ax.set_xlim(0, 1)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    out_path = OUTPUT_DIR / "common_support.png"
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    print(f"  저장: {out_path.relative_to(PROJECT_ROOT)}")

    # ------------------------------------------------------------------
    banner("STEP 6", "요약")
    # ------------------------------------------------------------------
    print(f"  공통지지영역: [{lo:.4f}, {hi:.4f}]")
    print(f"  영역 안: 처치 {len(t) - n_t_out}/{len(t)} ({(len(t)-n_t_out)/len(t):.1%}), "
          f"대조 {len(c) - n_c_out}/{len(c)} ({(len(c)-n_c_out)/len(c):.1%})")
    print(f"  영역 밖: 처치 {n_t_out}가구, 대조 {n_c_out}가구 (합 {n_t_out + n_c_out}가구)")
    print("  ※ 이 단계에서는 트리밍하지 않았다. 분석표(analysis_data.csv)는 변경하지 않았고")
    print("    in_support 플래그는 이 스크립트 안에서만 계산했다(매칭 단계에서 다시 계산해 사용).")


if __name__ == "__main__":
    main()
