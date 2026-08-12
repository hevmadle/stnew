"""캠페인 18번 효과 추정치 시각화 — 매칭 전/후 차이와 95% 신뢰구간 (forest plot).

analysis/campaign_18_effect_estimate.py 가 저장한 outputs/campaign_18/results.json 을
그대로 읽어(재계산하지 않음) 주요·보조 결과 6개의 매칭 전/후 차이와 95% CI를
가로 오차막대(forest plot)로 그린다. 0을 지나는 막대는 그 신뢰수준에서 차이가
통계적으로 구분되지 않는다는 뜻이다(0을 포함 = 회색 배경으로 표시).

산출: outputs/campaign_18/effect_forest_plot.png (진단·보고용 이미지)

실행:
    .venv/bin/python analysis/campaign_18_effect_forest_plot.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from campaign_18_groups import TARGET_CAMPAIGN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / f"campaign_{TARGET_CAMPAIGN}"
RESULTS_FILE = OUTPUT_DIR / "results.json"

COLOR_MATCHED = "#2a78d6"      # 매칭 후 (slot1 blue)
COLOR_UNADJUSTED = "#eb6834"   # 매칭 전 (slot2 orange)
COLOR_ZERO_BAND = "#898781"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

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

ORDER = [
    ("primary_target_product", "target_purchase"),
    ("primary_target_product", "target_sales"),
    ("primary_target_product", "target_quantity"),
    ("secondary_overall", "any_purchase"),
    ("secondary_overall", "total_sales"),
    ("secondary_overall", "baskets"),
]


def banner(step: str, title: str) -> None:
    print("\n" + "=" * 92)
    print(f"[{step}] {title}")
    print("=" * 92)


def draw_row(ax, y, diff, ci, color, row_label):
    lo, hi = ci
    ax.errorbar(
        diff, y, xerr=[[diff - lo], [hi - diff]],
        fmt="o", color=color, ecolor=color, elinewidth=2, capsize=4, markersize=6, zorder=4,
    )
    sign = "+" if diff >= 0 else ""
    ax.text(hi, y, f"  {sign}{diff:.2f} [{lo:.2f}, {hi:.2f}]", va="center", ha="left",
            fontsize=7.5, color=INK_SECONDARY)


def main() -> None:
    # ------------------------------------------------------------------
    banner("STEP 1", "results.json 불러오기")
    # ------------------------------------------------------------------
    if not RESULTS_FILE.exists():
        raise SystemExit(f"{RESULTS_FILE} 가 없다. campaign_18_effect_estimate.py 를 먼저 실행하라.")

    with open(RESULTS_FILE, encoding="utf-8") as f:
        results = json.load(f)
    print(f"  입력: {RESULTS_FILE.relative_to(PROJECT_ROOT)}")
    print(f"  캠페인 {results['campaign']} / 기간 DAY {results['campaign_period_day']['start']}~"
          f"{results['campaign_period_day']['end']} / 매칭 {results['sample']['matching']['n_pairs']}쌍")

    # ------------------------------------------------------------------
    banner("STEP 2", "그래프 — 매칭 전/후 차이와 95% CI (forest plot)")
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 6.5), dpi=150)
    axes = axes.flatten()

    for i, (bucket, key) in enumerate(ORDER):
        ax = axes[i]
        eff = results["effects"][bucket][key]
        label = eff["label"]

        ax.axvline(0, color=AXIS, linewidth=1, zorder=1)

        draw_row(ax, 1, eff["matched_diff"], eff["matched_ci95"], COLOR_MATCHED, "매칭 후")
        draw_row(ax, 0, eff["unadjusted_diff"], eff["unadjusted_ci95"], COLOR_UNADJUSTED, "매칭 전")

        ax.set_yticks([0, 1])
        ax.set_yticklabels(["매칭 전\n(845가구)", "매칭 후\n(228쌍)"], fontsize=8.5)
        ax.set_ylim(-0.6, 1.6)

        all_vals = eff["unadjusted_ci95"] + eff["matched_ci95"] + [0]
        pad = (max(all_vals) - min(all_vals)) * 0.35 + 1e-6
        ax.set_xlim(min(all_vals) - pad, max(all_vals) + pad * 1.8)

        prefix = "[주요] " if bucket == "primary_target_product" else "[보조] "
        ax.set_title(prefix + label, fontsize=10, color=INK_PRIMARY, loc="left")
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(left=False)
        ax.set_axisbelow(True)

    handles = [
        plt.Line2D([0], [0], marker="o", color=COLOR_MATCHED, linestyle="", markersize=7, label="매칭 후 (대응표본 CI)"),
        plt.Line2D([0], [0], marker="o", color=COLOR_UNADJUSTED, linestyle="", markersize=7, label="매칭 전 (독립표본 CI)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04), frameon=False, fontsize=10)
    fig.suptitle(
        f"캠페인 {results['campaign']} 처치-대조 차이와 95% 신뢰구간 — 점선(x=0) 통과 시 차이가 유의하지 않음",
        fontsize=11, color=INK_PRIMARY, y=1.10,
    )
    fig.tight_layout()

    out_path = OUTPUT_DIR / "effect_forest_plot.png"
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {out_path.relative_to(PROJECT_ROOT)}")

    # ------------------------------------------------------------------
    banner("STEP 3", "0을 포함하는 CI 요약")
    # ------------------------------------------------------------------
    for bucket, key in ORDER:
        eff = results["effects"][bucket][key]
        u_lo, u_hi = eff["unadjusted_ci95"]
        m_lo, m_hi = eff["matched_ci95"]
        u_sig = "유의" if not (u_lo <= 0 <= u_hi) else "0 포함"
        m_sig = "유의" if not (m_lo <= 0 <= m_hi) else "0 포함"
        print(f"  {eff['label']:14s} 매칭전={u_sig:6s} 매칭후={m_sig}")


if __name__ == "__main__":
    main()
