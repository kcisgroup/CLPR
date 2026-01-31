#!/usr/bin/env python3
"""
重排消融实验可视化：Query-Only vs Profile-Only vs Hybrid (Profile+Query)

学习参考图片的柱状图+折线图风格，蓝色系配色
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np


def resolve_font(preferred_fonts: list[str]) -> str:
    """Pick the first installed font from the preferred serif stack."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for font in preferred_fonts:
        if font in installed:
            return font
    return preferred_fonts[-1]


# --------- 全局排版：IEEE JBHI (Times) ----------
SERIF_STACK = ["Times New Roman", "CMU Serif", "DejaVu Serif"]
PRIMARY_FONT = resolve_font(SERIF_STACK)
plt.rcParams.update(
    {
        "font.family": PRIMARY_FONT,
        "font.serif": SERIF_STACK,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
    }
)

# --------- 数据定义：保留重排消融数据 ----------
# 数据：从 RERANKING_COMPARISON_REPORT.md 提取，转换为百分比
med_full = {
    "MAP@10": 0.5440 * 100,
    "P@1": 0.9206 * 100,
    "NDCG@10": 0.9078 * 100,
}
med_ablate = {
    "Query-Only": {
        "MAP@10": 0.5110 * 100,
        "P@1": 0.8154 * 100,
        "NDCG@10": 0.8272 * 100,
    },
    "Hybrid": {
        "MAP@10": 0.5435 * 100,
        "P@1": 0.9203 * 100,
        "NDCG@10": 0.9070 * 100,
    },
}

lit_full = {
    "MAP@10": 0.2003 * 100,
    "P@1": 0.3400 * 100,
    "NDCG@10": 0.4336 * 100,
}
lit_ablate = {
    "Query-Only": {
        "MAP@10": 0.0877 * 100,
        "P@1": 0.0687 * 100,
        "NDCG@10": 0.2588 * 100,
    },
    "Hybrid": {
        "MAP@10": 0.1948 * 100,
        "P@1": 0.3266 * 100,
        "NDCG@10": 0.4251 * 100,
    },
}

datasets = [
    ("MedCorpus", med_full, med_ablate, (50, 95)),
    ("LitSearch", lit_full, lit_ablate, (0, 50)),
]

methods_order = ["Query-Only", "Profile-Only", "Hybrid"]
# 蓝色系配色：深蓝、中蓝、浅蓝（对应 MAP@10, P@1, NDCG@10）
colors_bar = ["#1E3A8A", "#3B82F6", "#93C5FD"]  # 深蓝、中蓝、浅蓝


def main() -> None:
    # 创建左右子图（a/b）
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.subplots_adjust(wspace=0.3)

    for idx, (ax, (title, full_vals, ablate, ylim)) in enumerate(zip(axes, datasets)):
        # 准备数据：每个方法显示三个指标（MAP@10, P@1, NDCG@10）
        map_values = []
        p1_values = []
        ndcg_values = []

        for method in methods_order:
            if method == "Profile-Only":
                map_values.append(full_vals["MAP@10"])
                p1_values.append(full_vals["P@1"])
                ndcg_values.append(full_vals["NDCG@10"])
            else:
                map_values.append(ablate[method]["MAP@10"])
                p1_values.append(ablate[method]["P@1"])
                ndcg_values.append(ablate[method]["NDCG@10"])

        # 分组柱状图：每个方法显示三个指标
        x_pos = np.arange(len(methods_order))
        width = 0.25  # 单个柱子宽度
        x1 = x_pos - width
        x2 = x_pos
        x3 = x_pos + width

        bars1 = ax.bar(
            x1,
            map_values,
            width,
            label="MAP@10",
            color=colors_bar[0],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
        )
        bars2 = ax.bar(
            x2,
            p1_values,
            width,
            label="P@1",
            color=colors_bar[1],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
        )
        bars3 = ax.bar(
            x3,
            ndcg_values,
            width,
            label="NDCG@10",
            color=colors_bar[2],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
        )

        # 折线图：为三个指标分别添加对应配色的折线
        # MAP@10 折线（深蓝色）
        line1 = ax.plot(
            x1,
            map_values,
            color=colors_bar[0],
            linestyle="--",
            marker="o",
            markersize=5,
            linewidth=2.0,
            zorder=3,
        )
        # P@1 折线（中蓝色）
        line2 = ax.plot(
            x2,
            p1_values,
            color=colors_bar[1],
            linestyle="--",
            marker="s",
            markersize=5,
            linewidth=2.0,
            zorder=3,
        )
        
        # NDCG@10 折线（浅蓝色）
        line3 = ax.plot(
            x3,
            ndcg_values,
            color=colors_bar[2],
            linestyle="--",
            marker="^",
            markersize=5,
            linewidth=2.0,
            zorder=3,
        )
        
        # 在折线点上方标注数值（纯黑色，zorder=5 确保在最上层）
        for x, y in zip(x1, map_values):
            ax.text(
                x,
                y + (ylim[1] - ylim[0]) * 0.02,
                f"{y:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
                fontweight="bold",
                zorder=5,
            )
        
        for x, y in zip(x2, p1_values):
            ax.text(
                x,
                y + (ylim[1] - ylim[0]) * 0.02,
                f"{y:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
                fontweight="bold",
                zorder=5,
            )
        
        for x, y in zip(x3, ndcg_values):
            ax.text(
                x,
                y + (ylim[1] - ylim[0]) * 0.02,
                f"{y:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
                fontweight="bold",
                zorder=5,
            )

        # 设置轴
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods_order, fontsize=10, color="black")
        ax.set_ylabel("Performance (%)", fontsize=10, color="black", fontweight="bold")
        ax.set_ylim(ylim[0] * 0.95, ylim[1] * 1.05)
        ax.yaxis.grid(True, color="#EEEEEE", lw=0.5, zorder=0, linestyle="-")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("black")
        ax.spines["bottom"].set_color("black")
        ax.tick_params(axis="y", colors="black", labelsize=9)
        ax.tick_params(axis="x", colors="black", labelsize=9)

        # 添加子图标签 (a) (b)
        ax.text(
            -0.08,
            1.02,
            f"({'a' if idx == 0 else 'b'}) {title}",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="bottom",
        )

    # 创建统一图例：每个指标只出现一次（用柱子表示）
    from matplotlib.patches import Rectangle

    legend_elements = [
        Rectangle((0, 0), 1, 1, fc=colors_bar[0], label="MAP@10", edgecolor="black", linewidth=0.5),
        Rectangle((0, 0), 1, 1, fc=colors_bar[1], label="P@1", edgecolor="black", linewidth=0.5),
        Rectangle((0, 0), 1, 1, fc=colors_bar[2], label="NDCG@10", edgecolor="black", linewidth=0.5),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 1.0),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # 保存 SVG
    out = Path(
        "/mnt/data/zsy-data/PerMed/experiments/reranking_ablation/reranking_ablation.svg"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"✅ Saved reranking ablation figure -> {out}")


if __name__ == "__main__":
    main()

