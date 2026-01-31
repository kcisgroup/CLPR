#!/usr/bin/env python3
"""
生成 Profile 质量评估的 SVG 图 - CVPR 风格配色
参考 CVPR 2025 论文雷达图配色
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import numpy as np
from math import pi


def resolve_font(preferred_fonts: list[str]) -> str:
    """Pick the first installed font from the preferred serif stack."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for font in preferred_fonts:
        if font in installed:
            return font
    return preferred_fonts[-1]


# --------- 全局排版：与 memory_ablation 风格统一 ----------
SERIF_STACK = ["Times New Roman", "CMU Serif", "DejaVu Serif"]
PRIMARY_FONT = resolve_font(SERIF_STACK)
plt.rcParams.update({
    "font.family": PRIMARY_FONT,
    "font.serif": SERIF_STACK,
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

# 数据
models = ['Human', 'Claude 4.5 Haiku', 'Gemini 2.5 Pro', 'GPT-4o-mini']
dimensions = ['Relevance', 'Accuracy', 'Informativeness', 'Coherence']

scores = {
    'Human': [4.86, 3.95, 2.96, 3.68],
    'Claude 4.5 Haiku': [4.52, 3.04, 2.75, 3.00],
    'Gemini 2.5 Pro': [4.92, 3.87, 3.08, 3.55],
    'GPT-4o-mini': [4.83, 3.90, 3.57, 3.03],
}

mae_data = {
    'Claude 4.5 Haiku': 0.54,
    'Gemini 2.5 Pro': 0.10,
    'GPT-4o-mini': 0.34,
}

# 直接拄 IMG_4862 的配色 - 和谐柔和
colors = {
    'Human': '#666666',           # 灰色虚线（基准）
    'Claude 4.5 Haiku': '#E07B54',    # 珊瑚橙红
    'Gemini 2.5 Pro': '#5B8AC2',  # 柔和蓝
    'GPT-4o-mini': '#7CB872',     # 柔和绿
}

# 填充色（更浅的同色系）
fill_colors = {
    'Human': 'none',              # 不填充
    'Claude 4.5 Haiku': '#F5C4B3',    # 浅珊瑚
    'Gemini 2.5 Pro': '#B8D4F0',  # 浅蓝
    'GPT-4o-mini': '#C5E5C0',     # 浅绿
}

# ============ 综合图 ============
def create_combined_chart():
    fig = plt.figure(figsize=(11, 4.8))
    
    # ===== 左图：雷达图 =====
    ax1 = fig.add_subplot(121, polar=True)
    
    angles = [n / float(len(dimensions)) * 2 * pi for n in range(len(dimensions))]
    angles += angles[:1]
    
    # 绘制顺序：先画填充大的，后画填充小的
    # 注意：Claude 4.5 Haiku 面积可能被遮挡，所以把它放在最后画（除了 Human）
    draw_order = ['GPT-4o-mini', 'Gemini 2.5 Pro', 'Claude 4.5 Haiku', 'Human']
    
    for model in draw_order:
        values = scores[model] + scores[model][:1]
        if model == 'Human':
            # Human 用虚线，不填充
            ax1.plot(angles, values, '--', linewidth=2.5, label=model, 
                     color=colors[model], zorder=10)
        else:
            # 其他模型用实线 + 浅色填充
            ax1.plot(angles, values, '-', linewidth=2.2, label=model, 
                     color=colors[model])
            ax1.fill(angles, values, color=fill_colors[model], alpha=0.6)
    
    # 设置维度标签
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(dimensions, fontweight='bold', fontsize=10)
    
    # 设置刻度
    ax1.set_ylim(0, 5.5)
    ax1.set_yticks([1, 2, 3, 4, 5])
    ax1.set_yticklabels(['1', '2', '3', '4', '5'], color='#666666', fontsize=8)
    
    # 网格样式 - 细灰线
    ax1.grid(True, linestyle='-', alpha=0.5, color='#AAAAAA', linewidth=0.8)
    ax1.spines['polar'].set_visible(False)
    
    # 使用 fig.text 统一添加标题，确保水平对齐
    fig.text(0.26, 0.9, '(a) Quality Scores by Dimension', ha='center', fontsize=13, fontweight='bold')
    fig.text(0.76, 0.9, '(b) Agreement with Human Evaluation', ha='center', fontsize=13, fontweight='bold')
    
    # 移除原来的 ax.set_title
    # ax1.set_title(...)
    # ax2.set_title(...)
    
    # 调整布局以适应新标题
    plt.subplots_adjust(top=0.85, bottom=0.15, wspace=0.25, left=0.05, right=0.95)

    # ===== 右图：柱状图 =====
    ax2 = fig.add_subplot(122)
    
    # 按 MAE 从低到高排序（最好的在最右边）
    models_mae = ['Claude 4.5 Haiku', 'GPT-4o-mini', 'Gemini 2.5 Pro']
    mae_values = [mae_data[m] for m in models_mae]
    bar_colors = [colors[m] for m in models_mae]
    
    x_pos = np.arange(len(models_mae))
    bars = ax2.bar(x_pos, mae_values, color=bar_colors, 
                   edgecolor='none',  # 无边框
                   width=0.55)
    
    # 数值标注
    for bar, val in zip(bars, mae_values):
        height = bar.get_height()
        ax2.annotate(f'{val:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=11,
                    color='#333333')
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models_mae, fontsize=10)
    ax2.set_ylabel('Mean Absolute Error (|Δ|)', fontweight='bold')
    ax2.set_ylim(0, 0.72)
    
    # 移除原来的 ax.set_title
    # ax2.set_title(...)
    
    # 移除上右边框
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#666666')
    ax2.spines['bottom'].set_color('#666666')
    
    # 添加说明：越低越接近 Human
    ax2.text(1, -0.12, '↓ Lower = closer to Human', ha='center', 
             fontsize=9, color='#666666', style='italic',
             transform=ax2.transAxes)
    
    # 统一图例
    legend_elements = [
        plt.Line2D([0], [0], color=colors['Human'], linewidth=2, linestyle='--', label='Human'),
        plt.Line2D([0], [0], color=colors['Claude 4.5 Haiku'], linewidth=2, label='Claude 4.5 Haiku'),
        plt.Line2D([0], [0], color=colors['Gemini 2.5 Pro'], linewidth=2, label='Gemini 2.5 Pro'),
        plt.Line2D([0], [0], color=colors['GPT-4o-mini'], linewidth=2, label='GPT-4o-mini'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.01),
               ncol=4, frameon=False, fontsize=9)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.14, wspace=0.28)
    
    plt.savefig('profile_quality_combined.svg', format='svg', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("✅ Saved: profile_quality_combined.svg")
    plt.close()

# ============ 单独雷达图（CVPR风格） ============
def create_radar_chart():
    fig, ax = plt.subplots(figsize=(6, 5.5), subplot_kw=dict(polar=True))
    
    angles = [n / float(len(dimensions)) * 2 * pi for n in range(len(dimensions))]
    angles += angles[:1]
    
    draw_order = ['Claude 4.5 Haiku', 'GPT-4o-mini', 'Gemini 2.5 Pro', 'Human']
    
    for model in draw_order:
        values = scores[model] + scores[model][:1]
        if model == 'Human':
            # Human 用虚线，不填充
            ax.plot(angles, values, '--', linewidth=2.5, label=model, 
                    color=colors[model], zorder=10)
        else:
            # 其他模型用实线 + 浅色填充
            ax.plot(angles, values, '-', linewidth=2.2, label=model, 
                    color=colors[model])
            ax.fill(angles, values, color=fill_colors[model], alpha=0.6)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontweight='bold', fontsize=10)
    ax.set_ylim(0, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'], color='#666666', fontsize=8)
    ax.grid(True, linestyle='-', alpha=0.5, color='#AAAAAA', linewidth=0.8)
    ax.spines['polar'].set_visible(False)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.28, 1.08), frameon=False, fontsize=9)
    
    plt.tight_layout()
    plt.savefig('profile_quality_radar.svg', format='svg', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("✅ Saved: profile_quality_radar.svg")
    plt.close()

# ============ 单独柱状图 ============
def create_mae_bar_chart():
    fig, ax = plt.subplots(figsize=(5, 4))
    
    models_mae = ['Claude 4.5 Haiku', 'GPT-4o-mini', 'Gemini 2.5 Pro']
    mae_values = [mae_data[m] for m in models_mae]
    bar_colors = [colors[m] for m in models_mae]
    edge_colors = ['#D95F02', '#666666', '#4575B4']
    
    x_pos = np.arange(len(models_mae))
    bars = ax.bar(x_pos, mae_values, color=bar_colors, 
                  edgecolor='none', width=0.55)
    
    for bar, val in zip(bars, mae_values):
        height = bar.get_height()
        ax.annotate(f'{val:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 5),
                   textcoords="offset points",
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models_mae, fontsize=10)
    ax.set_ylabel('Mean Absolute Error (|Δ|)', fontweight='bold')
    ax.set_ylim(0, 0.72)
    ax.set_title('Agreement with Human Evaluation', fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 无需参考线
    
    plt.tight_layout()
    plt.savefig('profile_quality_mae.svg', format='svg', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("✅ Saved: profile_quality_mae.svg")
    plt.close()

if __name__ == "__main__":
    create_combined_chart()
    create_radar_chart()
    create_mae_bar_chart()
    print("\n🎨 All figures generated with CVPR-style colors!")
