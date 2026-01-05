import matplotlib.pyplot as plt
import numpy as np

# ---------------------- 全局专业配置 ----------------------
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']  # 兼容中英文（论文优先Arial）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.linewidth'] = 1.2  # 坐标轴线条宽度
plt.rcParams['grid.alpha'] = 0.3  # 网格透明度
plt.rcParams['font.weight'] = 'normal'

# ---------------------- 数据整理 ----------------------
# 指标名称（对应表格列）
metrics = ["Positive Score", "Negative Score", "Discrimination Gap"]
# 微调前/后数据
before_fine = [0.91, 0.89, 0.02]
after_fine = [0.82, 0.35, 0.47]

# ---------------------- 绘图配置 ----------------------
fig, ax = plt.subplots(figsize=(9, 6))  # 适中画布尺寸
x = np.arange(len(metrics))  # x轴指标位置
width = 0.35  # 柱状图宽度

# 专业配色（低饱和度+区分度高）
color_before = '#2E86AB'  # 微调前：深蓝
color_after = '#F24236'   # 微调后：深红橙

# ---------------------- 绘制分组柱状图 ----------------------
# 微调前柱状图（添加白色边框提升质感）
rects1 = ax.bar(x - width/2, before_fine, width,
                label='Before Fine-tuning', color=color_before,
                alpha=0.9, edgecolor='white', linewidth=1.0)
# 微调后柱状图
rects2 = ax.bar(x + width/2, after_fine, width,
                label='After Fine-tuning', color=color_after,
                alpha=0.9, edgecolor='white', linewidth=1.0)

# ---------------------- 添加数据标签 ----------------------
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 5),  # 向上偏移5点
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=10, fontweight='medium',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

add_labels(rects1)
add_labels(rects2)

# ---------------------- 图表美化 ----------------------
# 标题（加粗+适中字号）
ax.set_title('Performance Comparison Before/After Fine-tuning',
             fontsize=14, fontweight='bold', pad=20, color='#333333')
# x轴标签
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11, color='#333333')
# y轴范围（预留余量，更舒展）
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score / Gap Value', fontsize=12, color='#333333')

# 添加y轴网格（仅横向，提升可读性）
ax.grid(axis='y', linestyle='--', linewidth=0.8, color='#CCCCCC')
ax.set_axisbelow(True)  # 网格置于底层

# 隐藏顶部/右侧边框（学术图表常用）
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 图例（精致化）
legend = ax.legend(loc='upper right', fontsize=11, frameon=True,
                   framealpha=0.9, borderpad=0.8)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('#EEEEEE')

# ---------------------- 保存+显示 ----------------------
plt.tight_layout()
plt.savefig('./fine_tuning_comparison.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()