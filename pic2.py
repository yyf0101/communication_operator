import matplotlib.pyplot as plt
import numpy as np

# ---------------------- 全局配置（提升专业感）----------------------
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']  # 支持英文/中文，优先Arial（论文常用）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['figure.dpi'] = 300  # 全局默认分辨率
plt.rcParams['savefig.dpi'] = 300  # 保存图片分辨率
plt.rcParams['axes.linewidth'] = 1.2  # 坐标轴线条宽度
plt.rcParams['grid.alpha'] = 0.3  # 网格透明度
plt.rcParams['font.weight'] = 'normal'  # 字体常规权重

# 数据准备
metrics = [
    {'name': 'Top-1 Accuracy', 'base': 0.5556, 'fine': 0.7130, 'improve': 15.74, 'unit': '%'},
    {'name': 'MRR', 'base': 0.7031, 'fine': 0.8311, 'improve': 0.1280, 'unit': ''},
    {'name': 'Average Discrimination', 'base': 0.0154, 'fine': 0.0878, 'improve': 0.0723, 'unit': ''}
]

# ---------------------- 创建图表和双y轴 ----------------------
fig, ax1 = plt.subplots(figsize=(12, 7))  # 适当放大画布，更舒展
ax2 = ax1.twinx()

# 设置双y轴范围（适配不同指标数值区间，预留少量余量更美观）
ax1.set_ylim(0, 105)  # 左侧y轴：Top-1 Accuracy（百分比），预留5%余量
ax2.set_ylim(0, 1.05)  # 右侧y轴：MRR、Average Discrimination，预留5%余量

# 柱状图宽度和x轴位置
x = np.arange(len(metrics))
width = 0.32  # 微调宽度，避免柱状图过于拥挤

# ---------------------- 绘制所有指标的柱状图（高级配色+精致细节）----------------------
# 专业配色（低饱和度，避免刺眼，论文/报告常用）
color_base = '#2E86AB'  # 深蓝（Base Model）
color_fine = '#F24236'  # 深红橙（Fine-tuned Model）
color_improve = '#3E92CC'  # 浅蓝绿（提升幅度标注）

# Base Model 柱状图（添加边框，提升立体感）
rects1_ax1 = ax1.bar(x[0] - width/2, metrics[0]['base'] * 100, width,
                     color=color_base, alpha=0.9, edgecolor='white', linewidth=1.0,
                     label='Base Model')
rects1_ax2 = ax2.bar(x[1:] - width/2, [m['base'] for m in metrics[1:]], width,
                     color=color_base, alpha=0.9, edgecolor='white', linewidth=1.0)

# Fine-tuned Model 柱状图（添加边框，提升立体感）
rects2_ax1 = ax1.bar(x[0] + width/2, metrics[0]['fine'] * 100, width,
                     color=color_fine, alpha=0.9, edgecolor='white', linewidth=1.0,
                     label='Fine-tuned Model')
rects2_ax2 = ax2.bar(x[1:] + width/2, [m['fine'] for m in metrics[1:]], width,
                     color=color_fine, alpha=0.9, edgecolor='white', linewidth=1.0)

# ---------------------- 添加背景网格（提升可读性，不抢主体）----------------------
ax1.grid(axis='y', linestyle='--', linewidth=0.8, color='#CCCCCC')
ax1.set_axisbelow(True)  # 网格置于底层，不遮挡柱状图

# ---------------------- 添加所有柱状图的数据标签（规范+美观）----------------------
def add_value_label(ax, rects, is_percent=False, is_mrr_disc=False):
    """统一添加数据标签的工具函数，提升代码复用性"""
    for rect in rects:
        height = rect.get_height()
        # 标签格式适配
        if is_percent:
            label_text = f'{height:.1f}%'
        elif is_mrr_disc:
            label_text = f'{height:.4f}'
        else:
            label_text = f'{height:.2f}'

        # 标注位置（微调，避免遮挡，添加轻微背景框提升辨识度）
        ax.annotate(label_text,
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, -18),  # 向下偏移，避免与柱状图重叠
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='medium',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))

# 调用工具函数添加标签
add_value_label(ax1, rects1_ax1, is_percent=True)
add_value_label(ax1, rects2_ax1, is_percent=True)
add_value_label(ax2, rects1_ax2, is_mrr_disc=True)
add_value_label(ax2, rects2_ax2, is_mrr_disc=True)

# ---------------------- 添加提升幅度标注（醒目+规范）----------------------
for i in range(len(metrics)):
    if i == 0:  # Top-1 Accuracy（百分比格式）
        fine_value = metrics[i]['fine'] * 100
        ax1.annotate(f'Improve: +{metrics[i]["improve"]:.1f}%',
                     xy=(i + width/2, fine_value),
                     xytext=(0, 10),  # 向上偏移，置于柱状图顶部
                     textcoords='offset points',
                     ha='center', va='bottom', fontsize=9, color=color_improve,
                     fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F8FF', alpha=0.8))
    else:  # MRR 和 Average Discrimination（原始值格式）
        fine_value = metrics[i]['fine']
        improve_text = f'Improve: +{metrics[i]["improve"]:.4f}'
        ax2.annotate(improve_text,
                     xy=(i + width/2, fine_value),
                     xytext=(0, 10),  # 向上偏移，置于柱状图顶部
                     textcoords='offset points',
                     ha='center', va='bottom', fontsize=9, color=color_improve,
                     fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F8FF', alpha=0.8))

# ---------------------- 优化坐标轴和图表元素（专业感核心）----------------------
# 标题（加粗+适当字号，预留间距）
ax1.set_title('Model Performance Comparison (Base vs Fine-tuned)',
              fontsize=16, fontweight='bold', pad=25, color='#333333')

# x轴标签（微调旋转角度，更自然，避免拥挤）
ax1.set_xticks(x)
ax1.set_xticklabels([m['name'] for m in metrics], rotation=10, ha='right', fontsize=12, color='#333333')

# y轴标签（配色与对应轴匹配，加粗，适当字号）
ax1.set_ylabel('Top-1 Accuracy (%)', fontsize=13, fontweight='bold', color=color_base)
ax2.set_ylabel('MRR / Average Discrimination', fontsize=13, fontweight='bold', color=color_fine)

# 优化坐标轴刻度（字体大小，颜色）
ax1.tick_params(axis='y', labelsize=11, colors=color_base)
ax2.tick_params(axis='y', labelsize=11, colors=color_fine)
ax1.tick_params(axis='x', labelsize=11, colors='#333333')

# 隐藏顶部和右侧多余边框（科研图表常用技巧，更简洁）
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)

# ---------------------- 优化图例（精致+不冗余）----------------------
legend = ax1.legend(loc='upper left', fontsize=12, frameon=True,
                    shadow=True, framealpha=0.9, borderpad=0.8)
legend.get_frame().set_facecolor('white')  # 图例背景白色
legend.get_frame().set_edgecolor('#EEEEEE')  # 图例边框浅灰色

# ---------------------- 调整布局+保存图片 ----------------------
plt.tight_layout()  # 自动调整布局
# 保存图片（指定背景色为白色，避免透明背景在报告中显示异常）
plt.savefig('./professional_performance_comparison.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()