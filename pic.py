import matplotlib.pyplot as plt
import numpy as np

# 数据准备
models = ['First Version', 'Second Version', 'Now Version']
correct = [22, 48, 53]
total = [57, 56, 55]
accuracy = [round(c/t*100, 1) for c,t in zip(correct, total)]
improvements = [
    round(accuracy[1]-accuracy[0], 1),
    round(accuracy[2]-accuracy[1], 1)
]

# 配色方案（采用Material Design渐变色）
colors = ['#FF7043', '#FFA726', '#66BB6A']
arrow_colors = ['#4CAF50', '#2E7D32']  # 提升箭头颜色（深绿到浅绿渐变）

# 创建画布
plt.figure(figsize=(10, 6), dpi=100)
ax = plt.subplot()

# 绘制柱状图
bars = ax.bar(models, accuracy,
              color=colors,
              edgecolor='white',
              linewidth=1.5,
              zorder=3)

# 添加数据标签
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height+1,
            f'{height}%',
            ha='center', va='bottom',
            fontsize=10, fontweight='bold')

# 添加提升箭头和标签
for i in range(len(improvements)):
    # 箭头起点和终点
    start = accuracy[i]
    end = accuracy[i+1]
    x_start = i + 0.7  # 箭头x起始位置（柱子右侧）
    x_end = i + 1.3    # 箭头x结束位置（下一柱子左侧）

    # 绘制箭头
    ax.annotate('', xy=(x_end, end), xytext=(x_start, start),
                arrowprops=dict(arrowstyle='->',
                                color=arrow_colors[i],
                                lw=2,
                                connectionstyle='arc3,rad=0.3'))

    # 添加提升百分比标签
    mid_x = (x_start + x_end)/2
    mid_y = start + (end-start)/2
    ax.text(mid_x, mid_y,
            f'+{improvements[i]}%',
            ha='center', va='center',
            fontsize=10, fontweight='bold',
            color=arrow_colors[i])

# 装饰图表
ax.set_ylim(0, 110)
ax.set_yticks(range(0, 110, 10))
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Performance Comparison', fontsize=14, pad=20)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 调整布局
plt.tight_layout()
plt.show()