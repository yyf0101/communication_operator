import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体，确保中文正常显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 原始数据定义
lte_operators = ["fir_filter", "correlation", "fft_operation", "cmul", "cdiv", "interpolation_linear"]
nr_operators = ["fir_filter", "correlation", "fft_operation", "cmul", "cdiv", "interpolation_linear"]
wifi6_operators = ["fir_filter", "correlation", "fft_operation", "cmul", "cdiv"]

operator_cost_database = {
    "fir_filter":         {"multipliers": 16, "adders": 15, "memory_bits": 544,  "control_gates": 5000},
    "correlation":        {"multipliers": 32, "adders": 31, "memory_bits": 2048, "control_gates": 8000},
    "fft_operation":      {"multipliers": 0,  "adders": 512,"memory_bits": 32768,"control_gates": 10000},
    "cmul":               {"multipliers": 4,  "adders": 2,  "memory_bits": 64,   "control_gates": 1000},
    "cdiv":               {"multipliers": 8,  "adders": 4,  "memory_bits": 128,  "control_gates": 5000},
    "interpolation_linear":{"multipliers": 2,  "adders": 2,  "memory_bits": 64,   "control_gates": 2000},
}

# 计算函数
def calculate_monolithic_cost():
    total_cost = {"multipliers": 0, "adders": 0, "memory_bits": 0, "control_gates": 0}
    for op in lte_operators + nr_operators + wifi6_operators:
        cost = operator_cost_database[op]
        for key in total_cost:
            total_cost[key] += cost[key]
    return total_cost

def calculate_shared_cost():
    total_cost = {"multipliers": 0, "adders": 0, "memory_bits": 0, "control_gates": 0}
    all_unique_operators = set(lte_operators + nr_operators + wifi6_operators)
    for op in all_unique_operators:
        cost = operator_cost_database[op]
        for key in total_cost:
            total_cost[key] += cost[key]
    # 增加调度器成本
    scheduler_cost = {"multipliers": 0, "adders": 10, "memory_bits": 8192, "control_gates": 20000}
    for key in total_cost:
        total_cost[key] += scheduler_cost[key]
    return total_cost

# 获取计算结果
mono_cost = calculate_monolithic_cost()
shared_cost = calculate_shared_cost()

# 计算节省百分比
savings = {}
for key in mono_cost:
    savings[key] = (mono_cost[key] - shared_cost[key]) / mono_cost[key] * 100

# 数据准备
resource_labels = {
    "multipliers": "乘法器",
    "adders": "加法器",
    "memory_bits": "存储器 (bits)",
    "control_gates": "控制逻辑 (gates)"
}

# 提取标签和数据
labels = [resource_labels[key] for key in mono_cost.keys()]
mono_values = list(mono_cost.values())
shared_values = list(shared_cost.values())
savings_values = list(savings.values())

# 1. 资源对比柱状图
def plot_resource_comparison():
    plt.figure(figsize=(12, 6))  # 独立窗口
    x = np.arange(len(labels))
    width = 0.35

    plt.bar(x - width/2, mono_values, width, label='不复用方案', color='#ff9999', edgecolor='black')
    plt.bar(x + width/2, shared_values, width, label='复用方案', color='#66b3ff', edgecolor='black')

    plt.ylabel('资源数量', fontsize=12)
    plt.title('两种方案的硬件资源占用对比', fontsize=14)
    plt.xticks(x, labels, rotation=15, ha='right')
    plt.legend()

    # 添加数值标签
    def autolabel(heights, x_positions):
        for height, x in zip(heights, x_positions):
            plt.text(x, height + max(mono_values)*0.01, f'{height}',
                     ha='center', va='bottom', fontsize=9)

    autolabel(mono_values, x - width/2)
    autolabel(shared_values, x + width/2)

    plt.tight_layout()
    return plt.gcf()  # 返回当前图表

# 2. 节省百分比柱状图
def plot_savings():
    plt.figure(figsize=(10, 6))  # 独立窗口
    x = np.arange(len(labels))

    plt.bar(x, savings_values, color='#99ff99', edgecolor='black')

    plt.ylabel('节省百分比 (%)', fontsize=12)
    plt.title('复用方案相比不复用方案的资源节省百分比', fontsize=14)
    plt.xticks(x, labels, rotation=15, ha='right')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    # 添加数值标签
    for i, v in enumerate(savings_values):
        plt.text(i, v + max(savings_values)*0.01, f'{v:.2f}%',
                 ha='center', va='bottom', fontsize=10)

    plt.ylim(0, max(savings_values) * 1.1)
    plt.tight_layout()
    return plt.gcf()  # 返回当前图表

# 3. 资源对比折线图（新增）
def plot_resource_linechart():
    plt.figure(figsize=(12, 6))  # 独立窗口
    x = np.arange(len(labels))

    # 绘制折线图，添加标记点
    plt.plot(x, mono_values, marker='o', linestyle='-', color='#ff6666', label='不复用方案', linewidth=2, markersize=8)
    plt.plot(x, shared_values, marker='s', linestyle='--', color='#3399ff', label='复用方案', linewidth=2, markersize=8)

    plt.ylabel('资源数量', fontsize=12)
    plt.title('两种方案的硬件资源占用趋势对比', fontsize=14)
    plt.xticks(x, labels, rotation=15, ha='right')
    plt.legend()
    plt.grid(alpha=0.3)  # 添加网格线

    # 添加数值标签
    for i, (m_val, s_val) in enumerate(zip(mono_values, shared_values)):
        plt.text(i, m_val + max(mono_values)*0.01, f'{m_val}',
                 ha='center', va='bottom', fontsize=9, color='#ff6666')
        plt.text(i, s_val - max(mono_values)*0.03, f'{s_val}',
                 ha='center', va='top', fontsize=9, color='#3399ff')

    plt.tight_layout()
    return plt.gcf()  # 返回当前图表

# 4. 节省百分比折线图（新增）
def plot_savings_linechart():
    plt.figure(figsize=(10, 6))  # 独立窗口
    x = np.arange(len(labels))

    # 绘制节省百分比折线图
    plt.plot(x, savings_values, marker='^', linestyle='-', color='#33cc33',
             linewidth=2, markersize=8, markerfacecolor='#33cc33', markeredgecolor='black')

    plt.ylabel('节省百分比 (%)', fontsize=12)
    plt.title('各资源类型的节省百分比趋势', fontsize=14)
    plt.xticks(x, labels, rotation=15, ha='right')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(alpha=0.3)  # 添加网格线

    # 添加数值标签
    for i, v in enumerate(savings_values):
        plt.text(i, v + max(savings_values)*0.01, f'{v:.2f}%',
                 ha='center', va='bottom', fontsize=10, color='#008000')

    plt.ylim(0, max(savings_values) * 1.1)
    plt.tight_layout()
    return plt.gcf()  # 返回当前图表

# 主函数：生成并独立显示所有图表
def main():
    # 计算综合节省百分比
    overall_savings = sum(savings.values()) / len(savings)

    # 生成并显示资源对比柱状图
    fig1 = plot_resource_comparison()
    fig1.suptitle(f'综合面积节省: {overall_savings:.2f}%', fontsize=14, y=1.02)
    plt.figure(fig1.number)  # 激活第一个窗口
    plt.show(block=False)    # 非阻塞显示

    # 生成并显示节省百分比柱状图
    fig2 = plot_savings()
    plt.figure(fig2.number)  # 激活第二个窗口
    plt.show(block=False)    # 非阻塞显示

    # 生成并显示资源对比折线图
    fig3 = plot_resource_linechart()
    plt.figure(fig3.number)  # 激活第三个窗口
    plt.show(block=False)    # 非阻塞显示

    # 生成并显示节省百分比折线图
    fig4 = plot_savings_linechart()
    plt.figure(fig4.number)  # 激活第四个窗口
    plt.show()  # 最后一个使用阻塞显示，保持窗口打开

if __name__ == "__main__":
    main()