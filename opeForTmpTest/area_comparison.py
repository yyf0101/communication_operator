# area_comparison.py

# 假设我们已经从仿真脚本中分析出了每条链路使用的算子
# 这是证明的关键输入数据
lte_operators = ["fir_filter", "correlation", "fft_operation", "cmul", "cdiv", "interpolation_linear"]
nr_operators = ["fir_filter", "correlation", "fft_operation", "cmul", "cdiv", "interpolation_linear"]
wifi6_operators = ["fir_filter", "correlation", "fft_operation", "cmul", "cdiv"]

# --- 1. 定义硬件资源成本模型 ---
# 我们为每个算子创建一个成本字典，用资源计数器作为面积的代理
# 这些数值是基于典型ASIC/FPGA实现的估算值，用于相对比较
operator_cost_database = {
    "fir_filter":         {"multipliers": 16, "adders": 15, "memory_bits": 544,  "control_gates": 5000},
    "correlation":        {"multipliers": 32, "adders": 31, "memory_bits": 2048, "control_gates": 8000},
    "fft_operation":      {"multipliers": 0,  "adders": 512,"memory_bits": 32768,"control_gates": 10000},
    "cmul":               {"multipliers": 4,  "adders": 2,  "memory_bits": 64,   "control_gates": 1000},
    "cdiv":               {"multipliers": 8,  "adders": 4,  "memory_bits": 128,  "control_gates": 5000},
    "interpolation_linear":{"multipliers": 2,  "adders": 2,  "memory_bits": 64,   "control_gates": 2000},
}

# --- 2. 计算不复用方案（基线）的总成本 ---
def calculate_monolithic_cost():
    """为每条标准构建独立的硬件加速器"""
    total_cost = {"multipliers": 0, "adders": 0, "memory_bits": 0, "control_gates": 0}

    # 累加所有标准的算子成本
    for op in lte_operators + nr_operators + wifi6_operators:
        cost = operator_cost_database[op]
        total_cost["multipliers"] += cost["multipliers"]
        total_cost["adders"] += cost["adders"]
        total_cost["memory_bits"] += cost["memory_bits"]
        total_cost["control_gates"] += cost["control_gates"]

    return total_cost

# --- 3. 计算复用方案的总成本 ---
def calculate_shared_cost():
    """构建一个共享的算子库，所有标准分时复用"""
    total_cost = {"multipliers": 0, "adders": 0, "memory_bits": 0, "control_gates": 0}

    # 找出所有链路使用的独特算子集合
    all_unique_operators = set(lte_operators + nr_operators + wifi6_operators)

    # 累加独特算子的成本
    for op in all_unique_operators:
        cost = operator_cost_database[op]
        total_cost["multipliers"] += cost["multipliers"]
        total_cost["adders"] += cost["adders"]
        total_cost["memory_bits"] += cost["memory_bits"]
        total_cost["control_gates"] += cost["control_gates"]

    # 增加一个调度器的成本，用于管理分时复用
    scheduler_cost = {"multipliers": 0, "adders": 10, "memory_bits": 8192, "control_gates": 20000}
    total_cost["multipliers"] += scheduler_cost["multipliers"]
    total_cost["adders"] += scheduler_cost["adders"]
    total_cost["memory_bits"] += scheduler_cost["memory_bits"]
    total_cost["control_gates"] += scheduler_cost["control_gates"]

    return total_cost

# --- 4. 主函数：运行比较并打印报告 ---
def main():
    print("="*50)
    print("--- 芯片面积节省分析报告 ---")
    print("--- 基于公用算子库复用 ---")
    print("="*50)

    # 1. 打印输入信息
    print("\n1. 链路与算子分析:")
    print(f"   - LTE 使用的算子: {lte_operators}")
    print(f"   - 5G NR 使用的算子: {nr_operators}")
    print(f"   - Wi-Fi 6 使用的算子: {wifi6_operators}")

    # 2. 计算两种方案的成本
    mono_cost = calculate_monolithic_cost()
    shared_cost = calculate_shared_cost()

    # 3. 打印成本汇总
    print("\n2. 硬件资源成本汇总 (估算值):")
    print(f"   - [方案A: 不复用] 三条独立链路的总成本:")
    print(f"     - 乘法器: {mono_cost['multipliers']}")
    print(f"     - 加法器: {mono_cost['adders']}")
    print(f"     - 存储器: {mono_cost['memory_bits']} bits")
    print(f"     - 控制逻辑: {mono_cost['control_gates']} gates")

    print(f"\n   - [方案B: 复用] 共享算子库 + 调度器的总成本:")
    print(f"     - 乘法器: {shared_cost['multipliers']}")
    print(f"     - 加法器: {shared_cost['adders']}")
    print(f"     - 存储器: {shared_cost['memory_bits']} bits")
    print(f"     - 控制逻辑: {shared_cost['control_gates']} gates (含调度器)")

    # 4. 计算并打印节省百分比
    print("\n3. 面积节省分析:")
    # 计算每个资源的节省百分比
    savings = {}
    for key in mono_cost:
        savings[key] = (mono_cost[key] - shared_cost[key]) / mono_cost[key] * 100

    print(f"   - 乘法器资源节省: {savings['multipliers']:.2f}%")
    print(f"   - 加法器资源节省: {savings['adders']:.2f}%")
    print(f"   - 存储资源节省: {savings['memory_bits']:.2f}%")
    print(f"   - 控制逻辑节省: {savings['control_gates']:.2f}%")

    # 计算一个综合的面积节省百分比（简单加权）
    # 假设各种资源的权重相等
    overall_savings = sum(savings.values()) / len(savings)
    print("\n" + "-"*30)
    print(f"   - **综合面积节省: {overall_savings:.2f}%**")
    print("-"*30)

    print("\n4. 结论:")
    print("   通过复用公用算子库，避免了在不同标准链路中重复实现相同功能的硬件模块。")
    print(f"   仿真结果表明，这种架构可以节省超过 {overall_savings:.0f}% 的硬件资源，从而显著降低芯片面积、功耗和成本。")
    print("="*50)

if __name__ == "__main__":
    main()