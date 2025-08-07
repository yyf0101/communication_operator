import numpy as np
from typing import List, Dict, Tuple, Set

# 复用基础算子（依赖前文中的通用算子）
from ope_data import (
    xor_array,
    sparse_matrix_multiply,
    IterationControl,
    llr_awgn,
    hard_decision,
    sparse_data_access
)

# ------------------------------
# 一、LDPC编码专用算子
# ------------------------------

def construct_ldpc_h_matrix(
        code_len: int,
        info_len: int,
        row_weight: int,
        col_weight: int,
        avoid_short_cycles: bool = True
) -> Dict[int, Dict[int, int]]:
    """
    LDPC校验矩阵H构造算子（稀疏表示）
    :param code_len: 码长n
    :param info_len: 信息位长度k
    :param row_weight: 每行1的个数（行重）
    :param col_weight: 每列1的个数（列重）
    :param avoid_short_cycles: 是否避免短环（提升解码性能）
    :return: 稀疏矩阵H，格式为{行索引: {列索引: 1}}
    """
    parity_len = code_len - info_len  # 校验位长度
    h_matrix = {i: {} for i in range(parity_len)}  # 初始化校验行

    # 基础构造：均匀分布1的位置（简化版，实际可采用PEG算法优化）
    for col in range(code_len):
        # 为当前列分配row_weight个行
        rows = set()
        while len(rows) < col_weight:
            row = np.random.randint(0, parity_len)
            # 避免行重超过限制，且尽量避免短环（简单判断：与已有列的行交集最小）
            if len(h_matrix[row]) < row_weight:
                if avoid_short_cycles:
                    overlap = sum(1 for r in rows if r in h_matrix)
                    if overlap < 2:  # 限制行交集，减少4环
                        rows.add(row)
                else:
                    rows.add(row)
        # 更新H矩阵
        for row in rows:
            h_matrix[row][col] = 1

    return h_matrix

def ldpc_encode(
        info_bits: List[int],
        h_matrix: Dict[int, Dict[int, int]],
        code_len: int
) -> List[int]:
    """
    LDPC编码算子（基于校验矩阵H生成校验位）
    :param info_bits: 信息位（长度k）
    :param h_matrix: 稀疏校验矩阵H
    :param code_len: 码长n
    :return: 完整码字（信息位+校验位，长度n）
    """
    info_len = len(info_bits)
    parity_len = code_len - info_len
    if info_len + parity_len != code_len:
        raise ValueError("信息位长度+校验位长度必须等于码长")

    # 分割H矩阵为信息位部分H_i和校验位部分H_p
    h_i = {}  # H中对应信息位的列
    h_p = {}  # H中对应校验位的列
    for row in h_matrix:
        h_i[row] = {col: 1 for col in h_matrix[row] if col < info_len}
        h_p[row] = {col - info_len: 1 for col in h_matrix[row] if col >= info_len}

    # 校验位计算：p = (info_bits * H_i) * H_p^{-1} (模2)
    # 简化实现：通过高斯消元求H_p的逆，此处用矩阵伪逆近似（适用于小规模矩阵）
    # 实际工程中会采用预计算的生成矩阵G直接编码
    info_matrix = [info_bits]
    product = sparse_matrix_multiply(h_i, info_matrix)  # 中间结果：H_i * info_bits
    parity_bits = [0] * parity_len

    # 求解校验方程：H_p * p = product (模2)
    # 此处用简单迭代法求解（适用于稀疏H_p）
    for _ in range(10):  # 迭代次数
        new_parity = []
        for row in range(parity_len):
            # 计算当前行的校验和
            sum_val = product[0][row] if row in product[0] else 0
            for col in h_p.get(row, {}):
                sum_val ^= parity_bits[col]
            new_parity.append(sum_val)
        parity_bits = new_parity

    return info_bits + parity_bits

# ------------------------------
# 二、LDPC解码专用算子（基于置信传播BP算法）
# ------------------------------

def tanner_graph_construction(
        h_matrix: Dict[int, Dict[int, int]]
) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    """
    Tanner图构造算子（建立变量节点与校验节点的连接）
    :param h_matrix: 稀疏校验矩阵H
    :return:
        var_to_check: {变量节点: {校验节点集合}}
        check_to_var: {校验节点: {变量节点集合}}
    """
    var_to_check = {}
    check_to_var = {row: set(cols.keys()) for row, cols in h_matrix.items()}

    for check_node, var_nodes in check_to_var.items():
        for var_node in var_nodes:
            if var_node not in var_to_check:
                var_to_check[var_node] = set()
            var_to_check[var_node].add(check_node)

    return var_to_check, check_to_var

def check_node_update(
        check_node: int,
        var_nodes: Set[int],
        var_messages: Dict[Tuple[int, int], float],  # (变量节点, 校验节点) -> 消息
        min_sum_approx: bool = True
) -> Dict[int, float]:
    """
    校验节点更新算子（计算从校验节点到变量节点的消息）
    :param check_node: 校验节点索引
    :param var_nodes: 与该校验节点相连的变量节点集合
    :param var_messages: 变量节点到校验节点的消息
    :param min_sum_approx: 是否使用最小和近似（降低复杂度）
    :return: {变量节点: 校验节点发送的消息}
    """
    check_messages = {}
    for v in var_nodes:
        # 排除当前变量节点v，收集其他变量节点的消息
        other_vars = [u for u in var_nodes if u != v]
        if not other_vars:
            check_messages[v] = 0.0
            continue

        # 标准BP算法：使用双曲正切函数乘积
        if not min_sum_approx:
            products = [np.tanh(abs(var_messages[(u, check_node)]) / 2) for u in other_vars]
            product = np.prod(products)
            sign = np.prod([np.sign(var_messages[(u, check_node)]) for u in other_vars])
            message = 2 * sign * np.arctanh(product)
        # 最小和近似：取绝对值最小值与符号乘积
        else:
            abs_vals = [abs(var_messages[(u, check_node)]) for u in other_vars]
            min_abs = min(abs_vals)
            sign = np.prod([np.sign(var_messages[(u, check_node)]) for u in other_vars])
            message = sign * min_abs

        check_messages[v] = message

    return check_messages

def variable_node_update(
        var_node: int,
        check_nodes: Set[int],
        check_messages: Dict[Tuple[int, int], float],  # (校验节点, 变量节点) -> 消息
        channel_llr: float  # 信道初始LLR
) -> Dict[int, float]:
    """
    变量节点更新算子（计算从变量节点到校验节点的消息）
    :param var_node: 变量节点索引
    :param check_nodes: 与该变量节点相连的校验节点集合
    :param check_messages: 校验节点到变量节点的消息
    :param channel_llr: 该变量节点的初始信道LLR
    :return: {校验节点: 变量节点发送的消息}
    """
    var_messages = {}
    for c in check_nodes:
        # 排除当前校验节点c，累加其他校验节点的消息
        other_checks = [d for d in check_nodes if d != c]
        sum_check = sum(check_messages[(d, var_node)] for d in other_checks)
        # 变量消息 = 信道LLR + 其他校验节点消息之和
        var_messages[(var_node, c)] = channel_llr + sum_check

    return var_messages

def bp_decoder(
        received_llr: List[float],
        h_matrix: Dict[int, Dict[int, int]],
        max_iter: int = 50,
        min_sum_approx: bool = True
) -> List[int]:
    """
    LDPC置信传播（BP）解码算子
    :param received_llr: 接收序列的LLR（长度n）
    :param h_matrix: 稀疏校验矩阵H
    :param max_iter: 最大迭代次数
    :param min_sum_approx: 是否使用最小和近似
    :return: 解码后的比特序列
    """
    code_len = len(received_llr)
    var_to_check, check_to_var = tanner_graph_construction(h_matrix)
    iteration_ctrl = IterationControl(max_iter=max_iter, threshold=1e-4)

    # 初始化消息：变量节点到校验节点的消息 = 信道LLR
    var_messages = {}  # (var, check) -> message
    for var in var_to_check:
        for check in var_to_check[var]:
            var_messages[(var, check)] = received_llr[var]

    # 初始化校验节点到变量节点的消息（初始为0）
    check_messages = {}  # (check, var) -> message
    for check in check_to_var:
        for var in check_to_var[check]:
            check_messages[(check, var)] = 0.0

    prev_llr = [0.0] * code_len
    while True:
        # 1. 校验节点更新
        new_check_messages = {}
        for check in check_to_var:
            vars_connected = check_to_var[check]
            cm = check_node_update(check, vars_connected, var_messages, min_sum_approx)
            for var, msg in cm.items():
                new_check_messages[(check, var)] = msg
        check_messages = new_check_messages

        # 2. 变量节点更新
        new_var_messages = {}
        for var in var_to_check:
            checks_connected = var_to_check[var]
            vm = variable_node_update(var, checks_connected, check_messages, received_llr[var])
            new_var_messages.update(vm)
        var_messages = new_var_messages

        # 3. 计算后验LLR并判断是否收敛
        posterior_llr = []
        for var in range(code_len):
            # 后验LLR = 信道LLR + 所有校验节点消息之和
            sum_check = sum(check_messages[(c, var)] for c in var_to_check.get(var, []))
            posterior_llr.append(received_llr[var] + sum_check)

        # 检查收敛（后验LLR变化量）
        llr_diff = np.mean(np.abs(np.array(posterior_llr) - np.array(prev_llr)))
        if not iteration_ctrl.step(prev_value=llr_diff, current_value=0):
            break
        prev_llr = posterior_llr

    # 硬判决并返回结果
    return hard_decision(posterior_llr)

# ------------------------------
# 测试示例
# ------------------------------
if __name__ == "__main__":
    # 配置参数
    code_len = 16  # 码长n
    info_len = 8   # 信息位长度k
    row_weight = 4 # 行重
    col_weight = 2 # 列重

    # 1. 构造H矩阵
    h_matrix = construct_ldpc_h_matrix(code_len, info_len, row_weight, col_weight)
    print("H矩阵（稀疏表示，行: {列:1}）:")
    for row in h_matrix:
        print(f"行{row}: {h_matrix[row]}")

    # 2. 生成随机信息位并编码
    info_bits = [np.random.randint(0, 2) for _ in range(info_len)]
    codeword = ldpc_encode(info_bits, h_matrix, code_len)
    print("\n信息位:", info_bits)
    print("编码后码字:", codeword)

    # 3. 模拟AWGN信道（添加噪声）
    snr_db = 3.0  # 信噪比3dB
    snr = 10 **(snr_db / 10)
    sigma = 1 / np.sqrt(2 * snr)
    received_symbols = [(-1)** b + np.random.normal(0, sigma) for b in codeword]
    received_llr = [llr_awgn(r, sigma**2) for r in received_symbols]
    print("\n接收符号（含噪声）:", [round(x, 2) for x in received_symbols])

    # 4. BP解码
    decoded_bits = bp_decoder(received_llr, h_matrix, max_iter=20)
    print("解码结果:", decoded_bits)
    print("解码是否正确:", decoded_bits == codeword)
