import numpy as np
from typing import List, Tuple, Dict

# 复用基础算子（同一目录下的ope_data.py）
from ope_data import (
    xor_operator,
    interleave,
    deinterleave,
    IterationControl,
    llr_awgn,
    hard_decision,
    puncturing
)

# ------------------------------
# 一、Turbo码编码专用算子
# ------------------------------

class RSCEncoder:
    """递归系统卷积码（RSC）编码器算子"""
    def __init__(self, generator_poly: Tuple[int, int], constraint_length: int):
        """
        :param generator_poly: 生成多项式（八进制表示，如(0o13, 0o15)）
        :param constraint_length: 约束长度K（移位寄存器级数+1）
        """
        self.gen1, self.gen2 = generator_poly
        self.constraint_length = constraint_length
        self.shift_reg = [0] * (constraint_length - 1)  # 移位寄存器状态
        self.tail_length = constraint_length - 1  # 尾部比特长度

    def reset(self):
        """重置移位寄存器状态"""
        self.shift_reg = [0] * (len(self.shift_reg))

    def encode_bit(self, info_bit: int) -> Tuple[int, int]:
        """编码单个信息位"""
        # 递归反馈：信息位与移位寄存器输出异或
        feedback = xor_operator(info_bit, self._tap_output(self.gen2))
        # 更新移位寄存器（左移，新反馈填入末位）
        self.shift_reg = [feedback] + self.shift_reg[:-1]
        # 计算校验位
        parity_bit = self._tap_output(self.gen1)
        return info_bit, parity_bit  # 系统位=信息位

    def _tap_output(self, poly: int) -> int:
        """根据生成多项式计算移位寄存器抽头输出（模2和）"""
        output = 0
        # 多项式每一位对应移位寄存器的一个抽头
        for i in range(self.constraint_length):
            if (poly >> (self.constraint_length - 1 - i)) & 1:
                output = xor_operator(output, self.shift_reg[i] if i < len(self.shift_reg) else 0)
        return output

    def encode_sequence(self, info_bits: List[int], add_tail: bool = True) -> Tuple[List[int], List[int]]:
        """
        编码信息序列
        :param info_bits: 信息位序列
        :param add_tail: 是否添加尾部比特
        :return: (系统位序列, 校验位序列)
        """
        self.reset()
        sys_bits = []
        parity_bits = []

        # 编码信息位
        for bit in info_bits:
            sys, par = self.encode_bit(bit)
            sys_bits.append(sys)
            parity_bits.append(par)

        # 编码尾部比特（使移位寄存器归零）
        if add_tail:
            # 计算需要的尾部比特（使移位寄存器归零）
            tail_bits = []
            current_reg = self.shift_reg.copy()
            for _ in range(self.tail_length):
                # 尾部比特等于反馈值才能归零
                tail_bit = current_reg[-1]  # 最后一位作为尾部比特输入
                tail_bits.append(tail_bit)
                sys, par = self.encode_bit(tail_bit)
                sys_bits.append(sys)
                parity_bits.append(par)
                current_reg = self.shift_reg.copy()

        return sys_bits, parity_bits

    def _tap_output_from_reg(self, poly: int, reg: List[int]) -> int:
        """根据给定寄存器状态计算抽头输出（用于解码）"""
        output = 0
        for i in range(self.constraint_length):
            if (poly >> (self.constraint_length - 1 - i)) & 1:
                output = xor_operator(output, reg[i] if i < len(reg) else 0)
        return output

def turbo_encode(
        info_bits: List[int],
        rsc1: RSCEncoder,
        rsc2: RSCEncoder,
        interleave_table: List[int],
        puncture_pattern: List[Tuple[int, int]] = None
) -> List[int]:
    """
    Turbo码编码算子（并行级联结构）
    :param info_bits: 原始信息位序列
    :param rsc1: 第一个RSC编码器
    :param rsc2: 第二个RSC编码器
    :param interleave_table: 交织表（长度=信息位长度）
    :param puncture_pattern: 删余图案
    :return: 编码后的Turbo码字
    """
    # 1. 第一个RSC编码（原始信息）
    sys1, par1 = rsc1.encode_sequence(info_bits)

    # 2. 交织后通过第二个RSC编码
    interleaved_info = interleave(info_bits, interleave_table)
    sys2, par2 = rsc2.encode_sequence(interleaved_info)

    # 验证系统位一致性（仅验证信息位部分，不包括尾部）
    info_length = len(info_bits)
    assert sys1[:info_length] == sys2[:info_length], "信息位部分的系统位必须一致"

    # 3. 删余配置
    if puncture_pattern is None:
        puncture_pattern = [(1, 1, 1)] * len(sys1)  # 无删余：保留所有位

    # 4. 拼接码字：系统位 + 校验位1 + 校验位2（按删余图案）
    codeword = []
    for i in range(len(sys1)):
        s, p1, p2 = puncture_pattern[i % len(puncture_pattern)]
        if s:
            codeword.append(sys1[i])
        if p1:
            codeword.append(par1[i])
        if p2:
            codeword.append(par2[i])

    return codeword

# ------------------------------
# 二、Turbo码解码专用算子（保持不变）
# ------------------------------

def siso_decoder(
        received_sys: List[float],
        received_parity: List[float],
        rsc: RSCEncoder,
        a_priori: List[float] = None,
        max_log_map: bool = True
) -> Tuple[List[float], List[float]]:
    """软输入软输出（SISO）解码器（基于BCJR算法）"""
    n = len(received_sys)
    a_priori = a_priori or [0.0] * n
    constraint_length = rsc.constraint_length
    num_states = 1 << (constraint_length - 1)  # 状态数=2^(K-1)

    # 1. 预计算状态转移表
    trans_table = {}
    for state in range(num_states):
        trans_table[state] = {}
        for input_bit in [0, 1]:
            temp_reg = list(np.binary_repr(state, width=constraint_length-1))
            temp_reg = [int(bit) for bit in temp_reg]
            feedback = xor_operator(input_bit, rsc._tap_output_from_reg(rsc.gen2, temp_reg))
            next_reg = [feedback] + temp_reg[:-1]
            next_state = int(''.join(map(str, next_reg)), 2)
            parity = rsc._tap_output_from_reg(rsc.gen1, next_reg)
            trans_table[state][input_bit] = (next_state, parity)

    # 2. 分支度量计算
    def branch_metric(state: int, input_bit: int, i: int) -> float:
        next_state, parity = trans_table[state][input_bit]
        metric_sys = received_sys[i] * (1 - 2 * input_bit)
        metric_par = received_parity[i] * (1 - 2 * parity)
        metric_apriori = a_priori[i] * (1 - 2 * input_bit)
        return metric_sys + metric_par + metric_apriori

    # 3. 前向递归（α）
    alpha = np.full((n, num_states), -np.inf)
    alpha[0, 0] = 0.0  # 初始状态为全0
    for i in range(n - 1):
        for state in range(num_states):
            if alpha[i, state] == -np.inf:
                continue
            for input_bit in [0, 1]:
                next_state, _ = trans_table[state][input_bit]
                bm = branch_metric(state, input_bit, i)
                new_alpha = alpha[i, state] + bm
                if new_alpha > alpha[i+1, next_state]:
                    alpha[i+1, next_state] = new_alpha
        # 归一化避免数值溢出
        max_alpha = np.max(alpha[i+1])
        if max_alpha != -np.inf:
            alpha[i+1] -= max_alpha

    # 4. 后向递归（β）
    beta = np.full((n, num_states), -np.inf)
    beta[-1, :] = 0.0  # 最后一个状态的后向度量为0
    for i in range(n-2, -1, -1):
        for state in range(num_states):
            for input_bit in [0, 1]:
                next_state, _ = trans_table[state][input_bit]
                if beta[i+1, next_state] == -np.inf:
                    continue
                bm = branch_metric(state, input_bit, i+1)
                new_beta = beta[i+1, next_state] + bm
                if new_beta > beta[i, state]:
                    beta[i, state] = new_beta
        # 归一化
        max_beta = np.max(beta[i])
        if max_beta != -np.inf:
            beta[i] -= max_beta

    # 5. 计算后验LLR和外部信息
    posterior_llr = []
    extrinsic = []
    for i in range(n):
        metric0 = -np.inf
        metric1 = -np.inf
        for state in range(num_states):
            for input_bit in [0, 1]:
                next_state, _ = trans_table[state][input_bit]
                if alpha[i, state] == -np.inf or beta[i, next_state] == -np.inf:
                    continue
                total = alpha[i, state] + branch_metric(state, input_bit, i) + beta[i, next_state]
                if input_bit == 0 and total > metric0:
                    metric0 = total
                if input_bit == 1 and total > metric1:
                    metric1 = total

        # 计算LLR
        if max_log_map:
            llr = metric0 - metric1
        else:
            llr = np.log(np.exp(metric0) + 1e-10) - np.log(np.exp(metric1) + 1e-10)

        posterior_llr.append(llr)
        extrinsic.append(llr - a_priori[i] - received_sys[i])

    return posterior_llr, extrinsic

def turbo_decoder(
        received: List[float],
        rsc1: RSCEncoder,
        rsc2: RSCEncoder,
        interleave_table: List[int],
        puncture_pattern: List[Tuple[int, int]] = None,
        max_iter: int = 10,
        max_log_map: bool = True
) -> List[int]:
    """Turbo码迭代解码器"""
    if puncture_pattern is None:
        puncture_pattern = [(1, 1, 1)]  # 无删余
    # 1. 解删余
    sys_bits, par1_bits, par2_bits = [], [], []
    idx = 0
    while idx < len(received):
        s, p1, p2 = puncture_pattern[len(sys_bits) % len(puncture_pattern)]
        if s and idx < len(received):
            sys_bits.append(received[idx])
            idx += 1
        if p1 and idx < len(received):
            par1_bits.append(received[idx])
            idx += 1
        if p2 and idx < len(received):
            par2_bits.append(received[idx])
            idx += 1
    n = len(sys_bits)

    # 2. 初始化先验信息
    extrinsic1 = [0.0] * n
    iteration_ctrl = IterationControl(max_iter=max_iter, threshold=1e-3)

    # 3. 迭代解码
    prev_llr = [0.0] * n
    while True:
        # a. 第二个SISO的先验信息
        a_priori2 = interleave(extrinsic1, interleave_table)

        # b. 第二个SISO解码
        _, extrinsic2 = siso_decoder(
            received_sys=sys_bits,
            received_parity=par2_bits,
            rsc=rsc2,
            a_priori=a_priori2,
            max_log_map=max_log_map
        )

        # c. 第一个SISO的先验信息
        a_priori1 = deinterleave(extrinsic2, interleave_table)

        # d. 第一个SISO解码
        posterior_llr, extrinsic1 = siso_decoder(
            received_sys=sys_bits,
            received_parity=par1_bits,
            rsc=rsc1,
            a_priori=a_priori1,
            max_log_map=max_log_map
        )

        # e. 判断收敛
        llr_diff = np.mean(np.abs(np.array(posterior_llr) - np.array(prev_llr)))
        if not iteration_ctrl.step(prev_value=llr_diff, current_value=0):
            break
        prev_llr = posterior_llr

    # 4. 硬判决并去除尾部比特
    decoded = hard_decision(posterior_llr)
    return decoded[:-rsc1.tail_length]  # 去除尾部比特

# ------------------------------
# 测试示例
# ------------------------------
if __name__ == "__main__":
    # 1. 配置参数
    info_len = 16  # 信息位长度
    constraint_length = 4  # RSC约束长度K=4
    generator_poly = (0o13, 0o15)  # 生成多项式(1+D+D^3, 1+D^2+D^3)
    puncture_pattern = [(1, 1, 0), (1, 0, 1)]  # 码率1/2的删余图案
    interleave_table = np.random.permutation(info_len).tolist()  # 随机交织表

    # 2. 初始化RSC编码器
    rsc1 = RSCEncoder(generator_poly, constraint_length)
    rsc2 = RSCEncoder(generator_poly, constraint_length)

    # 3. 生成随机信息位并编码
    info_bits = [np.random.randint(0, 2) for _ in range(info_len)]
    print("原始信息位:", info_bits)

    codeword = turbo_encode(info_bits, rsc1, rsc2, interleave_table, puncture_pattern)
    print("编码后码字长度:", len(codeword))

    # 4. 模拟AWGN信道
    snr_db = 2.0
    snr = 10 **(snr_db / 10)
    sigma = 1 / np.sqrt(2 * snr)
    modulated = [1 - 2 * b for b in codeword]  # BPSK调制
    received_symbols = [s + np.random.normal(0, sigma) for s in modulated]
    received_llr = [llr_awgn(r, sigma**2) for r in received_symbols]

    # 5. Turbo解码
    decoded_bits = turbo_decoder(
        received_llr, rsc1, rsc2, interleave_table, puncture_pattern, max_iter=8
    )
    print("解码结果:", decoded_bits)
    print("解码是否正确:", decoded_bits == info_bits)
