import numpy as np
from typing import List, Tuple, Dict, Callable

# ------------------------------
# 一、基础数学运算算子
# ------------------------------

def xor_operator(a: int, b: int) -> int:
    """模2加/异或算子（单个比特）"""
    return (a + b) % 2

def xor_array(arr1: List[int], arr2: List[int]) -> List[int]:
    """模2加/异或算子（序列）"""
    if len(arr1) != len(arr2):
        raise ValueError("输入序列长度必须相等")
    return [(a + b) % 2 for a, b in zip(arr1, arr2)]

class FiniteFieldOperator:
    """有限域GF(2^m)运算算子"""
    def __init__(self, m: int, primitive_poly: int):
        """
        初始化GF(2^m)算子
        :param m: 有限域阶数（2^m）
        :param primitive_poly: 本原多项式（整数表示，如GF(2^3)的本原多项式x^3+x+1表示为0b1011=11）
        """
        self.m = m
        self.size = 1 << m  # 2^m
        self.primitive_poly = primitive_poly
        self._build_tables()

    def _build_tables(self):
        """预计算指数表和对数表，加速乘法运算"""
        self.exp_table = [0] * (self.size * 2)  # 指数表：alpha^i
        self.log_table = [0] * self.size        # 对数表：log_alpha(x)
        alpha = 1  # 本原元的幂
        for i in range(self.size - 1):
            self.exp_table[i] = alpha
            self.log_table[alpha] = i
            # 计算alpha^(i+1) = alpha^i * 2 mod primitive_poly
            alpha <<= 1
            if alpha & self.size:  # 若超过m位，则模本原多项式
                alpha ^= self.primitive_poly
        # 扩展指数表（处理i >= 2^m -1的情况）
        for i in range(self.size - 1, self.size * 2 - 2):
            self.exp_table[i] = self.exp_table[i - (self.size - 1)]

    def add(self, a: int, b: int) -> int:
        """GF(2^m)加法（等价于异或）"""
        return a ^ b

    def multiply(self, a: int, b: int) -> int:
        """GF(2^m)乘法（基于指数表和对数表）"""
        if a == 0 or b == 0:
            return 0
        return self.exp_table[self.log_table[a] + self.log_table[b]]

def matrix_multiply(A: List[List[int]], B: List[List[int]], mod: int = 2) -> List[List[int]]:
    """矩阵乘法算子（支持模运算，默认模2）"""
    rows_A, cols_A = len(A), len(A[0]) if A else 0
    rows_B, cols_B = len(B), len(B[0]) if B else 0
    if cols_A != rows_B:
        raise ValueError("A的列数必须等于B的行数")

    result = [[0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for k in range(cols_A):
            if A[i][k] == 0:
                continue  # 稀疏优化：跳过0元素
            for j in range(cols_B):
                result[i][j] = (result[i][j] + A[i][k] * B[k][j]) % mod
    return result

def sparse_matrix_multiply(
        sparse_A: Dict[int, Dict[int, int]],  # 稀疏矩阵A：{行: {列: 值}}
        B: List[List[int]],
        mod: int = 2
) -> List[List[int]]:
    """稀疏矩阵乘法算子（针对LDPC等稀疏场景优化）"""
    rows_A = max(sparse_A.keys()) + 1 if sparse_A else 0
    cols_B = len(B[0]) if B else 0

    result = [[0] * cols_B for _ in range(rows_A)]
    for i in sparse_A:
        for k, val_A in sparse_A[i].items():
            for j in range(cols_B):
                result[i][j] = (result[i][j] + val_A * B[k][j]) % mod
    return result


# ------------------------------
# 二、数据处理与控制算子
# ------------------------------

def interleave(data: List[int], interleave_table: List[int]) -> List[int]:
    """交织算子（基于交织表重排数据）"""
    if len(data) != len(interleave_table):
        raise ValueError("数据长度必须与交织表长度一致")
    return [data[idx] for idx in interleave_table]

def deinterleave(interleaved_data: List[int], interleave_table: List[int]) -> List[int]:
    """解交织算子（恢复原始数据顺序）"""
    if len(interleaved_data) != len(interleave_table):
        raise ValueError("交织数据长度必须与交织表长度一致")
    deinterleave_table = [0] * len(interleave_table)
    for i, idx in enumerate(interleave_table):
        deinterleave_table[idx] = i
    return [interleaved_data[idx] for idx in deinterleave_table]

class IterationControl:
    """迭代控制算子"""
    def __init__(self, max_iter: int, threshold: float = 1e-6):
        """
        :param max_iter: 最大迭代次数
        :param threshold: 收敛阈值（两次迭代差值小于该值则停止）
        """
        self.max_iter = max_iter
        self.threshold = threshold
        self.current_iter = 0

    def step(self, prev_value: float, current_value: float) -> bool:
        """
        迭代一步并判断是否停止
        :return: True表示继续迭代，False表示停止
        """
        self.current_iter += 1
        if self.current_iter >= self.max_iter:
            return False  # 达到最大迭代次数
        if abs(current_value - prev_value) < self.threshold:
            return False  # 收敛
        return True

def puncturing(data: List[int], puncture_pattern: List[int]) -> List[int]:
    """码率调整算子（删余）"""
    # 删余图案：1表示保留，0表示删除
    return [bit for bit, keep in zip(data, puncture_pattern) if keep]

def repetition(data: List[int], rep_factor: int) -> List[int]:
    """码率调整算子（重复，降低码率）"""
    # 重复因子：每个比特重复rep_factor次
    return [bit for bit in data for _ in range(rep_factor)]


# ------------------------------
# 三、概率与度量计算算子
# ------------------------------

def llr_awgn(r: float, sigma2: float) -> float:
    """
    AWGN信道下的对数似然比（LLR）计算算子
    :param r: 接收信号
    :param sigma2: 噪声方差
    :return: LLR = log(P(x=0|r)/P(x=1|r))
    """
    return (2 * r) / sigma2

def hamming_distance(a: List[int], b: List[int]) -> int:
    """汉明距离度量算子（硬判决）"""
    if len(a) != len(b):
        raise ValueError("输入序列长度必须相等")
    return sum(xor_operator(ai, bi) for ai, bi in zip(a, b))

def euclidean_distance(a: List[float], b: List[float]) -> float:
    """欧氏距离度量算子（软判决）"""
    if len(a) != len(b):
        raise ValueError("输入序列长度必须相等")
    return sum((x - y) **2 for x, y in zip(a, b))** 0.5

def hard_decision(llr: List[float]) -> List[int]:
    """硬判决算子（基于LLR）"""
    return [0 if val >= 0 else 1 for val in llr]


# ------------------------------
# 四、辅助性通用算子
# ------------------------------

def sparse_data_access(sparse_data: Dict[int, int], index: int) -> int:
    """稀疏数据访问算子（获取指定索引的值，默认0）"""
    return sparse_data.get(index, 0)

class CRCOperator:
    """CRC校验算子"""
    def __init__(self, crc_poly: int, crc_len: int):
        """
        :param crc_poly: CRC多项式（整数表示，如CRC-8: 0x07）
        :param crc_len: CRC校验位长度（比特）
        """
        self.crc_poly = crc_poly
        self.crc_len = crc_len
        self.mask = (1 << crc_len) - 1  # 掩码，确保结果在crc_len位内

    def compute(self, data: List[int]) -> List[int]:
        """计算CRC校验位"""
        crc = 0
        for bit in data:
            # 左移1位，加入新比特
            crc = ((crc << 1) | bit) & self.mask
            # 若最高位为1，则与多项式异或
            if crc & (1 << (self.crc_len - 1)):
                crc ^= self.crc_poly
        # 补全剩余移位
        for _ in range(self.crc_len):
            crc = (crc << 1) & self.mask
            if crc & (1 << (self.crc_len - 1)):
                crc ^= self.crc_poly
        # 转换为比特列表
        return [(crc >> i) & 1 for i in range(self.crc_len - 1, -1, -1)]

    def check(self, data: List[int], crc_bits: List[int]) -> bool:
        """验证CRC校验"""
        computed_crc = self.compute(data)
        return computed_crc == crc_bits


# ------------------------------
# 测试示例
# ------------------------------
if __name__ == "__main__":
    # 1. 模2加测试
    print("XOR测试:", xor_operator(1, 0), xor_array([1,0,1], [0,1,1]))  # 1, [1,1,0]

    # 2. 有限域运算测试（GF(2^3)，多项式x^3+x+1=0b1011）
    ff = FiniteFieldOperator(3, 0b1011)
    print("GF(2^3)加法:", ff.add(0b010, 0b101))  # 0b111=7
    print("GF(2^3)乘法:", ff.multiply(0b010, 0b101))  # 0b110=6

    # 3. 交织/解交织测试
    interleave_table = [2, 0, 3, 1]  # 示例交织表
    data = [1,2,3,4]
    print("交织后:", interleave(data, interleave_table))  # [3,1,4,2]
    print("解交织后:", deinterleave(interleave(data, interleave_table), interleave_table))  # [1,2,3,4]

    # 4. LLR计算测试（AWGN信道，r=0.5，sigma²=1）
    print("LLR值:", llr_awgn(0.5, 1.0))  # 1.0

    # 5. CRC校验测试（CRC-8，多项式0x07）
    crc = CRCOperator(0x07, 8)
    data = [1,0,1,0,1,0]
    crc_bits = crc.compute(data)
    print("CRC校验位:", crc_bits)
    print("CRC验证结果:", crc.check(data, crc_bits))  # True
