import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Tuple
import json

# 算子元数据的数据类
@dataclass
class OperatorMetadata:
    """算子的元数据信息"""
    input_params: List[str] = field(default_factory=list)
    output_params: List[str] = field(default_factory=list)
    computation_complexity: str = ""  # 例如 "O(N)", "O(N^3)"
    delay_cycles: int = 0             # 延迟周期
    throughput: float = 0.0           # 吞吐量，单位：符号/秒
    resource_usage: Dict[str, float] = field(default_factory=dict)  # 资源使用情况
    description: str = ""             # 算子描述

# 算子基类
class BaseOperator:
    """所有符号级算子的基类"""

    def __init__(self, name: str, operator_type: str):
        self.name = name
        self.operator_type = operator_type  # 例如："基础运算", "组合算子"
        self.metadata = OperatorMetadata()
        self.dependencies = []  # 依赖的其他算子

    def process(self, *args, **kwargs) -> Any:
        """处理数据的抽象方法，子类必须实现"""
        raise NotImplementedError("子类必须实现process方法")

    def get_metadata(self) -> OperatorMetadata:
        """获取算子元数据"""
        return self.metadata

    def add_dependency(self, operator: 'BaseOperator'):
        """添加依赖的算子"""
        if operator not in self.dependencies:
            self.dependencies.append(operator)

    def to_dict(self) -> Dict[str, Any]:
        """将算子信息转换为字典，用于序列化"""
        return {
            "name": self.name,
            "type": self.operator_type,
            "metadata": {
                "input_params": self.metadata.input_params,
                "output_params": self.metadata.output_params,
                "computation_complexity": self.metadata.computation_complexity,
                "delay_cycles": self.metadata.delay_cycles,
                "throughput": self.metadata.throughput,
                "resource_usage": self.metadata.resource_usage,
                "description": self.metadata.description
            },
            "dependencies": [op.name for op in self.dependencies]
        }

    def __str__(self) -> str:
        return f"{self.operator_type}: {self.name}"

# 基础运算算子
class BasicOperator(BaseOperator):
    """基础运算算子类"""

    def __init__(self, name: str):
        super().__init__(name, "基础运算")

# 组合算子
class CompositeOperator(BaseOperator):
    """组合算子类，由多个基础算子或其他组合算子构成"""

    def __init__(self, name: str):
        super().__init__(name, "组合算子")
        self.sub_operators = []  # 包含的子算子

    def add_sub_operator(self, operator: BaseOperator):
        """添加子算子"""
        if operator not in self.sub_operators:
            self.sub_operators.append(operator)
            self.add_dependency(operator)

    def to_dict(self) -> Dict[str, Any]:
        """扩展父类方法，包含子算子信息"""
        base_dict = super().to_dict()
        base_dict["sub_operators"] = [op.name for op in self.sub_operators]
        return base_dict

# 具体算子实现 - 相位旋转（复数乘法）
class PhaseRotationOperator(BasicOperator):
    """相位旋转算子（复数乘法）"""

    def __init__(self):
        super().__init__("相位旋转")

        # 设置元数据
        self.metadata.input_params = ["复数符号序列", "相位旋转角度"]
        self.metadata.output_params = ["相位旋转后的复数符号序列"]
        self.metadata.computation_complexity = "O(N)"
        self.metadata.delay_cycles = 1
        self.metadata.throughput = 1e6  # 示例值
        self.metadata.resource_usage = {"LUT": 500, "FF": 200}
        self.metadata.description = "对复数符号序列应用相位旋转，实现相位补偿"

    def process(self, symbols: np.ndarray, phase_angles: np.ndarray) -> np.ndarray:
        """
        对符号应用相位旋转

        参数:
            symbols: 复数符号序列
            phase_angles: 相位旋转角度（弧度）

        返回:
            相位旋转后的复数符号序列
        """
        # 计算旋转因子：e^(-jθ) = cosθ - j*sinθ
        rotation_factors = np.exp(-1j * phase_angles)
        # 应用相位旋转（复数乘法）
        return symbols * rotation_factors

# 具体算子实现 - 载波频偏补偿
class CFOCompensationOperator(CompositeOperator):
    """载波频偏补偿算子"""

    def __init__(self):
        super().__init__("载波频偏补偿")

        # 添加子算子（相位旋转）
        self.phase_rotator = PhaseRotationOperator()
        self.add_sub_operator(self.phase_rotator)

        # 设置元数据
        self.metadata.input_params = ["含频偏的复数符号序列", "估计频偏"]
        self.metadata.output_params = ["补偿后的复数符号序列"]
        self.metadata.computation_complexity = "O(N)"
        self.metadata.delay_cycles = 3
        self.metadata.throughput = 8e5  # 示例值
        self.metadata.resource_usage = {"LUT": 1200, "FF": 500}
        self.metadata.description = "对接收符号进行载波频偏补偿，消除频率偏移影响"

    def process(self, symbols: np.ndarray, freq_offset: float, sampling_freq: float, symbol_period: float) -> np.ndarray:
        """
        执行载波频偏补偿

        参数:
            symbols: 含频偏的复数符号序列
            freq_offset: 估计的频率偏移
            sampling_freq: 采样频率
            symbol_period: 符号周期

        返回:
            补偿后的复数符号序列
        """
        # 计算每个符号的相位偏移
        n = np.arange(len(symbols))
        phase_offsets = 2 * np.pi * freq_offset * (n * symbol_period)

        # 使用相位旋转算子进行补偿
        return self.phase_rotator.process(symbols, phase_offsets)

# 具体算子实现 - FFT算子
class FFTOperator(BasicOperator):
    """FFT变换算子"""

    def __init__(self, size: int = 1024):
        super().__init__(f"FFT_{size}")
        self.size = size

        # 设置元数据
        self.metadata.input_params = ["时域复数符号序列"]
        self.metadata.output_params = ["频域复数符号序列"]
        self.metadata.computation_complexity = f"O(N log N) where N={size}"
        self.metadata.delay_cycles = size // 2
        self.metadata.throughput = 5e5  # 示例值
        self.metadata.resource_usage = {"LUT": 8000, "FF": 3000, "BRAM": 10}
        self.metadata.description = f"{size}点快速傅里叶变换，将时域信号转换到频域"

    def process(self, time_domain_symbols: np.ndarray) -> np.ndarray:
        """
        执行FFT变换

        参数:
            time_domain_symbols: 时域复数符号序列

        返回:
            频域复数符号序列
        """
        # 如果输入长度不足FFT大小，进行补零
        if len(time_domain_symbols) < self.size:
            time_domain_symbols = np.pad(time_domain_symbols, (0, self.size - len(time_domain_symbols)), mode='constant')
        # 执行FFT
        return np.fft.fft(time_domain_symbols)

# 具体算子实现 - LMMSE MIMO检测器
class LMMSEDetectorOperator(CompositeOperator):
    """LMMSE MIMO检测器"""

    def __init__(self):
        super().__init__("LMMSE MIMO检测器")

        # 设置元数据
        self.metadata.input_params = ["MIMO接收符号矢量", "信道矩阵", "噪声方差"]
        self.metadata.output_params = ["估计的发送符号"]
        self.metadata.computation_complexity = "O(N^3)"
        self.metadata.delay_cycles = 50
        self.metadata.throughput = 1e5  # 示例值
        self.metadata.resource_usage = {"LUT": 15000, "FF": 8000, "DSP": 50}
        self.metadata.description = "基于线性最小均方误差准则的MIMO信号检测器"

    def process(self, received_symbols: np.ndarray, channel_matrix: np.ndarray, noise_variance: float) -> np.ndarray:
        """
        执行LMMSE检测

        参数:
            received_symbols: 接收符号矢量，形状为[Nr, 1]
            channel_matrix: 信道矩阵，形状为[Nr, Nt]
            noise_variance: 噪声方差

        返回:
            估计的发送符号，形状为[Nt, 1]
        """
        Nr, Nt = channel_matrix.shape

        # 计算LMMSE权重矩阵: W = (H^H H + σ² I)^(-1) H^H
        # H^H表示H的共轭转置
        h_hermitian = np.conj(channel_matrix.T)
        # 计算H^H H
        hh_product = np.dot(h_hermitian, channel_matrix)
        # 添加正则项
        regularization = noise_variance * np.eye(Nt)
        # 矩阵求逆
        inverse_term = np.linalg.inv(hh_product + regularization)
        # 计算权重矩阵
        weight_matrix = np.dot(inverse_term, h_hermitian)
        # 估计发送符号
        transmitted_estimate = np.dot(weight_matrix, received_symbols)

        return transmitted_estimate

# 算子管理器，用于提取、分类和管理算子
class OperatorManager:
    """算子管理器，负责算子的提取、分类和管理"""

    def __init__(self):
        self.operators = {}  # 存储所有算子，键为算子名称
        self.basic_operators = []  # 基础算子列表
        self.composite_operators = []  # 组合算子列表

    def add_operator(self, operator: BaseOperator):
        """添加算子到管理器"""
        if operator.name not in self.operators:
            self.operators[operator.name] = operator

            # 分类存储
            if isinstance(operator, BasicOperator):
                self.basic_operators.append(operator)
            elif isinstance(operator, CompositeOperator):
                self.composite_operators.append(operator)

    def get_operator(self, name: str) -> BaseOperator:
        """根据名称获取算子"""
        return self.operators.get(name)

    def extract_operators(self, operator_classes: List[Callable]) -> List[BaseOperator]:
        """
        提取并实例化一组算子

        参数:
            operator_classes: 算子类的列表

        返回:
            实例化的算子列表
        """
        extracted = []
        for op_class in operator_classes:
            # 对于需要参数的算子，可以在这里处理
            if op_class == FFTOperator:
                # 实例化不同大小的FFT
                for size in [256, 512, 1024, 2048]:
                    op = op_class(size)
                    self.add_operator(op)
                    extracted.append(op)
            else:
                op = op_class()
                self.add_operator(op)
                extracted.append(op)
        return extracted

    def get_operators_by_type(self, operator_type: str) -> List[BaseOperator]:
        """根据类型获取算子"""
        if operator_type == "基础运算":
            return self.basic_operators
        elif operator_type == "组合算子":
            return self.composite_operators
        return []

    def save_operator_metadata(self, file_path: str):
        """将算子元数据保存到JSON文件"""
        operator_data = {name: op.to_dict() for name, op in self.operators.items()}
        with open(file_path, 'w') as f:
            json.dump(operator_data, f, indent=4)

    def load_operator_metadata(self, file_path: str) -> Dict[str, Any]:
        """从JSON文件加载算子元数据"""
        with open(file_path, 'r') as f:
            return json.load(f)

    def __str__(self) -> str:
        return f"算子管理器: 共{len(self.operators)}个算子，其中基础运算{len(self.basic_operators)}个，组合算子{len(self.composite_operators)}个"

# 示例使用
if __name__ == "__main__":
    # 创建算子管理器
    op_manager = OperatorManager()

    # 定义要提取的算子类
    operator_classes = [
        PhaseRotationOperator,
        CFOCompensationOperator,
        FFTOperator,
        LMMSEDetectorOperator
    ]

    # 提取算子
    extracted_ops = op_manager.extract_operators(operator_classes)
    print(op_manager)

    # 打印所有算子
    print("\n所有算子:")
    for op in op_manager.operators.values():
        print(f"- {op}, 输入: {op.metadata.input_params}, 输出: {op.metadata.output_params}")

    # 演示CFO补偿算子的使用
    print("\n演示CFO补偿算子:")
    # 生成测试数据
    np.random.seed(42)
    num_symbols = 100
    # 生成随机QPSK符号
    symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], num_symbols)
    # 加入频率偏移
    freq_offset = 1000  # 1000 Hz
    sampling_freq = 1e6  # 1 MHz
    symbol_period = 1e-3  # 1 ms
    n = np.arange(num_symbols)
    cfo_symbols = symbols * np.exp(2j * np.pi * freq_offset * (n * symbol_period))

    # 获取CFO补偿算子并处理
    cfo_op = op_manager.get_operator("载波频偏补偿")
    compensated_symbols = cfo_op.process(cfo_symbols, freq_offset, sampling_freq, symbol_period)

    # 计算补偿前后的误差
    original_error = np.mean(np.abs(cfo_symbols - symbols))
    compensated_error = np.mean(np.abs(compensated_symbols - symbols))
    print(f"补偿前平均误差: {original_error:.4f}")
    print(f"补偿后平均误差: {compensated_error:.4f}")

    # 保存算子元数据
    op_manager.save_operator_metadata("baseband_operators_metadata.json")
    print("\n算子元数据已保存到baseband_operators_metadata.json")
