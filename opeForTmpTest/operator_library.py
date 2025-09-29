import numpy as np
from scipy.signal import firwin, lfilter
from scipy.fft import fft, ifft
from scipy.linalg import qr, svd, toeplitz

class SymbolLevelOperatorLibrary:
    """
    符号级共性算子库。
    所有函数都以NumPy数组作为输入和输出，以模拟硬件中的数据流。
    """

    @staticmethod
    def cmac(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """复数乘加算子: z = a * b + c"""
        return a * b + c

    @staticmethod
    def cmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """复数乘法算子: z = a * b"""
        return a * b

    @staticmethod
    def fir_filter(input_signal: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        """FIR滤波器"""
        # 使用scipy的lfilter模拟硬件实现的FIR
        return lfilter(coefficients, 1.0, input_signal)

    @staticmethod
    def iq_correction_and_gain(input_signal: np.ndarray, gain: float, iq_matrix: np.ndarray) -> np.ndarray:
        """IQ校正与增益调整"""
        # input_signal: shape (N,), complex
        # iq_matrix: shape (2, 2), real, e.g., [[a, -b], [b, a]] for rotation and scaling
        corrected_signal = np.empty_like(input_signal, dtype=np.complex128)
        for i, s in enumerate(input_signal):
            re, im = np.real(s), np.imag(s)
            corrected_re = iq_matrix[0, 0] * re + iq_matrix[0, 1] * im
            corrected_im = iq_matrix[1, 0] * re + iq_matrix[1, 1] * im
            corrected_signal[i] = corrected_re + 1j * corrected_im
        return corrected_signal * gain

    @staticmethod
    def correlation(reference_sequence: np.ndarray, input_signal: np.ndarray) -> np.ndarray:
        """相关运算 (滑动互相关)"""
        # 为简化，使用numpy的卷积实现。注意卷积与相关的关系: corr(a,b) = conv(a, reversed(b))
        reversed_ref = np.conj(np.flipud(reference_sequence))
        return np.convolve(input_signal, reversed_ref, mode='valid')

    @staticmethod
    def peak_detection(correlation_results: np.ndarray, threshold: float = 0.7) -> int:
        """峰值检测"""
        # 找到超过阈值的第一个峰值位置
        peak_values = np.where(correlation_results > threshold * np.max(correlation_results))[0]
        if peak_values.size > 0:
            return peak_values[0]
        return np.argmax(correlation_results) # 如果没有超过阈值的，返回最大值位置

    @staticmethod
    def fft_operation(input_signal: np.ndarray) -> np.ndarray:
        """FFT运算"""
        return fft(input_signal)

    @staticmethod
    def ifft_operation(input_signal: np.ndarray) -> np.ndarray:
        """IFFT运算"""
        return ifft(input_signal)

    @staticmethod
    def channel_estimation_ls(received_pilots: np.ndarray, known_pilots: np.ndarray) -> np.ndarray:
        """基于LS (最小二乘) 的信道估计"""
        # h_hat = Y_pilot * X_pilot^* / ||X_pilot||^2
        # 为简化，假设导频功率为1
        return received_pilots * np.conj(known_pilots)

    @staticmethod
    def interpolation_linear(input_samples: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """线性插值"""
        # 这是一个简化版本，假设positions是0到len(input_samples)-1之间的浮点数索引
        # 实际硬件实现会更复杂，需要地址生成和分数延迟滤波器
        return np.interp(positions, np.arange(len(input_samples)), input_samples)

    @staticmethod
    def matrix_inversion(matrix: np.ndarray) -> np.ndarray:
        """矩阵求逆"""
        return np.linalg.inv(matrix)

    @staticmethod
    def matrix_multiplication(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
        """矩阵乘法"""
        return np.dot(matrix_a, matrix_b)

    @staticmethod
    def mmse_equalization(received_symbols: np.ndarray, channel_matrix: np.ndarray, noise_variance: float) -> np.ndarray:
        """MMSE均衡"""
        # W = (H^H * H + sigma^2 * I)^-1 * H^H
        # s_hat = W * y
        H = channel_matrix
        H_H = np.conj(H.T)
        I = np.eye(H.shape[0])
        W = SymbolLevelOperatorLibrary.matrix_multiplication(
            SymbolLevelOperatorLibrary.matrix_inversion(
                SymbolLevelOperatorLibrary.matrix_multiplication(H_H, H) + noise_variance * I
            ),
            H_H
        )
        # 假设received_symbols是一个向量 (N, 1)
        return SymbolLevelOperatorLibrary.matrix_multiplication(W, received_symbols.reshape(-1, 1))

    @staticmethod
    def llr_calculation(symbol: complex, constellation: np.ndarray, noise_variance: float) -> np.ndarray:
        """
        计算单个符号的LLR (简化版)。
        假设BPSK调制。
        """
        # 对于更复杂的QAM，需要计算每个比特的似然比
        # 这里以BPSK为例，星座点为[-1, 1]
        # LLR = ln(P(b=1|r)/P(b=0|r)) = 2 * Re(r * s^*) / sigma^2
        # 对于BPSK, s^* 对于b=1是1, 对于b=0是-1
        llr = np.zeros(1)
        # 计算到两个星座点的距离
        dist0 = np.abs(symbol - constellation[0])**2
        dist1 = np.abs(symbol - constellation[1])**2
        llr[0] = (dist0 - dist1) / noise_variance
        return llr