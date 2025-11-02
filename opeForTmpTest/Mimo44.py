import numpy as np
from scipy.signal import firwin, lfilter
from scipy.fft import fft, ifft
from scipy.linalg import inv


class SymbolLevelOperatorLibrary:
    """符号级共性算子库"""

    @staticmethod
    def cmac(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        return a * b + c

    @staticmethod
    def cmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a * b

    @staticmethod
    def fir_filter(input_signal: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        return lfilter(coefficients, 1.0, input_signal)

    @staticmethod
    def iq_correction_and_gain(input_signal: np.ndarray, gain: float, iq_matrix: np.ndarray) -> np.ndarray:
        corrected_signal = np.empty_like(input_signal, dtype=np.complex128)
        for i, s in enumerate(input_signal):
            re, im = np.real(s), np.imag(s)
            corrected_re = iq_matrix[0, 0] * re + iq_matrix[0, 1] * im
            corrected_im = iq_matrix[1, 0] * re + iq_matrix[1, 1] * im
            corrected_signal[i] = corrected_re + 1j * corrected_im
        return corrected_signal * gain

    @staticmethod
    def correlation(reference_sequence: np.ndarray, input_signal: np.ndarray) -> np.ndarray:
        reversed_ref = np.conj(np.flipud(reference_sequence))
        return np.convolve(input_signal, reversed_ref, mode='valid')

    @staticmethod
    def peak_detection(correlation_results: np.ndarray, threshold: float = 0.7) -> int:
        peak_values = np.where(correlation_results > threshold * np.max(correlation_results))[0]
        if peak_values.size > 0:
            return peak_values[0]
        return np.argmax(correlation_results)

    @staticmethod
    def fft_operation(input_signal: np.ndarray) -> np.ndarray:
        if len(input_signal) != 1024:
            raise ValueError(f"FFT输入需1024点（当前{len(input_signal)}点）")
        return fft(input_signal)

    @staticmethod
    def ifft_operation(input_signal: np.ndarray) -> np.ndarray:
        return ifft(input_signal)

    @staticmethod
    def channel_estimation_ls(received_pilots: np.ndarray, known_pilots: np.ndarray) -> np.ndarray:
        return received_pilots * np.conj(known_pilots)

    @staticmethod
    def interpolation_linear(input_samples: np.ndarray, positions: np.ndarray) -> np.ndarray:
        if len(input_samples) == 0:
            raise ValueError("插值输入需非空序列")
        return np.interp(positions, np.arange(len(input_samples)), input_samples)

    @staticmethod
    def matrix_inversion(matrix: np.ndarray) -> np.ndarray:
        return np.linalg.inv(matrix)

    @staticmethod
    def matrix_multiplication(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
        return np.dot(matrix_a, matrix_b)

    @staticmethod
    def mmse_equalization(received_symbols: np.ndarray, channel_matrix: np.ndarray, noise_variance: float) -> np.ndarray:
        H = channel_matrix
        H_H = np.conj(H.T)
        I = np.eye(H.shape[0])
        W = SymbolLevelOperatorLibrary.matrix_multiplication(
            SymbolLevelOperatorLibrary.matrix_inversion(
                SymbolLevelOperatorLibrary.matrix_multiplication(H_H, H) + noise_variance * I
            ),
            H_H
        )
        return SymbolLevelOperatorLibrary.matrix_multiplication(W, received_symbols.reshape(-1, 1))

    @staticmethod
    def llr_calculation(symbol: complex, constellation: np.ndarray, noise_variance: float) -> np.ndarray:
        llr = np.zeros(1)
        dist0 = np.abs(symbol - constellation[0])**2
        dist1 = np.abs(symbol - constellation[1])** 2
        llr[0] = (dist0 - dist1) / noise_variance
        return llr

    @staticmethod
    def mimo_channel_generation(num_subcarriers: int, tx_ant: int = 4, rx_ant: int = 4) -> np.ndarray:
        H = np.random.normal(0, np.sqrt(0.5), (num_subcarriers, rx_ant, tx_ant)) + \
            1j * np.random.normal(0, np.sqrt(0.5), (num_subcarriers, rx_ant, tx_ant))
        return H

    @staticmethod
    def channel_estimation_ls_mimo(received_pilots: np.ndarray, known_pilots: np.ndarray) -> np.ndarray:
        rx_ant, num_pilots = received_pilots.shape
        tx_ant = known_pilots.shape[0]
        h_pilot = np.zeros((rx_ant, tx_ant, num_pilots), dtype=np.complex128)

        for k in range(num_pilots):
            rx_pilot_k = received_pilots[:, k:k+1]
            known_pilot_k = known_pilots[:, k:k+1]
            denominator = np.dot(known_pilot_k.conj().T, known_pilot_k) + 1e-6
            h_pilot[:, :, k] = np.dot(rx_pilot_k, known_pilot_k.conj().T) / denominator

        return h_pilot

    @staticmethod
    def llr_calculation_qpsk(symbol: np.ndarray, constellation: np.ndarray, noise_variance: float) -> np.ndarray:
        N = len(symbol)
        llr = np.zeros((N, 2))
        bit_mapping = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        for i in range(N):
            dist = np.abs(symbol[i] - constellation) **2
            llr[i, 0] = (dist[bit_mapping[:, 0] == 1].sum() - dist[bit_mapping[:, 0] == 0].sum()) / noise_variance
            llr[i, 1] = (dist[bit_mapping[:, 1] == 1].sum() - dist[bit_mapping[:, 1] == 0].sum()) / noise_variance
        return llr

    @staticmethod
    def generate_zc_sequence(length: int, root: int) -> np.ndarray:
        n = np.arange(length)
        return np.exp(-1j * np.pi * root * n * (n + 1) / length)


# 基础参数配置
tx_ant = 4
rx_ant = 4
fft_size = 1024
cp_length = 128
sym_length = fft_size + cp_length  # 1152点
subcarrier_spacing = 15e3
pilot_interval = 16
num_symbols = 100
num_symbols_tx = num_symbols + 1  # 预留1个符号冗余
qpsk_const = np.array([1+1j, -1+1j, -1-1j, 1-1j])
snr_db = 15
pilot_pos = np.arange(0, fft_size, pilot_interval)  # 导频位置：0,16,...,1008（共64个）
data_pos = np.setdiff1d(np.arange(fft_size), pilot_pos)  # 数据子载波位置（共960个）
num_pilots = len(pilot_pos)  # 64
num_data_per_symbol = len(data_pos)  # 960
interp_positions = data_pos / pilot_interval  # 插值位置


# 发送端处理
def tx_mimo_signal(operator_lib: SymbolLevelOperatorLibrary):
    num_bits_per_ant = num_symbols_tx * num_data_per_symbol * 2
    tx_bits = [np.random.randint(0, 2, num_bits_per_ant) for _ in range(tx_ant)]
    tx_bits_valid = [bits[:num_symbols * num_data_per_symbol * 2] for bits in tx_bits]

    tx_symbols_list = []
    for bits in tx_bits:
        symbol_idx = bits.reshape(-1, 2).dot([1, 2])
        tx_symbols = qpsk_const[symbol_idx]
        tx_symbols_list.append(tx_symbols)

    tx_freq_sym = np.zeros((num_symbols_tx, fft_size, tx_ant), dtype=np.complex128)
    for ant in range(tx_ant):
        zc_pilot = operator_lib.generate_zc_sequence(num_pilots, root=ant+1)
        for sym in range(num_symbols_tx):
            tx_freq_sym[sym, data_pos, ant] = tx_symbols_list[ant][sym*num_data_per_symbol : (sym+1)*num_data_per_symbol]
            tx_freq_sym[sym, pilot_pos, ant] = zc_pilot

    tx_time_sym = np.zeros((num_symbols_tx, sym_length, tx_ant), dtype=np.complex128)
    for ant in range(tx_ant):
        for sym in range(num_symbols_tx):
            ifft_out = operator_lib.ifft_operation(tx_freq_sym[sym, :, ant])
            cp = ifft_out[-cp_length:]
            tx_time_sym[sym, :, ant] = np.hstack([cp, ifft_out])

    tx_time_combined = tx_time_sym.reshape(-1, tx_ant)
    return tx_bits_valid, tx_time_combined


# 信道传输
def mimo_channel_transmit(operator_lib: SymbolLevelOperatorLibrary, tx_time_combined: np.ndarray, snr_db: float):
    h_mimo_freq = operator_lib.mimo_channel_generation(fft_size, tx_ant, rx_ant)
    h_mimo_time = operator_lib.ifft_operation(h_mimo_freq)

    rx_time_combined = np.zeros_like(tx_time_combined, dtype=np.complex128)
    for rx in range(rx_ant):
        rx_signal = np.zeros(len(tx_time_combined), dtype=np.complex128)
        for tx in range(tx_ant):
            h_truncated = h_mimo_time[:sym_length, rx, tx]
            rx_signal += np.convolve(tx_time_combined[:, tx], h_truncated, mode="same")
        rx_time_combined[:, rx] = rx_signal

    signal_power = np.mean(np.abs(rx_time_combined) **2)
    noise_var = signal_power / (10** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_var/2), rx_time_combined.shape) + \
            1j * np.random.normal(0, np.sqrt(noise_var/2), rx_time_combined.shape)
    return rx_time_combined + noise, h_mimo_freq, noise_var


# 接收端处理
def rx_mimo_signal(operator_lib: SymbolLevelOperatorLibrary, rx_time_combined: np.ndarray, h_mimo_freq: np.ndarray, noise_var: float):
    rx_bits = []
    fir_coeff = firwin(numtaps=31, cutoff=subcarrier_spacing*1.2, fs=fft_size*subcarrier_spacing)
    iq_matrix = np.array([[1.0, -0.05], [0.05, 1.0]])

    # 1. 数字前端
    rx_dfe_combined = np.zeros_like(rx_time_combined)
    for rx in range(rx_ant):
        rx_iq = operator_lib.iq_correction_and_gain(rx_time_combined[:, rx], gain=1.1, iq_matrix=iq_matrix)
        rx_dfe_combined[:, rx] = operator_lib.fir_filter(rx_iq, fir_coeff)

    # 2. 同步
    sync_offset = np.zeros(rx_ant, dtype=int)
    max_sync_offset = sym_length
    min_sync_offset = 0
    for rx in range(rx_ant):
        sync_seq = rx_dfe_combined[:2*sym_length, rx]
        corr_vals = []
        valid_offset_range = range(min_sync_offset, max_sync_offset + 1)
        for i in valid_offset_range:
            cp_ref = sync_seq[i:i+cp_length]
            cp_recv = sync_seq[i+fft_size : i+fft_size+cp_length]
            corr = np.abs(operator_lib.correlation(cp_ref, cp_recv))
            corr_vals.append(corr.max())
        peak_idx_in_range = operator_lib.peak_detection(np.array(corr_vals))
        sync_offset[rx] = valid_offset_range[peak_idx_in_range]
    final_sync_offset = int(np.mean(sync_offset))
    assert final_sync_offset + num_symbols * sym_length <= len(rx_dfe_combined), \
        f"同步偏移{final_sync_offset}过大"
    print(f"[同步结果] 采样点级定时偏移：{final_sync_offset}（0~{sym_length}点）")

    # 3. OFDM解调
    rx_freq = np.zeros((num_symbols, fft_size, rx_ant), dtype=np.complex128)
    for rx in range(rx_ant):
        for sym in range(num_symbols):
            sym_start_samp = final_sync_offset + sym * sym_length
            sym_samp = rx_dfe_combined[sym_start_samp + cp_length : sym_start_samp + sym_length, rx]
            if len(sym_samp) != fft_size:
                raise ValueError(f"符号{sym}提取{len(sym_samp)}点（需1024点），同步偏移{final_sync_offset}超界")
            rx_freq[sym, :, rx] = operator_lib.fft_operation(sym_samp)

    # 4. MIMO信道估计
    h_est_full = np.zeros((num_symbols, fft_size, rx_ant, tx_ant), dtype=np.complex128)
    for sym in range(num_symbols):
        rx_pilot = rx_freq[sym, pilot_pos, :].T
        known_pilot = np.array([operator_lib.generate_zc_sequence(num_pilots, root=ant+1) for ant in range(tx_ant)])
        h_pilot = operator_lib.channel_estimation_ls_mimo(rx_pilot, known_pilot)

        # 线性插值补全数据子载波信道
        for rx in range(rx_ant):
            for tx in range(tx_ant):
                h_pilot_seq = h_pilot[rx, tx, :]
                h_data = operator_lib.interpolation_linear(h_pilot_seq, interp_positions)
                h_est_full[sym, data_pos, rx, tx] = h_data
            h_est_full[sym, pilot_pos, rx, :] = h_pilot[rx, :, :].T

    # 5. MMSE均衡 - 修复矩阵维度不匹配问题
    rx_data_sym = np.zeros((num_symbols, num_data_per_symbol, tx_ant), dtype=np.complex128)
    for sym in range(num_symbols):
        # 对每个数据子载波单独进行均衡（每个子载波有独立的信道矩阵）
        for i, data_idx in enumerate(data_pos):
            # 提取当前子载波的接收信号（rx_ant×1）
            rx_data = rx_freq[sym, data_idx, :].reshape(-1, 1)
            # 提取当前子载波的信道矩阵（rx_ant×tx_ant）
            h = h_est_full[sym, data_idx, :, :]
            # MMSE均衡：恢复tx_ant个发射流的符号
            equalized = operator_lib.mmse_equalization(rx_data, h, noise_var)
            # 存储均衡结果
            rx_data_sym[sym, i, :] = equalized.squeeze()

    # 6. QPSK-LLR计算与比特恢复
    rx_bits = [[] for _ in range(tx_ant)]  # 为每个发射天线单独存储比特
    for sym in range(num_symbols):
        for i in range(num_data_per_symbol):
            # 每个数据子载波对应tx_ant个发射天线的符号
            for tx in range(tx_ant):
                symbol = rx_data_sym[sym, i, tx]
                # 计算单个符号的LLR
                llr = operator_lib.llr_calculation_qpsk(np.array([symbol]), qpsk_const, noise_var)
                # 判决并添加到对应发射天线的比特序列
                rx_bits[tx].append((llr[0, 0] < 0).astype(int))
                rx_bits[tx].append((llr[0, 1] < 0).astype(int))

    # 转换为numpy数组
    rx_bits = [np.array(bits) for bits in rx_bits]

    return rx_bits


# BER计算
def calculate_ber(tx_bits: list, rx_bits: list) -> float:
    total_err = 0
    total_bits = 0
    for tx_bit, rx_bit in zip(tx_bits, rx_bits):
        min_len = min(len(tx_bit), len(rx_bit))
        total_err += np.sum(tx_bit[:min_len] != rx_bit[:min_len])
        total_bits += min_len
    return total_err / total_bits if total_bits > 0 else 1.0


# 主函数：全链路执行
if __name__ == "__main__":
    op_lib = SymbolLevelOperatorLibrary()

    print("="*60)
    print("4×4 MIMO链路仿真启动")
    print(f"参数配置：FFT={fft_size}点，SCS={subcarrier_spacing/1e3}kHz，QPSK，SNR={snr_db}dB")
    print(f"工程化配置：发送端预留1个符号冗余，同步偏移限制0~{sym_length}点")
    print("="*60)

    # 1. 发送端
    print("\n[1/4] 发送端：生成4×4 MIMO QPSK信号（含1个符号冗余）...")
    tx_bits_valid, tx_time_combined = tx_mimo_signal(op_lib)

    # 2. 信道
    print("[2/4] 信道：4×4瑞利衰落+AWGN噪声（长度与发送端一致）...")
    rx_time_combined, h_mimo, noise_var = mimo_channel_transmit(op_lib, tx_time_combined, snr_db)

    # 3. 接收端
    print("[3/4] 接收端：数字前端→同步→均衡→LLR计算...")
    rx_bits = rx_mimo_signal(op_lib, rx_time_combined, h_mimo, noise_var)

    # 4. 性能验证
    print("[4/4] 性能验证：计算有效符号BER...")
    ber = calculate_ber(tx_bits_valid, rx_bits)

    # 输出结果
    print("\n" + "="*60)
    print("4×4 MIMO链路仿真结果")
    print("="*60)
    print(f"平均误码率（BER）：{ber:.6f}")
    print(f"链路状态：{'✅ 跑通' if ber < 1e-3 else '❌ 未跑通'}（工业级要求BER<1e-3）")
    print("="*60)