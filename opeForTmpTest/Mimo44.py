import numpy as np
from scipy.signal import firwin, lfilter
from scipy.fft import fft, ifft


class SymbolLevelOperatorLibrary:
    """符号级共性算子库"""

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
    def peak_detection(correlation_results: np.ndarray, threshold: float = 0.5) -> int:
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
        return received_pilots * np.conj(known_pilots) / (np.abs(known_pilots)**2 + 1e-12)

    @staticmethod
    def interpolation_linear(input_samples: np.ndarray, positions: np.ndarray) -> np.ndarray:
        if len(input_samples) == 0:
            raise ValueError("插值输入需非空序列")
        return np.interp(positions, np.arange(len(input_samples)), input_samples)

    @staticmethod
    def matrix_multiplication(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
        return np.dot(matrix_a, matrix_b)

    @staticmethod
    def mmse_equalization(received_symbols: np.ndarray, channel_matrix: np.ndarray, noise_variance: float) -> np.ndarray:
        H = channel_matrix  # rx_ant × tx_ant
        H_H = np.conj(H.T)  # tx_ant × rx_ant
        I = np.eye(H.shape[1])  # 单位矩阵维度为发射天线数（tx_ant）

        H_H_H = SymbolLevelOperatorLibrary.matrix_multiplication(H_H, H)  # tx_ant × tx_ant
        W = SymbolLevelOperatorLibrary.matrix_multiplication(
            np.linalg.inv(H_H_H + noise_variance * I + 1e-3 * np.eye(H_H_H.shape[0])),
            H_H
        )
        return SymbolLevelOperatorLibrary.matrix_multiplication(W, received_symbols.reshape(-1, 1))

    @staticmethod
    def llr_calculation_qpsk(symbol: np.ndarray, constellation: np.ndarray, noise_variance: float) -> np.ndarray:
        N = len(symbol)
        llr = np.zeros((N, 2))
        bit_mapping = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        for i in range(N):
            dist = np.abs(symbol[i] - constellation)** 2
            llr[i, 0] = (np.sum(dist[bit_mapping[:, 0] == 1]) - np.sum(dist[bit_mapping[:, 0] == 0])) / noise_variance
            llr[i, 1] = (np.sum(dist[bit_mapping[:, 1] == 1]) - np.sum(dist[bit_mapping[:, 1] == 0])) / noise_variance
        return llr

    @staticmethod
    def generate_zc_sequence(length: int, root: int) -> np.ndarray:
        n = np.arange(length)
        zc = np.exp(-1j * np.pi * root * n * (n + 1) / length)
        return zc / np.sqrt(length)  # 归一化

    @staticmethod
    def mimo_channel_generation(num_subcarriers: int, tx_ant: int = 4, rx_ant: int = 4) -> np.ndarray:
        H = np.random.normal(0, np.sqrt(0.5), (num_subcarriers, rx_ant, tx_ant)) + \
            1j * np.random.normal(0, np.sqrt(0.5), (num_subcarriers, rx_ant, tx_ant))
        return H / np.sqrt(tx_ant)

    @staticmethod
    def channel_estimation_ls_mimo(received_pilots: np.ndarray, known_pilots: np.ndarray) -> np.ndarray:
        rx_ant, num_pilots = received_pilots.shape
        tx_ant = known_pilots.shape[0]
        h_pilot = np.zeros((rx_ant, tx_ant, num_pilots), dtype=np.complex128)

        for k in range(num_pilots):
            rx_pilot_k = received_pilots[:, k:k+1]
            known_pilot_k = known_pilots[:, k:k+1]
            h_pilot[:, :, k] = rx_pilot_k @ np.conj(known_pilot_k.T) / (np.abs(known_pilot_k)**2 + 1e-12)
        return h_pilot


# 基础参数配置
tx_ant = 4
rx_ant = 4
fft_size = 1024
cp_length = 128
sym_length = fft_size + cp_length  # 1152点
subcarrier_spacing = 15e3
pilot_interval = 16
num_symbols = 100
num_sync_symbols = 2  # 专用同步符号数量
num_symbols_tx = num_sync_symbols + num_symbols + 2  # 同步+数据+冗余
qpsk_const = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
snr_db = 15
pilot_pos = np.arange(0, fft_size, pilot_interval)
data_pos = np.setdiff1d(np.arange(fft_size), pilot_pos)
num_pilots = len(pilot_pos)
num_data_per_symbol = len(data_pos)
interp_positions = np.linspace(0, num_pilots-1, len(data_pos))


# 发送端处理
def tx_mimo_signal(operator_lib: SymbolLevelOperatorLibrary):
    # 生成发送比特
    num_bits_per_ant = (num_symbols_tx - num_sync_symbols) * num_data_per_symbol * 2  # 扣除同步符号
    tx_bits = [np.random.randint(0, 2, num_bits_per_ant) for _ in range(tx_ant)]
    tx_bits_valid = [bits[:num_symbols * num_data_per_symbol * 2] for bits in tx_bits]

    # 生成QPSK符号（降低幅度，×2）
    tx_symbols_list = []
    for ant in range(tx_ant):
        bits = tx_bits[ant]
        symbol_idx = bits.reshape(-1, 2) @ [2, 1]
        tx_symbols = qpsk_const[symbol_idx] * 2  # 降低幅度，控制功率
        tx_symbols_list.append(tx_symbols)

    # 构造频域数据符号（数据+导频）
    tx_freq_sym = np.zeros(((num_symbols_tx - num_sync_symbols), fft_size, tx_ant), dtype=np.complex128)
    for ant in range(tx_ant):
        zc_pilot = operator_lib.generate_zc_sequence(num_pilots, root=3+ant*2) * 2  # 导频幅度同步降低
        for sym in range(num_symbols_tx - num_sync_symbols):
            tx_freq_sym[sym, data_pos, ant] = tx_symbols_list[ant][sym*num_data_per_symbol : (sym+1)*num_data_per_symbol]
            tx_freq_sym[sym, pilot_pos, ant] = zc_pilot

    # 数据符号转换到时域（含CP）
    tx_time_sym = np.zeros(((num_symbols_tx - num_sync_symbols), sym_length, tx_ant), dtype=np.complex128)
    for ant in range(tx_ant):
        for sym in range(num_symbols_tx - num_sync_symbols):
            ifft_out = operator_lib.ifft_operation(tx_freq_sym[sym, :, ant]) * fft_size
            cp = ifft_out[-cp_length:]
            tx_time_sym[sym, :, ant] = np.hstack([cp, ifft_out])

    # 生成专用同步符号（ZC序列，长度=sym_length）
    sync_seq_length = sym_length
    sync_zc = [operator_lib.generate_zc_sequence(sync_seq_length, root=10+ant) for ant in range(tx_ant)]
    sync_symbols = [seq * 2 for seq in sync_zc]  # 同步符号幅度

    # 组合所有符号（同步符号+数据符号）
    tx_time_combined = np.zeros((num_symbols_tx * sym_length, tx_ant), dtype=np.complex128)
    # 填充同步符号
    for ant in range(tx_ant):
        for sym in range(num_sync_symbols):
            start_idx = sym * sym_length
            tx_time_combined[start_idx:start_idx+sym_length, ant] = sync_symbols[ant]
    # 填充数据符号（偏移同步符号长度）
    data_start = num_sync_symbols * sym_length
    for ant in range(tx_ant):
        for sym in range(num_symbols_tx - num_sync_symbols):
            start_idx = data_start + sym * sym_length
            end_idx = start_idx + sym_length
            tx_time_combined[start_idx:end_idx, ant] = tx_time_sym[sym, :, ant]

    # 日志：发送信号功率
    tx_power = np.mean(np.abs(tx_time_combined)**2)
    print(f"[发送端] 信号功率: {tx_power:.4f}, 总长度: {len(tx_time_combined)}采样点")
    print(f"[发送端] 天线0前10比特: {tx_bits[0][:20].reshape(-1,2)}")
    print(f"[发送端] 对应QPSK符号索引: {symbol_idx[:10].flatten()}")

    return tx_bits_valid, tx_symbols_list, tx_time_combined


# 信道传输（不变）
def mimo_channel_transmit(operator_lib: SymbolLevelOperatorLibrary, tx_time_combined: np.ndarray, snr_db: float):
    h_mimo_freq = operator_lib.mimo_channel_generation(fft_size, tx_ant, rx_ant)
    rx_time_combined = np.zeros((len(tx_time_combined), rx_ant), dtype=np.complex128)
    total_symbols = len(tx_time_combined) // sym_length

    for sym in range(total_symbols):
        start_idx = sym * sym_length
        end_idx = start_idx + sym_length
        tx_sym = tx_time_combined[start_idx:end_idx, :]

        tx_freq = np.zeros((fft_size, tx_ant), dtype=np.complex128)
        for ant in range(tx_ant):
            tx_no_cp = tx_sym[cp_length:, ant]
            tx_freq[:, ant] = operator_lib.fft_operation(tx_no_cp) / fft_size

        rx_freq = np.zeros((fft_size, rx_ant), dtype=np.complex128)
        for sc in range(fft_size):
            H = h_mimo_freq[sc, :, :]
            rx_freq[sc, :] = H @ tx_freq[sc, :]

        rx_time_sym = np.zeros((sym_length, rx_ant), dtype=np.complex128)
        for ant in range(rx_ant):
            ifft_out = operator_lib.ifft_operation(rx_freq[:, ant]) * fft_size
            cp = ifft_out[-cp_length:]
            rx_time_sym[:, ant] = np.hstack([cp, ifft_out])
        rx_time_combined[start_idx:end_idx, :] = rx_time_sym

    signal_power = np.mean(np.abs(rx_time_combined)** 2)
    if signal_power < 1e-6:
        scale_factor = 1e-6 / signal_power if signal_power != 0 else 1e6
        rx_time_combined *= scale_factor
        signal_power = np.mean(np.abs(rx_time_combined)**2)

    noise_power = signal_power / (10 **(snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power/2), rx_time_combined.shape) + \
            1j * np.random.normal(0, np.sqrt(noise_power/2), rx_time_combined.shape)
    rx_signal = rx_time_combined + noise
    print(f"[信道] 信号功率: {signal_power:.4f}, 噪声功率: {noise_power:.6f}, SNR: {10*np.log10(signal_power/noise_power):.1f}dB")
    return rx_signal, h_mimo_freq, noise_power


# 接收端处理（优化同步）
def rx_mimo_signal(operator_lib: SymbolLevelOperatorLibrary, rx_time_combined: np.ndarray,
                   h_mimo_freq: np.ndarray, noise_var: float, tx_symbols_list: list):
    # 1. 数字前端
    fir_coeff = firwin(numtaps=15, cutoff=subcarrier_spacing*0.95, fs=fft_size*subcarrier_spacing)  # 简化滤波器
    iq_matrix = np.array([[1.0, -0.01], [0.01, 1.0]])
    rx_dfe_combined = np.zeros_like(rx_time_combined)

    for rx in range(rx_ant):
        rx_iq = operator_lib.iq_correction_and_gain(rx_time_combined[:, rx], gain=1.0, iq_matrix=iq_matrix)
        rx_dfe_combined[:, rx] = operator_lib.fir_filter(rx_iq, fir_coeff)

    rx_power = np.mean(np.abs(rx_dfe_combined)**2)
    print(f"[接收端] 数字前端处理完成，信号功率: {rx_power:.4f}, 范围: [{np.min(np.abs(rx_dfe_combined)):.3f}, {np.max(np.abs(rx_dfe_combined)):.3f}]")

    # 2. 同步（基于专用同步符号）
    sync_seq_length = sym_length
    local_sync = [operator_lib.generate_zc_sequence(sync_seq_length, root=10+ant) * 2 for ant in range(rx_ant)]  # 本地同步序列
    sync_offset = np.zeros(rx_ant, dtype=int)
    max_sync_offset = sym_length  # 搜索范围：1个符号长度
    min_sync_offset = 0

    for rx in range(rx_ant):
        sync_search_range = rx_dfe_combined[:(num_sync_symbols + 2)*sym_length, rx]  # 搜索前N个符号
        corr_vals = []
        valid_offsets = range(min_sync_offset, max_sync_offset + 1)
        local_seq = local_sync[rx]

        for offset in valid_offsets:
            if offset + sync_seq_length > len(sync_search_range):
                corr_vals.append(0)
                continue
            rx_segment = sync_search_range[offset:offset+sync_seq_length]
            # 归一化相关性
            corr = np.abs(np.sum(rx_segment * np.conj(local_seq)))
            corr /= (np.linalg.norm(rx_segment) * np.linalg.norm(local_seq) + 1e-12)
            corr_vals.append(corr)

        peak_idx = operator_lib.peak_detection(np.array(corr_vals), threshold=0.3)
        sync_offset[rx] = valid_offsets[peak_idx]

    final_sync_offset = int(np.median(sync_offset))  # 取中值更稳健
    sync_quality = np.max(corr_vals) / np.mean(corr_vals) if np.mean(corr_vals) > 0 else 0
    print(f"[同步] 偏移: {final_sync_offset}, 相关性峰值/均值: {sync_quality:.1f} (正常应>5)")
    data_start = final_sync_offset + num_sync_symbols * sym_length  # 数据符号起始位置（跳过同步符号）
    assert data_start + num_symbols * sym_length <= len(rx_dfe_combined), "同步偏移过大"

    # 3. OFDM解调（从数据符号起始位置开始）
    rx_freq = np.zeros((num_symbols, fft_size, rx_ant), dtype=np.complex128)
    for rx in range(rx_ant):
        for sym in range(num_symbols):
            sym_start = data_start + sym * sym_length
            sym_end = sym_start + sym_length
            sym_samp = rx_dfe_combined[sym_start + cp_length : sym_start + sym_length, rx]
            if len(sym_samp) != fft_size:
                raise ValueError(f"符号{sym}长度错误: {len(sym_samp)}点")
            rx_freq[sym, :, rx] = operator_lib.fft_operation(sym_samp) / fft_size

    # 4. MIMO信道估计（不变）
    h_est_full = np.zeros((num_symbols, fft_size, rx_ant, tx_ant), dtype=np.complex128)
    for sym in range(num_symbols):
        rx_pilot = rx_freq[sym, pilot_pos, :].T
        known_pilot = np.array([operator_lib.generate_zc_sequence(num_pilots, root=3+ant*2) * 2 for ant in range(tx_ant)])
        h_pilot = operator_lib.channel_estimation_ls_mimo(rx_pilot, known_pilot)

        for rx in range(rx_ant):
            for tx in range(tx_ant):
                h_pilot_seq = h_pilot[rx, tx, :]
                h_data = operator_lib.interpolation_linear(h_pilot_seq, interp_positions)
                h_est_full[sym, data_pos, rx, tx] = h_data
            h_est_full[sym, pilot_pos, rx, :] = h_pilot[rx, :, :].T

        if sym == 0:
            h_true_pilot = h_mimo_freq[pilot_pos, :, :]
            h_est_pilot = h_est_full[sym, pilot_pos, :, :]
            mse_pilot = np.mean(np.abs(h_true_pilot - h_est_pilot)**2)
            print(f"[信道估计] 导频位置MSE: {mse_pilot:.6f} (越小越好)")

    # 5. MMSE均衡（不变）
    rx_data_sym = np.zeros((num_symbols, num_data_per_symbol, tx_ant), dtype=np.complex128)
    for sym in range(num_symbols):
        for i, data_idx in enumerate(data_pos):
            rx_data = rx_freq[sym, data_idx, :].reshape(-1, 1)
            h = h_est_full[sym, data_idx, :, :]
            equalized = operator_lib.mmse_equalization(rx_data, h, noise_var)
            rx_data_sym[sym, i, :] = equalized.squeeze()

        if sym == 0:
            tx_sym_0 = tx_symbols_list[0][:num_data_per_symbol]
            rx_sym_0 = rx_data_sym[0, :, 0]
            mse_eq = np.mean(np.abs(tx_sym_0 - rx_sym_0)**2)
            print(f"[均衡] 第0符号MSE: {mse_eq:.6f} (越小越好)")
            print(f"[均衡] 发送符号前5个: {np.round(tx_sym_0[:5], 3)}")
            print(f"[均衡] 均衡后前5个: {np.round(rx_sym_0[:5], 3)}")

    # 6. LLR计算与比特恢复（不变）
    rx_bits = [[] for _ in range(tx_ant)]
    for sym in range(num_symbols):
        for i in range(num_data_per_symbol):
            for tx in range(tx_ant):
                symbol = rx_data_sym[sym, i, tx]
                llr = operator_lib.llr_calculation_qpsk(np.array([symbol]), qpsk_const * 2, noise_var)
                rx_bits[tx].append((llr[0, 0] < 0).astype(int))
                rx_bits[tx].append((llr[0, 1] < 0).astype(int))

        if sym == 0:
            for tx in range(1):
                rx_bits_0 = np.array(rx_bits[tx][:20]).reshape(-1, 2)
                print(f"[比特判决] 天线{tx}第0符号前10比特判决: {rx_bits_0[:10]}")

    rx_bits = [np.array(bits) for bits in rx_bits]
    return rx_bits


# BER计算（不变）
def calculate_ber(tx_bits: list, rx_bits: list) -> float:
    total_err = 0
    total_bits = 0
    for ant, (tx_bit, rx_bit) in enumerate(zip(tx_bits, rx_bits)):
        min_len = min(len(tx_bit), len(rx_bit))
        if min_len == 0:
            continue
        err = np.sum(tx_bit[:min_len] != rx_bit[:min_len])
        total_err += err
        total_bits += min_len
        ber_ant = err / min_len
        print(f"[BER] 天线{ant}的BER: {ber_ant:.6f}")
    return total_err / total_bits if total_bits > 0 else 1.0


# 主函数（不变）
if __name__ == "__main__":
    op_lib = SymbolLevelOperatorLibrary()

    print("="*60)
    print("4×4 MIMO链路仿真启动")
    print(f"参数配置：FFT={fft_size}, SCS={subcarrier_spacing/1e3}kHz, SNR={snr_db}dB")
    print("="*60)

    print("\n[1/4] 发送端处理...")
    tx_bits_valid, tx_symbols_list, tx_time_combined = tx_mimo_signal(op_lib)

    print("\n[2/4] 信道传输...")
    rx_time_combined, h_mimo, noise_var = mimo_channel_transmit(op_lib, tx_time_combined, snr_db)

    print("\n[3/4] 接收端处理...")
    rx_bits = rx_mimo_signal(op_lib, rx_time_combined, h_mimo, noise_var, tx_symbols_list)

    print("\n[4/4] 计算BER...")
    ber = calculate_ber(tx_bits_valid, rx_bits)

    print("\n" + "="*60)
    print("仿真结果")
    print("="*60)
    print(f"平均BER：{ber:.6f}")
    print(f"链路状态：{'✅ 跑通' if ber < 1e-3 else '❌ 未跑通'}")
    print("="*60)