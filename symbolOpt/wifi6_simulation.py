import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import lfilter, correlate
from scipy.fft import fft, ifft, fftshift, ifftshift

# --------------------------
# WiFi6标准固定参数
# --------------------------
WIFI6_PARAMS = {
    'n_fft': 2048,
    'cp_length': 320,
    'fir_taps': 64,
    'fir_rolloff': 0.35,
    'symbol_rate': 11.428e6,
    'fs': 45.712e6,     # 4倍过采样
    'bits_per_symbol': 4,  # 16QAM
    'sync_seq': np.array([1+0j,-1+0j,1+0j,-1+0j,1+0j,-1+0j,1+0j,-1+0j]*24),  # 192点
    'num_data_sym': 50
}

# --------------------------
# WiFi6专用算子（固化参数）
# --------------------------
def design_wifi6_rrc():
    """设计WiFi6专用RRC滤波器"""
    taps = WIFI6_PARAMS['fir_taps']
    rolloff = WIFI6_PARAMS['fir_rolloff']
    R = WIFI6_PARAMS['symbol_rate']
    fs = WIFI6_PARAMS['fs']
    t = np.arange(-taps//2, taps//2) / fs
    Rt = R * t
    beta = rolloff
    rrc = (np.sin(np.pi * (1 - beta) * Rt) + 4 * beta * Rt * np.cos(np.pi * (1 + beta) * Rt)) / \
          (np.pi * Rt * (1 - (4 * beta * Rt)**2 + 1e-10))
    return rrc / np.sqrt(np.sum(rrc**2))

def wifi6_freq_map(tx_symbols):
    """WiFi6频域映射"""
    n_fft = WIFI6_PARAMS['n_fft']
    freq_sym = np.zeros(n_fft, dtype=complex)
    all_data_subcarriers = np.arange(1, n_fft//2)  # 1~1023，共1022个点
    used_subcarriers = all_data_subcarriers[:len(tx_symbols)]
    freq_sym[used_subcarriers] = tx_symbols
    freq_sym[n_fft - used_subcarriers] = np.conj(tx_symbols)
    return freq_sym, used_subcarriers

def generate_wifi6_constellation():
    """WiFi6 16QAM星座图"""
    return generate_lte_constellation()  # 16QAM星座图通用

def generate_lte_constellation():
    # 复用LTE的16QAM星座图（通用）
    constellation = np.array([
        3+3j, 3+1j, 1+3j, 1+1j,
        3-3j, 3-1j, 1-3j, 1-1j,
        -3+3j, -3+1j, -1+3j, -1+1j,
        -3-3j, -3-1j, -1-3j, -1-1j
    ]) / np.sqrt(10)
    return constellation

def wifi6_soft_demod(rx_symbols, noise_var):
    """WiFi6 16QAM软解调"""
    bits_per_symbol = WIFI6_PARAMS['bits_per_symbol']
    constellation = generate_wifi6_constellation()
    llr = []
    for sym in rx_symbols:
        distances = np.abs(sym - constellation)**2
        for k in range(bits_per_symbol):
            mask = (np.arange(len(constellation)) >> (bits_per_symbol - 1 - k)) & 1
            d0 = np.min(distances[mask == 0]) if np.any(mask == 0) else 1e6
            d1 = np.min(distances[mask == 1]) if np.any(mask == 1) else 1e6
            llr.append((d0 - d1) / noise_var)
    return np.array(llr)

def wifi6_frame_sync(rx_signal, sync_seq):
    """WiFi6帧同步"""
    corr = correlate(np.abs(rx_signal), np.abs(sync_seq), mode='same')
    frame_start = np.argmax(corr) - len(sync_seq) // 2
    return np.clip(frame_start, 0, len(rx_signal) - len(sync_seq))

def wifi6_freq_offset_est(rx_sync_seq, fs):
    """WiFi6频偏估计"""
    N = len(rx_sync_seq)
    half = N // 2
    phase_diff = np.mean(np.angle(rx_sync_seq[half:] * np.conj(rx_sync_seq[:half])))
    freq_offset = phase_diff / (2 * np.pi * (half / fs))
    return freq_offset

def wifi6_freq_compensate(x, freq_offset, fs):
    """WiFi6时域频偏补偿"""
    n = np.arange(len(x))
    return x * np.exp(-1j * 2 * np.pi * freq_offset * n / fs)

# --------------------------
# WiFi6完整链路仿真
# --------------------------
def simulate_wifi6(snr_db):
    # 1. 初始化算子
    rrc_coeff = design_wifi6_rrc()
    sync_seq = WIFI6_PARAMS['sync_seq']
    n_fft = WIFI6_PARAMS['n_fft']
    cp_length = WIFI6_PARAMS['cp_length']
    bits_per_symbol = WIFI6_PARAMS['bits_per_symbol']
    num_data_sym = WIFI6_PARAMS['num_data_sym']
    fs = WIFI6_PARAMS['fs']

    # 2. 生成发送数据
    tx_data_bits = np.random.randint(0, 2, num_data_sym * bits_per_symbol)
    constellation = generate_wifi6_constellation()
    bit_matrix = tx_data_bits.reshape(-1, bits_per_symbol)
    weights = 2 ** np.arange(bits_per_symbol - 1, -1, -1)
    sym_indices = np.dot(bit_matrix, weights)
    tx_data_sym = constellation[sym_indices]

    # 3. 发射端处理
    freq_sym, used_subcarriers = wifi6_freq_map(tx_data_sym)
    tx_ifft = np.real(ifft(ifftshift(freq_sym), n_fft))
    tx_cp = np.concatenate([tx_ifft[-cp_length:], tx_ifft])
    tx_signal = lfilter(rrc_coeff, 1.0, tx_cp)
    tx_signal = np.concatenate([sync_seq, tx_signal])

    # 4. 信道传输
    signal_power = np.var(tx_signal)
    snr = 10 ** (snr_db / 10)
    noise_power = signal_power / snr
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(tx_signal)) + 1j * np.random.randn(len(tx_signal)))
    true_freq_offset = np.random.randint(-3, 4)
    n = np.arange(len(tx_signal))
    rx_signal = tx_signal * np.exp(1j * 2 * np.pi * true_freq_offset * n / fs) + noise

    # 5. 接收端处理
    rx_filtered = lfilter(rrc_coeff, 1.0, rx_signal)
    frame_start = wifi6_frame_sync(rx_filtered, sync_seq)
    rx_sync_seq = rx_filtered[frame_start:frame_start+len(sync_seq)]
    freq_offset = wifi6_freq_offset_est(rx_sync_seq, fs)
    rx_data_time = rx_filtered[frame_start+len(sync_seq):]
    rx_data_time_comp = wifi6_freq_compensate(rx_data_time, freq_offset, fs)

    if len(rx_data_time_comp) < cp_length + n_fft:
        rx_data_time_comp = np.pad(rx_data_time_comp, (0, cp_length + n_fft - len(rx_data_time_comp)))
    rx_no_cp = rx_data_time_comp[cp_length:cp_length+n_fft]
    rx_fft = fftshift(fft(rx_no_cp, n_fft))
    rx_data_sym = rx_fft[used_subcarriers]
    rx_llr = wifi6_soft_demod(rx_data_sym, noise_var=noise_power)
    rx_data_bits = (rx_llr > 0).astype(int)

    # 6. 计算BER
    if len(rx_data_bits) != len(tx_data_bits):
        return 0.5
    return np.mean(rx_data_bits != tx_data_bits)

# --------------------------
# WiFi6仿真验证与可视化
# --------------------------
if __name__ == "__main__":
    snr_range = np.arange(0, 21, 3)
    num_trials = 5
    wifi6_ber = []

    print("=== WiFi6标准链路仿真 ===")
    for snr in snr_range:
        avg_ber = 0.0
        for _ in range(num_trials):
            ber = simulate_wifi6(snr)
            avg_ber += ber / num_trials
        wifi6_ber.append(avg_ber)
        print(f"SNR={snr:2d}dB | 平均BER={avg_ber:.6f}")

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_range, wifi6_ber, marker='s', linewidth=2, label='WiFi6 (16QAM)')
    plt.xlabel('信噪比 SNR (dB)', fontsize=12)
    plt.ylabel('误码率 BER', fontsize=12)
    plt.title('WiFi6标准通信链路误码率性能', fontsize=14)
    plt.grid(True, which='both', ls='--', alpha=0.7)
    plt.legend(fontsize=11)
    plt.ylim(1e-6, 0.5)
    plt.show()