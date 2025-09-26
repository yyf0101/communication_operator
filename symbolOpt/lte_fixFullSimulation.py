import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, correlate
from scipy.fft import fft, ifft, fftshift, ifftshift
import matplotlib
matplotlib.use('TkAgg')
# --- 中文显示配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --------------------------
# 1. 核心参数配置
# --------------------------
class LTEParams:
    def __init__(self):
        self.n_fft = 1024          # FFT大小
        self.cp_length = 144       # 常规CP长度（采样点）
        self.symbol_rate = 15.36e6 # 符号率（1倍采样率）
        self.fs = 4 * self.symbol_rate  # 4倍过采样率
        self.rrc_alpha = 0.22      # RRC滤波器滚降系数
        self.rrc_taps = 65         # RRC滤波器抽头数（奇数）
        self.mod_order = 16        # 16QAM调制
        self.bits_per_sym = int(np.log2(self.mod_order))  # 4bit/符号
        self.data_sym_num = 120    # 数据符号数 (调整为12的倍数，模拟10个RB)
        # 关键参数：适配小频偏估计
        self.zc_seq_len_1x = 64    # ZC序列长度（1倍采样），增加长度提高估计精度
        self.sync_seq_len_4x = self.zc_seq_len_1x * 4  # 同步序列长度（4倍采样）
        # 仿真参数
        self.snr_range = [8, 11, 14, 17, 20]
        self.num_trials = 5
        self.true_cfo = 3000       # 真实载波频偏

# 初始化参数
lte = LTEParams()

# --------------------------
# 2. 发射端模块 (Tx)
# --------------------------
def generate_zc_sequence(length, root_idx=3):
    """生成Zadoff-Chu序列"""
    n = np.arange(length)
    zc = np.exp(-1j * np.pi * root_idx * n * (n + 1) / length)
    return zc / np.sqrt(length)

def generate_16qam_constellation():
    """生成标准的16QAM格雷编码星座图"""
    # 标准格雷编码映射，相邻点只有1比特差异
    const = np.array([
        -3-3j, -3-1j, -3+1j, -3+3j,
        -1-3j, -1-1j, -1+1j, -1+3j,
        1-3j,  1-1j,  1+1j,  1+3j,
        3-3j,  3-1j,  3+1j,  3+3j
    ])
    # 归一化，使平均功率为1
    return const / np.sqrt(10)

def tx_source_gen(lte_params):
    """生成随机比特流"""
    return np.random.randint(0, 2, lte_params.data_sym_num * lte_params.bits_per_sym)

def tx_modulate(tx_bits, constellation, lte_params):
    """16QAM调制（与标准星座图匹配）"""
    # 格雷编码映射表，用于将2进制比特组转换为星座点索引
    gray_code = [0, 1, 3, 2, 6, 7, 5, 4, 12, 13, 15, 14, 10, 11, 9, 8]
    bit_groups = tx_bits.reshape(-1, lte_params.bits_per_sym)
    # 将比特组转换为十进制整数
    bit_values = np.dot(bit_groups, 2 ** np.arange(lte_params.bits_per_sym-1, -1, -1))
    # 使用格雷码映射获取最终的星座点索引
    sym_indices = [np.where(gray_code == val)[0][0] for val in bit_values]
    return constellation[sym_indices]

def tx_resource_map(tx_sym, lte_params):
    """资源映射到子载波（优化版）"""
    freq_sym = np.zeros(lte_params.n_fft, dtype=complex)
    # 假设使用中心频带附近的子载波，避开DC子载波(索引为n_fft//2)
    # 我们将数据映射到正频率轴，并对称地映射到负频率轴以保持信号实值性
    # 这是一种简化的映射，非严格LTE标准，但能保证链路通畅
    half_fft = lte_params.n_fft // 2
    # 从DC子载波右侧开始映射
    start_idx = half_fft + 1
    end_idx = start_idx + lte_params.data_sym_num
    pos_subcarriers = np.arange(start_idx, end_idx)

    # 检查是否越界，如果越界则调整
    if end_idx > lte_params.n_fft:
        raise ValueError("数据符号过多，超出FFT大小限制。请减小data_sym_num或增大n_fft。")

    freq_sym[pos_subcarriers] = tx_sym
    # 负频率轴对应位置，实现Hermitian对称
    neg_subcarriers = lte_params.n_fft - pos_subcarriers
    freq_sym[neg_subcarriers] = np.conj(tx_sym)

    return freq_sym, pos_subcarriers

def tx_ifft_cp(freq_sym, lte_params):
    """IFFT变换并添加循环前缀"""
    freq_sym_shift = ifftshift(freq_sym)
    time_sym = ifft(freq_sym_shift, lte_params.n_fft)
    cp = time_sym[-lte_params.cp_length:]
    return np.concatenate([cp, time_sym])

def tx_pulse_shaping(time_signal, lte_params):
    """RRC脉冲成形（优化版）"""
    N, alpha, Ts, Fs = lte_params.rrc_taps, lte_params.rrc_alpha, 1/lte_params.symbol_rate, lte_params.fs
    T_sample = 1 / Fs
    t = np.arange(-N // 2, N // 2 + 1) * T_sample

    rrc_coeff = np.zeros_like(t)
    for i, t_val in enumerate(t):
        if t_val == 0:
            rrc_coeff[i] = (1 - alpha + 4 * alpha / np.pi)
        elif np.abs(t_val) == Ts / (4 * alpha):
            rrc_coeff[i] = alpha / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) +
                                                 (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha)))
        else:
            numerator = np.sin(np.pi * t_val * (1 - alpha) / Ts) + \
                        4 * alpha * (t_val / Ts) * np.cos(np.pi * t_val * (1 + alpha) / Ts)
            denominator = np.pi * t_val * (1 - (4 * alpha * t_val / Ts) ** 2) / Ts
            rrc_coeff[i] = numerator / denominator
    rrc_coeff /= np.sqrt(np.sum(rrc_coeff ** 2))

    interp_signal = np.zeros(len(time_signal) * 4, dtype=complex)
    interp_signal[::4] = time_signal
    # 使用卷积代替lfilter，效果更精确，mode='same'保证输出长度一致
    shaped_signal = np.convolve(interp_signal, rrc_coeff, mode='same')
    return shaped_signal, rrc_coeff

def tx_add_sync(signal, rrc_coeff, lte_params):
    """添加纯ZC序列作为同步头"""
    zc_seq_1x = generate_zc_sequence(lte_params.zc_seq_len_1x)
    zc_interp = np.zeros(len(zc_seq_1x) * 4, dtype=complex)
    zc_interp[::4] = zc_seq_1x
    zc_shaped_4x = np.convolve(zc_interp, rrc_coeff, mode='same')
    # 截取有效长度
    zc_shaped_4x = zc_shaped_4x[:lte_params.sync_seq_len_4x]
    return np.concatenate([zc_shaped_4x, signal]), zc_seq_1x

# --------------------------
# 3. 信道模块 (Channel)
# --------------------------
def channel_awgn_cfo(signal, snr_db, lte_params):
    """添加AWGN噪声和固定频偏"""
    signal_power = np.var(signal)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    n = np.arange(len(signal))
    cfo_factor = np.exp(1j * 2 * np.pi * lte_params.true_cfo * n / lte_params.fs)
    return signal * cfo_factor + noise, noise_power

# --------------------------
# 4. 接收端模块 (Rx)
# --------------------------
def rx_matched_filter(signal, rrc_coeff):
    """匹配滤波并降采样（优化版）"""
    # 匹配滤波器是发射端滤波器的时域反转共轭
    filtered = np.convolve(signal, rrc_coeff[::-1].conj(), mode='same')
    return filtered[::4]

def rx_frame_sync(signal_1x, local_zc_seq_1x, lte_params):
    """帧同步，提取接收的ZC序列"""
    # 使用'valid'模式进行相关，只返回有意义的重叠部分
    corr = correlate(signal_1x, local_zc_seq_1x, mode='valid')
    corr_abs = np.abs(corr)
    peak_idx = np.argmax(corr_abs)
    # 同步头在相关结果峰值处结束，因此其开始位置需要减去序列长度
    zc_start = peak_idx
    rx_zc_seq_1x = signal_1x[zc_start : zc_start + lte_params.zc_seq_len_1x]

    # 健壮性处理：如果提取的序列太短，则进行填充
    if len(rx_zc_seq_1x) < lte_params.zc_seq_len_1x:
        rx_zc_seq_1x = np.pad(rx_zc_seq_1x, (0, lte_params.zc_seq_len_1x - len(rx_zc_seq_1x)))

    # 计算峰均比（峰值/平均能量）
    peak_to_avg = np.max(corr_abs) / np.mean(corr_abs) if np.mean(corr_abs) > 0 else 0
    return rx_zc_seq_1x, peak_to_avg

def rx_freq_offset_est(rx_zc_seq_1x, local_zc_seq_1x, lte_params):
    """基于延迟相关的高精度频偏估计（优化版）"""
    corr_peaks = []
    # 扩大延迟搜索范围以提高频偏估计范围和精度
    delay_range = np.arange(-20, 21)
    for delay in delay_range:
        delayed_local = np.roll(local_zc_seq_1x, delay)
        corr = np.abs(np.sum(rx_zc_seq_1x * np.conj(delayed_local)))
        corr_peaks.append(corr)
    best_delay_idx = np.argmax(corr_peaks)
    best_delay = delay_range[best_delay_idx]

    # 频偏计算公式：f = (best_delay * 符号率) / ZC序列长度
    # 这里的符号率是1倍采样率下的速率
    cfo_est = (best_delay * lte_params.symbol_rate) / lte_params.zc_seq_len_1x

    return cfo_est, best_delay

def rx_freq_compensate(signal, freq_offset, lte_params):
    """频偏补偿（修复版）"""
    n = np.arange(len(signal))
    # 补偿因子必须使用实际的采样率fs，而不是符号率
    comp_factor = np.exp(-1j * 2 * np.pi * freq_offset * n / lte_params.fs)
    return signal * comp_factor

def rx_remove_cp(signal, lte_params):
    """移除循环前缀"""
    return signal[lte_params.cp_length : lte_params.cp_length + lte_params.n_fft]

def rx_fft_resource_extract(time_signal, pos_subcarriers, lte_params):
    """FFT变换并提取数据子载波"""
    freq_signal = fftshift(fft(time_signal, lte_params.n_fft))
    return freq_signal[pos_subcarriers]

def rx_demodulate(rx_sym, constellation, noise_power, lte_params):
    """16QAM软解调"""
    llr = []
    # 标准格雷码，用于解调时判断每个比特
    gray_code = [0, 1, 3, 2, 6, 7, 5, 4, 12, 13, 15, 14, 10, 11, 9, 8]
    for sym in rx_sym:
        distances = np.abs(sym - constellation)**2
        for k in range(lte_params.bits_per_sym):
            # 根据格雷码确定每个比特位为0或1的星座点
            mask = np.array([(g >> (lte_params.bits_per_sym-1 - k)) & 1 for g in gray_code])
            d0 = np.min(distances[mask == 0]) if np.any(mask == 0) else 1e6
            d1 = np.min(distances[mask == 1]) if np.any(mask == 1) else 1e6
            # LLR = (d0 - d1) / sigma^2
            llr.append((d0 - d1) / noise_power)
    # 硬判决：LLR > 0 判为1，否则为0
    return (np.array(llr) > 0).astype(int)

# --------------------------
# 5. 完整链路仿真
# --------------------------
def lte_full_link(snr_db, trial_idx, lte_params):
    """单轮完整链路仿真"""
    # --- 发射端 ---
    tx_bits = tx_source_gen(lte_params)
    constellation = generate_16qam_constellation()
    tx_sym = tx_modulate(tx_bits, constellation, lte_params)
    freq_sym, pos_subcarriers = tx_resource_map(tx_sym, lte_params)
    time_with_cp = tx_ifft_cp(freq_sym, lte_params)
    tx_shaped, rrc_coeff = tx_pulse_shaping(time_with_cp, lte_params)
    tx_final, local_zc_seq_1x = tx_add_sync(tx_shaped, rrc_coeff, lte_params)

    # --- 信道 ---
    rx_channel, noise_power = channel_awgn_cfo(tx_final, snr_db, lte_params)

    # --- 接收端 ---
    rx_filtered_1x = rx_matched_filter(rx_channel, rrc_coeff)
    rx_zc_seq_1x, peak_ratio = rx_frame_sync(rx_filtered_1x, local_zc_seq_1x, lte_params)
    cfo_est, best_delay = rx_freq_offset_est(rx_zc_seq_1x, local_zc_seq_1x, lte_params)

    # 从同步头结束后开始提取数据，注意这里的信号已经是1倍采样率
    data_start = rx_filtered_1x.tolist().index(rx_zc_seq_1x[0], max(0, peak_ratio-100)) + lte_params.zc_seq_len_1x
    rx_data_1x = rx_filtered_1x[data_start:]

    # 对1倍采样率的数据进行频偏补偿
    rx_data_comp = rx_freq_compensate(rx_data_1x, cfo_est, lte_params)

    # 确保有足够的数据进行后续处理
    required_length = lte_params.cp_length + lte_params.n_fft
    if len(rx_data_comp) < required_length:
        # 使用循环扩展以保持信号特性，优于填充零
        rx_data_comp = np.tile(rx_data_comp, (required_length // len(rx_data_comp) + 1))[:required_length]

    rx_no_cp = rx_remove_cp(rx_data_comp, lte_params)
    rx_extracted_sym = rx_fft_resource_extract(rx_no_cp, pos_subcarriers, lte_params)
    rx_bits = rx_demodulate(rx_extracted_sym, constellation, noise_power, lte_params)

    # --- 结果计算与日志 ---
    # 确保比较的比特长度一致
    min_len = min(len(rx_bits), len(tx_bits))
    ber = np.mean(rx_bits[:min_len] != tx_bits[:min_len]) if min_len > 0 else 0.5
    cfo_error = abs(cfo_est - lte_params.true_cfo)
    print(f"SNR={snr_db:2d}dB | 第{trial_idx+1}轮 | 同步峰值比={peak_ratio:.2f} | "
          f"真实频偏={lte_params.true_cfo:4d}Hz | 估计频偏={cfo_est:4.0f}Hz | 误差={cfo_error:4.0f}Hz | "
          f"最佳延迟={best_delay:2d} | BER={ber:.6f}")
    return ber

# --------------------------
# 6. 批量仿真与可视化
# --------------------------
def batch_simulation():
    """批量仿真并绘制BER曲线"""
    avg_ber = {snr: 0.0 for snr in lte.snr_range}
    print("="*150)
    print(f"LTE完整链路仿真开始（真实频偏={lte.true_cfo}Hz）")
    print("="*150)

    for snr in lte.snr_range:
        total_ber = 0.0
        for trial in range(lte.num_trials):
            total_ber += lte_full_link(snr, trial, lte)
        avg_ber[snr] = total_ber / lte.num_trials
        print("="*150)
        print(f"SNR={snr:2d}dB | 平均BER={avg_ber[snr]:.6f}")
        print("="*150)

    plt.figure(figsize=(10, 6))
    snrs = sorted(lte.snr_range)
    bers = [avg_ber[snr] for snr in snrs]
    plt.semilogy(snrs, bers, marker='o', linewidth=2, markersize=8, label=f'LTE 16QAM (CFO={lte.true_cfo}Hz)')
    plt.xlabel('信噪比 SNR (dB)')
    plt.ylabel('误码率 BER')
    plt.title('LTE下行链路误码率性能（优化后）')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    # plt.ylim(1e-6, 0.5)
    plt.show()

if __name__ == "__main__":
    batch_simulation()