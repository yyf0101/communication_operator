import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import lfilter, correlate
from scipy.fft import fft, ifft, fftshift, ifftshift

# --------------------------
# LTE标准固定参数（优化同步序列+信号功率）
# --------------------------
LTE_PARAMS = {
    'n_fft': 1024,
    'cp_length': 144,
    'fir_taps': 64,
    'fir_rolloff': 0.22,
    'symbol_rate': 15.36e6,
    'fs': 61.44e6,
    'bits_per_symbol': 4,  # 16QAM
    # 优化1：换用自相关性更强的m序列（4级m序列，周期15，重复13次→195点，接近原192点）
    'sync_seq': np.array([1,0,0,0,1,0,0,1,1,0,1,0,1,1,1]*13, dtype=complex) * 10,  # 乘10放大，提升相关性
    'num_data_sym': 50,
    'signal_gain': 100  # 优化2：发射端信号放大系数，避免被噪声淹没
}

# --------------------------
# 工具函数：日志格式化
# --------------------------
def log_step(step_name, content):
    print(f"\033[1;34m【{step_name}】\033[0m {content}")

# --------------------------
# LTE专用算子（全量优化）
# --------------------------
def design_lte_rrc():
    """设计RRC滤波器（不变，但确保系数正确）"""
    taps = LTE_PARAMS['fir_taps']
    rolloff = LTE_PARAMS['fir_rolloff']
    R = LTE_PARAMS['symbol_rate']
    fs = LTE_PARAMS['fs']
    t = np.arange(-taps//2, taps//2) / fs
    Rt = R * t
    beta = rolloff

    rrc = np.zeros_like(Rt, dtype=float)
    zero_idx = np.where(np.abs(Rt) < 1e-10)[0]
    if zero_idx.size > 0:
        rrc[zero_idx] = (1 - beta) + (4 * beta / np.pi)  # t=0处的极限值

    non_zero_mask = np.abs(Rt) >= 1e-10
    denominator = np.pi * Rt[non_zero_mask] * (1 - (4 * beta * Rt[non_zero_mask])**2)
    numerator = np.sin(np.pi * (1 - beta) * Rt[non_zero_mask]) + 4 * beta * Rt[non_zero_mask] * np.cos(np.pi * (1 + beta) * Rt[non_zero_mask])
    rrc[non_zero_mask] = numerator / denominator

    return rrc / np.sqrt(np.sum(rrc**2))  # 能量归一化

def lte_freq_map(tx_symbols):
    """频域映射（不变，但确保子载波索引正确）"""
    n_fft = LTE_PARAMS['n_fft']
    freq_sym = np.zeros(n_fft, dtype=complex)
    all_data_subcarriers = np.arange(1, n_fft//2)  # 排除直流和负频率起始点
    used_subcarriers = all_data_subcarriers[:len(tx_symbols)]
    freq_sym[used_subcarriers] = tx_symbols
    freq_sym[n_fft - used_subcarriers] = np.conj(tx_symbols)  # 共轭对称
    return freq_sym, used_subcarriers

def generate_lte_constellation():
    """16QAM星座图（不变，但确保幅值正确）"""
    constellation = np.array([
        3+3j, 3+1j, 1+3j, 1+1j,
        3-3j, 3-1j, 1-3j, 1-1j,
        -3+3j, -3+1j, -1+3j, -1+1j,
        -3-3j, -3-1j, -1-3j, -1-1j
    ]) / np.sqrt(10)  # 能量归一化（每个符号平均功率为1）
    return constellation

def lte_soft_demod(rx_symbols, noise_var):
    """软解调（不变，但确保LLR计算正确）"""
    bits_per_symbol = LTE_PARAMS['bits_per_symbol']
    constellation = generate_lte_constellation()
    llr = []
    log_step("解调前-星座图", f"接收符号数：{len(rx_symbols)}，符号均值：{np.mean(rx_symbols):.4f}，符号方差：{np.var(rx_symbols):.4f}")

    for sym in rx_symbols:
        distances = np.abs(sym - constellation)**2
        for k in range(bits_per_symbol):
            mask = (np.arange(len(constellation)) >> (bits_per_symbol - 1 - k)) & 1
            d0 = np.min(distances[mask == 0]) if np.any(mask == 0) else 1e6
            d1 = np.min(distances[mask == 1]) if np.any(mask == 1) else 1e6
            llr.append((d0 - d1) / noise_var)

    llr_np = np.array(llr)
    log_step("解调-LLR", f"LLR均值：{np.mean(llr_np):.4f}，LLR方差：{np.var(llr_np):.4f}，LLR>0占比：{np.mean(llr_np>0):.4f}")
    return llr_np

# --------------------------
# 优化3：帧同步逻辑（增加峰锐度判断，避免选错峰）
# --------------------------
def lte_frame_sync(rx_signal, sync_seq):
    # 复数相关（保留相位信息）
    corr = correlate(rx_signal, sync_seq[::-1].conj(), mode='same')
    corr_abs = np.abs(corr)
    peak_value = np.max(corr_abs)
    # 计算峰锐度：峰值/次峰值（确保主峰唯一）
    sorted_corr = np.sort(corr_abs)[::-1]
    main_peak = sorted_corr[0]
    sub_peak = sorted_corr[1]
    peak_sharpness = main_peak / sub_peak if sub_peak > 1e-6 else 100  # 峰锐度（>2为优）

    log_step("帧同步-相关峰", f"峰值：{main_peak:.4f}，次峰值：{sub_peak:.4f}，峰锐度：{peak_sharpness:.2f}（>2为优）")
    log_step("帧同步-相关峰", f"相关峰均值：{np.mean(corr_abs):.4f}，峰均比：{main_peak/np.mean(corr_abs):.2f}（>10为优）")

    # 仅选择峰锐度合格的主峰位置
    if peak_sharpness < 1.5:
        log_step("帧同步-警告", "峰锐度不足，可能同步错误！")
    frame_start = np.argmax(corr_abs) - len(sync_seq) // 2
    frame_start = np.clip(frame_start, 0, len(rx_signal) - len(sync_seq))
    log_step("帧同步-结果", f"估计帧起始位置：{frame_start}，同步序列长度：{len(sync_seq)}")
    return frame_start

# --------------------------
# 优化4：频偏估计（移除硬限制+异常相位差剔除）
# --------------------------
def lte_freq_offset_est(rx_sync_seq, fs):
    """
    优化版频偏估计：相位差累积法（无硬限制+异常值剔除）
    解决：相位跳变导致的频偏计算异常
    """
    # 1. 预处理：移除噪声点（幅值过小的点）
    valid_mask = np.abs(rx_sync_seq) > np.mean(np.abs(rx_sync_seq)) * 0.3  # 保留幅值>30%均值的点
    rx_sync_valid = rx_sync_seq[valid_mask]
    if len(rx_sync_valid) < 50:
        log_step("频偏估计-警告", "有效同步序列点过少，频偏估计可能不准！")
        return 0.0

    # 2. 计算相位差（并剔除异常跳变）
    phase = np.angle(rx_sync_valid)
    phase_diff = np.diff(phase)
    # 剔除相位差绝对值>π/2的异常点（避免-π~π跳变导致的错误）
    phase_diff = phase_diff[np.abs(phase_diff) < np.pi/2]
    if len(phase_diff) == 0:
        return 0.0

    # 3. 计算平均相位差→频偏
    avg_phase_diff = np.mean(phase_diff)
    freq_offset = avg_phase_diff * fs / (2 * np.pi)  # 核心公式：频偏=平均相位差*fs/(2π)

    # 4. 仅做“合理范围”限制（±50kHz，远大于真实频偏）
    freq_offset = np.clip(freq_offset, -50e3, 50e3) if abs(freq_offset) > 50e3 else freq_offset
    log_step("频偏估计", f"平均相位差：{avg_phase_diff:.4f} rad，估计频偏：{freq_offset:.2f} Hz")
    return freq_offset

def lte_freq_compensate(x, freq_offset, fs):
    """频偏补偿（不变，但确保相位补偿正确）"""
    n = np.arange(len(x))
    compensated = x * np.exp(-1j * 2 * np.pi * freq_offset * n / fs)
    # 更准确的补偿效果判断：对比补偿前后的信号幅值稳定性
    amp_before = np.abs(x[np.abs(x) > 1e-6][:20])
    amp_after = np.abs(compensated[np.abs(compensated) > 1e-6][:20])
    amp_var_before = np.var(amp_before)
    amp_var_after = np.var(amp_after)
    log_step("频偏补偿", f"补偿前幅值方差：{amp_var_before:.6f}，补偿后：{amp_var_after:.6f}（越小越稳定）")
    return compensated

# --------------------------
# LTE完整链路仿真（终极优化版）
# --------------------------
def simulate_lte(snr_db, trial_idx):
    log_step("仿真开始", f"第{trial_idx+1}次试验，SNR：{snr_db} dB")

    # 1. 初始化算子
    rrc_coeff = design_lte_rrc()
    sync_seq_base = LTE_PARAMS['sync_seq']
    n_fft = LTE_PARAMS['n_fft']
    cp_length = LTE_PARAMS['cp_length']
    bits_per_symbol = LTE_PARAMS['bits_per_symbol']
    num_data_sym = LTE_PARAMS['num_data_sym']
    fs = LTE_PARAMS['fs']
    signal_gain = LTE_PARAMS['signal_gain']

    # 2. 发射端：生成数据（放大信号，避免被噪声淹没）
    tx_data_bits = np.random.randint(0, 2, num_data_sym * bits_per_symbol)
    constellation = generate_lte_constellation()
    bit_matrix = tx_data_bits.reshape(-1, bits_per_symbol)
    weights = 2 ** np.arange(bits_per_symbol - 1, -1, -1)
    sym_indices = np.dot(bit_matrix, weights)
    tx_data_sym = constellation[sym_indices] * signal_gain  # 优化：放大信号
    log_step("发射端-星座图", f"发射符号数：{len(tx_data_sym)}，符号均值：{np.mean(tx_data_sym):.4f}，符号方差：{np.var(tx_data_sym):.4f}")

    # 3. 发射端：频域映射→IFFT→加CP→成形滤波（同步序列也成形）
    freq_sym, used_subcarriers = lte_freq_map(tx_data_sym)
    tx_ifft = ifft(ifftshift(freq_sym), n_fft) * signal_gain  # IFFT后也放大
    log_step("发射端-IFFT", f"IFFT后信号长度：{len(tx_ifft)}，均值：{np.mean(tx_ifft):.4f}，方差：{np.var(tx_ifft):.4f}")

    # 加CP
    tx_cp = np.concatenate([tx_ifft[-cp_length:], tx_ifft])
    # 数据信号成形
    tx_data_signal = lfilter(rrc_coeff, 1.0, tx_cp)
    # 同步序列成形（优化：和数据信号频谱一致）
    tx_sync_signal = lfilter(rrc_coeff, 1.0, sync_seq_base)
    # 拼接同步序列和数据信号
    tx_signal = np.concatenate([tx_sync_signal, tx_data_signal])
    log_step("发射端-最终信号", f"发射信号总长度：{len(tx_signal)}，同步序列长度：{len(tx_sync_signal)}，数据信号长度：{len(tx_data_signal)}")

    # 4. 信道：AWGN+频偏（真实频偏±5kHz内，更贴近实际）
    signal_power = np.var(tx_signal)
    snr = 10 ** (snr_db / 10)
    noise_power = signal_power / snr
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(tx_signal)) + 1j * np.random.randn(len(tx_signal)))
    true_freq_offset = np.random.randint(-5000, 5000)  # 真实频偏±5kHz
    n = np.arange(len(tx_signal))
    rx_signal = tx_signal * np.exp(1j * 2 * np.pi * true_freq_offset * n / fs) + noise
    log_step("信道", f"真实频偏：{true_freq_offset:.2f} Hz，信号功率：{signal_power:.4f}，噪声功率：{noise_power:.4f}，实际SNR：{10*np.log10(signal_power/noise_power):.2f} dB")

    # 5. 接收端：匹配滤波→同步→频偏补偿→FFT→解调（全流程优化）
    rx_filtered = lfilter(rrc_coeff, 1.0, rx_signal)
    log_step("接收端-滤波", f"滤波后信号长度：{len(rx_filtered)}，均值：{np.mean(rx_filtered):.4f}，方差：{np.var(rx_filtered):.4f}")

    # 帧同步（优化版，峰锐度判断）
    frame_start = lte_frame_sync(rx_filtered, tx_sync_signal)
    # 提取同步序列（并计算真实相关性）
    rx_sync_seq = rx_filtered[frame_start:frame_start+len(tx_sync_signal)]
    # 计算归一化相关性（避免幅值影响）
    corr_sync = np.abs(np.sum(rx_sync_seq * np.conj(tx_sync_signal))) / \
                np.sqrt(np.sum(np.abs(rx_sync_seq)**2) * np.sum(np.abs(tx_sync_signal)**2))
    log_step("接收端-同步序列", f"提取同步序列长度：{len(rx_sync_seq)}，与本地序列相关性：{corr_sync:.4f}（>0.5为优）")

    # 频偏估计（优化版，无硬限制）
    estimated_freq = lte_freq_offset_est(rx_sync_seq, fs)
    # 频偏补偿（全信号补偿）
    rx_compensated = lte_freq_compensate(rx_filtered, estimated_freq, fs)

    # 提取数据（补偿滤波器延迟）
    rrc_delay = len(rrc_coeff) // 2  # RRC滤波器群延迟（64/2=32点）
    data_start = frame_start + len(tx_sync_signal) + rrc_delay
    rx_data_time = rx_compensated[data_start:]
    # 确保数据长度足够（补零）
    if len(rx_data_time) < cp_length + n_fft:
        rx_data_time = np.pad(rx_data_time, (0, cp_length + n_fft - len(rx_data_time)))
    log_step("接收端-数据提取", f"数据起始位置：{data_start}，提取数据长度：{len(rx_data_time)}，补零后长度：{len(rx_data_time)}")

    # 去CP→FFT
    rx_no_cp = rx_data_time[cp_length:cp_length+n_fft]
    rx_fft = fftshift(fft(rx_no_cp, n_fft))
    rx_data_sym = rx_fft[used_subcarriers]
    # 日志：对比发射/接收子载波（关键！）
    log_step("接收端-FFT", f"FFT后提取子载波数：{len(rx_data_sym)}，前3个接收子载波：{[f'{s:.4f}' for s in rx_data_sym[:3]]}")
    log_step("接收端-FFT", f"前3个发射子载波：{[f'{s:.4f}' for s in tx_data_sym[:3]]}（对比是否一致）")

    # 软解调→BER计算
    rx_llr = lte_soft_demod(rx_data_sym, noise_var=noise_power)
    rx_data_bits = (rx_llr > 0).astype(int)
    log_step("接收端-比特对比", f"发射比特数：{len(tx_data_bits)}，接收比特数：{len(rx_data_bits)}")

    if len(rx_data_bits) == len(tx_data_bits):
        error_count = np.sum(rx_data_bits != tx_data_bits)
        ber = error_count / len(tx_data_bits)
        log_step("接收端-BER", f"错误比特数：{error_count}，BER：{ber:.6f}")
    else:
        ber = 0.5
        log_step("接收端-BER", f"比特长度不匹配！BER强制设为0.5")

    log_step("仿真结束", f"第{trial_idx+1}次试验完成，BER：{ber:.6f}\n" + "-"*80)
    return ber

# --------------------------
# 主函数（测试高SNR，快速验证效果）
# --------------------------
if __name__ == "__main__":
    snr_range = [10, 13, 16, 19]  # 高SNR点，更容易看到BER下降
    num_trials = 1  # 1次试验即可验证效果
    lte_ber = []

    print(f"\033[1;32m===== 开始 LTE 标准仿真（共{len(snr_range)}个SNR点，每个点{num_trials}次试验） =====\033[0m")
    for snr in snr_range:
        avg_ber = 0.0
        for trial in range(num_trials):
            ber = simulate_lte(snr_db=snr, trial_idx=trial)
            avg_ber += ber / num_trials
        lte_ber.append(avg_ber)
        print(f"\033[1;33mLTE - SNR={snr}dB：平均BER={avg_ber:.6f}\033[0m")

    # 绘制BER曲线（对数坐标，符合通信性能展示习惯）
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_range, lte_ber, marker='o', linewidth=2, markersize=8, label='LTE (16QAM)')
    plt.xlabel('信噪比 SNR (dB)', fontsize=12)
    plt.ylabel('误码率 BER', fontsize=12)
    plt.title('LTE标准通信链路误码率性能（终极修复版）', fontsize=14)
    plt.grid(True, which='both', ls='--', alpha=0.7)
    plt.legend(fontsize=11)
    plt.ylim(1e-6, 0.5)  # BER范围：从随机猜测到几乎无错
    plt.show()