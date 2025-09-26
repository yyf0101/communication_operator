import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, correlate
from scipy.fft import fft, ifft, fftshift, ifftshift

# --------------------------
# 1. LTE核心参数配置（符合标准，适配小频偏估计）
# --------------------------
class LTEParams:
    def __init__(self):
        # 物理层基础参数
        self.n_fft = 1024          # FFT点数（OFDM子载波总数）
        self.cp_length = 144       # 常规循环前缀长度（采样点）
        self.symbol_rate = 15.36e6 # 符号率（1倍采样率，LTE下行标准）
        self.fs = 4 * self.symbol_rate  # 4倍过采样率（61.44MHz，抑制频谱泄漏）
        self.rrc_alpha = 0.22      # RRC滤波器滚降系数（0.22为LTE常用值）
        self.rrc_taps = 65         # RRC滤波器抽头数（奇数，保证线性相位）

        # 调制与数据参数
        self.mod_order = 16        # 16QAM调制（4bit/符号，平衡效率与误码）
        self.bits_per_sym = int(np.log2(self.mod_order))
        self.data_sym_num = 50     # 数据符号数（对应50×4=200bit，便于统计BER）

        # 同步与频偏参数
        self.zc_seq_len_1x = 32    # ZC同步序列长度（1倍采样，适配小频偏估计）
        self.sync_seq_len_4x = self.zc_seq_len_1x * 4  # 4倍过采样同步序列长度
        self.true_cfo = 3000       # 真实载波频偏（3kHz，LTE典型场景）

        # 仿真参数
        self.snr_range = [8, 11, 14, 17, 20]  # 仿真SNR点（覆盖16QAM误码门限到低误码）
        self.num_trials = 3        # 每个SNR重复次数（取平均，降低随机误差）

# 初始化参数实例
lte = LTEParams()

# --------------------------
# 2. 发射端模块（Tx）：从比特到射频前信号
# --------------------------
def generate_zc_sequence(length, root_idx=3):
    """生成Zadoff-Chu序列（LTE PSS核心，自相关性极强）"""
    n = np.arange(length)
    # ZC序列公式：exp(-jπ·root·n(n+1)/length)，能量归一化
    zc_seq = np.exp(-1j * np.pi * root_idx * n * (n + 1) / length)
    return zc_seq / np.sqrt(length)

def generate_16qam_constellation():
    """生成16QAM格雷编码星座图（能量归一化，避免多bit误码）"""
    constellation = np.array([
        3+3j, 3+1j, 1+1j, 1+3j, 1-3j, 1-1j, 3-1j, 3-3j,
        -3+3j, -3+1j, -1+1j, -1+3j, -1-3j, -1-1j, -3-1j, -3-3j
    ])
    return constellation / np.sqrt(10)  # 平均功率=1

def tx_source_gen():
    """信源生成：随机二进制比特流"""
    return np.random.randint(0, 2, lte.data_sym_num * lte.bits_per_sym)

def tx_modulate(tx_bits, constellation):
    """16QAM调制：比特分组→格雷编码→星座点映射"""
    # 格雷编码表（4bit→星座点索引）
    gray_code = [0, 1, 3, 2, 6, 7, 5, 4, 8, 9, 11, 10, 14, 15, 13, 12]
    # 4bit分组
    bit_groups = tx_bits.reshape(-1, lte.bits_per_sym)
    # 分组→格雷码整数（如[0,0,0,0]→0，[0,0,0,1]→1）
    gray_vals = np.dot(bit_groups, 2 ** np.arange(lte.bits_per_sym-1, -1, -1))
    # 格雷码→星座点索引→星座点
    sym_indices = [np.where(gray_code == g)[0][0] for g in gray_vals]
    return constellation[sym_indices]

def tx_resource_map(tx_sym):
    """资源映射：调制符号→频域子载波（排除直流/保护子载波）"""
    freq_sym = np.zeros(lte.n_fft, dtype=complex)
    # 数据子载波索引（正频率：1~50，负频率：n_fft-50~n_fft-1，共轭对称）
    pos_subcarriers = np.arange(1, 1 + lte.data_sym_num)
    neg_subcarriers = lte.n_fft - pos_subcarriers
    # 填充正频率子载波，负频率子载波共轭填充（保证实信号）
    freq_sym[pos_subcarriers] = tx_sym
    freq_sym[neg_subcarriers] = np.conj(tx_sym)
    return freq_sym, pos_subcarriers

def tx_ifft_cp(freq_sym):
    """IFFT+加CP：频域信号→时域OFDM符号（消除ISI）"""
    # IFFT前shift：将负频率移到左侧（符合IFFT输入要求）
    freq_sym_shift = ifftshift(freq_sym)
    # IFFT变换（频域→时域）
    time_sym = ifft(freq_sym_shift, lte.n_fft)
    # 加CP：取时域符号末尾CP长度的采样点，拼接到开头
    cp = time_sym[-lte.cp_length:]
    return np.concatenate([cp, time_sym])

def tx_pulse_shaping(time_signal):
    """脉冲成形：RRC滤波器（4倍过采样，限制带宽）"""
    N = lte.rrc_taps
    alpha = lte.rrc_alpha
    Ts = 1 / lte.symbol_rate  # 符号周期
    Fs = lte.fs              # 采样率
    T_sample = 1 / Fs        # 采样周期

    # 生成RRC滤波器系数
    t = np.arange(-N // 2, N // 2 + 1) * T_sample  # 时间轴
    rrc_coeff = np.zeros_like(t)
    for i, t_val in enumerate(t):
        if t_val == 0:
            # 极限情况：t=0时，避免除以零
            rrc_coeff[i] = (1 - alpha + 4 * alpha / np.pi)
        elif np.abs(t_val) == Ts / (4 * alpha):
            # 特殊点：t=Ts/(4α)，单独计算
            rrc_coeff[i] = alpha / np.sqrt(2) * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) +
                    (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
            )
        else:
            # 通用公式
            numerator = np.sin(np.pi * t_val * (1 - alpha) / Ts) + \
                        4 * alpha * (t_val / Ts) * np.cos(np.pi * t_val * (1 + alpha) / Ts)
            denominator = np.pi * t_val * (1 - (4 * alpha * t_val / Ts) ** 2) / Ts
            rrc_coeff[i] = numerator / denominator
    # 能量归一化
    rrc_coeff /= np.sqrt(np.sum(rrc_coeff ** 2))

    # 4倍过采样：每个采样点后补3个零（插值）
    interp_signal = np.zeros(len(time_signal) * 4, dtype=complex)
    interp_signal[::4] = time_signal
    # 滤波成形（应用RRC滤波器）
    shaped_signal = lfilter(rrc_coeff, 1.0, interp_signal)
    return shaped_signal, rrc_coeff

def tx_add_sync(data_signal, rrc_coeff):
    """添加同步序列：ZC序列（4倍过采样+RRC成形，与数据信号一致）"""
    # 生成1倍采样ZC序列
    zc_seq_1x = generate_zc_sequence(lte.zc_seq_len_1x)
    # 4倍过采样（补零插值）
    zc_interp = np.zeros(len(zc_seq_1x) * 4, dtype=complex)
    zc_interp[::4] = zc_seq_1x
    # RRC成形（与数据信号滤波一致）
    zc_shaped = lfilter(rrc_coeff, 1.0, zc_interp)
    # 截取固定长度（避免过长）
    zc_shaped = zc_shaped[:lte.sync_seq_len_4x]
    # 拼接：同步序列 + 数据信号
    tx_final = np.concatenate([zc_shaped, data_signal])
    return tx_final, zc_seq_1x  # 返回本地ZC序列（用于接收端同步）

# --------------------------
# 3. 信道模块（Channel）：模拟无线传输
# --------------------------
def channel_awgn_cfo(signal, snr_db):
    """添加AWGN噪声+载波频偏（CFO），模拟真实无线信道"""
    # 计算信号功率（仅数据部分，排除同步序列）
    signal_power = np.var(signal)
    # 计算噪声功率（按设定SNR）
    noise_power = signal_power / (10 ** (snr_db / 10))
    # 生成复噪声（实部/虚部分别服从N(0, noise_power/2)）
    noise = np.sqrt(noise_power / 2) * (
            np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    )
    # 加载频偏：相位旋转因子（f=3000Hz）
    n = np.arange(len(signal))
    cfo_factor = np.exp(1j * 2 * np.pi * lte.true_cfo * n / lte.fs)
    # 信道输出：信号+频偏+噪声
    return signal * cfo_factor + noise, noise_power

# --------------------------
# 4. 接收端模块（Rx）：从信道信号到恢复比特
# --------------------------
def rx_matched_filter(channel_signal, rrc_coeff):
    """匹配滤波：与发射端RRC滤波器一致，恢复信号能量+4倍降采样"""
    # 匹配滤波（抑制噪声，提取有用信号）
    filtered_signal = lfilter(rrc_coeff, 1.0, channel_signal)
    # 4倍降采样：恢复1倍采样率（与发射端一致）
    return filtered_signal[::4]

def rx_frame_sync(rx_1x_signal, local_zc_seq_1x):
    """帧同步：基于ZC序列互相关，找到帧起始位置并提取纯净ZC序列"""
    # 互相关：接收信号与本地ZC序列（共轭反转，提高相关性）
    corr = correlate(rx_1x_signal, local_zc_seq_1x[::-1].conj(), mode='same')
    corr_abs = np.abs(corr)
    # 峰值检测：找到相关峰位置（ZC序列中心）
    peak_idx = np.argmax(corr_abs)
    # 计算帧起始位置（ZC序列起始）
    zc_start = peak_idx - len(local_zc_seq_1x) // 2
    # 提取纯净ZC序列（避免越界，补零）
    rx_zc_seq_1x = rx_1x_signal[zc_start : zc_start + lte.zc_seq_len_1x]
    if len(rx_zc_seq_1x) != lte.zc_seq_len_1x:
        rx_zc_seq_1x = np.pad(rx_zc_seq_1x, (0, lte.zc_seq_len_1x - len(rx_zc_seq_1x)))
    # 计算峰均比（验证同步质量，>10为优）
    peak_avg_ratio = np.max(corr_abs) / np.mean(corr_abs)
    return rx_zc_seq_1x, peak_avg_ratio

def rx_freq_offset_est(rx_zc_seq_1x, local_zc_seq_1x):
    """
    修复版频偏估计：基于FFT的高精度小数延迟估计（解决小频偏问题）
    步骤：1. 频域互相关 → 2. 有效峰值筛选 → 3. 小数延迟计算 → 4. 频偏转换
    """
    N_zc = len(rx_zc_seq_1x)
    fft_size = N_zc * 32  # 32倍补零，提高频域分辨率（平衡精度与计算量）

    # 1. 频域互相关（正确公式：接收FFT共轭 × 本地FFT，避免镜像）
    rx_fft = fft(rx_zc_seq_1x, fft_size)
    local_fft = fft(local_zc_seq_1x, fft_size)
    corr_freq = np.conj(rx_fft) * local_fft
    # 转换回时域：得到高分辨率互相关函数
    corr_time = ifft(corr_freq)
    corr_abs = np.abs(corr_time)

    # 2. 有效峰值筛选：排除FFT边界虚假峰（仅保留中心±1/4范围）
    center_idx = fft_size // 2
    valid_range = [center_idx - fft_size//4, center_idx + fft_size//4]
    valid_corr_abs = corr_abs[valid_range[0]:valid_range[1]]
    valid_indices = np.arange(valid_range[0], valid_range[1])

    # 峰锐度判断：排除噪声导致的平缓峰（峰值/次峰值>2为优）
    sorted_vals = np.sort(valid_corr_abs)[::-1]
    peak_sharpness = sorted_vals[0] / sorted_vals[1] if sorted_vals[1] > 1e-6 else 100
    if peak_sharpness < 2:
        return 0.0, 0.0  # 峰锐度不足，返回默认值

    # 3. 小数延迟计算（核心修复：保留[-0.5, 0.5]采样点，符合小频偏物理意义）
    max_val_idx = valid_indices[np.argmax(valid_corr_abs)]
    # 原始延迟 = (FFT索引 / FFT大小) × ZC序列长度（包含整数+小数部分）
    raw_delay = (max_val_idx / fft_size) * N_zc
    # 仅保留小数部分（整数延迟不影响频偏，ZC序列周期性吸收）
    decimal_delay = raw_delay - np.round(raw_delay)

    # 4. 频偏转换（正确公式：f = (小数延迟 × 符号率) / ZC序列长度）
    cfo_est = (decimal_delay * lte.symbol_rate) / N_zc
    # 过滤极端异常值（±5kHz，避免计算错误）
    cfo_est = cfo_est if abs(cfo_est) < 5000 else 0.0

    return cfo_est, decimal_delay

def rx_freq_compensate(rx_data_1x, cfo_est):
    """频偏补偿：消除频偏导致的相位旋转"""
    n = np.arange(len(rx_data_1x))
    # 补偿因子：与频偏相位旋转相反
    comp_factor = np.exp(-1j * 2 * np.pi * cfo_est * n / lte.symbol_rate)
    return rx_data_1x * comp_factor

def rx_remove_cp(rx_time_signal):
    """去CP：移除发射端添加的循环前缀，保留IFFT主体"""
    # 取CP后的1024个采样点（IFFT大小）
    return rx_time_signal[lte.cp_length : lte.cp_length + lte.n_fft]

def rx_fft_resource_extract(rx_no_cp, pos_subcarriers):
    """FFT+资源提取：时域信号→频域，提取数据子载波"""
    # FFT变换（时域→频域）
    rx_fft = fft(rx_no_cp, lte.n_fft)
    # fftshift：转换为[-511, 512]子载波索引（与发射端一致）
    rx_fft_shifted = fftshift(rx_fft)
    # 提取数据子载波（正频率：1~50）
    return rx_fft_shifted[pos_subcarriers]

def rx_demodulate(rx_sym, constellation, noise_power):
    """16QAM软解调：接收符号→LLR→二进制比特"""
    llr_list = []
    for sym in rx_sym:
        # 计算接收符号到各星座点的距离
        distances = np.abs(sym - constellation) ** 2
        # 每个bit的LLR计算（格雷编码）
        for k in range(lte.bits_per_sym):
            # 格雷码第k位为0/1的星座点掩码
            gray_code = [0, 1, 3, 2, 6, 7, 5, 4, 8, 9, 11, 10, 14, 15, 13, 12]
            mask = np.array([(g >> (lte.bits_per_sym-1 - k)) & 1 for g in gray_code])
            # 最小距离（0bit和1bit）
            min_dist_0 = np.min(distances[mask == 0]) if np.any(mask == 0) else 1e6
            min_dist_1 = np.min(distances[mask == 1]) if np.any(mask == 1) else 1e6
            # LLR公式：(d0 - d1)/噪声功率（噪声功率归一化）
            llr = (min_dist_0 - min_dist_1) / noise_power
            llr_list.append(llr)
    # LLR判决：>0为1，否则为0
    return (np.array(llr_list) > 0).astype(int)

# --------------------------
# 5. 完整链路仿真：整合Tx→Channel→Rx
# --------------------------
def lte_full_link(snr_db, trial_idx):
    """单轮LTE完整链路仿真，返回BER"""
    # --------------------------
    # 发射端流程
    # --------------------------
    # 1. 信源生成
    tx_bits = tx_source_gen()
    # 2. 16QAM调制
    constellation = generate_16qam_constellation()
    tx_sym = tx_modulate(tx_bits, constellation)
    # 3. 资源映射
    freq_sym, pos_subcarriers = tx_resource_map(tx_sym)
    # 4. IFFT+加CP
    time_with_cp = tx_ifft_cp(freq_sym)
    # 5. 脉冲成形
    tx_shaped, rrc_coeff = tx_pulse_shaping(time_with_cp)
    # 6. 添加同步序列
    tx_final, local_zc_seq_1x = tx_add_sync(tx_shaped, rrc_coeff)

    # --------------------------
    # 信道流程
    # --------------------------
    rx_channel, noise_power = channel_awgn_cfo(tx_final, snr_db)

    # --------------------------
    # 接收端流程
    # --------------------------
    # 1. 匹配滤波+降采样
    rx_1x = rx_matched_filter(rx_channel, rrc_coeff)
    # 2. 帧同步：提取ZC序列
    rx_zc_seq_1x, peak_avg_ratio = rx_frame_sync(rx_1x, local_zc_seq_1x)
    # 3. 频偏估计
    cfo_est, decimal_delay = rx_freq_offset_est(rx_zc_seq_1x, local_zc_seq_1x)
    # 4. 频偏补偿（数据信号）
    data_start = lte.zc_seq_len_1x + lte.cp_length  # 数据起始位置：同步序列+CP
    rx_data_1x = rx_1x[data_start:]
    rx_data_comp = rx_freq_compensate(rx_data_1x, cfo_est)
    # 5. 去CP
    if len(rx_data_comp) < lte.cp_length + lte.n_fft:
        rx_data_comp = np.pad(rx_data_comp, (0, lte.cp_length + lte.n_fft - len(rx_data_comp)))
    rx_no_cp = rx_remove_cp(rx_data_comp)
    # 6. FFT+资源提取
    rx_extracted_sym = rx_fft_resource_extract(rx_no_cp, pos_subcarriers)
    # 7. 解调：符号→比特
    rx_bits = rx_demodulate(rx_extracted_sym, constellation, noise_power)

    # --------------------------
    # BER计算与日志输出
    # --------------------------
    if len(rx_bits) != len(tx_bits):
        ber = 0.5  # 长度不匹配，返回随机BER
    else:
        ber = np.mean(rx_bits != tx_bits)  # 错误比特率
    # 打印关键信息（便于调试）
    cfo_error = abs(cfo_est - lte.true_cfo)
    print(f"SNR={snr_db:2d}dB | 第{trial_idx+1}轮 | 峰均比={peak_avg_ratio:.2f} | "
          f"真实频偏={lte.true_cfo:4d}Hz | 估计频偏={cfo_est:6.1f}Hz | 误差={cfo_error:5.1f}Hz | "
          f"小数延迟={decimal_delay:6.4f} | BER={ber:.6f}")
    return ber

# --------------------------
# 6. 批量仿真与结果可视化
# --------------------------
def batch_simulation():
    """批量仿真：多SNR点+多轮重复，绘制BER曲线"""
    # 存储各SNR的平均BER
    avg_ber_dict = {snr: 0.0 for snr in lte.snr_range}
    # 打印仿真 header
    print("="*180)
    print(f"LTE完整链路仿真开始（真实频偏={lte.true_cfo}Hz，每个SNR重复{lte.num_trials}轮）")
    print("="*180)

    # 遍历每个SNR点
    for snr in lte.snr_range:
        total_ber = 0.0
        for trial in range(lte.num_trials):
            # 单轮仿真
            ber = lte_full_link(snr, trial)
            total_ber += ber
        # 计算平均BER
        avg_ber_dict[snr] = total_ber / lte.num_trials
        # 打印该SNR的平均结果
        print("="*180)
        print(f"SNR={snr:2d}dB | 平均BER={avg_ber_dict[snr]:.6f}")
        print("="*180)

    # 绘制BER曲线（对数坐标，符合通信性能展示习惯）
    plt.figure(figsize=(10, 6))
    snrs = sorted(lte.snr_range)
    avg_bers = [avg_ber_dict[snr] for snr in snrs]
    # 绘制BER曲线（标记点+实线）
    plt.semilogy(snrs, avg_bers, marker='o', linewidth=2, markersize=8,
                 label=f'LTE 16QAM（真实频偏={lte.true_cfo}Hz）')

    # 图表配置
    plt.xlabel('信噪比 SNR (dB)', fontsize=12)
    plt.ylabel('误码率 BER', fontsize=12)
    plt.title('LTE下行链路误码率性能（修复版FFT频偏估计）', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)  # 网格线（便于读数）
    plt.legend(fontsize=11)  # 图例
    plt.ylim(1e-6, 0.5)  # BER范围：从随机猜测（0.5）到几乎无错（1e-6）
    plt.xlim(lte.snr_range[0]-1, lte.snr_range[-1]+1)  # SNR范围微调
    plt.show()

    return avg_ber_dict

# --------------------------
# 7. 启动仿真（主函数）
# --------------------------
if __name__ == "__main__":
    batch_simulation()