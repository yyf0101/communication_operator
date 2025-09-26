import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 独立窗口显示
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, correlate
from scipy.fft import fft, ifft, fftshift, ifftshift

# --- 中文显示配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --------------------------
# 1. 可复用符号级算子定义（最终修正版）
# --------------------------
class ConfigurableFFT:
    """可配置FFT/IFFT算子，修正OFDM转换逻辑"""
    def __init__(self, n_fft=1024):
        self.n_fft = n_fft
        self.all_data_subcarriers = self._get_all_data_subcarriers()
        self.used_data_subcarriers = None

    def _get_all_data_subcarriers(self):
        """生成正频率数据子载波索引（符合OFDM实信号传输）"""
        # 排除直流(0)和第一个负频率(n_fft-1)
        return np.arange(1, self.n_fft // 2)

    def update_fft_size(self, n_fft):
        self.n_fft = n_fft
        self.all_data_subcarriers = self._get_all_data_subcarriers()
        self.used_data_subcarriers = None

    def freq_map(self, tx_symbols):
        """频域映射：填充正频率子载波，并共轭对称填充负频率"""
        freq_sym = np.zeros(self.n_fft, dtype=complex)
        # 确保符号数量不超过可用子载波
        num_symbols = min(len(tx_symbols), len(self.all_data_subcarriers))
        self.used_data_subcarriers = self.all_data_subcarriers[:num_symbols]

        # 填充正频率子载波
        freq_sym[self.used_data_subcarriers] = tx_symbols[:num_symbols]
        # 共轭对称填充负频率子载波
        freq_sym[self.n_fft - self.used_data_subcarriers] = np.conj(tx_symbols[:num_symbols])
        return freq_sym

    def process(self, x, is_fft=True):
        """修正处理流程：发射端(ifftshift->IFFT)，接收端(FFT->fftshift)"""
        if is_fft:
            # 接收端：时域 -> FFT -> fftshift
            return fftshift(fft(x, self.n_fft))
        else:
            # 发射端：频域 -> ifftshift -> IFFT -> 实部
            return np.real(ifft(ifftshift(x), self.n_fft))

class MultiModeFIR:
    """多模式FIR滤波器，用于脉冲成形和匹配滤波"""
    def __init__(self, taps=64, rolloff=0.22, symbol_rate=15.36e6):
        self.taps = taps
        self.rolloff = rolloff
        self.symbol_rate = symbol_rate
        self.coeffs = self._design_rrc_filter()

    def _design_rrc_filter(self):
        """设计根升余弦滤波器"""
        fs = 4 * self.symbol_rate  # 4倍过采样
        t = np.arange(-self.taps//2, self.taps//2) / fs
        rrc = np.sinc(self.symbol_rate * t) * (1 - 4 * self.rolloff**2 * self.symbol_rate**2 * t**2 + 1e-10)
        rrc /= (1 - 4 * self.rolloff**2 * self.symbol_rate**2 * t**2 + 1e-10)
        rrc *= np.cos(np.pi * self.rolloff * self.symbol_rate * t) / (1 - (2 * self.rolloff * self.symbol_rate * t)**2 + 1e-10)
        return rrc / np.sqrt(np.sum(rrc**2))

    def update_params(self, rolloff, symbol_rate):
        self.rolloff = rolloff
        self.symbol_rate = symbol_rate
        self.coeffs = self._design_rrc_filter()

    def process(self, x):
        return lfilter(self.coeffs, 1.0, x)

class SyncProcessor:
    """同步处理算子，用于帧同步和频偏估计"""
    def __init__(self, sync_seq=None, fs=61.44e6):
        self.sync_seq = sync_seq
        self.fs = fs
        self.freq_offset = 0.0

    def frame_sync(self, rx_signal):
        """帧同步：通过相关找到同步序列位置"""
        corr = correlate(np.abs(rx_signal), np.abs(self.sync_seq), mode='same')
        frame_start = np.argmax(corr) - len(self.sync_seq) // 2
        return np.clip(frame_start, 0, len(rx_signal) - len(self.sync_seq))

    def freq_offset_est(self, rx_sync_seq):
        """频偏估计：基于同步序列前后半段的相位差"""
        N = len(rx_sync_seq)
        half = N // 2
        phase_diff = np.mean(np.angle(rx_sync_seq[half:] * np.conj(rx_sync_seq[:half])))
        self.freq_offset = phase_diff / (2 * np.pi * (half / self.fs))
        return self.freq_offset

    def freq_compensate(self, x):
        """频偏补偿"""
        n = np.arange(len(x))
        return x * np.exp(-1j * 2 * np.pi * self.freq_offset * n / self.fs)

class SoftDemodulator:
    """软解调算子，支持多阶调制"""
    def __init__(self, mod_order=4):
        self.mod_order = mod_order
        self.bits_per_symbol = mod_order // 2
        self.constellation = self._generate_constellation()

    def _generate_constellation(self):
        """生成标准格雷编码星座图"""
        if self.mod_order == 2:  # QPSK
            return np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        elif self.mod_order == 4:  # 16QAM
            return np.array([
                3+3j, 3+1j, 1+3j, 1+1j,
                3-3j, 3-1j, 1-3j, 1-1j,
                -3+3j, -3+1j, -1+3j, -1+1j,
                -3-3j, -3-1j, -1-3j, -1-1j
            ]) / np.sqrt(10)
        else:  # 64QAM
            const = []
            for i in range(8):
                for j in range(8):
                    gray_i = i ^ (i >> 1)
                    gray_j = j ^ (j >> 1)
                    const.append((gray_i - 3.5) + (gray_j - 3.5)*1j)
            return np.array(const) / np.sqrt(42)

    def update_mod_order(self, mod_order):
        self.mod_order = mod_order
        self.bits_per_symbol = mod_order // 2
        self.constellation = self._generate_constellation()

    def process(self, rx_symbols, noise_var=0.1):
        """计算LLR（对数似然比）"""
        llr = []
        for sym in rx_symbols:
            distances = np.abs(sym - self.constellation)**2
            bits = []
            for k in range(self.bits_per_symbol):
                mask = (np.arange(len(self.constellation)) >> (self.bits_per_symbol - 1 - k)) & 1
                d0 = np.min(distances[mask == 0]) if np.any(mask == 0) else 1e6
                d1 = np.min(distances[mask == 1]) if np.any(mask == 1) else 1e6
                bits.append((d0 - d1) / noise_var)
            llr.extend(bits)
        return np.array(llr)

# --------------------------
# 2. 多标准配置控制器
# --------------------------
class StandardConfig:
    def __init__(self, standard='LTE'):
        self.standard = standard
        self.params = self._get_params()

    def _get_params(self):
        if self.standard == 'LTE':
            return {
                'n_fft': 1024,
                'cp_length': 144,
                'fir_rolloff': 0.22,
                'symbol_rate': 15.36e6,
                'fs': 61.44e6,
                'mod_order': 4,  # 16QAM
                'sync_seq': np.array([1,1,-1,-1,1,-1,-1,1,-1,1,1,-1]*16, dtype=complex)
            }
        elif self.standard == 'WiFi6':
            return {
                'n_fft': 2048,
                'cp_length': 320,
                'fir_rolloff': 0.35,
                'symbol_rate': 11.428e6,
                'fs': 45.712e6,
                'mod_order': 4,  # 16QAM
                'sync_seq': np.array([1,-1,1,-1,1,-1,1,-1]*24, dtype=complex)
            }
        elif self.standard == '5G':
            return {
                'n_fft': 4096,
                'cp_length': 512,
                'fir_rolloff': 0.22,
                'symbol_rate': 19.2e6,
                'fs': 76.8e6,
                'mod_order': 6,  # 64QAM
                'sync_seq': np.array([1,1,1,-1,-1,1,-1,1,-1,-1,-1,1]*20, dtype=complex)
            }

    def configure_operators(self, fft_op, fir_op, sync_op, demod_op):
        fft_op.update_fft_size(self.params['n_fft'])
        fir_op.update_params(self.params['fir_rolloff'], self.params['symbol_rate'])
        sync_op.sync_seq = self.params['sync_seq']
        sync_op.fs = self.params['fs']
        demod_op.update_mod_order(self.params['mod_order'])
        return self.params['cp_length'], self.params['fs']

# --------------------------
# 3. 完整通信链路仿真（最终修正版）
# --------------------------
def simulate_communication(standard='LTE', snr_db=10):
    # 1. 初始化算子
    fft_op = ConfigurableFFT()
    fir_op = MultiModeFIR()
    sync_op = SyncProcessor()
    demod_op = SoftDemodulator()

    # 2. 配置标准参数
    config = StandardConfig(standard)
    cp_length, fs = config.configure_operators(fft_op, fir_op, sync_op, demod_op)
    n_fft = fft_op.n_fft
    bits_per_symbol = demod_op.bits_per_symbol

    # 3. 生成发送数据
    num_data_sym = 50  # 发送50个数据符号
    tx_data_bits = np.random.randint(0, 2, num_data_sym * bits_per_symbol)

    # 3.1 调制
    bit_matrix = tx_data_bits.reshape(-1, bits_per_symbol)
    weights = 2 ** np.arange(bits_per_symbol - 1, -1, -1)
    sym_indices = np.dot(bit_matrix, weights)
    tx_data_sym = demod_op.constellation[sym_indices]

    # 4. 发射端处理
    # 4.1 频域映射
    freq_sym = fft_op.freq_map(tx_data_sym)
    # 4.2 IFFT
    tx_ifft = fft_op.process(freq_sym, is_fft=False)
    # 4.3 添加CP
    tx_cp = np.concatenate([tx_ifft[-cp_length:], tx_ifft])
    # 4.4 脉冲成形
    tx_signal = fir_op.process(tx_cp)

    # 5. 信道传输（简化为AWGN+小频偏）
    snr = 10 **(snr_db / 10)
    signal_power = np.var(tx_signal)
    noise_power = signal_power / snr
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(tx_signal)) + 1j * np.random.randn(len(tx_signal)))
    true_freq_offset = np.random.randint(-5, 6)  # 小频偏
    n = np.arange(len(tx_signal))
    rx_signal = tx_signal * np.exp(1j * 2 * np.pi * true_freq_offset * n / fs) + noise

    # 6. 接收端处理
    # 6.1 匹配滤波
    rx_filtered = fir_op.process(rx_signal)
    # 6.2 帧同步
    frame_start = sync_op.frame_sync(rx_filtered)
    rx_sync_seq = rx_filtered[frame_start:frame_start+len(sync_op.sync_seq)]
    # 6.3 频偏估计与补偿（在频域符号上进行补偿）
    sync_op.freq_offset_est(rx_sync_seq)
    # 6.4 提取数据并去CP
    rx_data = rx_filtered[frame_start+len(sync_op.sync_seq):]
    if len(rx_data) < cp_length + n_fft:
        rx_data = np.pad(rx_data, (0, cp_length + n_fft - len(rx_data)))
    rx_no_cp = rx_data[cp_length:cp_length+n_fft]
    # 6.5 FFT
    rx_fft = fft_op.process(rx_no_cp, is_fft=True)
    # 6.6 提取数据子载波
    rx_data_sym_freq = rx_fft[fft_op.used_data_subcarriers]
    # 6.7 频偏补偿（在频域进行）
    rx_data_sym_compensated = sync_op.freq_compensate(rx_data_sym_freq)
    # 6.8 简化均衡（假设理想信道）
    rx_equalized_sym = rx_data_sym_compensated
    # 6.9 软解调
    rx_llr = demod_op.process(rx_equalized_sym, noise_var=noise_power)
    rx_data_bits = (rx_llr > 0).astype(int)

    # 7. 计算误码率
    if len(rx_data_bits) != len(tx_data_bits):
        return 0.5  # 长度不匹配，返回随机BER
    ber = np.mean(rx_data_bits != tx_data_bits)
    return ber

# --------------------------
# 4. 仿真验证与结果可视化
# --------------------------
if __name__ == "__main__":
    standards = ['LTE', 'WiFi6', '5G']
    snr_range = np.arange(0, 21, 3)
    results = {std: [] for std in standards}
    num_trials = 5  # 每个SNR点重复5次取平均

    for std in standards:
        print(f"正在仿真 {std} 标准...")
        for snr in snr_range:
            avg_ber = 0.0
            for _ in range(num_trials):
                ber = simulate_communication(standard=std, snr_db=snr)
                avg_ber += ber / num_trials
            results[std].append(avg_ber)
            print(f"  SNR={snr}dB, 平均BER={avg_ber:.6f}")

    # 绘制误码率曲线
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^']
    for i, std in enumerate(standards):
        plt.semilogy(snr_range, results[std], marker=markers[i], label=std, linewidth=2)

    plt.xlabel('信噪比 SNR (dB)', fontsize=12)
    plt.ylabel('误码率 BER', fontsize=12)
    plt.title('多标准通信系统误码率性能（复用符号级算子）', fontsize=14)
    plt.grid(True, which='both', ls='--', alpha=0.7)
    plt.legend(fontsize=11)
    # plt.ylim(1e-6, 0.5)
    plt.show()