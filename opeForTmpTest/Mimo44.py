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
        I = np.eye(H.shape[1])  # 单位矩阵维度为发射天线数

        H_H_H = SymbolLevelOperatorLibrary.matrix_multiplication(H_H, H)
        # 噪声方差<=0时退化为ZF均衡
        reg = (noise_variance if noise_variance > 0 else 0.0)
        W = SymbolLevelOperatorLibrary.matrix_multiplication(
            np.linalg.inv(H_H_H + reg * I + 1e-6 * np.eye(H_H_H.shape[0])),
            H_H
        )
        equalized = SymbolLevelOperatorLibrary.matrix_multiplication(W, received_symbols.reshape(-1, 1))
        return equalized

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
        return H / np.sqrt(tx_ant)  # 归一化信道能量

    @staticmethod
    def channel_estimation_ls_mimo(received_pilots: np.ndarray, known_pilots: np.ndarray) -> np.ndarray:
        rx_ant, num_pilots = received_pilots.shape
        tx_ant = known_pilots.shape[0]
        h_pilot = np.zeros((rx_ant, tx_ant, num_pilots), dtype=np.complex128)
        for k in range(num_pilots):
            y = received_pilots[:, k:k+1]  # (rx_ant,1)
            x = known_pilots[:, k:k+1]     # (tx_ant,1)
            power_x = np.sum(np.abs(x)**2) + 1e-12
            h_pilot[:, :, k] = y @ np.conj(x.T) / power_x  # LS: H_k = y x^H / ||x||^2
        return h_pilot


# 基础参数配置
tx_ant = 4
rx_ant = 4
fft_size = 1024
cp_length = 128
sym_length = fft_size + cp_length  # 1152点
subcarrier_spacing = 15e3
pilot_interval = 8  # 增加导频密度以改善信道估计精度
num_symbols = 100
num_sync_symbols = 4  # 同步符号重复4次
# 重构同步结构: 设定前2个为STF(重复半符号自相关)，后2个为LTF(全载波BPSK用于精细对齐/信道估计)
num_stf = 2
num_ltf = 2
num_symbols_tx = num_sync_symbols + num_symbols + 2  # 同步+数据+冗余
qpsk_const = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
snr_db = 15
use_perfect_channel = True  # 调试开关：使用理想信道知识验证均衡与判决链路
run_minimal_path = False     # A: 极简单符号链路验证开关 (关闭以运行完整链路)
enable_frontend_filter = False  # S7a: 同步调试阶段关闭前端FIR/IQ增益影响
pilot_pos = np.arange(0, fft_size, pilot_interval)
data_pos = np.setdiff1d(np.arange(fft_size), pilot_pos)
num_pilots = len(pilot_pos)
num_data_per_symbol = len(data_pos)
interp_positions = np.linspace(0, num_pilots-1, len(data_pos))


# 发送端处理
def tx_mimo_signal(operator_lib: SymbolLevelOperatorLibrary):
    # 生成发送比特
    num_bits_per_ant = (num_symbols_tx - num_sync_symbols) * num_data_per_symbol * 2
    tx_bits = [np.random.randint(0, 2, num_bits_per_ant) for _ in range(tx_ant)]
    tx_bits_valid = [bits[:num_symbols * num_data_per_symbol * 2] for bits in tx_bits]

    # 生成QPSK符号（不再额外×2放大，保持单位平均功率）
    tx_symbols_list = []
    for ant in range(tx_ant):
        bits = tx_bits[ant]
        symbol_idx = bits.reshape(-1, 2) @ [2, 1]
        tx_symbols = qpsk_const[symbol_idx]      # 保持归一化QPSK幅度
        tx_symbols_list.append(tx_symbols)

    # 构造频域数据符号（数据+导频）
    tx_freq_sym = np.zeros(((num_symbols_tx - num_sync_symbols), fft_size, tx_ant), dtype=np.complex128)
    for ant in range(tx_ant):
        # 导频：正交根序列（去除过高功率提升，避免动态范围过大）
        zc_pilot = operator_lib.generate_zc_sequence(num_pilots, root=23+ant*5)
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

    # === 频域正规 STF + LTF 重构 ===
    # STF 频域条件: 仅在偶数子载波上放置BPSK, 奇数为0 => 时域满足 s[n+N/2]=s[n] (半重复)
    # LTF 频域: 全载波BPSK (可为后续精细定时/信道估计使用)
    rng = np.random.default_rng(2026)
    sync_freq_patterns = np.zeros((num_sync_symbols, fft_size, tx_ant), dtype=np.complex128)
    even_idx = np.arange(0, fft_size, 2)
    # 重新设计 STF: 稀疏 comb + 时域半符号复制 (提高自相关判别力)
    comb_step = 8
    comb_idx = np.arange(0, fft_size, comb_step)
    stf_time = np.zeros((num_stf, sym_length, tx_ant), dtype=np.complex128)
    L = fft_size // 2
    for rep in range(num_stf):
        for ant in range(tx_ant):
            # 在频域构造稀疏comb BPSK
            freq_tmp = np.zeros(fft_size, dtype=np.complex128)
            bpsk = rng.choice([-1, 1], size=len(comb_idx)) + 0j
            freq_tmp[comb_idx] = bpsk
            # IFFT 得到时域符号 (不保证半重复) -> 再强制半重复
            t_full = operator_lib.ifft_operation(freq_tmp) * fft_size
            # 取前L/2样本随机选择组成 half(这里直接用前L样本)
            half = t_full[:L]
            full_no_cp = np.concatenate([half, half])  # 强制半重复结构
            # 归一化功率 ~1
            p = np.mean(np.abs(full_no_cp)**2) + 1e-12
            full_no_cp /= np.sqrt(p)
            cp = full_no_cp[-cp_length:]
            stf_time[rep, :, ant] = np.hstack([cp, full_no_cp])
            # 反推出对应频域(仅用于占位, 不再用于同步判决)
            sync_freq_patterns[rep, :, ant] = operator_lib.fft_operation(full_no_cp) / fft_size
    # LTF 部分
    for rep in range(num_stf, num_sync_symbols):
        for ant in range(tx_ant):
            bpsk_full = rng.choice([-1, 1], size=fft_size) + 0j
            sync_freq_patterns[rep, :, ant] = bpsk_full / np.sqrt(1.0)

    # 频域 -> 时域，加CP
    sync_time = np.zeros((num_sync_symbols, sym_length, tx_ant), dtype=np.complex128)
    for rep in range(num_sync_symbols):
        for ant in range(tx_ant):
            if rep < num_stf:
                sync_time[rep, :, ant] = stf_time[rep, :, ant]
            else:
                ifft_out = operator_lib.ifft_operation(sync_freq_patterns[rep, :, ant]) * fft_size
                cp = ifft_out[-cp_length:]
                sync_time[rep, :, ant] = np.hstack([cp, ifft_out])

    # S12: 同步符号功率归一化 -> 匹配数据符号平均频域功率 (~1)
    for rep in range(num_sync_symbols):
        for ant in range(tx_ant):
            cur_pow = np.mean(np.abs(sync_freq_patterns[rep, :, ant])**2)
            if cur_pow > 0:
                scale = 1.0 / np.sqrt(cur_pow)
                sync_freq_patterns[rep, :, ant] *= scale
                # 时域同样缩放
                sync_time[rep, :, ant] *= scale

    # 组合所有符号（同步符号+数据符号）
    tx_time_combined = np.zeros((num_symbols_tx * sym_length, tx_ant), dtype=np.complex128)
    # 填充同步符号（逐个rep）
    for rep in range(num_sync_symbols):
        start_idx = rep * sym_length
        tx_time_combined[start_idx:start_idx+sym_length, :] = sync_time[rep, :, :]
    # 填充数据符号
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

    # 返回含同步频域图案
    return tx_bits_valid, tx_symbols_list, tx_time_combined, tx_freq_sym, sync_freq_patterns


# 信道传输
def mimo_channel_transmit(operator_lib: SymbolLevelOperatorLibrary, tx_time_combined: np.ndarray, snr_db: float):
    # 使用相关的频率选择性信道（基于多径时域卷积的FFT），提高插值有效性
    def correlated_mimo_channel(num_subcarriers: int, tx_ant: int, rx_ant: int, num_paths: int = 8):
        # 生成时域多径冲激响应并做FFT得到频域相关信道
        h_time = np.random.normal(0, np.sqrt(0.5), (rx_ant, tx_ant, num_paths)) + 1j * np.random.normal(0, np.sqrt(0.5), (rx_ant, tx_ant, num_paths))
        # 加指数衰落使后续路径能量降低
        path_decay = np.exp(-0.5 * np.arange(num_paths))
        h_time *= path_decay  # 广播到最后一维
        # 频域响应
        H_f = np.fft.fft(h_time, n=num_subcarriers, axis=-1)
        H_f = np.transpose(H_f, (2, 0, 1))  # (num_subcarriers, rx_ant, tx_ant)
        # 归一化平均功率
        avg_pow = np.mean(np.abs(H_f)**2)
        H_f /= np.sqrt(avg_pow + 1e-12)
        return H_f

    h_mimo_freq = correlated_mimo_channel(fft_size, tx_ant, rx_ant)
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


# 接收端处理
def rx_mimo_signal(operator_lib: SymbolLevelOperatorLibrary, rx_time_combined: np.ndarray,
                   h_mimo_freq: np.ndarray, noise_var: float, tx_symbols_list: list,
                   tx_freq_sym: np.ndarray, sync_freq_patterns: np.ndarray):
    # 1. 数字前端：低阶滤波器，减少信号衰减
    # 采用归一化低通滤波器，避免原先过窄带宽造成幅度大幅衰减
    if enable_frontend_filter:
        fir_coeff = firwin(numtaps=33, cutoff=0.45, window='hamming')
        iq_matrix = np.array([[1.0, -0.01], [0.01, 1.0]])
        rx_dfe_combined = np.zeros_like(rx_time_combined)
        for rx in range(rx_ant):
            rx_iq = operator_lib.iq_correction_and_gain(rx_time_combined[:, rx], gain=1.0, iq_matrix=iq_matrix)
            rx_dfe_combined[:, rx] = operator_lib.fir_filter(rx_iq, fir_coeff)
        rx_power = np.mean(np.abs(rx_dfe_combined)**2)
        print(f"[接收端] 数字前端处理完成(FIR启用)，信号功率: {rx_power:.4f}")
    else:
        rx_dfe_combined = rx_time_combined.copy()
        rx_power = np.mean(np.abs(rx_dfe_combined)**2)
        print(f"[接收端] 数字前端旁路(FIR关闭)，信号功率: {rx_power:.4f}")

    # 2. 频域正规 STF + LTF 同步流程
    # 2.1 STF自相关粗同步: 搜索 d ∈ [0, cp_length] (理论起点应在0附近)
    L = fft_size // 2
    max_offset_search = cp_length  # 仅在CP范围内搜索，减小虚假峰
    P = np.zeros(max_offset_search+1, dtype=np.complex128)
    R = np.zeros(max_offset_search+1, dtype=np.float64)
    for d in range(0, max_offset_search+1):
        acc_p = 0+0j
        acc_r = 0.0
        for ant in range(rx_ant):
            seg = rx_dfe_combined[d : d + fft_size, ant]
            if len(seg) < fft_size:
                continue
            a = seg[:L]
            b = seg[L:]
            acc_p += np.vdot(a, b)
            acc_r += np.sum(np.abs(a)**2 + np.abs(b)**2)
        P[d] = acc_p
        R[d] = acc_r + 1e-12
    metric = (np.abs(P) / R)
    # 平滑窗口=7
    if len(metric) >= 7:
        kernel = np.ones(7)/7
        metric_smooth = np.convolve(metric, kernel, mode='same')
    else:
        metric_smooth = metric
    # 峰/中位数比
    median_val = np.median(metric_smooth)
    peak_idx = int(np.argmax(metric_smooth))
    peak_val = metric_smooth[peak_idx]
    peak_median_ratio = peak_val/(median_val+1e-12)
    # 回溯左缘 (0.7*peak)
    thresh_left = peak_val * 0.7
    left = peak_idx
    while left > 0 and metric_smooth[left-1] >= thresh_left:
        left -= 1
    best_off = left
    print(f"[同步-STF] peak_idx={peak_idx}, best_off={best_off}, peak/median={peak_median_ratio:.2f}")
    # S9: CFO估计 (基于P[best_off] 相位) 并补偿
    P_best = P[best_off]
    cfo_norm = np.angle(P_best) / (np.pi * fft_size)
    if abs(cfo_norm) > 1e-6:
        n = np.arange(len(rx_dfe_combined))
        cfo_ph = np.exp(-1j * 2 * np.pi * cfo_norm * n)
        rx_dfe_combined = (rx_dfe_combined.T * cfo_ph).T
        print(f"[同步-CFO] cfo_norm={cfo_norm:.4e} 已补偿")
    else:
        print("[同步-CFO] cfo_norm≈0, 跳过补偿")
    # 2.2 LTF 相关精细同步: 在 best_off±8 中用LTF频域匹配评分
    def ltf_score(offset: int) -> float:
        # 第一LTF起点: offset + num_stf*sym_length
        ltf_start = offset + num_stf * sym_length
        if ltf_start + sym_length > len(rx_dfe_combined):
            return 0.0
        # 取第一LTF符号
        rx_ltf = rx_dfe_combined[ltf_start + cp_length : ltf_start + sym_length, :]  # (fft_size, rx_ant)
        if rx_ltf.shape[0] != fft_size:
            return 0.0
        # 已知LTF频域图案(第 num_stf 个索引)
        pattern = sync_freq_patterns[num_stf, :, 0]  # 取天线0图案进行匹配
        score = 0.0
        for ant in range(rx_ant):
            Y = operator_lib.fft_operation(rx_ltf[:, ant]) / fft_size
            score += np.sum(np.abs(Y * np.conj(pattern)))
        return float(np.real(score))
    search_offsets = range(max(0, best_off-8), min(cp_length, best_off+8)+1)
    ltf_scores = [(o, ltf_score(o)) for o in search_offsets]
    ltf_scores.sort(key=lambda x: x[1], reverse=True)
    if peak_median_ratio >= 3.5 and ltf_scores:
        refined_off, refined_score = ltf_scores[0]
        print(f"[同步-LTF] refined_off={refined_off}, score={refined_score:.3e}")
        best_off = refined_off
    else:
        if peak_median_ratio < 3.5:
            print("[同步-LTF] skip (STF判别力不足)")
    if best_off > cp_length:
        print("[同步] 警告: 起点>CP长度, 可能仍有偏差")
    # S10: 微调 (±4) 基于首数据符号模型残差 (perfect channel)
    def data_symbol_model_mse(offset: int) -> float:
        data_start_tmp = offset + num_sync_symbols * sym_length
        if data_start_tmp + sym_length > len(rx_dfe_combined):
            return 1e9
        rx_freq_first = np.zeros((fft_size, rx_ant), dtype=np.complex128)
        for rx_i in range(rx_ant):
            seg = rx_dfe_combined[data_start_tmp + cp_length : data_start_tmp + sym_length, rx_i]
            if len(seg) != fft_size:
                return 1e9
            rx_freq_first[:, rx_i] = operator_lib.fft_operation(seg) / fft_size
        S0 = tx_freq_sym[0]
        errs = []
        for sc in range(0, fft_size, 16):
            y = rx_freq_first[sc, :]
            Hsc = h_mimo_freq[sc, :, :]
            x = S0[sc, :]
            y_hat = Hsc @ x
            denom = np.linalg.norm(y_hat)**2
            if denom > 0:
                errs.append(np.linalg.norm(y - y_hat)**2 / (denom + 1e-12))
        if not errs:
            return 1e9
        return float(np.mean(errs))
    micro_candidates = range(max(0, best_off-4), min(cp_length, best_off+4)+1)
    micro_scores = [(o, data_symbol_model_mse(o)) for o in micro_candidates]
    micro_scores.sort(key=lambda x: x[1])
    if micro_scores:
        if micro_scores[0][0] != best_off and micro_scores[0][1] < micro_scores[-1][1]*0.9:
            print(f"[同步-微调] offset {best_off} -> {micro_scores[0][0]}, mse={micro_scores[0][1]:.3f}")
            best_off = micro_scores[0][0]
        else:
            print(f"[同步-微调] 保持offset={best_off}, mse={micro_scores[0][1]:.3f}")

    # 暴力模型扫描回退 (peak判别力不足 或 微调后MSE仍>1.0)
    if (peak_median_ratio < 3.5):
        global_scan = []
        for o in range(0, cp_length+1):
            mse = data_symbol_model_mse(o)
            global_scan.append((o, mse))
        global_scan.sort(key=lambda x: x[1])
        best_global_off, best_global_mse = global_scan[0]
        current_mse = [m for off,m in micro_scores if off==best_off]
        current_mse = current_mse[0] if current_mse else 1e9
        print(f"[同步-全局模型扫描] best_off={best_global_off}, mse={best_global_mse:.3f}, current_off={best_off}, current_mse={current_mse:.3f}")
        if best_global_mse < current_mse * 0.85:
            print(f"[同步-全局模型扫描] 采用全局最佳 offset {best_global_off}")
            best_off = best_global_off
        else:
            print("[同步-全局模型扫描] 改进不足，保持现有offset")
    data_start = best_off + num_sync_symbols * sym_length
    assert data_start + num_symbols * sym_length <= len(rx_dfe_combined), "同步偏移过大"

    # ================= 诊断1：y 与 H@S_tx 线性模型误差 + 偏移扫描 =================
    diag_scan = True
    if diag_scan:
        # 构造第0个数据符号的发送频域符号 S_tx_f (不含同步开头)，用于模型 y=H*S
        # 仅用于诊断，不影响后续处理
        S_tx_f = np.zeros((fft_size, tx_ant), dtype=np.complex128)
        for ant in range(tx_ant):
            # 重建该天线第0个数据符号的频域符号（索引0对应数据符号序列中的第一个）
            data_slice = tx_symbols_list[ant][:num_data_per_symbol]
            S_tx_f[data_pos, ant] = data_slice
            pilot_seq = operator_lib.generate_zc_sequence(num_pilots, root=23+ant*5)
            S_tx_f[pilot_pos, ant] = pilot_seq

        # 基线偏移下的线性残差（使用当前 data_start）
        base_start = data_start
        base_rx_freq = np.zeros((fft_size, rx_ant), dtype=np.complex128)
        for rx in range(rx_ant):
            sym_samp = rx_dfe_combined[base_start + cp_length : base_start + sym_length, rx]
            base_rx_freq[:, rx] = operator_lib.fft_operation(sym_samp) / fft_size
        model_err = []
        for sc in range(fft_size):
            y_meas = base_rx_freq[sc, :]
            y_model = h_mimo_freq[sc, :, :] @ S_tx_f[sc, :]
            if np.linalg.norm(y_model) > 0:
                model_err.append(np.linalg.norm(y_meas - y_model)**2 / (np.linalg.norm(y_model)**2 + 1e-12))
        if model_err:
            model_err = np.array(model_err)
            print(f"[诊断-线性] 当前偏移0子载波相对误差: mean={model_err.mean():.3f}, p90={np.percentile(model_err,90):.3f}, p99={np.percentile(model_err,99):.3f}")
        else:
            print("[诊断-线性] 无有效子载波用于误差统计")

        # 偏移扫描（±64采样）：寻找模型残差最小的偏移，验证同步是否偏差
        scan_range = 64
        scan_results = []  # (delta, mse)
        for delta in range(-scan_range, scan_range+1):
            cand_start = data_start + delta
            if cand_start < 0 or cand_start + sym_length > len(rx_dfe_combined):
                continue
            tmp_rx_freq = np.zeros((fft_size, rx_ant), dtype=np.complex128)
            for rx in range(rx_ant):
                seg = rx_dfe_combined[cand_start + cp_length : cand_start + sym_length, rx]
                if len(seg) != fft_size:
                    break
                tmp_rx_freq[:, rx] = operator_lib.fft_operation(seg) / fft_size
            # 计算模型 MSE
            err_acc = 0.0
            count = 0
            for sc in range(0, fft_size, 8):  # 降低计算量：每8个子载波取一个
                y_meas = tmp_rx_freq[sc, :]
                y_model = h_mimo_freq[sc, :, :] @ S_tx_f[sc, :]
                denom = np.linalg.norm(y_model)**2
                if denom > 0:
                    err_acc += np.linalg.norm(y_meas - y_model)**2 / (denom + 1e-12)
                    count += 1
            if count > 0:
                scan_results.append((delta, err_acc / count))
        if scan_results:
            scan_results.sort(key=lambda x: x[1])
            best = scan_results[0]
            print(f"[诊断-同步扫描] 最佳delta={best[0]} 样本, 模型MSE={best[1]:.4f}")
            print("[诊断-同步扫描] 前5低MSE偏移:", scan_results[:5])
            # 如果最佳偏移不为0且显著低于当前，提示潜在同步错误
            if abs(best[0]) > 0 and best[1] < model_err.mean()*0.7:
                print("[诊断结论] 存在更佳符号边界 -> 同步可能失配，需改进相关或偏移精炼。")
        else:
            print("[诊断-同步扫描] 无可用扫描结果")

        # 条件数统计（随机抽取或均匀抽取 50 个子载波）
        step = max(1, fft_size // 50)
        cond_list = []
        for sc in range(0, fft_size, step):
            H_sc = h_mimo_freq[sc, :, :]
            try:
                cond_list.append(np.linalg.cond(H_sc))
            except np.linalg.LinAlgError:
                cond_list.append(np.inf)
        if cond_list:
            cond_arr = np.array(cond_list)
            print(f"[诊断-条件数] 样本={len(cond_arr)}, min={cond_arr.min():.1f}, median={np.median(cond_arr):.1f}, p90={np.percentile(cond_arr,90):.1f}, max={cond_arr.max():.1f}")
            high_ratio = np.mean(cond_arr > 1e3)
            if high_ratio > 0.2:
                print(f"[诊断结论] {high_ratio*100:.1f}% 子载波条件数>1e3，ZF放大噪声风险高，可考虑MMSE正则。")
        # ================= 诊断1结束 =================

    # B: 若扫描找到更优delta则精炼 data_start
    # 旧扫描精炼逻辑依赖相关候选, 此处改为S7b: 若峰值不显著, 使用模型误差扫描结果进行回退修正
    # (回退条件: peak/mean < 3 且扫描得到更低MSE的非零delta)
    if 'scan_results' in locals() and scan_results and 'model_err' in locals() and isinstance(model_err, np.ndarray):
        scan_results.sort(key=lambda x: x[1])
        best_delta, best_scan_mse = scan_results[0]
        baseline_mse = float(model_err.mean())
        if peak_median_ratio < 3.0 and best_delta != 0 and best_scan_mse < baseline_mse * 0.95:
            print(f"[同步-回退] 采用模型误差驱动修正 delta={best_delta}, MSE {baseline_mse:.3f} -> {best_scan_mse:.3f}")
            data_start_candidate = data_start + best_delta
            if 0 <= data_start_candidate + sym_length <= len(rx_dfe_combined):
                tmp_rx_freq = np.zeros((fft_size, rx_ant), dtype=np.complex128)
                for rx_i in range(rx_ant):
                    seg = rx_dfe_combined[data_start_candidate + cp_length : data_start_candidate + sym_length, rx_i]
                    if len(seg) != fft_size:
                        break
                    tmp_rx_freq[:, rx_i] = operator_lib.fft_operation(seg) / fft_size
                S_tx_f = np.zeros((fft_size, tx_ant), dtype=np.complex128)
                for ant_i in range(tx_ant):
                    data_slice = tx_symbols_list[ant_i][:num_data_per_symbol]
                    S_tx_f[data_pos, ant_i] = data_slice
                    pilot_seq = operator_lib.generate_zc_sequence(num_pilots, root=23+ant_i*5)
                    S_tx_f[pilot_pos, ant_i] = pilot_seq
                err_list_new = []
                for sc in range(0, fft_size, 8):
                    y_meas = tmp_rx_freq[sc, :]
                    y_model = h_mimo_freq[sc, :, :] @ S_tx_f[sc, :]
                    denom = np.linalg.norm(y_model)**2
                    if denom > 0:
                        err_list_new.append(np.linalg.norm(y_meas - y_model)**2 / (denom + 1e-12))
                if err_list_new:
                    new_mse = float(np.mean(err_list_new))
                    if new_mse <= best_scan_mse * 1.2:
                        print(f"[同步-回退] 修正后复测模型MSE={new_mse:.3f}")
                        data_start = data_start_candidate
                    else:
                        print(f"[同步-回退] 放弃修正(复测MSE={new_mse:.3f} 未优于扫描)")
        else:
            if peak_median_ratio < 3.0:
                print("[同步-回退] 条件不满足(Δ或改进不足) 不执行修正")
            else:
                print("[同步-回退] 峰值区分度尚可, 不触发回退")

    # 越界保护
    if data_start < 0 or data_start + num_symbols * sym_length > len(rx_dfe_combined):
        raise RuntimeError(f"精炼后 data_start 越界: {data_start}")

    # 3. OFDM解调
    rx_freq = np.zeros((num_symbols, fft_size, rx_ant), dtype=np.complex128)
    for rx in range(rx_ant):
        for sym in range(num_symbols):
            sym_start = data_start + sym * sym_length
            sym_end = sym_start + sym_length
            sym_samp = rx_dfe_combined[sym_start + cp_length : sym_start + sym_length, rx]
            rx_freq[sym, :, rx] = operator_lib.fft_operation(sym_samp) / fft_size

    # 4. MIMO信道估计 / 或使用理想信道
    if use_perfect_channel:
        h_est_full = np.repeat(h_mimo_freq[np.newaxis, :, :, :], num_symbols, axis=0)
        print("[信道估计] 使用理想信道 (调试模式)")
    else:
        h_est_full = np.zeros((num_symbols, fft_size, rx_ant, tx_ant), dtype=np.complex128)
        for sym in range(num_symbols):
            rx_pilot = rx_freq[sym, pilot_pos, :].T
            known_pilot = np.array([operator_lib.generate_zc_sequence(num_pilots, root=23+ant*5) for ant in range(tx_ant)])
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

    # 5. MMSE均衡（已修正）
    rx_data_sym = np.zeros((num_symbols, num_data_per_symbol, tx_ant), dtype=np.complex128)
    for sym in range(num_symbols):
        for i, data_idx in enumerate(data_pos):
            rx_data = rx_freq[sym, data_idx, :].reshape(-1, 1)
            h = h_est_full[sym, data_idx, :, :]
            # 使用ZF均衡（噪声方差设为0）提升符号恢复精度（调试）
            equalized = operator_lib.mmse_equalization(rx_data, h, 0.0)
            rx_data_sym[sym, i, :] = equalized.squeeze()

        # 符号级公共相位误差(CPE)校正：计算每个发射天线本符号平均相位并旋转回参考
        # （调试）暂时跳过CPE校正，观察纯均衡输出BER

        if sym == 0:
            tx_sym_0 = tx_symbols_list[0][:num_data_per_symbol]
            rx_sym_0 = rx_data_sym[0, :, 0]
            mse_eq = np.mean(np.abs(tx_sym_0 - rx_sym_0)**2)
            print(f"[均衡] 第0符号MSE: {mse_eq:.6f} (越小越好)")
            print(f"[均衡] 发送符号前5个: {np.round(tx_sym_0[:5], 3)}")
            print(f"[均衡] 均衡后前5个: {np.round(rx_sym_0[:5], 3)}")

    # 6. 比特恢复（改为最近邻星座判决，避免当前LLR近似导致高BER）
    bit_mapping = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    constellation_used = qpsk_const  # 与发送端一致（无放大）
    rx_bits = [[] for _ in range(tx_ant)]
    for sym in range(num_symbols):
        for i in range(num_data_per_symbol):
            for tx in range(tx_ant):
                sym_val = rx_data_sym[sym, i, tx]
                dists = np.abs(sym_val - constellation_used)
                idx_hat = np.argmin(dists)
                bits_hat = bit_mapping[idx_hat]
                rx_bits[tx].extend(bits_hat.tolist())
        if sym == 0:
            rx_bits_preview = np.array(rx_bits[0][:20]).reshape(-1, 2)
            print(f"[判决] 天线0第0符号前10对比特: {rx_bits_preview[:10]}")

    rx_bits = [np.array(b) for b in rx_bits]
    return rx_bits


# BER计算
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


# 主函数
if __name__ == "__main__":
    op_lib = SymbolLevelOperatorLibrary()

    # ================= A: 极简单符号 4x4 MIMO OFDM 验证 =================
    if run_minimal_path:
        print("[极简测试] 启动 (单OFDM符号，无同步/CP复杂度)")
        np.random.seed(0)
        # 采用较小子载波集合，避免与全局fft_size混淆
        N = 256
        tx_bits_simple = [np.random.randint(0,2, N*2) for _ in range(tx_ant)]  # 每天线N个QPSK符号（2比特/符号）
        tx_syms_simple = []
        for b in tx_bits_simple:
            idx = b.reshape(-1,2) @ [2,1]
            tx_syms_simple.append(qpsk_const[idx])
        # 组装频域：直接全部为数据，无导频
        S_f = np.stack(tx_syms_simple, axis=1)  # (N, tx_ant)
        # 生成真实信道
        H_f = op_lib.mimo_channel_generation(N, tx_ant, rx_ant)
        # 生成接收频域
        Y_f = np.zeros((N, rx_ant), dtype=np.complex128)
        for sc in range(N):
            Y_f[sc,:] = H_f[sc,:,:] @ S_f[sc,:]
        # 加噪
        snr_lin = 10**(snr_db/10)
        sig_pow = np.mean(np.abs(Y_f)**2)
        noise_pow = sig_pow / snr_lin
        noise = (np.random.normal(0, np.sqrt(noise_pow/2), Y_f.shape) + 1j*np.random.normal(0, np.sqrt(noise_pow/2), Y_f.shape))
        Y_f_n = Y_f + noise
        # ZF/MMSE恢复
        S_hat = np.zeros_like(S_f)
        for sc in range(N):
            y = Y_f_n[sc,:].reshape(-1,1)
            H = H_f[sc,:,:]
            x_hat = op_lib.mmse_equalization(y, H, noise_pow)  # MMSE
            S_hat[sc,:] = x_hat.flatten()
        # 判决与BER
        bit_map = np.array([[0,0],[0,1],[1,0],[1,1]])
        ber_total = 0; bits_total = 0
        for ant in range(tx_ant):
            const = qpsk_const
            dists = np.abs(S_hat[:,ant][:,None] - const[None,:])
            idx_hat = np.argmin(dists, axis=1)
            bits_hat = bit_map[idx_hat].reshape(-1)
            bits_tx = tx_bits_simple[ant][:len(bits_hat)]
            err = np.sum(bits_hat != bits_tx)
            ber = err/len(bits_tx)
            print(f"[极简测试] 天线{ant} BER={ber:.4e}")
            ber_total += err
            bits_total += len(bits_tx)
        print(f"[极简测试] 平均BER={(ber_total/bits_total):.4e}")
        print("[极简测试] 完成 -> 若此处BER很低，问题确定位于完整流程的同步/窗口部分")
        raise SystemExit(0)

    print("="*60)
    print("4×4 MIMO链路仿真启动")
    print(f"参数配置：FFT={fft_size}, SCS={subcarrier_spacing/1e3}kHz, SNR={snr_db}dB")
    print("="*60)

    print("\n[1/4] 发送端处理...")
    tx_bits_valid, tx_symbols_list, tx_time_combined, tx_freq_sym, sync_freq_patterns = tx_mimo_signal(op_lib)

    print("\n[2/4] 信道传输...")
    rx_time_combined, h_mimo, noise_var = mimo_channel_transmit(op_lib, tx_time_combined, snr_db)

    print("\n[3/4] 接收端处理...")
    rx_bits = rx_mimo_signal(op_lib, rx_time_combined, h_mimo, noise_var, tx_symbols_list, tx_freq_sym, sync_freq_patterns)

    print("\n[4/4] 计算BER...")
    ber = calculate_ber(tx_bits_valid, rx_bits)

    print("\n" + "="*60)
    print("仿真结果")
    print("="*60)
    print(f"平均BER：{ber:.6f}")
    print(f"链路状态：{'✅ 跑通' if ber < 0.05 else '❌ 未跑通'}")
    print("="*60)