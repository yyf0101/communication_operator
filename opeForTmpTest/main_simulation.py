import numpy as np
from scipy.signal import firwin
from operator_library import SymbolLevelOperatorLibrary

# --- 1. 仿真参数设置 ---
num_subcarriers_pilot = 64
num_subcarriers_data = 64
fft_size = num_subcarriers_pilot + num_subcarriers_data  # 128
cp_length = 16

# 使用固定种子以保证结果可复现
np.random.seed(42)

# 生成发送符号 (BPSK)
pilot_symbol = np.ones(num_subcarriers_pilot, dtype=np.complex128)
data_symbols = np.ones(num_subcarriers_data, dtype=np.complex128)
tx_frame = np.concatenate([pilot_symbol, data_symbols])

# 多径信道
channel_taps = np.array([0.8, 0.3 + 0.2j, -0.1 + 0.1j], dtype=np.complex128)
num_taps = len(channel_taps)

# 射频前端参数
noise_power = 0.05  # 适当增加噪声以测试鲁棒性
iq_imbalance_matrix = np.array([[1.0, 0.05], [-0.03, 0.95]])
gain_factor = 1.0

# --- 2. 生成发送信号 ---
tx_time_signal = np.fft.ifft(tx_frame)
tx_time_signal_cp = np.concatenate([tx_time_signal[-cp_length:], tx_time_signal])

# --- 3. 信号通过信道和射频前端 ---
rx_time_signal_channelized = np.convolve(tx_time_signal_cp, channel_taps, mode='same')
noise = np.sqrt(noise_power/2) * (np.random.randn(len(rx_time_signal_channelized)) + 1j * np.random.randn(len(rx_time_signal_channelized)))
rx_time_signal_noisy = rx_time_signal_channelized + noise
rx_signal_dfe_input = SymbolLevelOperatorLibrary.iq_correction_and_gain(rx_time_signal_noisy, gain_factor, iq_imbalance_matrix)

# --- 4. 正确的符号级处理链路 ---
op_lib = SymbolLevelOperatorLibrary
print("--- 正确的符号级处理链路仿真 ---")

# 4.1 数字前端DFE (FIR滤波)
fir_coeffs = firwin(16, 0.4)
rx_signal_dfe_output = op_lib.fir_filter(rx_signal_dfe_input, fir_coeffs)
print(f"[DFE] 输入长度: {len(rx_signal_dfe_input)}, 输出长度: {len(rx_signal_dfe_output)}")

# 4.2 初始同步 (修正版：修正循环范围，防止数组越界)
# 计算整个信号的自相关，寻找最大峰值
# 【错误修正】修正循环的上限，确保两个切片都不会越界
corr_length = len(rx_signal_dfe_output) - fft_size - cp_length + 1
corr = np.zeros(corr_length)
for i in range(corr_length):
    # 确保两个切片的长度都是 cp_length
    segment1 = rx_signal_dfe_output[i:i+cp_length]
    segment2 = rx_signal_dfe_output[i+fft_size:i+fft_size+cp_length]
    corr[i] = np.abs(np.dot(segment1, np.conj(segment2)))

symbol_start_index = np.argmax(corr)
print(f"[同步] 检测到的符号起始索引: {symbol_start_index}")

# 4.3 多址解调 (OFDM-FFT)
rx_time_symbol = rx_signal_dfe_output[symbol_start_index : symbol_start_index + fft_size]
rx_freq_symbol = op_lib.fft_operation(rx_time_symbol)
print(f"[OFDM解调] FFT输入长度: {len(rx_time_symbol)}, 输出长度: {len(rx_freq_symbol)}")

# 4.4 信道估计 (修正版：频域插值 -> IFFT -> 时域滤波 -> FFT)
# 4.4.1 LS估计 (仅在导频位置)
h_ls_pilot = rx_freq_symbol[:num_subcarriers_pilot] / pilot_symbol

# 4.4.2 频域线性插值，得到完整的频域信道响应
h_freq_est_full = np.zeros(fft_size, dtype=np.complex128)
# 简单线性插值
freq_axis = np.arange(fft_size)
pilot_freq_indices = np.arange(num_subcarriers_pilot)
h_freq_est_full_re = np.interp(freq_axis, pilot_freq_indices, np.real(h_ls_pilot))
h_freq_est_full_im = np.interp(freq_axis, pilot_freq_indices, np.imag(h_ls_pilot))
h_freq_est_full = h_freq_est_full_re + 1j * h_freq_est_full_im

# 4.4.3 IFFT变换到时域，得到信道冲激响应 (CIR)
h_time_est = np.fft.ifft(h_freq_est_full)

# 4.4.4 时域滤波/门限，抑制噪声和小的多径分量
threshold = 0.1 * np.max(np.abs(h_time_est))
h_time_est_filtered = h_time_est * (np.abs(h_time_est) > threshold)

# 4.4.5 FFT变换回频域，得到最终用于均衡的信道估计
h_freq_final = np.fft.fft(h_time_est_filtered)
print(f"[信道估计] 最终信道估计在DC子载波处的幅度: {np.abs(h_freq_final[0]):.3f}")

# 4.5 均衡检测 (迫零均衡)
# 为了防止除以零，给信道估计值加上一个小的偏移
h_freq_final += 1e-6  # 数值稳定性
equalized_symbol = rx_freq_symbol / h_freq_final
print(f"[均衡] 第一个数据符号（均衡后）: {equalized_symbol[num_subcarriers_pilot]:.3f}")

# 4.6 LLR计算
constellation_bpsk = np.array([-1+0j, 1+0j])
llr = op_lib.llr_calculation(equalized_symbol[num_subcarriers_pilot], constellation_bpsk, noise_power)
print(f"[LLR计算] 第一个数据符号LLR: {llr[0]:.2f}")
print("--- 正确的符号级处理链路仿真结束 ---")

# --- 5. 最终结果验证 ---
print("\n--- 最终结果验证 ---")
print(f"发送的第一个数据符号: {data_symbols[0]}")
print(f"均衡后的第一个数据符号: {equalized_symbol[num_subcarriers_pilot]:.3f}")
print(f"判决结果（LLR>0→1，LLR<0→-1）: {1 if llr[0] > 0 else -1}")
print(f"判决是否正确: {'是' if (1 if llr[0] > 0 else -1) == np.real(data_symbols[0]) else '否'}")