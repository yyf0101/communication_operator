import numpy as np
from scipy.signal import firwin

from operator_library import SymbolLevelOperatorLibrary

# --- 1. Wi-Fi 6 参数配置 ---
FFT_SIZE = 256  # Wi-Fi 6 20MHz channel uses 256 FFT
NUM_SUBCARRIERS = 234 # Number of data/tone subcarriers in 20MHz
RU_SIZE_TONES = 52 # We simulate a user assigned to a 52-tone RU
CP_LENGTH = 32

# --- 2. Wi-Fi 6 信号生成 ---
def generate_wifi6_signal():
    # 资源网格: [时间(符号), 频率(子载波)]
    # 简化为一个数据符号
    resource_grid = np.zeros((1, FFT_SIZE), dtype=np.complex128)

    # Wi-Fi子载波索引从-117到+117 (DC子载波为0)
    # 我们选择一个位于中间的52-tone RU
    ru_start_tone_index = -26
    ru_end_tone_index = 25
    ru_tone_indices = np.arange(ru_start_tone_index, ru_end_tone_index + 1)
    # 将[-117, 117]的索引映射到[0, 255]的数组索引
    ru_array_indices = ru_tone_indices + (FFT_SIZE // 2)

    # --- 在RU中插入导频 (Pilots) ---
    # Wi-Fi 6的RU内导频位置是固定的
    ru_pilot_positions_in_ru = [7, 21, 35, 49] # 相对于RU起始位置的索引
    ru_pilot_array_indices = ru_array_indices[ru_pilot_positions_in_ru]
    pilot_symbols = np.array([1+0j, -1+0j, -1+0j, 1+0j]) # Known BPSK pilots
    resource_grid[0, ru_pilot_array_indices] = pilot_symbols

    # --- 在RU中插入数据 ---
    ru_data_positions_in_ru = np.setdiff1d(np.arange(RU_SIZE_TONES), ru_pilot_positions_in_ru)
    ru_data_array_indices = ru_array_indices[ru_data_positions_in_ru]
    data_symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], size=len(ru_data_positions_in_ru)) / np.sqrt(2)
    resource_grid[0, ru_data_array_indices] = data_symbols

    print(f"[Wi-Fi 6信号生成] 资源网格大小: {resource_grid.shape}")
    print(f"[Wi-Fi 6信号生成] 仿真一个 {RU_SIZE_TONES}-tone RU。")
    print(f"[Wi-Fi 6信号生成] RU内导频位置: {ru_pilot_positions_in_ru}")
    print(f"[Wi-Fi 6信号生成] 生成的数据符号数量: {len(data_symbols)}")

    tx_time_signal = np.fft.ifft(resource_grid[0, :])
    tx_time_signal_cp = np.concatenate([tx_time_signal[-CP_LENGTH:], tx_time_signal])

    # --- 生成HE-LTF用于信道估计 (简化) ---
    # 实际HE-LTF更复杂，这里用一个已知序列模拟
    he_ltf_sequence = np.ones(FFT_SIZE, dtype=np.complex128) # 简化的LTF序列
    he_ltf_time = np.fft.ifft(he_ltf_sequence)
    he_ltf_time_cp = np.concatenate([he_ltf_time[-CP_LENGTH:], he_ltf_time])

    # 组合前导码和数据
    # 注意：真实的PPDU有更长的前导码，这里简化为一个LTF符号+一个数据符号
    tx_ppdu = np.concatenate([he_ltf_time_cp, tx_time_signal_cp])

    return tx_ppdu, resource_grid, he_ltf_sequence, ru_data_array_indices, ru_pilot_array_indices, pilot_symbols, data_symbols

# --- 3. 主仿真流程 ---
def main_wifi6():
    op_lib = SymbolLevelOperatorLibrary

    tx_ppdu, resource_grid, he_ltf_sequence, ru_data_indices, ru_pilot_indices, pilot_symbols, data_symbols = generate_wifi6_signal()

    channel_taps = np.array([0.0+0.0j, 0.45+0.0j, 0.8+0.3j, 0.2+0.2j])
    rx_ppdu_channelized = np.convolve(tx_ppdu, channel_taps, mode='same')

    noise_power = 0.05 # Wi-Fi环境噪声通常更大
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(rx_ppdu_channelized)) + 1j * np.random.randn(len(rx_ppdu_channelized)))
    rx_ppdu_noisy = rx_ppdu_channelized + noise

    print("\n--- Wi-Fi 6符号级处理链路仿真 ---")

    fir_coeffs = firwin(16, 0.4)
    rx_ppdu_dfe = op_lib.fir_filter(rx_ppdu_noisy, fir_coeffs)
    print(f"[DFE] 完成。")

    # --- 同步 (简化) ---
    # 假设已通过前导码完成同步
    ltf_start_index = CP_LENGTH
    data_symbol_start_index = len(rx_ppdu_dfe) - (FFT_SIZE + CP_LENGTH)

    rx_ltf_time = rx_ppdu_dfe[ltf_start_index : ltf_start_index + FFT_SIZE]
    rx_data_time = rx_ppdu_dfe[data_symbol_start_index : data_symbol_start_index + FFT_SIZE]
    print(f"[同步] 假设LTF和数据符号起始位置已找到。")

    rx_ltf_freq = op_lib.fft_operation(rx_ltf_time)
    rx_data_freq = op_lib.fft_operation(rx_data_time)
    print(f"[OFDM解调] FFT完成。")

    # --- 信道估计 (基于HE-LTF) ---
    # 简化的频域信道估计
    h_est_freq = rx_ltf_freq / he_ltf_sequence
    # 提取RU内的信道估计
    h_est_ru = h_est_freq[ru_data_indices]
    print(f"[信道估计] 基于HE-LTF的估计完成。")

    # --- 均衡 (ZF) ---
    # Wi-Fi接收机常用ZF均衡
    h_est_ru += 1e-10 # 数值稳定
    equalized_symbols = rx_data_freq[ru_data_indices] / h_est_ru
    print(f"[均衡] ZF均衡完成。均衡后第一个符号: {equalized_symbols[0]:.3f}")

    print("\n--- 结果验证 ---")
    tx_data_symbol = data_symbols[0]
    print(f"发送的第一个数据符号: {tx_data_symbol:.3f}")
    print(f"均衡后的第一个数据符号: {equalized_symbols[0]:.3f}")
    constellation_qpsk = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    distances = np.abs(equalized_symbols[0] - constellation_qpsk)
    detected_symbol = constellation_qpsk[np.argmin(distances)]
    print(f"检测到的符号 (最近邻判决): {detected_symbol:.3f}")
    print(f"判决是否正确: {'是' if np.allclose(detected_symbol, tx_data_symbol, atol=0.3) else '否'}")

if __name__ == "__main__":
    # main_nr()
    main_wifi6()