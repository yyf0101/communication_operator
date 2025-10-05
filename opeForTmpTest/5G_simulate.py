import numpy as np
from scipy.signal import firwin

from operator_library import SymbolLevelOperatorLibrary

# --- 1. 5G NR 参数配置 ---
FFT_SIZE = 1024
SCS_KHZ = 30  # 30 kHz subcarrier spacing
NUM_RB = 52   # 52 RBs -> 52 * 12 = 624 subcarriers (corresponds to ~20 MHz)
NUM_SUBCARRIERS = NUM_RB * 12
CP_LENGTH = 80  # For SCS 30kHz, normal CP
SYMBOL_LENGTH = FFT_SIZE + CP_LENGTH

# --- 2. 5G NR 信号生成 ---
def generate_nr_signal():
    # 资源网格: [时间(符号), 频率(子载波)]
    resource_grid = np.zeros((1, FFT_SIZE), dtype=np.complex128)

    start_idx = (FFT_SIZE - NUM_SUBCARRIERS) // 2
    end_idx = start_idx + NUM_SUBCARRIERS

    # --- 插入DMRS ---
    # DMRS在频域上每隔一个RB的第3个子载波
    dmrs_positions_freq = np.arange(start_idx + 3, end_idx, 12)
    dmrs_symbols = np.exp(1j * 2 * np.pi * np.random.rand(len(dmrs_positions_freq)))
    resource_grid[0, dmrs_positions_freq] = dmrs_symbols

    # --- 确定数据位置并生成数据 ---
    all_subcarrier_indices_in_band = np.arange(start_idx, end_idx)
    data_indices = np.setdiff1d(all_subcarrier_indices_in_band, dmrs_positions_freq)
    data_symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], size=len(data_indices)) / np.sqrt(2)
    resource_grid[0, data_indices] = data_symbols

    print(f"[5G NR信号生成] 资源网格大小: {resource_grid.shape}")
    print(f"[5G NR信号生成] SCS: {SCS_KHZ} kHz, 数据子载波范围: [{start_idx}, {end_idx-1}]")
    print(f"[5G NR信号生成] DMRS插入位置数量: {len(dmrs_positions_freq)}")
    print(f"[5G NR信号生成] 生成的数据符号数量: {len(data_symbols)}")

    tx_time_signal = np.fft.ifft(resource_grid[0, :])
    tx_time_signal_cp = np.concatenate([tx_time_signal[-CP_LENGTH:], tx_time_signal])

    return tx_time_signal_cp, resource_grid, dmrs_positions_freq, dmrs_symbols, data_indices

# --- 3. 主仿真流程 ---
def main_nr():
    op_lib = SymbolLevelOperatorLibrary

    tx_signal, resource_grid, dmrs_freq_pos, dmrs_syms, data_indices = generate_nr_signal()

    channel_taps = np.array([0.0+0.0j, 0.45+0.0j, 0.8+0.3j, 0.2+0.2j])
    rx_signal_channelized = np.convolve(tx_signal, channel_taps, mode='same')

    noise_power = 0.01
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(rx_signal_channelized)) + 1j * np.random.randn(len(rx_signal_channelized)))
    rx_signal_noisy = rx_signal_channelized + noise

    print("\n--- 5G NR符号级处理链路仿真 ---")

    fir_coeffs = firwin(16, 0.4)
    rx_signal_dfe = op_lib.fir_filter(rx_signal_noisy, fir_coeffs)
    print(f"[DFE] 完成。")

    symbol_start_index = CP_LENGTH
    rx_time_symbol = rx_signal_dfe[symbol_start_index : symbol_start_index + FFT_SIZE]
    print(f"[同步] 假设符号起始索引: {symbol_start_index}")

    rx_freq_symbol = op_lib.fft_operation(rx_time_symbol)
    print(f"[OFDM解调] FFT完成。")

    # --- 信道估计 (基于DMRS) ---
    h_ls_dmrs = rx_freq_symbol[dmrs_freq_pos] / dmrs_syms
    h_est_data = np.interp(data_indices, dmrs_freq_pos, h_ls_dmrs)
    print(f"[信道估计] 基于DMRS的估计完成。")

    rx_data_freq = rx_freq_symbol[data_indices]
    h_conj = np.conj(h_est_data)
    h_mag_sq = np.abs(h_est_data)**2
    mmse_weights = h_conj / (h_mag_sq + noise_power + 1e-10)
    equalized_symbols = rx_data_freq * mmse_weights
    print(f"[均衡] MMSE均衡完成。均衡后第一个符号: {equalized_symbols[0]:.3f}")

    print("\n--- 结果验证 ---")
    tx_data_symbol = resource_grid[0, data_indices[0]]
    print(f"发送的第一个数据符号: {tx_data_symbol:.3f}")
    print(f"均衡后的第一个数据符号: {equalized_symbols[0]:.3f}")
    constellation_qpsk = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    distances = np.abs(equalized_symbols[0] - constellation_qpsk)
    detected_symbol = constellation_qpsk[np.argmin(distances)]
    print(f"检测到的符号 (最近邻判决): {detected_symbol:.3f}")
    print(f"判决是否正确: {'是' if np.allclose(detected_symbol, tx_data_symbol, atol=0.3) else '否'}")

if __name__ == "__main__":
    main_nr()