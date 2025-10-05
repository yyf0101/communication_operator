import numpy as np
from scipy.signal import firwin

# --- 导入我们之前定义的符号级算子库 ---
from operator_library import SymbolLevelOperatorLibrary

# --- 1. LTE 参数配置 ---
FFT_SIZE = 1024
NUM_RB = 50
NUM_SUBCARRIERS = NUM_RB * 12
CP_LENGTH = 160
SYMBOL_LENGTH = FFT_SIZE + CP_LENGTH

# --- 2. LTE 信号生成 (使用DMRS，修复shape mismatch错误) ---
def generate_lte_signal_with_dmrs():
    resource_grid = np.zeros((1, FFT_SIZE), dtype=np.complex128)

    start_idx = (FFT_SIZE - NUM_SUBCARRIERS) // 2
    end_idx = start_idx + NUM_SUBCARRIERS

    # --- 1. 插入解调参考信号 (DMRS) ---
    dmrs_positions_freq = np.arange(start_idx + 6, end_idx, 12)
    dmrs_symbols = np.exp(1j * 2 * np.pi * np.random.rand(len(dmrs_positions_freq)))
    resource_grid[0, dmrs_positions_freq] = dmrs_symbols

    # --- 2. 确定可用的数据子载波位置 ---
    all_subcarrier_indices_in_band = np.arange(start_idx, end_idx)
    data_indices = np.setdiff1d(all_subcarrier_indices_in_band, dmrs_positions_freq)
    num_data_symbols_needed = len(data_indices) # 计算需要的数据符号数量

    # --- 3. 根据可用位置数量生成数据符号 ---
    data_symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], size=num_data_symbols_needed) / np.sqrt(2)

    # --- 4. 将数据符号映射到资源网格 ---
    resource_grid[0, data_indices] = data_symbols

    print(f"[LTE信号生成] 资源网格大小: {resource_grid.shape}")
    print(f"[LTE信号生成] 数据子载波范围: [{start_idx}, {end_idx-1}]")
    print(f"[LTE信号生成] DMRS插入位置数量: {len(dmrs_positions_freq)}")
    print(f"[LTE信号生成] 生成的数据符号数量: {len(data_symbols)}")

    tx_time_signal = np.fft.ifft(resource_grid[0, :])
    tx_time_signal_cp = np.concatenate([tx_time_signal[-CP_LENGTH:], tx_time_signal])

    return tx_time_signal_cp, resource_grid, start_idx, dmrs_positions_freq, dmrs_symbols, data_indices

# --- 3. 主仿真流程 (此部分无需修改) ---
def main():
    op_lib = SymbolLevelOperatorLibrary

    # --- 3.1 生成发送信号 (带DMRS) ---
    tx_signal, resource_grid, data_start_idx, dmrs_freq_pos, dmrs_syms, data_indices = generate_lte_signal_with_dmrs()

    # --- 3.2 信号通过信道和射频前端 ---
    channel_taps = np.array([0.0+0.0j, 0.45+0.0j, 0.8+0.3j, 0.2+0.2j])
    rx_signal_channelized = np.convolve(tx_signal, channel_taps, mode='same')

    noise_power = 0.01
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(rx_signal_channelized)) + 1j * np.random.randn(len(rx_signal_channelized)))
    rx_signal_noisy = rx_signal_channelized + noise

    # --- 3.3 符号级接收处理链路 ---
    print("\n--- LTE符号级处理链路仿真 (最终正确版) ---")

    # 3.3.1 数字前端DFE
    fir_coeffs = firwin(16, 0.4)
    rx_signal_dfe = op_lib.fir_filter(rx_signal_noisy, fir_coeffs)
    print(f"[DFE] 完成。")

    # 3.3.2 初始同步 (假设已完成)
    symbol_start_index = CP_LENGTH
    rx_time_symbol = rx_signal_dfe[symbol_start_index : symbol_start_index + FFT_SIZE]
    print(f"[同步] 假设符号起始索引: {symbol_start_index}")

    # 3.3.3 OFDM解调 (FFT)
    rx_freq_symbol = op_lib.fft_operation(rx_time_symbol)
    print(f"[OFDM解调] FFT完成。")

    # 3.3.4 信道估计 (基于DMRS)
    h_ls_dmrs = rx_freq_symbol[dmrs_freq_pos] / dmrs_syms
    h_est_data = np.interp(data_indices, dmrs_freq_pos, h_ls_dmrs)
    print(f"[信道估计] 基于DMRS的估计完成。")

    # 3.3.5 均衡 (MMSE均衡)
    rx_data_freq = rx_freq_symbol[data_indices]
    h_conj = np.conj(h_est_data)
    h_mag_sq = np.abs(h_est_data)**2
    mmse_weights = h_conj / (h_mag_sq + noise_power + 1e-10)
    equalized_symbols = rx_data_freq * mmse_weights
    print(f"[均衡] MMSE均衡完成。均衡后第一个符号: {equalized_symbols[0]:.3f}")

    # 3.3.6 LLR计算 (简化)
    print(f"[LLR计算] (简化) 第一个数据符号均衡后: {equalized_symbols[0]:.3f}")
    print("--- LTE符号级处理链路仿真 (最终正确版) 结束 ---")

    # --- 3.4 结果验证 ---
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
    main()