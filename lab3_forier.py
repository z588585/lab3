import additional_code
import pandas as pd
import numpy as np
import os
from scipy.fft import fft, ifft
from sklearn.preprocessing import MinMaxScaler


def fourier_search(file_path, budget, output_file, top_k=10, random_seed=42):
    """"
    使用傅里叶变换平滑性能曲线并预测最佳配置。"
    本模型假设性能曲线是周期性的，或者线性且在连续的"
    但是，实际上设置不是这样的，于是这个模型在所有测试中都表现不佳，不如随机搜索。"
    """
    
    np.random.seed(random_seed)

    # 加载数据
    data = pd.read_csv(file_path)
    config_columns = data.columns[:-1]
    performance_column = data.columns[-1]
    system_name = os.path.basename(file_path).split('.')[0]
    maximization = False  # 默认都是最小化问题

    worst_value = data[performance_column].max() * 2
    best_performance = np.inf
    best_solution = []

    # 生成 budget 数量的随机配置（确保在有效取值内）
    sampled_configs = []
    for i in range(budget):
        np.random.seed(random_seed + i)
        sampled_config = [int(np.random.choice(data[col].unique())) for col in config_columns]
        sampled_configs.append(sampled_config)

    # 查找存在于数据中的有效配置及其性能
    valid_configs = []
    valid_performances = []
    for config in sampled_configs:
        matched_row = data.loc[(data[config_columns] == pd.Series(config, index=config_columns)).all(axis=1)]
        if not matched_row.empty:
            perf = matched_row[performance_column].iloc[0]
            valid_configs.append(config)
            valid_performances.append(perf)

    if len(valid_configs) < 3:
        print(f"[{system_name}] 有效配置不足，无法进行傅里叶预测。")
        return [], worst_value

    # 对性能做归一化以适配傅里叶变换
    scaler = MinMaxScaler()
    norm_performance = scaler.fit_transform(np.array(valid_performances).reshape(-1, 1)).flatten()

    # 傅里叶平滑：保留前 top_k 个频率分量
    fft_coeffs = fft(norm_performance)
    fft_coeffs[top_k:-top_k] = 0
    smoothed_norm = ifft(fft_coeffs).real

    # 反归一化
    smoothed_performance = scaler.inverse_transform(smoothed_norm.reshape(-1, 1)).flatten()

    # 找到性能最好的配置
    # 找到预测性能最低的点
    min_idx = np.argmin(smoothed_performance)
    predicted_best_config = valid_configs[min_idx]

    # 查找该配置在原始数据中是否存在
    matched_row = data.loc[(data[config_columns] == pd.Series(predicted_best_config, index=config_columns)).all(axis=1)]

    if not matched_row.empty:
        best_solution = predicted_best_config
        best_performance = matched_row[performance_column].iloc[0]
    else:
        # 如果不存在，寻找 smoothed_performance 次小的，直到找到存在的配置
        sorted_idx = np.argsort(smoothed_performance)
        for idx in sorted_idx:
            candidate_config = valid_configs[idx]
            matched_row = data.loc[(data[config_columns] == pd.Series(candidate_config, index=config_columns)).all(axis=1)]
            if not matched_row.empty:
                best_solution = candidate_config
                best_performance = matched_row[performance_column].iloc[0]
                break


    # 保存搜索记录
    search_results = [conf + [perf] for conf, perf in zip(valid_configs, smoothed_performance)]
    columns = list(config_columns) + ["Performance"]
    pd.DataFrame(search_results, columns=columns).to_csv(output_file, index=False)

    return [int(x) for x in best_solution], best_performance


if __name__ == "__main__":
    print("傅里叶搜索算法")
    global_seed = 42
    datasets_folder = "datasets"
    output_folder = "results\\search_results_fourier"
    os.makedirs(output_folder, exist_ok=True)
    budget = 1000

    results = {}
    for file_name in os.listdir(datasets_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(datasets_folder, file_name)
            output_file = os.path.join(output_folder, f"{file_name.split('.')[0]}_search_results.csv")
            file_seed = additional_code.get_deterministic_seed(file_name, global_seed)

            best_solution, best_performance = fourier_search(
                file_path, budget, output_file, top_k=10, random_seed=file_seed
            )

            results[file_name] = {
                "Best Solution": best_solution,
                "Best Performance": best_performance,
                "Seed Used": file_seed
            }

    save_results_csv = pd.DataFrame(results)
    save_results_csv.to_csv(f'results\\fourier_{budget}.csv', index=False)

    for system, result in results.items():
        print(f"系统: {system} (种子: {result['Seed Used']})",
              f"  最佳配置:    [{', '.join(map(str, result['Best Solution']))}]",
              f"  最佳性能值: {result['Best Performance']}")
