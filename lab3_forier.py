import additional_code
import pandas as pd
import numpy as np
import os
from scipy.fft import fft, ifft
from sklearn.preprocessing import MinMaxScaler


def fourier_search(file_path, budget, output_file, top_k=10, random_seed=42):
    """"
    Use Fourier transform to smooth performance curves and predict optimal configurations.
    This model assumes that the performance curve is periodic, or linear and continuous.
    However, in reality, configurations are not like this, so this model performs poorly in all tests, worse than random search.
    """
    
    np.random.seed(random_seed)

    # Load data
    data = pd.read_csv(file_path)
    config_columns = data.columns[:-1]
    performance_column = data.columns[-1]
    system_name = os.path.basename(file_path).split('.')[0]
    maximization = False  # Default is minimization problems

    worst_value = data[performance_column].max() * 2
    best_performance = np.inf
    best_solution = []

    # Generate budget number of random configurations (ensure valid values)
    sampled_configs = []
    for i in range(budget):
        np.random.seed(random_seed + i)
        sampled_config = [int(np.random.choice(data[col].unique())) for col in config_columns]
        sampled_configs.append(sampled_config)

    # Find valid configurations in the data and their performance
    valid_configs = []
    valid_performances = []
    for config in sampled_configs:
        matched_row = data.loc[(data[config_columns] == pd.Series(config, index=config_columns)).all(axis=1)]
        if not matched_row.empty:
            perf = matched_row[performance_column].iloc[0]
            valid_configs.append(config)
            valid_performances.append(perf)

    if len(valid_configs) < 3:
        print(f"[{system_name}] Not enough valid configurations for Fourier prediction.")
        return [], worst_value

    # Normalize performance for Fourier transform
    scaler = MinMaxScaler()
    norm_performance = scaler.fit_transform(np.array(valid_performances).reshape(-1, 1)).flatten()

    # Fourier smoothing: preserve top_k frequency components
    fft_coeffs = fft(norm_performance)
    fft_coeffs[top_k:-top_k] = 0
    smoothed_norm = ifft(fft_coeffs).real

    # Inverse normalization
    smoothed_performance = scaler.inverse_transform(smoothed_norm.reshape(-1, 1)).flatten()

    # Find the best performing configuration
    # Find the point with lowest predicted performance
    min_idx = np.argmin(smoothed_performance)
    predicted_best_config = valid_configs[min_idx]

    # Check if this configuration exists in the original data
    matched_row = data.loc[(data[config_columns] == pd.Series(predicted_best_config, index=config_columns)).all(axis=1)]

    if not matched_row.empty:
        best_solution = predicted_best_config
        best_performance = matched_row[performance_column].iloc[0]
    else:
        # If not, look for the second lowest in smoothed_performance, and so on until finding an existing config
        sorted_idx = np.argsort(smoothed_performance)
        for idx in sorted_idx:
            candidate_config = valid_configs[idx]
            matched_row = data.loc[(data[config_columns] == pd.Series(candidate_config, index=config_columns)).all(axis=1)]
            if not matched_row.empty:
                best_solution = candidate_config
                best_performance = matched_row[performance_column].iloc[0]
                break

    # Save search records
    search_results = [conf + [perf] for conf, perf in zip(valid_configs, smoothed_performance)]
    columns = list(config_columns) + ["Performance"]
    pd.DataFrame(search_results, columns=columns).to_csv(output_file, index=False)

    return [int(x) for x in best_solution], best_performance


if __name__ == "__main__":
    print("Fourier Search Algorithm")
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
        print(f"System: {system} (Seed: {result['Seed Used']})",
              f"  Best Config:    [{', '.join(map(str, result['Best Solution']))}]",
              f"  Best Performance: {result['Best Performance']}")
