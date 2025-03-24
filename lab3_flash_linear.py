import additional_code

import tqdm
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import sys
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder


def flash_search_linear(file_path, budget, output_file, random_seed=42, initial_sample_size=10):

    '''
    I add linear DDS partition sampling to the original random sampling.
    '''

    if budget < 50:
        k = 1
    elif budget < 100:
        k = 3
    else:
        k = 5
    
    np.random.seed(random_seed)
    initial_budget = budget
    data = pd.read_csv(file_path)
    config_columns = data.columns[:-1]
    performance_column = data.columns[-1]
    system_name = os.path.basename(file_path).split('.')[0]
    maximization = False  # All are minimization problems

    worst_value = data[performance_column].max() * 2

    best_performance = np.inf
    best_solution = []
    search_results = []

    # --- Encode all parameter columns (if categorical) ---
    encoders = {}
    encoded_data = data.copy()
    for col in config_columns:
        if data[col].dtype == object or len(data[col].unique()) < 20:
            le = LabelEncoder()
            encoded_data[col] = le.fit_transform(data[col])
            encoders[col] = le

    # Initialize sampling points (randomly sample initial_sample_size points)
    # === DDS partition sampling instead of original random sampling ===
    intervals = {}
    for col in config_columns:
        unique_vals = sorted(encoded_data[col].unique())
        if len(unique_vals) <= k:
            intervals[col] = [[v] for v in unique_vals]
        else:
            chunks = np.array_split(unique_vals, k)
            intervals[col] = [list(chunk) for chunk in chunks]

    initial_samples = []
    for idx, col in enumerate(config_columns):
        col_intervals = intervals[col]
        for i, subrange in enumerate(col_intervals):
            np.random.seed(random_seed + idx * 100 + i)
            val = int(np.random.choice(subrange))
            sample = [encoded_data[c].mode()[0] for c in config_columns]  # Default
            sample[idx] = val
            initial_samples.append(sample)
            if len(initial_samples) >= initial_sample_size:
                break
        if len(initial_samples) >= initial_sample_size:
            break

    # Extract actual performance of initial_samples
    sampled_configs = []
    sampled_performances = []
    for sampled_config in initial_samples:
        matched_row = encoded_data.loc[(encoded_data[config_columns] == pd.Series(sampled_config, index=config_columns)).all(axis=1)]
        if not matched_row.empty:
            performance = matched_row[performance_column].iloc[0]
        else:
            performance = worst_value
        sampled_configs.append(sampled_config)
        sampled_performances.append(performance)
        search_results.append(sampled_config + [performance])
        budget -= 1
        if budget <= 0:
            break

    # Create progress bar
    progress_bar = tqdm(total=initial_budget, desc=f"Searching {system_name}", 
                        bar_format=" {l_bar}{bar}| Budget {n_fmt}/{total_fmt}  Time {remaining}  ")
    progress_bar.update(initial_sample_size)  # Update budget used for initial samples
    

    while budget > 0:
        progress_bar.set_postfix(remaining=budget, best=f"{best_performance:.4f}")
        # Train CART surrogate model
        model = DecisionTreeRegressor(random_state=random_seed)
        model.fit(sampled_configs, sampled_performances)

        # Predict all unsampled configurations
        all_configs = encoded_data[config_columns].values.tolist()
        remaining_configs = [cfg for cfg in all_configs if cfg not in sampled_configs]

        if not remaining_configs:
            break

        preds = model.predict(remaining_configs)

        # Use acquisition function (select configuration with lowest predicted value)
        selected_idx = int(np.argmin(preds))
        selected_config = remaining_configs[selected_idx]

        # Look up actual performance
        matched_row = encoded_data.loc[(encoded_data[config_columns] == pd.Series(selected_config, index=config_columns)).all(axis=1)]
        if not matched_row.empty:
            performance = matched_row[performance_column].iloc[0]
        else:
            performance = worst_value

        sampled_configs.append(selected_config)
        sampled_performances.append(performance)
        search_results.append(selected_config + [performance])

        if performance < best_performance:
            best_performance = performance
            best_solution = selected_config

        budget -= 1
        progress_bar.update(1)
    progress_bar.close()
    # Save search results
    columns = list(config_columns) + ["Performance"]
    pd.DataFrame(search_results, columns=columns).to_csv(output_file, index=False)

    return best_solution, best_performance


if __name__ == "__main__":
    global_seed = 42
    datasets_folder = "datasets"
    output_folder = "results\\search_results_flash_linear"
    os.makedirs(output_folder, exist_ok=True)
    budget = 20

    results = {}
    for file_name in os.listdir(datasets_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(datasets_folder, file_name)
            output_file = os.path.join(output_folder, f"{file_name.split('.')[0]}_search_results.csv")

            file_seed = additional_code.get_deterministic_seed(file_name, global_seed)

            best_solution, best_performance = flash_search_linear(file_path, budget, output_file, random_seed=file_seed, initial_sample_size = int(budget//10))
            results[file_name] = {
                "Best Solution": best_solution,
                "Best Performance": best_performance,
                "Seed Used": file_seed
            }

    save_results_csv = pd.DataFrame(results)
    save_results_csv.to_csv(f'results\\flash_linear{budget}.csv', index=False)

    for system, result in results.items():
        print(f"System: {system} (Seed: {result['Seed Used']})",
              f"  Best Config:    [{', '.join(map(str, result['Best Solution']))}]",
              f"  Best Performance: {result['Best Performance']}")
