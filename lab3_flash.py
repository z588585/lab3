import additional_code

import tqdm
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import sys
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder


def flash_search(file_path, budget, output_file, random_seed=42, initial_sample_size=10):
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
    initial_data = encoded_data.sample(n=min(initial_sample_size, len(encoded_data)), random_state=random_seed)
    sampled_configs = initial_data[config_columns].values.tolist()
    sampled_performances = initial_data[performance_column].tolist()

    search_results.extend([config + [perf] for config, perf in zip(sampled_configs, sampled_performances)])
    budget -= len(sampled_configs)

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
    output_folder = "results\\search_results_flash"
    os.makedirs(output_folder, exist_ok=True)
    budget = 500

    results = {}
    for file_name in os.listdir(datasets_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(datasets_folder, file_name)
            output_file = os.path.join(output_folder, f"{file_name.split('.')[0]}_search_results.csv")

            file_seed = additional_code.get_deterministic_seed(file_name, global_seed)

            best_solution, best_performance = flash_search(file_path, budget, output_file, random_seed=file_seed, initial_sample_size = int(budget//10))
            results[file_name] = {
                "Best Solution": best_solution,
                "Best Performance": best_performance,
                "Seed Used": file_seed
            }

    save_results_csv = pd.DataFrame(results)
    save_results_csv.to_csv(f'results\\flash_{budget}.csv', index=False)

    for system, result in results.items():
        print(f"System: {system} (Seed: {result['Seed Used']})",
              f"  Best Config:    [{', '.join(map(str, result['Best Solution']))}]",
              f"  Best Performance: {result['Best Performance']}")
