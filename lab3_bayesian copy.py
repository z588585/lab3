import additional_code 
import pandas as pd
import numpy as np
import os
from skopt import Optimizer
from skopt.space import Integer, Real, Categorical
import warnings
from tqdm import tqdm


# Define Bayesian optimization function
def bayesian_optimization(file_path, budget, output_file, random_seed=42):
    np.random.seed(random_seed)
    data = pd.read_csv(file_path)
    config_columns = data.columns[:-1]
    perf_column = data.columns[-1]
    system_name = os.path.basename(file_path).split('.')[0].lower()
    maximization = False  # All are minimization problems

    # Define search space
    space = []
    for col in config_columns:
        unique_values = sorted(data[col].unique())
        if len(unique_values) <= 20:
            # Discrete variables, more accurately represented by Categorical
            space.append(Categorical(unique_values))
        else:
            # Continuous or high-dimensional integer space
            space.append(Integer(min(unique_values), max(unique_values)))

    # Initialize BO optimizer
    optimizer = Optimizer(dimensions=space, random_state=random_seed, base_estimator="GP", n_initial_points=5)
    # Get maximum performance value for missing config cases
    worst_value = data[perf_column].max() * 2
    progress_bar = tqdm(total=budget, desc=f"Searching {system_name}", 
                    bar_format=" {l_bar}{bar}| Budget {n_fmt}/{total_fmt} Time {remaining}")

    evaluated = set()
    search_results = []
    best_performance = np.inf
    best_config = None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="skopt")
        for i in range(budget):
            config = optimizer.ask()
            config_key = tuple(config)

            # Check if configuration exists in data
            if config_key not in evaluated:
                query = (data[config_columns] == pd.Series(config, index=config_columns)).all(axis=1)
                match = data[query]

                if not match.empty:
                    perf = match[perf_column].iloc[0]
                else:
                    perf = worst_value
                progress_msg = f"Best performance: {best_performance:.4f}" if best_performance < np.inf else "No valid config found yet"
                progress_bar.set_description(f"Searching {system_name} [{progress_msg}]")
                progress_bar.update(1)
            else:
                # If configuration is duplicated, skip but don't update progress bar
                i -= 1  # Offset loop increment to ensure evaluating enough configurations

            evaluated.add(config_key)
            optimizer.tell(config, perf)

            # Update best
            if perf < best_performance:
                best_performance = perf
                best_config = config

            search_results.append(config + [perf])

    # Save results
    columns = list(config_columns) + ["Performance"]
    df = pd.DataFrame(search_results, columns=columns)
    df.to_csv(output_file, index=False)

    return best_config, best_performance



if __name__ == "__main__":
    # Set global random seed
    global_seed = 42
    
    datasets_folder = "datasets"  # Datasets folder
    output_folder = "results\\search_results_bayesian"  # Output folder
    os.makedirs(output_folder, exist_ok=True)  # Create output folder (if it doesn't exist)
    #budget = 1000  # Search budget (number of configurations to try)
    budgets = [20, 50, 100, 200, 500, 1000]  # Search budgets (number of configurations to try)
    
    for current_budget in budgets:  # Modified here to iterate through budgets list
        results = {}
        for file_name in os.listdir(datasets_folder):
            if file_name.endswith(".csv"):
                file_path = os.path.join(datasets_folder, file_name)
                # Include budget value in output filename
                output_file = os.path.join(output_folder, f"{file_name.split('.')[0]}_budget{current_budget}_search_results.csv")
                
                # Use different random seed for each file, while maintaining reproducibility
                file_seed = additional_code.get_deterministic_seed(file_name, global_seed)
                
                # Pass current budget value, not the entire list
                best_solution, best_performance = bayesian_optimization(file_path, current_budget, output_file, random_seed=file_seed)
                
                results[file_name] = {
                    "Best Solution": best_solution,
                    "Best Performance": best_performance,
                    "Seed Used": file_seed  # Record the seed used
                }

        # Include budget value in save filename
        save_results_csv = pd.DataFrame(results)
        save_results_csv.to_csv(f'results\\bayesian_{current_budget}.csv', index=False)
        
        # Print results
        # print(f"\n=== Search Budget: {current_budget} ===")
        # for system, result in results.items():
        #     print(f"System: {system} (Seed: {result['Seed Used']})",
        #           f"  Best Config:    [{', '.join(map(str, result['Best Solution']))}]",
        #           f"  Best Performance: {result['Best Performance']}")


