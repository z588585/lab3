import additional_code 
import pandas as pd
import numpy as np
import sys
import os
from itertools import product


def bestconfig_search(file_path, budget, output_file, k=2, top_n=1, random_seed=42):
    """
    Find the best configuration using divide-and-conquer sampling and recursive boundary search

    In fact, this model is poor because its time complexity is exponential, and the budget is only 100.
    In most cases, it needs to search for k^8 configurations (8 configurations), resulting in insufficient budget.
    Even if k is set to 2, it still needs to search for 2^8 configurations (256 configurations), exceeding 100.

    However, if we assume that each option's impact on performance is independent,
    when implementing bestconfig_search, we only need to search k*number_of_configs instead of k^number_of_configs in the DDS stage.
    This ensures linear time complexity (reducing complexity from O(k^d) to O(kd)).
    Thus, I completed my linear BestConfigSearch in the next file.

    Parameters:
        file_path: Path to the dataset file
        budget: Total sampling budget
        output_file: Path for result output file
        k: Number of parameter space partitions
        top_n: Return top N best solutions
        random_seed: Random seed
    """

    np.random.seed(random_seed)  # Set random seed to ensure reproducibility
    data = pd.read_csv(file_path)
    config_columns = data.columns[:-1]
    performance_column = data.columns[-1]
    system_name = os.path.basename(file_path).split('.')[0]
    if system_name.lower() == "---":
        maximization = True  # Maximization problem
    else:
        maximization = False  # Minimization problem

    if maximization:
        worst_value = data[performance_column].min() / 2  # For missing configurations, use half of the minimum value
    
    else:
        worst_value = data[performance_column].max() * 2  # For missing configurations, use twice the maximum value

    # Create a list to store the top N configurations
    top_configs = []  # Format: [(performance_value, [configuration])]

    worst_value = data[performance_column].max() * 2
    best_performance = -np.inf if maximization else np.inf  # Set initial value based on problem type - infinity or negative infinity
    best_solution = []
    search_results = []

# === Stage 1: DDS - Divide and Diverge Sampling stage ===
    intervals = {}  # Create a dictionary to store interval divisions for each configuration parameter
    for col in config_columns:  # Iterate through each configuration parameter column
        unique_vals = sorted(data[col].unique())  # Get all unique values for this parameter and sort them
        if len(unique_vals) <= k:  # If the number of unique values is less than or equal to k (division count)
            intervals[col] = [[v] for v in unique_vals]  # Each value forms its own interval
        else:  # If the number of unique values is greater than k
            chunks = np.array_split(unique_vals, k)  # Evenly divide the value range into k intervals
            intervals[col] = [list(chunk) for chunk in chunks]  # Convert each interval to a list and store it

    # Subspace combinations (selecting one interval for each parameter)
    # Generate all possible subspace combinations (Cartesian product)
    subspaces = list(product(*[range(len(intervals[col])) for col in config_columns]))
    np.random.shuffle(subspaces)  # Randomly shuffle the subspace access order to increase sampling diversity
    initial_samples = []  # Store initially sampled configurations

    # Select samples from each subspace
    for i, subspace in enumerate(subspaces):  # Iterate through each subspace
        random_seed_in_loop = random_seed + i  # Generate different random seeds for each iteration
        np.random.seed(random_seed_in_loop)
        if len(initial_samples) >= budget:  # Stop if the number of samples reaches the budget limit
            break
        sample = []  # Store the currently sampled configuration
        for idx, col in enumerate(config_columns):  # Iterate through each configuration parameter
            random_seed_in_loop = random_seed_in_loop + idx  # Generate different random seeds for each parameter
            subrange = intervals[col][subspace[idx]]  # Get the value range for this parameter in the current subspace
            sample.append(int(np.random.choice(subrange)))  # Randomly select a value from the range
        initial_samples.append(sample)  # Add the sampled configuration to the initial sample set
    np.random.seed(random_seed)  # Reset the random seed to ensure reproducibility

    # Evaluate the performance of initial samples
    for sampled_config in initial_samples:  # Iterate through each sampled configuration
        # Check if this configuration exists in the dataset
        matched_row = data.loc[(data[config_columns] == pd.Series(sampled_config, index=config_columns)).all(axis=1)]
        if not matched_row.empty:  # If the configuration exists in the dataset
            performance = matched_row[performance_column].iloc[0]  # Get its performance value
        else:  # If the configuration does not exist in the dataset
            performance = worst_value  # Use the worst performance value (twice the maximum value)
        
        # Update the top N configurations list
        update_top_configs(top_configs, sampled_config.copy(), performance, top_n, maximization)
        
        if maximization:
            if performance > best_performance:
                best_performance = performance
                best_solution = sampled_config
        else:
            if performance < best_performance:
                best_performance = performance
                best_solution = sampled_config

        
        search_results.append(sampled_config + [performance])  # Add the configuration and its performance to the search results

    remaining_budget = budget - len(initial_samples)  # Calculate the remaining sampling budget


    if remaining_budget <= 0:
        print(f"warning: budget is too small or k is too large {len(initial_samples)} / {budget}, in {file_path}")
    elif remaining_budget < 50:
        print(f"Initial samples: {len(initial_samples)} / {budget}, in {file_path}")
    else:
        print(f"Initial samples: {len(initial_samples)} / {budget}")
        #pass



    # === Stage 2: RBS - Recursive Bound and Search stage ===
    current_best = best_solution  # Set the current best configuration to the best configuration found in stage 1
    while remaining_budget > 0:  # Continue searching while there is remaining budget
        bounded_data = data.copy()  # Copy the original dataset for boundary restriction
        for i, col in enumerate(config_columns):  # Iterate through each configuration parameter
            ci = current_best[i]  # Get the value of the parameter in the current best configuration
            # Find the maximum value less than the current value, or use the current value if none
            smaller = data[data[col] < ci][col].max() if not data[data[col] < ci].empty else ci
            # Find the minimum value greater than the current value, or use the current value if none
            larger = data[data[col] > ci][col].min() if not data[data[col] > ci].empty else ci
            # Restrict the dataset within the found boundaries
            bounded_data = bounded_data[(bounded_data[col] >= smaller) & (bounded_data[col] <= larger)]

        # Randomly sample within the limited boundaries
        sampled_rows = bounded_data.sample(min(len(bounded_data), remaining_budget), replace=False)
        for _, row in sampled_rows.iterrows():  # Iterate through the sampled rows
            config = [int(row[col]) for col in config_columns]  # Extract the configuration parameter values
            performance = row[performance_column]  # Get the performance value
            
            # Update the top N configurations list
            update_top_configs(top_configs, config.copy(), performance, top_n, maximization)
            
            if performance < best_performance:  # If the performance is better than the historical best
                best_performance = performance  # Update the best performance
                best_solution = config  # Update the best configuration
                current_best = config  # Update the current best configuration for the next recursive search
            search_results.append(config + [performance])  # Record the search results
            remaining_budget -= 1  # Decrease the remaining budget
            if remaining_budget <= 0:  # Exit if the budget is exhausted
                break

    # Save the results
    columns = list(config_columns) + ["Performance"]
    pd.DataFrame(search_results, columns=columns).to_csv(output_file, index=False)
    
    # Save the top-n configurations to a separate file
    top_n_file = output_file.replace(".csv", "_top_n.csv")
    top_n_results = []
    for perf, conf in top_configs:
        top_n_results.append(conf + [perf])
    pd.DataFrame(top_n_results, columns=columns).to_csv(top_n_file, index=False)

    if len(top_configs) == 1:
        return [int(x) for x in best_solution], best_performance
    else:
        print("error , top n is not 1")
        sys.exit(1)
        return [int(x) for x in best_solution], best_performance


def update_top_configs(top_configs, config, performance, top_n, maximization=False):
    """
    Update the top N configurations list
    
    Parameters:
        top_configs: Current top N configurations list [(performance_value, [configuration])]
        config: New configuration
        performance: Performance value of the new configuration
        top_n: Number of best configurations to keep
        maximization: Whether it is a maximization problem
    """
    # If the list is not full or the new configuration is better than one in the list
    if len(top_configs) < top_n or (not maximization and performance < top_configs[-1][0]) or (maximization and performance > top_configs[0][0]):
        # Add the new configuration
        top_configs.append((performance, config))
        
        # Sort by performance (minimization or maximization)
        if maximization:
            top_configs.sort(key=lambda x: x[0], reverse=True)  # Sort in descending order (maximization)
        else:
            top_configs.sort(key=lambda x: x[0])  # Sort in ascending order (minimization)
        
        # If exceeding the top_n limit, remove the extra configuration
        if len(top_configs) > top_n:
            top_configs.pop()  # Remove the last one (worst performance)


# Main function to test on multiple datasets

if __name__ == "__main__":
    # Set global random seed
    global_seed = 42
    
    datasets_folder = "datasets"  # Datasets folder
    output_folder = "results\\search_results_bestconfig"  # Output folder
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it does not exist
    budget = 500  # Search budget (number of configurations to try)

    results = {}
    for file_name in os.listdir(datasets_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(datasets_folder, file_name)
            output_file = os.path.join(output_folder, f"{file_name.split('.')[0]}_search_results.csv")
            
            # Use a different random seed for each file, but still keep reproducibility

            file_seed = additional_code.get_deterministic_seed(file_name, global_seed)
            
            best_solution, best_performance = bestconfig_search (file_path, budget, output_file, random_seed=file_seed)
         
            results[file_name] = {
                "Best Solution": best_solution,
                "Best Performance": best_performance,
                "Seed Used": file_seed  # Record the seed used
            }
    
    save_results_csv = pd.DataFrame(results)
    save_results_csv.to_csv('results\\bestconfig_{}.csv'.format(budget), index=False)

    # Print the results
    for system, result in results.items():
        print(f"System: {system} (Seed: {result['Seed Used']})",f"  Best Configuration:    [{', '.join(map(str, result['Best Solution']))}]",f"  Best Performance: {result['Best Performance']}")


