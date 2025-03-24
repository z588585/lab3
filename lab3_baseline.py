import additional_code 
import pandas as pd
import numpy as np
import os


# Define random search function
def random_search(file_path, budget, output_file, random_seed=42):
    # Set random seed to ensure reproducibility
    
    # Load dataset
    data = pd.read_csv(file_path)
    # Identify configuration columns and performance column
    config_columns = data.columns[:-1]  # All columns except the last one are configuration parameters
    performance_column = data.columns[-1]  # The last column is the performance metric

    # Determine if this is a maximization or minimization problem
    # Maximize throughput and minimize runtime
    system_name = os.path.basename(file_path).split('.')[0]
    if system_name.lower() == "---":
        maximization = True  # Maximization problem
    else:
        maximization = False  # Minimization problem

    # Extract best and worst performance values
    if maximization:
        worst_value = data[performance_column].min() / 2  # For missing configurations, use half of the minimum value
    
    else:
        worst_value = data[performance_column].max() * 2  # For missing configurations, use twice the maximum value
    

    # Initialize best solution and performance
    best_performance = -np.inf if maximization else np.inf  # Set initial value based on problem type - negative infinity or infinity
    best_solution = []

    # Store all search results
    search_results = []

    for _ in range(budget):
        random_seed_in_loop = random_seed + _  # Generate different random seed for each iteration
        np.random.seed(random_seed_in_loop)
        # Randomly sample a configuration
        # For each configuration column, randomly select a value from the unique values available in the dataset
        # This ensures that the sampled configuration is within the valid domain for each parameter
        sampled_config = [int(np.random.choice(data[col].unique())) for col in config_columns]

        # Check if the configuration exists in the dataset
        # Create a Pandas Series from the sampled configuration and match it with all rows in the dataset
        # .all(axis=1) ensures that the matching is applied to all configuration columns
        matched_row = data.loc[(data[config_columns] == pd.Series(sampled_config, index=config_columns)).all(axis=1)]

        if not matched_row.empty:
            # Existing configuration
            performance = matched_row[performance_column].iloc[0]
        else:
            # Non-existent configuration
            performance = worst_value

        # Update best solution
        if maximization:
            if performance > best_performance:
                best_performance = performance
                best_solution = sampled_config
        else:
            if performance < best_performance:
                best_performance = performance
                best_solution = sampled_config

        # Record current search result
        search_results.append(sampled_config + [performance])

    # Save search results to CSV file
    columns = list(config_columns) + ["Performance"]
    search_df = pd.DataFrame(search_results, columns=columns)
    search_df.to_csv(output_file, index=False)

    return [int(x) for x in best_solution], best_performance




if __name__ == "__main__":
    # Set global random seed
    global_seed = 42
    
    datasets_folder = "datasets"  # Datasets folder
    output_folder = "results\\search_results_baseline"  # Output folder
    os.makedirs(output_folder, exist_ok=True)  # Create output folder (if it doesn't exist)
    budget = 1000  # Search budget (number of configurations to try)

    results = {}
    for file_name in os.listdir(datasets_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(datasets_folder, file_name)
            output_file = os.path.join(output_folder, f"{file_name.split('.')[0]}_search_results.csv")
            
            # Use different random seed for each file, while maintaining reproducibility

            file_seed = additional_code.get_deterministic_seed(file_name, global_seed)
            
            best_solution, best_performance = random_search(file_path, budget, output_file, random_seed=file_seed)
            results[file_name] = {
                "Best Solution": best_solution,
                "Best Performance": best_performance,
                "Seed Used": file_seed  # Record the seed used
            }

    save_results_csv = pd.DataFrame(results)
    save_results_csv.to_csv('results\\baseline_{}.csv'.format(budget), index=False)
    # Print results
    for system, result in results.items():
        print(f"System: {system} (Seed: {result['Seed Used']})",
              f"  Best Config:    [{', '.join(map(str, result['Best Solution']))}]",
              f"  Best Performance: {result['Best Performance']}")


