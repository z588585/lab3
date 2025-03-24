import additional_code
import os
import pandas 
import numpy 
import pandas as pd
import numpy as np
from itertools import product
import lab3_bestconfig_linear
import lab3_forier
import lab3_baseline
import lab3_bestconfig
import lab3_flash
import lab3_flash_linear
import lab3_bayesian
import argparse
from argparse import ArgumentParser



# Main function for testing on multiple datasets

def bestconfig_search(datasets_folder = "datasets" , budget=100, output_folder = "results\\search_results_bestconfig_fast" ,
                       global_seed=42 , search_method = lab3_bestconfig_linear.bestconfig_fast_search):

    print(f"Budget: {budget}, Method: {search_method}")

    # Set global random seed
    os.makedirs(output_folder, exist_ok=True)  # Create output folder (if it doesn't exist)

    results = {}
    for file_name in os.listdir(datasets_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(datasets_folder, file_name)
            output_file = os.path.join(output_folder, f"{file_name.split('.')[0]}_search_results.csv")
            
            # Use different random seed for each file, while maintaining reproducibility

            file_seed = additional_code.get_deterministic_seed(file_name, global_seed)
            
            best_solution, best_performance = search_method(file_path, budget, output_file, random_seed=file_seed)
         
            results[file_name] = {
                "Best Solution": best_solution,
                "Best Performance": best_performance,
                "Seed Used": file_seed  # Record the seed used
            }
    
    save_results_csv = pd.DataFrame(results)
    method_name = search_method.__name__.replace("_search", "")
    save_results_csv.to_csv(f'results\\{method_name}_{budget}.csv', index=False)

    # Print results
    for system, result in results.items():
        print(f"System: {system} (Seed: {result['Seed Used']})",
              f"  Best Config:    [{', '.join(map(str, result['Best Solution']))}]",
              f"  Best Performance: {result['Best Performance']}")



def main():

    arg_parser = ArgumentParser('Intelligent Software Engineering Lab — Intelligent Software Tuning Experiment')
    arg_parser.add_argument('-g','--global_seed', type=int, default=42, help='Global random seed')
    arg_parser.add_argument('-na','--not_search_all', action='store_true', dest='not_search_all', help='Disable searching all methods and budgets')

    args = arg_parser.parse_args()
    
    print(" note: no bayesian search, because it is too slow( take 6h ?) you can try it by -> py -3.10 lab3_bayesian.py")

    all_search_method = [lab3_bestconfig_linear.bestconfig_fast_search, 
                         lab3_forier.fourier_search, 
                         lab3_baseline.random_search, 
                         lab3_bestconfig.bestconfig_search, 
                         lab3_flash.flash_search,
                         lab3_flash_linear.flash_search_linear,
                         # lab3_bayesian.bayesian_optimization
                         # no bayesian search, because it is too slow( take 6h ?) you can try it by -> py -3.10 lab3_bayesian.py
                         ]
                        
    all_search_budget = [20,50,100, 200,500, 1000]
    output_folder = "results\\search_results_bestconfig_fast"
    datasets_folder = "datasets"
    print("global_seed: ", args.global_seed)

    if args.not_search_all == False:
        for search_method in all_search_method:
            for budget in all_search_budget:
                bestconfig_search(datasets_folder, budget, output_folder, search_method = search_method, global_seed=args.global_seed)
    else:
        print("please set you dataset folder:")
        print("for example: datasets")
        datasets_folder = input()
        print("please set you output folder:")
        print("for example: results\\my_search_results")
        output_folder = input()
        print("please set you budget:")
        budget = int(input())
        print("please set you search method:")
        print("you can choose from: 1.bestconfig_fast 2.fourier, 3.random, 4.bestconfig, 5.flash, 6.flash_linear, 7.bayesian")
        search_method_idx = int(input())
        if search_method_idx == 1:
            search_method = lab3_bestconfig_linear.bestconfig_fast_search
        elif search_method_idx == 2:
            search_method = lab3_forier.fourier_search
        elif search_method_idx == 3:
            search_method = lab3_baseline.random_search
        elif search_method_idx == 4:
            search_method = lab3_bestconfig.bestconfig_search
        elif search_method_idx == 5:
            search_method = lab3_flash.flash_search
        elif search_method_idx == 6:
            search_method = lab3_flash_linear.flash_search_linear
        elif search_method_idx == 7:
            search_method = lab3_bayesian.bayesian_optimization
        else:
            print("wrong search method")
            return

        bestconfig_search(datasets_folder, budget, output_folder, search_method = search_method, global_seed=args.global_seed)

    print("All search methods and budgets are done!")

    print_results_different_search_methods()



def print_results_different_search_methods():
    pass

if __name__ == "__main__":
    main()
