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
        


def test_search_methods(methods1 = lab3_baseline.random_search ,
                        methods2 = lab3_bestconfig_linear.bestconfig_fast_search,
                        budget = 100 , 
                        file_path = "datasets\\system1.csv" ,
                        output = "results\\my_search_results\\system_search_results.csv" ,
                        global_seed = [i for i in range(1, 101)]):
    different_performance_list = []
    for seed in global_seed:
        _ , bestperformance_search1 = methods1(file_path, budget, output, random_seed=seed)
        _ , bestperformance_search2 = methods2(file_path, budget, output, random_seed=seed)
        different_performance = bestperformance_search1 - bestperformance_search2
        different_performance_list.append(different_performance)

    # 计算统计信息
    avg_diff = np.mean(different_performance_list)
    pos_count = sum(1 for x in different_performance_list if x > 0)
    neg_count = sum(1 for x in different_performance_list if x < 0)
    zero_count = sum(1 for x in different_performance_list if x == 0)
    
    print(f"Statistical analysis for {methods1.__name__} vs {methods2.__name__}")
    print(f"Average difference: {avg_diff:.4f}")
    print(f"Positive differences (method1 > method2): {pos_count}")
    print(f"Negative differences (method1 < method2): {neg_count}")
    print(f"Zero differences (method1 = method2): {zero_count}")
    
    # 绘制差值分布图
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    
    # 绘制差值散点图
    plt.subplot(1, 2, 1)
    plt.scatter(range(len(different_performance_list)), different_performance_list)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.axhline(y=avg_diff, color='g', linestyle='--', label=f'Avg: {avg_diff:.4f}')
    plt.xlabel('Experiment Number')
    plt.ylabel('Performance Difference')
    plt.title(f'{methods1.__name__} - {methods2.__name__} Performance Difference')
    plt.legend()
    
    # 绘制直方图
    plt.subplot(1, 2, 2)
    plt.hist(different_performance_list, bins=20, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
    plt.axvline(x=avg_diff, color='g', linestyle='--', label=f'Avg: {avg_diff:.4f}')
    plt.xlabel('Performance Difference')
    plt.ylabel('Frequency')
    plt.title('Histogram of Performance Differences')
    plt.legend()
    
    plt.tight_layout()
    
    # 保存图像
    system_name = os.path.basename(file_path).split('.')[0]
    plt.savefig(f'results_img/comparison_{methods1.__name__}_vs_{methods2.__name__}_{system_name}_b{budget}.png')
    plt.show()
    
    return different_performance_list, avg_diff, pos_count, neg_count, zero_count




def main():

    arg_parser = ArgumentParser('Intelligent Software Engineering Lab — Intelligent Software Tuning Experiment')
    arg_parser.add_argument('-g','--global_seed', type=int, default=42, help='Global random seed')
    arg_parser.add_argument('-na','--not_search_all', action='store_true', dest='not_search_all', help='Disable searching all methods and budgets')
    arg_parser.add_argument('-t','--test_search_methods', action='store_true', dest='test_search_methods', help='Test search methods')

    args = arg_parser.parse_args()
    
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
    if args.not_search_all == False and args.test_search_methods == False:

        print("global_seed: ", args.global_seed)
        print(" note: no bayesian search, because it is too slow( take 6h ?) you can try it by -> py -3.10 lab3_bayesian.py")
        for search_method in all_search_method:
            for budget in all_search_budget:
                bestconfig_search(datasets_folder, budget, output_folder, search_method = search_method, global_seed=args.global_seed)
    elif args.not_search_all == True and args.test_search_methods == False:
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

    elif args.test_search_methods == True and args.not_search_all == False:
        file_path = ["datasets\\Apache.csv" ,
                    "datasets\\7z.csv",
                    "datasets\\LLVM.csv",
                    "datasets\\storm.csv",
                    "datasets\\brotli.csv",
                    "datasets\\x264.csv",
                    "datasets\\spear.csv",
                    "datasets\\postgreSQL.csv"]
        for file in file_path:
            test_search_methods(methods1 = lab3_baseline.random_search ,
                                methods2 = lab3_bestconfig_linear.bestconfig_fast_search,
                                budget = 100 , 
                                file_path = file ,
                                output = "results\\my_search_results\\system_search_results.csv" ,
                                global_seed = [i for i in range(1, 101)])
    else:
        print("wrong input")
        return


    print("All search methods and budgets are done!")



if __name__ == "__main__":
    main()
