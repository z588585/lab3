
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
from argparse import ArgumentParser



# 在多个数据集上测试的主函数

def bestconfig_search(datasets_folder = "datasets" , budget=100, output_folder = "results\\search_results_bestconfig_fast" ,
                       global_seed=42 , search_method = lab3_bestconfig_linear.bestconfig_fast_search):

    print(f"预算: {budget}, 方法: {search_method}")

    # 设置全局随机种子
    os.makedirs(output_folder, exist_ok=True)  # 创建输出文件夹（如果不存在）

    results = {}
    for file_name in os.listdir(datasets_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(datasets_folder, file_name)
            output_file = os.path.join(output_folder, f"{file_name.split('.')[0]}_search_results.csv")
            
            # 为每个文件使用不同的随机种子，但仍然保持可重现性

            file_seed = additional_code.get_deterministic_seed(file_name, global_seed)
            
            best_solution, best_performance = search_method (file_path, budget, output_file, random_seed=file_seed)
         
            results[file_name] = {
                "Best Solution": best_solution,
                "Best Performance": best_performance,
                "Seed Used": file_seed  # 记录使用的种子
            }
    
    save_results_csv = pd.DataFrame(results)
    method_name = search_method.__name__.replace("_search", "")
    save_results_csv.to_csv(f'results\\{method_name}_{budget}.csv', index=False)

    # 打印结果
    for system, result in results.items():
        print(f"系统: {system} (种子: {result['Seed Used']})",f"  最佳配置:    [{', '.join(map(str, result['Best Solution']))}]",f"  最佳性能值: {result['Best Performance']}")



def main():




    arg_parser = ArgumentParser('智能软件工程实验室——智能软件调优实验')
    #arg_parser.add_argument('--datasets_folder', type=str, default='datasets', help='数据集文件夹')
    #arg_parser.add_argument('--output_folder', type=str, default='results\\search_results_bestconfig_fast', help='输出文件夹')
    #arg_parser.add_argument('--budget', type=int, default=100, help='搜索预算（尝试的配置数量）')
    arg_parser.add_argument('--global_seed', type=int, default=42, help='全局随机种子')
    #arg_parser.add_argument('--search_method', type=str, default='lab3_bestconfig_linear.bestconfig_fast_search', help='搜索方法')
    args = arg_parser.parse_args()
    
    print(" note: no bayesian search, because it is too slow( take 6h ?) you can try it by -> py -3.10 lab3_bayesian.py")

    all_search_method = [lab3_bestconfig_linear.bestconfig_fast_search, 
                         lab3_forier.fourier_search, 
                         lab3_baseline.random_search, 
                         lab3_bestconfig.bestconfig_search, 
                         lab3_flash.flash_search]
                        # no bayesian search, because it is too slow( take 6h ?) you can try it by -> py -3.10 lab3_bayesian.py
    all_search_budget = [20,50,100, 200,500, 1000]
    output_folder = "results\\search_results_bestconfig_fast"
    datasets_folder = "datasets"

    for search_method in all_search_method:
        for budget in all_search_budget:
            bestconfig_search(datasets_folder, budget, output_folder, search_method = search_method)

    print("All search methods and budgets are done!")

    print_results_different_search_methods()



def print_results_different_search_methods():
    pass

if __name__ == "__main__":
    main()
