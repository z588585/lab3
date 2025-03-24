import additional_code 
import pandas as pd
import numpy as np
import os


# 定义随机搜索函数
def random_search(file_path, budget, output_file, random_seed=42):
    # 设置随机种子，确保结果可重现
    
    # 加载数据集
    data = pd.read_csv(file_path)
    # 识别配置列和性能列
    config_columns = data.columns[:-1]  # 除最后一列外都是配置参数
    performance_column = data.columns[-1]  # 最后一列是性能指标

    # 确定这是最大化问题还是最小化问题
    # 最大化吞吐量和最小化运行时间
    system_name = os.path.basename(file_path).split('.')[0]
    if system_name.lower() == "---":
        maximization = True  # 最大化问题
    else:
        maximization = False  # 最小化问题

    # 提取最佳和最差性能值
    if maximization:
        worst_value = data[performance_column].min() / 2  # 对于缺失配置，使用最小值的一半
    
    else:
        worst_value = data[performance_column].max() * 2  # 对于缺失配置，使用最大值的两倍
    

    # 初始化最佳解决方案和性能
    best_performance = -np.inf if maximization else np.inf  # 根据问题类型设置初始值 - 无穷大或无穷小
    best_solution = []

    # 存储所有搜索结果
    search_results = []

    for _ in range(budget):
        random_seed_in_loop = random_seed + _  # 为每次迭代生成不同的随机种子 
        np.random.seed(random_seed_in_loop)
        # 随机抽样一个配置
        # 对于每个配置列，从数据集中可用的唯一值中随机选择一个值
        # 这确保采样的配置在每个参数的有效域内
        sampled_config = [int(np.random.choice(data[col].unique())) for col in config_columns]

        # 检查该配置是否存在于数据集中
        # 从采样的配置创建一个Pandas Series并将其与数据集中的所有行进行匹配
        # .all(axis=1)确保匹配应用于所有配置列
        matched_row = data.loc[(data[config_columns] == pd.Series(sampled_config, index=config_columns)).all(axis=1)]

        if not matched_row.empty:
            # 现有配置
            performance = matched_row[performance_column].iloc[0]
        else:
            # 不存在的配置
            performance = worst_value

        # 更新最佳解决方案
        if maximization:
            if performance > best_performance:
                best_performance = performance
                best_solution = sampled_config
        else:
            if performance < best_performance:
                best_performance = performance
                best_solution = sampled_config

        # 记录当前搜索结果
        search_results.append(sampled_config + [performance])

    # 将搜索结果保存到CSV文件
    columns = list(config_columns) + ["Performance"]
    search_df = pd.DataFrame(search_results, columns=columns)
    search_df.to_csv(output_file, index=False)

    return [int(x) for x in best_solution], best_performance




if __name__ == "__main__":
    # 设置全局随机种子
    global_seed = 42
    
    datasets_folder = "datasets"  # 数据集文件夹
    output_folder = "results\\search_results_baseline"  # 输出文件夹
    os.makedirs(output_folder, exist_ok=True)  # 创建输出文件夹（如果不存在）
    budget = 1000  # 搜索预算（尝试的配置数量）

    results = {}
    for file_name in os.listdir(datasets_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(datasets_folder, file_name)
            output_file = os.path.join(output_folder, f"{file_name.split('.')[0]}_search_results.csv")
            
            # 为每个文件使用不同的随机种子，但仍然保持可重现性

            file_seed = additional_code.get_deterministic_seed(file_name, global_seed)
            
            best_solution, best_performance = random_search(file_path, budget, output_file, random_seed=file_seed)
            results[file_name] = {
                "Best Solution": best_solution,
                "Best Performance": best_performance,
                "Seed Used": file_seed  # 记录使用的种子
            }

    save_results_csv = pd.DataFrame(results)
    save_results_csv.to_csv('results\\baseline_{}.csv'.format(budget), index=False)
    # 打印结果
    for system, result in results.items():
        print(f"系统: {system} (种子: {result['Seed Used']})",f"  最佳配置:    [{', '.join(map(str, result['Best Solution']))}]",f"  最佳性能值: {result['Best Performance']}")


