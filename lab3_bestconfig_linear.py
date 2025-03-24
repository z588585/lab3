import additional_code 
import pandas as pd
import numpy as np
import sys
import os
from itertools import product


def bestconfig_fast_search(file_path, budget, output_file, k=5, top_n=1, random_seed=42):
    if budget < 50:
        k = 3
    if budget < 20:
        k = 1

    """
    使用优化版分治采样和递归边界搜索寻找最佳配置
    fast_dds=True 表示采用线性复杂度的 DDS,假设配置项影响独立
    
    参数:
        file_path: 数据集文件路径
        budget: 总采样预算
        output_file: 结果输出文件路径
        k: 参数空间划分数
        top_n: 返回前N个最佳解决方案
        random_seed: 随机种子
    """

    np.random.seed(random_seed)  # 设置随机种子，确保结果可重现
    data = pd.read_csv(file_path)
    config_columns = data.columns[:-1]
    performance_column = data.columns[-1]
    system_name = os.path.basename(file_path).split('.')[0]
    if system_name.lower() == "---":
        maximization = True  # 最大化问题
    else:
        maximization = False  # 最小化问题

    if maximization:
        worst_value = data[performance_column].min() / 2  # 对于缺失配置，使用最小值的一半
    
    else:
        worst_value = data[performance_column].max() * 2  # 对于缺失配置，使用最大值的两倍

    # 创建一个列表来存储前N个最佳配置
    top_configs = []  # 格式: [(性能值, [配置])]

    worst_value = data[performance_column].max() * 2
    best_performance = -np.inf if maximization else np.inf  # 根据问题类型设置初始值 - 无穷大或无穷小
    best_solution = []
    search_results = []

# === Stage 1: DDS - Divide and Diverge Sampling 分治发散采样阶段 ===
    intervals = {}
    for col in config_columns:
        unique_vals = sorted(data[col].unique())
        if len(unique_vals) <= k:
            intervals[col] = [[v] for v in unique_vals]
        else:
            chunks = np.array_split(unique_vals, k)
            intervals[col] = [list(chunk) for chunk in chunks]

    initial_samples = []

    # === 优化版本：线性复杂度 k*n 个采样 ===
    for idx, col in enumerate(config_columns):
        col_intervals = intervals[col]
        for i, subrange in enumerate(col_intervals):
            np.random.seed(random_seed + idx * 100 + i)
            val = int(np.random.choice(subrange))
            sample = [data[c].mode()[0] for c in config_columns]  # 初始默认配置
            sample[idx] = val
            initial_samples.append(sample)
            if len(initial_samples) >= budget:
                break
        if len(initial_samples) >= budget:
            break



    # 评估初始样本的性能
    for sampled_config in initial_samples:  # 遍历每个采样的配置
        # 检查该配置是否存在于数据集中
        matched_row = data.loc[(data[config_columns] == pd.Series(sampled_config, index=config_columns)).all(axis=1)]
        if not matched_row.empty:  # 如果配置存在于数据集中
            performance = matched_row[performance_column].iloc[0]  # 获取其性能值
        else:  # 如果配置不存在于数据集中
            performance = worst_value  # 使用最差性能值（最大值的两倍）
        
        # 更新前N个最佳配置列表
        update_top_configs(top_configs, sampled_config.copy(), performance, top_n, maximization)
        
        if maximization:
            if performance > best_performance:
                best_performance = performance
                best_solution = sampled_config
        else:
            if performance < best_performance:
                best_performance = performance
                best_solution = sampled_config

        
        search_results.append(sampled_config + [performance])  # 将配置及其性能添加到搜索结果

    remaining_budget = budget - len(initial_samples)  # 计算剩余采样预算


    if remaining_budget <= 0:
        print(f"warning: budget is too small or k is too large {len(initial_samples)} / {budget}, in {file_path}")
    elif remaining_budget < 50:
        print(f"Initial samples: {len(initial_samples)} / {budget}, in {file_path}")
    else:
        print(f"Initial samples: {len(initial_samples)} / {budget}")
        #pass



    # === Stage 2: RBS - Recursive Bound and Search 递归边界搜索阶段 ===
    current_best = best_solution  # 设置当前最佳配置为阶段1找到的最佳配置
    while remaining_budget > 0:  # 当还有采样预算时继续搜索
        bounded_data = data.copy()  # 复制原始数据集，用于边界限制
        for i, col in enumerate(config_columns):  # 遍历每个配置参数
            ci = current_best[i]  # 获取当前最佳配置中该参数的值
            # 找到小于当前值的最大值，如果没有则使用当前值
            smaller = data[data[col] < ci][col].max() if not data[data[col] < ci].empty else ci
            # 找到大于当前值的最小值，如果没有则使用当前值
            larger = data[data[col] > ci][col].min() if not data[data[col] > ci].empty else ci
            # 限制数据集在找到的上下边界内
            bounded_data = bounded_data[(bounded_data[col] >= smaller) & (bounded_data[col] <= larger)]

        # 在有限的边界内随机采样
        sampled_rows = bounded_data.sample(min(len(bounded_data), remaining_budget), replace=False)
        for _, row in sampled_rows.iterrows():  # 遍历采样的行
            config = [int(row[col]) for col in config_columns]  # 提取配置参数值
            performance = row[performance_column]  # 获取性能值
            
            # 更新前N个最佳配置列表
            update_top_configs(top_configs, config.copy(), performance, top_n, maximization)
            
            if performance < best_performance:  # 如果性能优于历史最佳
                best_performance = performance  # 更新最佳性能
                best_solution = config  # 更新最佳配置
                current_best = config  # 更新当前最佳配置，用于下一次递归搜索
            search_results.append(config + [performance])  # 记录搜索结果
            remaining_budget -= 1  # 减少剩余预算
            if remaining_budget <= 0:  # 如果预算用完则退出
                break

    # 保存结果
    columns = list(config_columns) + ["Performance"]
    pd.DataFrame(search_results, columns=columns).to_csv(output_file, index=False)
    
    # 保存top-n配置到单独文件
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
    更新前N个最佳配置列表
    
    参数:
        top_configs: 当前的前N个配置列表 [(性能值, [配置])]
        config: 新配置
        performance: 新配置的性能值
        top_n: 要保留的最佳配置数量
        maximization: 是否为最大化问题
    """
    # 如果列表未满或新配置优于列表中某个配置
    if len(top_configs) < top_n or (not maximization and performance < top_configs[-1][0]) or (maximization and performance > top_configs[0][0]):
        # 添加新配置
        top_configs.append((performance, config))
        
        # 按性能排序（最小化或最大化）
        if maximization:
            top_configs.sort(key=lambda x: x[0], reverse=True)  # 降序排列（最大化）
        else:
            top_configs.sort(key=lambda x: x[0])  # 升序排列（最小化）
        
        # 如果超出top_n限制，删除多余的配置
        if len(top_configs) > top_n:
            top_configs.pop()  # 删除最后一个（性能最差的）


# 在多个数据集上测试的主函数

if __name__ == "__main__":
    # 设置全局随机种子
    global_seed = 42
    
    datasets_folder = "datasets"  # 数据集文件夹
    output_folder = "results\\search_results_bestconfig_fast"  # 输出文件夹
    os.makedirs(output_folder, exist_ok=True)  # 创建输出文件夹（如果不存在）
    budget = 1000  # 搜索预算（尝试的配置数量）

    results = {}
    for file_name in os.listdir(datasets_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(datasets_folder, file_name)
            output_file = os.path.join(output_folder, f"{file_name.split('.')[0]}_search_results.csv")
            
            # 为每个文件使用不同的随机种子，但仍然保持可重现性

            file_seed = additional_code.get_deterministic_seed(file_name, global_seed)
            
            best_solution, best_performance = bestconfig_fast_search (file_path, budget, output_file, random_seed=file_seed)
         
            results[file_name] = {
                "Best Solution": best_solution,
                "Best Performance": best_performance,
                "Seed Used": file_seed  # 记录使用的种子
            }
    
    save_results_csv = pd.DataFrame(results)
    save_results_csv.to_csv('results\\bestconfig_fast_{}.csv'.format(budget), index=False)

    # 打印结果
    for system, result in results.items():
        print(f"系统: {system} (种子: {result['Seed Used']})",f"  最佳配置:    [{', '.join(map(str, result['Best Solution']))}]",f"  最佳性能值: {result['Best Performance']}")


