#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合统计分析脚本：按场景、任务、ring、模型组合统计指标
优化版：从CSV的method_name字段提取真实模型名，避免命名不一致问题
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import re


def parse_metric_with_std(value_str):
    """解析 'mean±std' 格式的字符串"""
    if pd.isna(value_str) or value_str == '':
        return np.nan, np.nan
    try:
        parts = str(value_str).split('±')
        mean = float(parts[0])
        std = float(parts[1]) if len(parts) > 1 else 0.0
        return mean, std
    except:
        return np.nan, np.nan


def extract_scenario(task_name):
    """从任务名称提取场景"""
    task_lower = str(task_name).lower()
    if 'motion' in task_lower:
        return 'motion'
    elif 'stationary' in task_lower:
        return 'stationary'
    elif 'spo2' in task_lower:
        return 'spo2'
    else:
        return 'unknown'


def extract_base_task(task_name):
    """从任务名称提取基础任务类型"""
    task_lower = str(task_name).lower()
    
    if 'bp_dia' in task_lower:
        return 'BP_dia'
    elif 'bp_sys' in task_lower:
        return 'BP_sys'
    elif 'resp_rr' in task_lower or task_lower.startswith('rr'):
        return 'resp_rr'
    elif task_lower.startswith('hr'):
        return 'hr'
    elif 'spo2' in task_lower:
        return 'spo2'
    
    return task_name


def collect_all_results(csv_dir):
    """收集所有CSV结果，从CSV的method_name字段提取真实模型名"""
    results = []
    
    csv_path = Path(csv_dir)
    
    # 遍历所有模型目录
    for model_dir in csv_path.iterdir():
        if not model_dir.is_dir() or model_dir.name == '.git':
            continue
        
        # 从目录名提取ring类型
        if 'ring1' in model_dir.name:
            ring = 'ring1'
        elif 'ring2' in model_dir.name:
            ring = 'ring2'
        else:
            ring = 'unknown'
        
        # 遍历任务子目录
        for task_dir in model_dir.iterdir():
            if not task_dir.is_dir():
                continue
            
            scenario = extract_scenario(task_dir.name)
            base_task = extract_base_task(task_dir.name)
            
            # 查找CSV文件
            csv_files = list(task_dir.glob('*.csv'))
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    
                    # 只提取平均值行
                    avg_row = df[df['fold'].str.contains('Average', na=False)]
                    
                    if not avg_row.empty:
                        row_data = avg_row.iloc[0]
                        
                        # ✅ 关键修复：从CSV中获取真实的模型名（method_name字段）
                        actual_model = row_data.get('method_name', 'unknown')
                        
                        # 解析指标
                        mae_mean, mae_std = parse_metric_with_std(row_data.get('mae_with_std', ''))
                        rmse_mean, rmse_std = parse_metric_with_std(row_data.get('rmse_with_std', ''))
                        mape_mean, mape_std = parse_metric_with_std(row_data.get('mape_with_std', ''))
                        pearson_mean, pearson_std = parse_metric_with_std(row_data.get('pearson_with_std', ''))
                        
                        results.append({
                            'model': actual_model,  # 使用CSV中的真实模型名
                            'ring_type': ring,
                            'scenario': scenario,
                            'base_task': base_task,
                            'mae_mean': mae_mean,
                            'mae_std': mae_std,
                            'rmse_mean': rmse_mean,
                            'rmse_std': rmse_std,
                            'mape_mean': mape_mean,
                            'mape_std': mape_std,
                            'pearson_mean': pearson_mean,
                            'pearson_std': pearson_std,
                            'csv_path': str(csv_file),
                            'exp_name': row_data.get('exp_name', ''),
                            'dir_name': model_dir.name
                        })
                except Exception as e:
                    # 忽略格式不匹配的CSV文件
                    if 'fold' in str(e):
                        continue
                    print(f"⚠️  处理文件时遇到问题: {csv_file.name}")
    
    return pd.DataFrame(results)


def generate_statistics(df, group_by_cols, output_file, description=""):
    """生成统计表格"""
    if df.empty:
        print(f"⚠️  {description}: 无数据")
        return None
    
    # 定义指标列
    metric_cols = ['mae_mean', 'mae_std', 'rmse_mean', 'rmse_std', 
                   'mape_mean', 'mape_std', 'pearson_mean', 'pearson_std']
    
    # 按指定列分组统计
    stats = df.groupby(group_by_cols)[metric_cols].agg(['mean', 'std', 'count'])
    
    # 展平多级列索引
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    
    # 重置索引
    stats = stats.reset_index()
    
    # 保存结果
    stats.to_csv(output_file, index=False)
    print(f"✅ {description}: {output_file}")
    
    return stats


def generate_detailed_table(df, output_file):
    """生成详细的模型性能对比表"""
    if df.empty:
        print(f"⚠️  详细对比表: 无数据")
        return None
    
    # 按场景、任务、ring、模型分组
    detailed = df.pivot_table(
        index=['scenario', 'base_task', 'ring_type'],
        columns='model',
        values=['mae_mean', 'rmse_mean', 'mape_mean', 'pearson_mean'],
        aggfunc='first'
    )
    
    detailed.to_csv(output_file)
    print(f"✅ 详细对比表: {output_file}")
    
    return detailed


def main():
    csv_dir = '/root/RingTool/csv'
    
    print("=" * 80)
    print("               📊 场景×任务×Ring×模型 综合统计分析")
    print("=" * 80)
    print(f"\n数据源: {csv_dir}")
    print("\n🔍 开始收集实验结果...")
    
    all_results = collect_all_results(csv_dir)
    
    if all_results.empty:
        print("❌ 未找到任何结果数据！")
        return
    
    print(f"\n✅ 共收集到 {len(all_results)} 条结果记录")
    
    # 显示模型分布
    print("\n📋 模型分布:")
    model_counts = all_results['model'].value_counts().sort_index()
    for model, count in model_counts.items():
        print(f"  • {model:15} : {count:3}条")
    
    print("\n" + "=" * 80)
    print("                    生成统计报表")
    print("=" * 80)
    
    # 1. 按场景+任务+ring统计（跨模型）
    generate_statistics(
        all_results,
        ['scenario', 'base_task', 'ring_type'],
        'stats_by_scenario_task_ring.csv',
        "按场景+任务+Ring统计"
    )
    
    # 2. 按场景+ring统计
    generate_statistics(
        all_results,
        ['scenario', 'ring_type'],
        'stats_by_scenario_ring.csv',
        "按场景+Ring统计"
    )
    
    # 3. 按任务+ring统计
    generate_statistics(
        all_results,
        ['base_task', 'ring_type'],
        'stats_by_task_ring.csv',
        "按任务+Ring统计"
    )
    
    # 4. 按模型+任务统计
    generate_statistics(
        all_results,
        ['model', 'base_task'],
        'stats_by_model_task.csv',
        "按模型+任务统计"
    )
    
    # 5. 按模型+场景统计
    generate_statistics(
        all_results,
        ['model', 'scenario'],
        'stats_by_model_scenario.csv',
        "按模型+场景统计"
    )
    
    # 6. 按模型+ring统计
    generate_statistics(
        all_results,
        ['model', 'ring_type'],
        'stats_by_model_ring.csv',
        "按模型+Ring统计"
    )
    
    # 7. 按场景+任务+ring+模型的详细对比表
    generate_detailed_table(
        all_results,
        'detailed_model_comparison.csv'
    )
    
    # 8. 保存原始汇总数据
    all_results.to_csv('all_results_summary.csv', index=False)
    print(f"✅ 原始汇总数据: all_results_summary.csv")
    
    # 9. 生成最佳模型报告
    print(f"\n✅ 生成最佳模型分析...")
    best_models = []
    
    for (scenario, task, ring), group in all_results.groupby(['scenario', 'base_task', 'ring_type']):
        if not group.empty:
            # 按MAE找最佳
            best_idx = group['mae_mean'].idxmin()
            best_row = group.loc[best_idx]
            
            best_models.append({
                'scenario': scenario,
                'base_task': task,
                'ring_type': ring,
                'best_model_by_mae': best_row['model'],
                'best_mae': best_row['mae_mean'],
                'best_rmse': best_row['rmse_mean'],
                'best_mape': best_row['mape_mean'],
                'best_pearson': best_row['pearson_mean']
            })
    
    best_df = pd.DataFrame(best_models)
    best_df.to_csv('best_models_by_scenario_task_ring.csv', index=False)
    print(f"✅ 最佳模型报告: best_models_by_scenario_task_ring.csv")
    
    print("\n" + "=" * 80)
    print("                      ✅ 统计完成!")
    print("=" * 80)
    print("\n📁 生成的文件:")
    print("  1. stats_by_scenario_task_ring.csv")
    print("  2. stats_by_scenario_ring.csv")
    print("  3. stats_by_task_ring.csv")
    print("  4. stats_by_model_task.csv")
    print("  5. stats_by_model_scenario.csv")
    print("  6. stats_by_model_ring.csv")
    print("  7. detailed_model_comparison.csv")
    print("  8. all_results_summary.csv")
    print("  9. best_models_by_scenario_task_ring.csv")
    print("=" * 80)


if __name__ == '__main__':
    main()

