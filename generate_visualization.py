#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成统计可视化图表
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
sns.set_style("whitegrid")
sns.set_palette("husl")

# 创建输出目录
output_dir = Path('statistics_plots')
output_dir.mkdir(exist_ok=True)


def plot_model_comparison():
    """模型性能对比图"""
    df = pd.read_csv('stats_by_model_task.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # MAE对比
    pivot_mae = df.pivot_table(values='mae_mean_mean', index='base_task', columns='model')
    pivot_mae.plot(kind='bar', ax=axes[0, 0], width=0.8)
    axes[0, 0].set_title('MAE Comparison by Model and Task', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Task')
    axes[0, 0].set_ylabel('MAE (mean)')
    axes[0, 0].legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # RMSE对比
    pivot_rmse = df.pivot_table(values='rmse_mean_mean', index='base_task', columns='model')
    pivot_rmse.plot(kind='bar', ax=axes[0, 1], width=0.8)
    axes[0, 1].set_title('RMSE Comparison by Model and Task', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Task')
    axes[0, 1].set_ylabel('RMSE (mean)')
    axes[0, 1].legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # MAPE对比
    pivot_mape = df.pivot_table(values='mape_mean_mean', index='base_task', columns='model')
    pivot_mape.plot(kind='bar', ax=axes[1, 0], width=0.8)
    axes[1, 0].set_title('MAPE Comparison by Model and Task', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Task')
    axes[1, 0].set_ylabel('MAPE (mean, %)')
    axes[1, 0].legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Pearson相关系数对比
    pivot_pearson = df.pivot_table(values='pearson_mean_mean', index='base_task', columns='model')
    pivot_pearson.plot(kind='bar', ax=axes[1, 1], width=0.8)
    axes[1, 1].set_title('Pearson Correlation by Model and Task', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Task')
    axes[1, 1].set_ylabel('Pearson Correlation (mean)')
    axes[1, 1].legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ {output_dir / 'model_comparison.png'}")


def plot_scenario_comparison():
    """场景性能对比图"""
    df = pd.read_csv('stats_by_model_scenario.csv')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # MAE对比
    pivot_mae = df.pivot_table(values='mae_mean_mean', index='scenario', columns='model')
    pivot_mae.plot(kind='bar', ax=axes[0], width=0.8)
    axes[0].set_title('MAE Comparison by Model and Scenario', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Scenario')
    axes[0].set_ylabel('MAE (mean)')
    axes[0].legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].tick_params(axis='x', rotation=45)
    
    # RMSE对比
    pivot_rmse = df.pivot_table(values='rmse_mean_mean', index='scenario', columns='model')
    pivot_rmse.plot(kind='bar', ax=axes[1], width=0.8)
    axes[1].set_title('RMSE Comparison by Model and Scenario', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Scenario')
    axes[1].set_ylabel('RMSE (mean)')
    axes[1].legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scenario_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ {output_dir / 'scenario_comparison.png'}")


def plot_ring_comparison():
    """Ring对比图"""
    df = pd.read_csv('stats_by_task_ring.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # MAE对比
    pivot_mae = df.pivot_table(values='mae_mean_mean', index='base_task', columns='ring_type')
    pivot_mae.plot(kind='bar', ax=axes[0, 0], width=0.6)
    axes[0, 0].set_title('MAE: Ring1 vs Ring2 by Task', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Task')
    axes[0, 0].set_ylabel('MAE (mean)')
    axes[0, 0].legend(title='Ring Type')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # RMSE对比
    pivot_rmse = df.pivot_table(values='rmse_mean_mean', index='base_task', columns='ring_type')
    pivot_rmse.plot(kind='bar', ax=axes[0, 1], width=0.6)
    axes[0, 1].set_title('RMSE: Ring1 vs Ring2 by Task', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Task')
    axes[0, 1].set_ylabel('RMSE (mean)')
    axes[0, 1].legend(title='Ring Type')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # MAPE对比
    pivot_mape = df.pivot_table(values='mape_mean_mean', index='base_task', columns='ring_type')
    pivot_mape.plot(kind='bar', ax=axes[1, 0], width=0.6)
    axes[1, 0].set_title('MAPE: Ring1 vs Ring2 by Task', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Task')
    axes[1, 0].set_ylabel('MAPE (mean, %)')
    axes[1, 0].legend(title='Ring Type')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Pearson对比
    pivot_pearson = df.pivot_table(values='pearson_mean_mean', index='base_task', columns='ring_type')
    pivot_pearson.plot(kind='bar', ax=axes[1, 1], width=0.6)
    axes[1, 1].set_title('Pearson: Ring1 vs Ring2 by Task', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Task')
    axes[1, 1].set_ylabel('Pearson Correlation (mean)')
    axes[1, 1].legend(title='Ring Type')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ring_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ {output_dir / 'ring_comparison.png'}")


def plot_best_models_heatmap():
    """最佳模型热力图"""
    df = pd.read_csv('best_models_by_scenario_task_ring.csv')
    
    # 为每个模型分配一个数字
    models = df['best_model_by_mae'].unique()
    model_to_num = {model: i for i, model in enumerate(sorted(models))}
    
    # 创建数据透视表
    df['model_num'] = df['best_model_by_mae'].map(model_to_num)
    
    for ring in ['ring1', 'ring2']:
        ring_df = df[df['ring_type'] == ring]
        pivot = ring_df.pivot_table(values='model_num', index='scenario', columns='base_task', aggfunc='first')
        
        plt.figure(figsize=(10, 6))
        
        # 使用自定义颜色映射
        cmap = sns.color_palette("Set2", n_colors=len(models))
        
        ax = sns.heatmap(pivot, annot=False, cmap=cmap, cbar=False, 
                        linewidths=1, linecolor='white')
        
        # 添加模型名称标注
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                if not np.isnan(val):
                    model_name = [k for k, v in model_to_num.items() if v == val][0]
                    ax.text(j + 0.5, i + 0.5, model_name, 
                           ha='center', va='center', fontsize=9, fontweight='bold')
        
        plt.title(f'Best Model by MAE - {ring.upper()}', fontsize=14, fontweight='bold')
        plt.xlabel('Task')
        plt.ylabel('Scenario')
        plt.tight_layout()
        plt.savefig(output_dir / f'best_models_heatmap_{ring}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ {output_dir / f'best_models_heatmap_{ring}.png'}")


def plot_mae_distribution():
    """MAE分布箱线图"""
    df = pd.read_csv('all_results_summary.csv')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 按模型分布
    df_sorted = df.sort_values('model')
    sns.boxplot(data=df_sorted, x='model', y='mae_mean', ax=axes[0])
    axes[0].set_title('MAE Distribution by Model', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('MAE')
    axes[0].tick_params(axis='x', rotation=45)
    
    # 按任务分布
    df_sorted = df.sort_values('base_task')
    sns.boxplot(data=df_sorted, x='base_task', y='mae_mean', ax=axes[1])
    axes[1].set_title('MAE Distribution by Task', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Task')
    axes[1].set_ylabel('MAE')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mae_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ {output_dir / 'mae_distribution.png'}")


def main():
    print("=" * 80)
    print("               📊 生成可视化图表")
    print("=" * 80)
    
    plot_model_comparison()
    plot_scenario_comparison()
    plot_ring_comparison()
    plot_best_models_heatmap()
    plot_mae_distribution()
    
    print("\n" + "=" * 80)
    print(f"✅ 所有图表已生成在 {output_dir} 目录中")
    print("=" * 80)


if __name__ == '__main__':
    main()
