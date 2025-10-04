#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical Analysis for Wearable Device Performance Evaluation
Computes performance metrics across models, tasks, scenarios, rings, and channels
All metrics reported as weighted averages (by sample size) with standard deviations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def calculate_metrics(predictions, targets):
    """
    Calculate regression metrics for prediction evaluation
    
    Args:
        predictions: Array of predicted values
        targets: Array of ground truth values
    
    Returns:
        Dictionary containing mae, rmse, mape, and pearson correlation
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    mask = ~(np.isnan(predictions) | np.isnan(targets))
    predictions = predictions[mask]
    targets = targets[mask]
    
    if len(predictions) == 0:
        return {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'pearson': np.nan}
    
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    mask_nonzero = targets != 0
    if np.sum(mask_nonzero) > 0:
        mape = np.mean(np.abs((targets[mask_nonzero] - predictions[mask_nonzero]) / targets[mask_nonzero])) * 100
    else:
        mape = np.nan
    
    if len(predictions) > 1 and np.std(predictions) > 0 and np.std(targets) > 0:
        pearson, _ = stats.pearsonr(predictions, targets)
    else:
        pearson = np.nan
    
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'pearson': pearson}


def get_scenario_group(scenario):
    """
    Categorize scenarios into groups
    
    Groups:
        motion: deepsquat, striding
        spo2: spo2
        stationary: sitting, talking, shaking_head, standing
    """
    scenario = str(scenario).lower()
    if scenario in ['deepsquat', 'striding']:
        return 'motion'
    elif scenario == 'spo2':
        return 'spo2'
    elif scenario in ['sitting', 'talking', 'shaking_head', 'standing']:
        return 'stationary'
    else:
        return 'other'


def normalize_model_name(model_name):
    """Standardize model names for consistency"""
    model_name = str(model_name).lower().strip()
    if model_name in ['inceptiontime', 'inception_time']:
        return 'inception-time'
    return model_name


def parse_experiment_name(exp_dir_name):
    """
    Parse experiment directory name to extract configuration
    
    Expected format: {model}-{ring}-{task}-{setting}-{channel}
    Example: resnet-ring1-hr-all-irred
    """
    parts = exp_dir_name.split('-')
    
    ring = None
    for part in parts:
        if 'ring' in part:
            ring = part
            break
    
    if ring:
        ring_idx = parts.index(ring)
        model = '-'.join(parts[:ring_idx])
    else:
        model = parts[0]
        ring = 'unknown'
    
    task = 'unknown'
    if 'bp' in exp_dir_name.lower():
        task = 'bp'
    elif 'hr' in exp_dir_name.lower():
        task = 'hr'
    elif 'rr' in exp_dir_name.lower():
        task = 'rr'
    elif 'spo2' in exp_dir_name.lower():
        task = 'spo2'
    
    channel = 'unknown'
    if 'all-' in exp_dir_name:
        channel = exp_dir_name.split('all-')[1]
    elif '-spo2-' in exp_dir_name:
        idx = exp_dir_name.find('-spo2-')
        channel = exp_dir_name[idx+6:]
    
    return {'model': model, 'ring': ring, 'task': task, 'channel': channel}


def process_predictions_directory(predictions_dir):
    """
    Process all prediction CSV files in the directory structure
    
    Returns:
        DataFrame with fold-level results for all experiments
    """
    predictions_path = Path(predictions_dir)
    all_results = []
    
    print("\nProcessing prediction files...")
    
    for exp_dir in sorted(predictions_path.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        exp_info = parse_experiment_name(exp_dir.name)
        
        csv_files = sorted(exp_dir.glob('Fold-*.csv'))
        if len(csv_files) == 0:
            continue
        
        fold_metrics = []
        
        for csv_file in csv_files:
            fold_num = csv_file.stem.split('-')[1]
            
            try:
                df = pd.read_csv(csv_file)
                
                if 'prediction' not in df.columns or 'target' not in df.columns:
                    continue
                
                if 'task' in df.columns:
                    tasks_in_csv = df['task'].unique()
                    
                    for task_name in tasks_in_csv:
                        task_df = df[df['task'] == task_name]
                        
                        if 'scenario' in task_df.columns:
                            scenarios = task_df['scenario'].unique()
                            
                            for scenario in scenarios:
                                scenario_df = task_df[task_df['scenario'] == scenario]
                                metrics = calculate_metrics(
                                    scenario_df['prediction'].values,
                                    scenario_df['target'].values
                                )
                                
                                fold_metrics.append({
                                    'model': exp_info['model'],
                                    'ring': exp_info['ring'],
                                    'task': task_name,
                                    'scenario': scenario,
                                    'scenario_group': get_scenario_group(scenario),
                                    'channel': exp_info['channel'],
                                    'fold': int(fold_num),
                                    'mae': metrics['mae'],
                                    'rmse': metrics['rmse'],
                                    'mape': metrics['mape'],
                                    'pearson': metrics['pearson'],
                                    'n_samples': len(scenario_df)
                                })
                
            except Exception as e:
                continue
        
        all_results.extend(fold_metrics)
    
    print(f"  Collected {len(all_results)} fold-level records")
    return pd.DataFrame(all_results)


def aggregate_by_folds(df):
    """
    Aggregate fold-level results to compute mean and standard deviation
    
    Each row in output represents the average across 5 folds for one
    specific configuration (model, task, scenario, channel, ring)
    """
    if df.empty:
        return pd.DataFrame()
    
    df['model'] = df['model'].apply(normalize_model_name)
    
    group_cols = ['model', 'ring', 'task', 'scenario', 'scenario_group', 'channel']
    
    agg_results = []
    
    for group_keys, group_df in df.groupby(group_cols):
        result = dict(zip(group_cols, group_keys))
        
        result['mae_mean'] = group_df['mae'].mean()
        result['mae_std'] = group_df['mae'].std()
        result['rmse_mean'] = group_df['rmse'].mean()
        result['rmse_std'] = group_df['rmse'].std()
        result['mape_mean'] = group_df['mape'].mean()
        result['mape_std'] = group_df['mape'].std()
        result['pearson_mean'] = group_df['pearson'].mean()
        result['pearson_std'] = group_df['pearson'].std()
        result['n_folds'] = len(group_df)
        result['total_samples'] = group_df['n_samples'].sum()
        
        agg_results.append(result)
    
    return pd.DataFrame(agg_results)


def save_with_description(df, filepath, description):
    """Save DataFrame to CSV with metadata description in separate file"""
    df.to_csv(filepath, index=False)
    print(f"  {filepath.name}")
    
    desc_file = filepath.parent / 'TABLE_DESCRIPTIONS.txt'
    with open(desc_file, 'a', encoding='utf-8') as f:
        f.write(f"{filepath.name}\n  {description}\n\n")


def generate_statistics_by_channel(df, output_dir):
    """
    Generate Table 1: Performance statistics by channel
    
    Each channel evaluated independently across all scenarios
    Weighted average by sample size
    """
    output_path = Path(output_dir)
    
    print("\n[Table 1] By Channel")
    
    channels = sorted(df['channel'].unique())
    
    for channel in channels:
        channel_df = df[df['channel'] == channel]
        
        stats_list = []
        for (model, task, ring), group in channel_df.groupby(['model', 'task', 'ring']):
            total_samples = group['total_samples'].sum()
            
            if total_samples > 0:
                mae_weighted = (group['mae_mean'] * group['total_samples']).sum() / total_samples
                mae_std = group['mae_std'].mean()
                rmse_weighted = (group['rmse_mean'] * group['total_samples']).sum() / total_samples
                rmse_std = group['rmse_std'].mean()
                mape_weighted = (group['mape_mean'] * group['total_samples']).sum() / total_samples
                mape_std = group['mape_std'].mean()
                pearson_weighted = (group['pearson_mean'] * group['total_samples']).sum() / total_samples
                pearson_std = group['pearson_std'].mean()
                
                stats_list.append({
                    'model': model,
                    'task': task,
                    'ring': ring,
                    'mae': f"{mae_weighted:.2f}±{mae_std:.2f}",
                    'rmse': f"{rmse_weighted:.2f}±{rmse_std:.2f}",
                    'mape': f"{mape_weighted:.2f}±{mape_std:.2f}",
                    'pearson': f"{pearson_weighted:.3f}±{pearson_std:.3f}",
                    'n_samples': int(total_samples)
                })
        
        stats_df = pd.DataFrame(stats_list)
        output_file = output_path / f'table1_{channel}_weighted.csv'
        
        description = f"Channel: {channel} | Calculation: Weighted average across all scenarios | Units: MAE/RMSE in task units, MAPE in %, Pearson [-1,1]"
        save_with_description(stats_df, output_file, description)


def generate_statistics_by_scenario_group(df, output_dir):
    """
    Generate Table 2: Performance statistics by scenario group and channel
    
    Each scenario group (motion/spo2/stationary) evaluated separately for each channel
    Weighted average across scenarios within the group
    """
    output_path = Path(output_dir)
    
    print("\n[Table 2] By Scenario Group x Channel")
    
    scenario_groups = ['motion', 'spo2', 'stationary']
    channels = sorted(df['channel'].unique())
    
    for group_name in scenario_groups:
        for channel in channels:
            subset_df = df[(df['scenario_group'] == group_name) & (df['channel'] == channel)]
            
            if subset_df.empty:
                continue
            
            stats_list = []
            for (model, task, ring), group in subset_df.groupby(['model', 'task', 'ring']):
                total_samples = group['total_samples'].sum()
                
                if total_samples > 0:
                    mae_weighted = (group['mae_mean'] * group['total_samples']).sum() / total_samples
                    mae_std = group['mae_std'].mean()
                    rmse_weighted = (group['rmse_mean'] * group['total_samples']).sum() / total_samples
                    rmse_std = group['rmse_std'].mean()
                    mape_weighted = (group['mape_mean'] * group['total_samples']).sum() / total_samples
                    mape_std = group['mape_std'].mean()
                    pearson_weighted = (group['pearson_mean'] * group['total_samples']).sum() / total_samples
                    pearson_std = group['pearson_std'].mean()
                    
                    stats_list.append({
                        'model': model,
                        'task': task,
                        'ring': ring,
                        'mae': f"{mae_weighted:.2f}±{mae_std:.2f}",
                        'rmse': f"{rmse_weighted:.2f}±{rmse_std:.2f}",
                        'mape': f"{mape_weighted:.2f}±{mape_std:.2f}",
                        'pearson': f"{pearson_weighted:.3f}±{pearson_std:.3f}",
                        'n_samples': int(total_samples)
                    })
            
            if not stats_list:
                continue
            
            stats_df = pd.DataFrame(stats_list)
            output_file = output_path / f'table2_{group_name}_{channel}.csv'
            
            description = f"Scenario Group: {group_name} | Channel: {channel} | Calculation: Weighted average across scenarios in group | Units: MAE/RMSE in task units, MAPE in %"
            save_with_description(stats_df, output_file, description)


def generate_channel_comparison_table(df, output_dir):
    """
    Generate Table 3: Side-by-side channel comparison
    
    Pivots data to show all channels for each model in one row
    Facilitates direct comparison of channel configurations
    """
    output_path = Path(output_dir)
    
    print("\n[Table 3] Channel Comparison")
    
    stats_list = []
    for (model, task, ring, channel), group in df.groupby(['model', 'task', 'ring', 'channel']):
        total_samples = group['total_samples'].sum()
        
        if total_samples > 0:
            mae_weighted = (group['mae_mean'] * group['total_samples']).sum() / total_samples
            mae_std = group['mae_std'].mean()
            rmse_weighted = (group['rmse_mean'] * group['total_samples']).sum() / total_samples
            rmse_std = group['rmse_std'].mean()
            mape_weighted = (group['mape_mean'] * group['total_samples']).sum() / total_samples
            mape_std = group['mape_std'].mean()
            pearson_weighted = (group['pearson_mean'] * group['total_samples']).sum() / total_samples
            pearson_std = group['pearson_std'].mean()
            
            stats_list.append({
                'model': model,
                'task': task,
                'ring': ring,
                'channel': channel,
                'mae': f"{mae_weighted:.2f}±{mae_std:.2f}",
                'rmse': f"{rmse_weighted:.2f}±{rmse_std:.2f}",
                'mape': f"{mape_weighted:.2f}±{mape_std:.2f}",
                'pearson': f"{pearson_weighted:.3f}±{pearson_std:.3f}",
                'n_samples': int(total_samples)
            })
    
    stats_df = pd.DataFrame(stats_list)
    
    # Create pivot table
    pivot_list = []
    for (model, task, ring), group in stats_df.groupby(['model', 'task', 'ring']):
        row = {'model': model, 'task': task, 'ring': ring}
        
        for _, ch_row in group.iterrows():
            ch = ch_row['channel']
            row[f'{ch}_mae'] = ch_row['mae']
            row[f'{ch}_mape'] = ch_row['mape']
            row[f'{ch}_samples'] = ch_row['n_samples']
        
        pivot_list.append(row)
    
    pivot_df = pd.DataFrame(pivot_list)
    output_file = output_path / 'table3_channel_comparison.csv'
    
    description = "Channel Comparison | Calculation: Weighted average across scenarios for each channel | Format: Side-by-side comparison of all channels"
    save_with_description(pivot_df, output_file, description)


def generate_all_channels_summary(df, output_dir):
    """
    Generate Table 4: Overall performance summary
    Note: This table is retained but NOT recommended for primary analysis
    as it mixes different input configurations
    """
    output_path = Path(output_dir)
    
    print("\n[Table 4] Overall Summary (reference only)")
    
    stats_list = []
    for (model, task, ring), group in df.groupby(['model', 'task', 'ring']):
        total_samples = group['total_samples'].sum()
        
        if total_samples > 0:
            mae_weighted = (group['mae_mean'] * group['total_samples']).sum() / total_samples
            mae_std = group['mae_std'].mean()
            rmse_weighted = (group['rmse_mean'] * group['total_samples']).sum() / total_samples
            rmse_std = group['rmse_std'].mean()
            mape_weighted = (group['mape_mean'] * group['total_samples']).sum() / total_samples
            mape_std = group['mape_std'].mean()
            pearson_weighted = (group['pearson_mean'] * group['total_samples']).sum() / total_samples
            pearson_std = group['pearson_std'].mean()
            
            stats_list.append({
                'model': model,
                'task': task,
                'ring': ring,
                'mae': f"{mae_weighted:.2f}±{mae_std:.2f}",
                'rmse': f"{rmse_weighted:.2f}±{rmse_std:.2f}",
                'mape': f"{mape_weighted:.2f}±{mape_std:.2f}",
                'pearson': f"{pearson_weighted:.3f}±{pearson_std:.3f}",
                'n_samples': int(total_samples),
                'n_channels': len(group['channel'].unique())
            })
    
    stats_df = pd.DataFrame(stats_list)
    output_file = output_path / 'table4_overall_weighted.csv'
    
    description = "Overall Performance | Calculation: Weighted average across all channels and scenarios | Note: Mixes different input configurations - use Table 1 for channel-specific analysis"
    save_with_description(stats_df, output_file, description)


def generate_best_models_table(df, output_dir):
    """
    Generate Table 5: Best model recommendations
    
    Identifies optimal model for each task, channel, scenario group, and ring combination
    Based on lowest weighted MAE
    """
    output_path = Path(output_dir)
    
    print("\n[Table 4] Best Model Recommendations")
    
    best_models = []
    
    for (channel, task, scenario_group, ring), group in df.groupby(['channel', 'task', 'scenario_group', 'ring']):
        model_stats = []
        for model in group['model'].unique():
            model_group = group[group['model'] == model]
            total_samples = model_group['total_samples'].sum()
            
            if total_samples > 0:
                mae = (model_group['mae_mean'] * model_group['total_samples']).sum() / total_samples
                model_stats.append({
                    'model': model,
                    'mae': mae,
                    'samples': total_samples
                })
        
        if model_stats:
            best = min(model_stats, key=lambda x: x['mae'])
            best_models.append({
                'task': task,
                'channel': channel,
                'scenario_group': scenario_group,
                'ring': ring,
                'best_model': best['model'],
                'mae': round(best['mae'], 2),
                'n_samples': best['samples']
            })
    
    best_df = pd.DataFrame(best_models)
    output_file = output_path / 'table4_best_models.csv'
    
    description = "Best Model Recommendations | Criterion: Lowest weighted MAE | Recommendation for each task+channel+scenario_group+ring combination"
    save_with_description(best_df, output_file, description)


def save_raw_data(fold_df, agg_df, output_dir):
    """Save raw data files for reference and verification"""
    output_path = Path(output_dir)
    
    print("\n[Raw Data]")
    
    output_file = output_path / 'raw_fold_level.csv'
    description = "Raw Fold-level Data | Each row represents one fold of one experimental configuration | Not aggregated"
    save_with_description(fold_df, output_file, description)
    
    output_file = output_path / 'raw_aggregated.csv'
    description = "Raw Aggregated Data | Each row represents 5-fold average for one configuration | Basis for all statistical tables | Columns: mean and std across 5 folds"
    save_with_description(agg_df, output_file, description)


def main():
    base_dir = Path(__file__).parent.parent
    predictions_dir = base_dir / 'predictions'
    output_dir = base_dir / 'statistics_results'
    
    print("="*80)
    print("Statistical Analysis for Wearable Device Performance")
    print("="*80)
    print(f"\nInput: {predictions_dir}")
    print(f"Output: {output_dir}\n")
    
    print("Step 1: Processing prediction files...")
    fold_df = process_predictions_directory(predictions_dir)
    
    if fold_df.empty:
        print("\nError: No data found")
        return
    
    print("\nStep 2: Aggregating across folds...")
    agg_df = aggregate_by_folds(fold_df)
    
    if agg_df.empty:
        print("\nError: Aggregation failed")
        return
    
    print(f"  Generated {len(agg_df)} aggregated records")
    print(f"\n  Models: {sorted(agg_df['model'].unique())}")
    print(f"  Tasks: {sorted(agg_df['task'].unique())}")
    print(f"  Rings: {sorted(agg_df['ring'].unique())}")
    print(f"  Channels: {sorted(agg_df['channel'].unique())}")
    print(f"  Scenario Groups: {sorted(agg_df['scenario_group'].unique())}")
    
    print("\n" + "="*80)
    print("Step 3: Generating statistical tables...")
    print("="*80)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize description file
    desc_file = output_path / 'TABLE_DESCRIPTIONS.txt'
    if desc_file.exists():
        desc_file.unlink()
    
    with open(desc_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("TABLE DESCRIPTIONS\n")
        f.write("="*80 + "\n\n")
    
    generate_statistics_by_channel(agg_df, output_dir)
    generate_statistics_by_scenario_group(agg_df, output_dir)
    generate_channel_comparison_table(agg_df, output_dir)
    generate_best_models_table(agg_df, output_dir)
    save_raw_data(fold_df, agg_df, output_dir)
    
    print("\n" + "="*80)
    print("Statistics Complete")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print("\n  Table 1: By Channel (4 files)")
    for ch in sorted(agg_df['channel'].unique()):
        print(f"    table1_{ch}_weighted.csv")
    print("\n  Table 2: By Scenario Group x Channel (12 files)")
    print("    Format: table2_[scenario_group]_[channel].csv")
    print("    Example: table2_stationary_irred.csv")
    print("\n  Table 3-4: Comparisons and Recommendations")
    print("    table3_channel_comparison.csv - Side-by-side channel comparison")
    print("    table4_best_models.csv - Best model recommendations")
    print("\n  Raw Data:")
    print("    raw_aggregated.csv - 5-fold averaged (basis for all tables)")
    print("    raw_fold_level.csv - Individual fold results")
    print("\n  Documentation:")
    print("    TABLE_DESCRIPTIONS.txt - Detailed description of each table")
    print("="*80)


if __name__ == '__main__':
    main()
