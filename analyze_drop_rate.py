#!/usr/bin/env python3
"""
统计各场景数据加载时的剔除率
分析质量过滤、长度过滤、生理范围过滤对不同场景的影响
"""
import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd

from constants.dataset import ALL_SCENARIOS, DatasetType
from dataset.load_dataset import RingToolDataset
from utils.utils import physiological_filter


def analyze_scenario_drop_rate(data_path: str, config_path: str, ring_type: str = None) -> Dict:
    """
    分析各场景的样本剔除率
    
    Args:
        data_path: 数据目录路径
        config_path: 配置文件路径
        ring_type: 指定ring类型 ('ring1' 或 'ring2')，如果为None则使用config中的设置
    
    Returns:
        统计结果字典
    """
    # 加载配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 确定ring类型
    if ring_type is None:
        ring_type = config["dataset"]["ring_type"]
    
    # 加载所有数据
    all_data = {}
    
    for filename in os.listdir(data_path):
        if filename.endswith('.pkl') and ring_type in filename:
            file_path = os.path.join(data_path, filename)
            try:
                data = pd.read_pickle(file_path)
                subject_id = filename.split('_')[0]
                all_data[subject_id] = data
            except Exception as e:
                logging.error(f"Error loading {filename}: {e}")
                continue
    
    all_data_df = pd.concat(all_data.values())
    logging.info(f"[{ring_type}] Loaded {len(all_data)} subjects, total {len(all_data_df)} raw samples")
    
    # 获取配置参数
    channels = config["dataset"]["input_type"]
    task = config["dataset"]["label_type"][0]
    target_fs = config.get("dataset", {}).get("target_fs", 100)
    window_duration = config.get("dataset", {}).get("window_duration", 30)
    target_length = target_fs * window_duration
    min_length = int(target_length * 0.95)
    quality_th = config.get("dataset", {}).get("quality_assessment", {}).get("th", 0)
    
    # 统计各场景的剔除情况
    results = {}
    
    for scenario in ALL_SCENARIOS:
        scenario_data = all_data_df[all_data_df['Label'] == scenario]
        
        stats = {
            'total': len(scenario_data),
            'filtered_quality': 0,
            'filtered_length': 0,
            'filtered_invalid_data': 0,
            'filtered_label': 0,
            'filtered_physiological': 0,
            'kept': 0,
        }
        
        for i in range(len(scenario_data)):
            row = scenario_data.iloc[i]
            
            # 质量过滤
            if row.get('ir-quality', 0) < quality_th or row.get('red-quality', 0) < quality_th:
                stats['filtered_quality'] += 1
                continue
            
            # 检查通道数据
            skip_invalid = False
            for channel in channels:
                channel_data = row.get(channel)
                if not isinstance(channel_data, np.ndarray):
                    skip_invalid = True
                    break
                if len(channel_data) < min_length:
                    stats['filtered_length'] += 1
                    skip_invalid = True
                    break
            
            if skip_invalid:
                if stats['filtered_length'] == 0:
                    stats['filtered_invalid_data'] += 1
                continue
            
            # 检查标签
            label_value = row.get(task)
            if label_value is None or (pd.isna(label_value) if isinstance(label_value, float) else False):
                stats['filtered_label'] += 1
                continue
            
            # 生理范围过滤
            try:
                mask = physiological_filter(label_value, task, behavior="mask")
                if isinstance(mask, np.ndarray):
                    if not mask.any():
                        stats['filtered_physiological'] += 1
                        continue
                elif not mask:
                    stats['filtered_physiological'] += 1
                    continue
            except Exception:
                # 如果过滤失败，保守地保留该样本
                pass
            
            stats['kept'] += 1
        
        # 计算剔除率
        if stats['total'] > 0:
            stats['drop_rate'] = (stats['total'] - stats['kept']) / stats['total'] * 100
            stats['quality_rate'] = stats['filtered_quality'] / stats['total'] * 100
            stats['length_rate'] = stats['filtered_length'] / stats['total'] * 100
            stats['invalid_rate'] = stats['filtered_invalid_data'] / stats['total'] * 100
            stats['label_rate'] = stats['filtered_label'] / stats['total'] * 100
            stats['physiological_rate'] = stats['filtered_physiological'] / stats['total'] * 100
        else:
            stats['drop_rate'] = 0
            stats['quality_rate'] = 0
            stats['length_rate'] = 0
            stats['invalid_rate'] = 0
            stats['label_rate'] = 0
            stats['physiological_rate'] = 0
        
        results[scenario] = stats
    
    return results


def print_results(results: Dict, ring_type: str = None):
    """打印统计结果"""
    title = f" Drop Rate Analysis for {ring_type.upper()} " if ring_type else " Drop Rate Analysis "
    print("\n" + "="*100)
    print(title.center(100, "="))
    print("="*100)
    print(f"{'Scenario':<20} {'Total':<8} {'Kept':<8} {'Drop%':<8} {'Quality%':<10} {'Length%':<10} {'Invalid%':<10} {'Label%':<10} {'Physio%':<10}")
    print("="*110)
    
    for scenario, stats in results.items():
        print(f"{scenario:<20} {stats['total']:<8} {stats['kept']:<8} "
              f"{stats['drop_rate']:<8.2f} {stats['quality_rate']:<10.2f} "
              f"{stats['length_rate']:<10.2f} {stats['invalid_rate']:<10.2f} "
              f"{stats['label_rate']:<10.2f} {stats['physiological_rate']:<10.2f}")
    
    print("="*100)
    
    # 汇总统计
    total_samples = sum(s['total'] for s in results.values())
    total_kept = sum(s['kept'] for s in results.values())
    total_dropped = total_samples - total_kept
    
    print(f"\n{'Overall Summary':<20} Total: {total_samples}, Kept: {total_kept}, "
          f"Dropped: {total_dropped}, Drop Rate: {total_dropped/total_samples*100:.2f}%")
    print("="*100 + "\n")


def save_results_to_csv(results: Dict, output_path: str):
    """保存结果到CSV"""
    rows = []
    for scenario, stats in results.items():
        rows.append({
            'scenario': scenario,
            'total': stats['total'],
            'kept': stats['kept'],
            'filtered_quality': stats['filtered_quality'],
            'filtered_length': stats['filtered_length'],
            'filtered_invalid_data': stats['filtered_invalid_data'],
            'filtered_label': stats['filtered_label'],
            'filtered_physiological': stats['filtered_physiological'],
            'drop_rate_%': stats['drop_rate'],
            'quality_rate_%': stats['quality_rate'],
            'length_rate_%': stats['length_rate'],
            'invalid_rate_%': stats['invalid_rate'],
            'label_rate_%': stats['label_rate'],
            'physiological_rate_%': stats['physiological_rate'],
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze scenario-wise drop rate for ring data')
    parser.add_argument('--data-path', type=str, required=True, help='Path to data directory')
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    parser.add_argument('--output', type=str, default='drop_rate_analysis.csv', help='Output CSV path')
    parser.add_argument('--ring-type', type=str, choices=['ring1', 'ring2', 'both'], default='both',
                        help='Ring type to analyze (ring1, ring2, or both)')
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 检查路径
    if not os.path.exists(args.data_path):
        logging.error(f"Data path not found: {args.data_path}")
        sys.exit(1)
    
    if not os.path.exists(args.config):
        logging.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    # 分析
    logging.info("Starting drop rate analysis...")
    
    if args.ring_type == 'both':
        # 分析 ring1 和 ring2
        for ring in ['ring1', 'ring2']:
            logging.info(f"\n{'='*50}")
            logging.info(f"Analyzing {ring.upper()}...")
            logging.info(f"{'='*50}")
            
            results = analyze_scenario_drop_rate(args.data_path, args.config, ring_type=ring)
            print_results(results, ring_type=ring)
            
            # 保存到独立文件
            output_file = args.output.replace('.csv', f'_{ring}.csv')
            save_results_to_csv(results, output_file)
    else:
        # 分析单个ring
        results = analyze_scenario_drop_rate(args.data_path, args.config, ring_type=args.ring_type)
        print_results(results, ring_type=args.ring_type)
        save_results_to_csv(results, args.output)
    
    logging.info("Analysis complete!")
