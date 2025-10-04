import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import gaussian_kde


def calculate_mae(predictions, targets):
    """Calculate Mean Absolute Error (MAE)"""
    abs_errors = torch.abs(predictions - targets)
    mae = torch.mean(abs_errors).item()
    standard_error = torch.std(abs_errors).item() / torch.sqrt(torch.tensor(predictions.numel()))
    return mae, standard_error

def calculate_rmse(predictions, targets):
    """Calculate Root Mean Square Error (RMSE)"""
    squared_errors = (predictions - targets) ** 2
    rmse = torch.sqrt(torch.mean(squared_errors)).item()
    standard_error = torch.sqrt(torch.std(squared_errors) / torch.sqrt(torch.tensor(predictions.numel()))).item()
    return rmse, standard_error

def calculate_mape(predictions, targets):
    """Calculate Mean Absolute Percentage Error (MAPE)"""
    mask = targets != 0
    if mask.sum() > 0:
        percent_errors = torch.abs((targets[mask] - predictions[mask]) / targets[mask])
        mape = torch.mean(percent_errors) * 100
        standard_error = torch.std(percent_errors) / torch.sqrt(torch.tensor(mask.sum())) * 100
        return mape.item(), standard_error.item()
    return float('inf'), float('inf')

'''
def calculate_pearson(predictions, targets):
    """Calculate Pearson correlation coefficient"""
    x = predictions.flatten()
    y = targets.flatten()
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    
    pearson = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    # Calculate standard error using the formula: sqrt((1-r²)/(n-2))
    n = torch.tensor(x.numel())
    if n > 2:
        standard_error = torch.sqrt((1 - pearson**2) / (n - 2))
        return pearson.item(), standard_error.item()
    return pearson.item(), float('inf')
'''
    
def calculate_pearson(x, y, eps=1e-5):
    # Flatten tensors to ensure we're computing a single correlation value
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    # Calculate means
    x_mean = torch.mean(x_flat)
    y_mean = torch.mean(y_flat)
    
    # Calculate Pearson correlation
    numerator = torch.sum((x_flat - x_mean) * (y_flat - y_mean))
    denominator = torch.sqrt(torch.sum((x_flat - x_mean) ** 2) * torch.sum((y_flat - y_mean) ** 2) + eps)
    
    pearson = numerator / denominator
    
    # Calculate standard error
    n = torch.tensor(x_flat.numel())
    if n > 2:
        standard_error = torch.sqrt((1 - pearson**2) / (n - 2))
        return pearson.item(), standard_error.item()
    return pearson.item(), float('inf')

def value_with_std(value, std):
    """Return value with standard error: value±std"""
    return f"{value:.2f}±{std:.2f}"

def calculate_metrics(predictions, targets) -> Dict:
    """Calculate all metrics"""
    mae, mae_std = calculate_mae(predictions, targets)
    rmse, rmse_std = calculate_rmse(predictions, targets)
    mape, mape_std = calculate_mape(predictions, targets)
    pearson, pearson_std = calculate_pearson(predictions, targets)
    sample_len = len(predictions)
    return {
        "sample_len": sample_len,
        "mae": mae,
        "mae_with_std": value_with_std(mae, mae_std),
        "rmse": rmse,
        "rmse_with_std": value_with_std(rmse, rmse_std),
        "mape": mape,
        "mape_with_std": value_with_std(mape, mape_std),
        "pearson": pearson,
        "pearson_with_std": value_with_std(pearson, pearson_std)
    }


def calculate_avg_metrics(all_preds_and_targets: List[Tuple]) -> Dict:
    all_preds = torch.cat([p_and_t[0] for p_and_t in all_preds_and_targets])
    all_targets = torch.cat([p_and_t[1] for p_and_t in all_preds_and_targets])

    assert len(all_preds) == len(all_targets), "Predictions and targets must have the same length"
    metrics = calculate_metrics(all_preds, all_targets)

    return metrics


def save_metrics_to_csv(metrics: Dict, config: Dict, task: str, result_csv_path=None, test_type: str = None) -> str:
    """
    Save evaluation metrics to a CSV file
    
    Args:
        metrics (dict): Dictionary containing model evaluation metrics
        config (dict): Configuration dictionary with experiment settings
        task (str): Task name 
        result_csv_path (str, optional): Path to save the CSV file. If None, a default path will be created.
        test_type (str, optional): Test type path like "all-scenarios", "by-group/motion", etc.
    
    Returns:
        str: Path where the CSV was saved
    """    
    exp_name = config.get("exp_name")
    
    if result_csv_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Determine test_type if not provided
        if test_type is None:
            from constants.scenarios import get_test_type_path
            dataset_tasks = config.get("dataset", {}).get("task", [])
            test_type = get_test_type_path(dataset_tasks)
        
        # New path structure: results/<exp_name>/<test_type>/summary.csv
        csv_dir = os.path.join(base_dir, "results", exp_name, test_type)
        os.makedirs(csv_dir, exist_ok=True)
        result_csv_path = os.path.join(csv_dir, "summary.csv")
    
    os.makedirs(os.path.dirname(result_csv_path), exist_ok=True)
    
    # Create a DataFrame with the metrics
    metrics_df = pd.DataFrame({
        'exp_name': [config.get("exp_name", "")],
        'mode': [config.get("mode", "")],
        'ring_type': [config.get("dataset", {}).get("ring_type", "")],
        'fold': [config.get("fold", "")],
        'task': [task],
        'input_type': [config.get("dataset", {}).get("input_type", "")],
        'dataset_task': [config.get("dataset", {}).get("task", "")],
        'method_name': [config.get("method", {}).get("name", "")],
        'epochs': [config.get("train", {}).get("epochs", "")],
        'lr': [config.get("train", {}).get("lr", "")],
        'criterion': [config.get("train", {}).get("criterion", "")],
        'batch_size': [config.get("dataset", {}).get("batch_size", "")],
        'sample_len': [metrics['sample_len']],
        'mae_with_std': [metrics['mae_with_std']],
        'rmse_with_std': [metrics['rmse_with_std']],
        'mape_with_std': [metrics['mape_with_std']],
        'pearson_with_std': [metrics['pearson_with_std']]
    })

    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(result_csv_path)

    # Write to CSV
    metrics_df.to_csv(result_csv_path, mode='a' if file_exists else 'w', 
              header=not file_exists, index=False)
    
    logging.info(f"Saved results to {result_csv_path}")
    return result_csv_path


def plot_and_save_metrics(predictions, targets, config, task, img_path_folder=None) -> str:
    """
    Generate and save visualization plots comparing predictions and targets
    
    Args:
        predictions (torch.Tensor): Model predictions
        targets (torch.Tensor): Ground truth values
        config (dict): Configuration dictionary with experiment settings
        task (str): Task name to include in the output filenames
        img_path_folder (str, optional): Path to save the generated plots. If None, a default path will be created.
    
    Returns:
        str: Path where the images were saved
    """
    exp_name = config.get("exp_name")
    
    if img_path_folder is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        img_path_folder = os.path.join(base_dir, "img", exp_name)
    
    # Create output directory
    os.makedirs(img_path_folder, exist_ok=True)
    task_img_folder = os.path.join(img_path_folder, task)
    os.makedirs(task_img_folder, exist_ok=True)
    
    # Convert tensors to numpy arrays for plotting
    pred_np = predictions.detach().cpu().numpy().flatten()
    target_np = targets.detach().cpu().numpy().flatten()
    
    # 1. Scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create density-based coloring for points
    xy = np.vstack([target_np, pred_np])
    try:
        z = gaussian_kde(xy)(xy)
        _ = ax.scatter(target_np, pred_np, c=z, s=30, alpha=0.8)
    except np.linalg.LinAlgError:
        # Fallback if KDE fails
        _ = ax.scatter(target_np, pred_np, s=30, alpha=0.5)
    
    # Draw the perfect prediction line
    max_val = max(np.max(pred_np), np.max(target_np))
    min_val = min(np.min(pred_np), np.min(target_np))
    padding = (max_val - min_val) * 0.05
    ax.plot([min_val-padding, max_val+padding], [min_val-padding, max_val+padding], 
            '--', color='red', label='Perfect Prediction')
    
    # Add labels and title
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'Predictions vs. Actual Values - {task}')
    
    # Add metrics to plot
    pearson, _ = calculate_pearson(predictions, targets)
    mae, _ = calculate_mae(predictions, targets)
    ax.text(0.05, 0.95, f'Pearson: {pearson:.4f}\nMAE: {mae:.4f}', 
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    scatter_path = os.path.join(task_img_folder, f'scatter_plot_{task}.png')
    plt.savefig(scatter_path, dpi=300)
    plt.close(fig)
    
    # 2. Difference plot (Bland-Altman style)
    fig, ax = plt.subplots(figsize=(8, 8))
    
    differences = target_np - pred_np
    averages = (target_np + pred_np) / 2
    
    # Plot with density coloring
    try:
        xy = np.vstack([averages, differences])
        z = gaussian_kde(xy)(xy)
        _ = ax.scatter(averages, differences, c=z, s=30, alpha=0.8)
    except np.linalg.LinAlgError:
        _ = ax.scatter(averages, differences, s=30, alpha=0.5)
    
    # Add mean and confidence intervals
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    ci_upper = mean_diff + 1.96 * std_diff
    ci_lower = mean_diff - 1.96 * std_diff
    
    ax.axhline(mean_diff, color='black', linestyle='-', label=f'Mean: {mean_diff:.4f}')
    ax.axhline(ci_upper, color='red', linestyle='--', label=f'+1.96 SD: {ci_upper:.4f}')
    ax.axhline(ci_lower, color='red', linestyle='--', label=f'-1.96 SD: {ci_lower:.4f}')
    
    ax.set_xlabel('Average of Predicted and Actual')
    ax.set_ylabel('Difference (Actual - Predicted)')
    ax.set_title(f'Bland-Altman Plot - {task}')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    diff_path = os.path.join(task_img_folder, f'difference_plot_{task}.png')
    plt.savefig(diff_path, dpi=300)
    plt.close(fig)
    
    # 3. Error distribution histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(differences, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(mean_diff, color='red', linestyle='-', label=f'Mean: {mean_diff:.4f}')
    
    ax.set_xlabel('Error (Actual - Predicted)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Error Distribution - {task}')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    hist_path = os.path.join(task_img_folder, f'error_histogram_{task}.png')
    plt.savefig(hist_path, dpi=300)
    plt.close(fig)
    
    logging.info(f"Saved visualization plots to {task_img_folder}")
    return task_img_folder


def save_config(config: Dict, config_path: str) -> Optional[str]:
    """
    Save the configuration dictionary to a JSON file.
    
    Args:
        config (dict): Configuration dictionary to save.
        config_path (str): Path where the configuration file will be saved.
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    try:
        with open(config_path, 'r') as f:
            _ = json.load(f)
        
        return config_path
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error loading configuration from {config_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return None


# ---------------- Physiological range filters ----------------
def _to_numpy(values):
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    if isinstance(values, list):
        return np.asarray(values)
    return values


def _from_numpy(values_np, like):
    if isinstance(like, torch.Tensor):
        return torch.from_numpy(values_np).to(dtype=like.dtype)
    if isinstance(like, list):
        return values_np.tolist()
    return values_np


def get_physiological_ranges() -> Dict[str, Tuple[float, float]]:
    """
    Return plausible physiological ranges for supported tasks.
    Values outside these ranges are considered outliers/noise.
    """
    return {
        # heart rate (bpm)
        "hr": (40.0, 200.0),
        "samsung_hr": (40.0, 200.0),
        "oura_hr": (40.0, 200.0),
        # respiratory rate (breaths per minute)
        "resp_rr": (4.0, 30.0),
        # oxygen saturation (%)
        "spo2": (75.0, 100.0),
        # blood pressure (mmHg)
        "BP_sys": (60.0, 260.0),
        "BP_dia": (30.0, 200.0),
    }


def filter_values_by_range(values, min_val: float, max_val: float, behavior: str = "filter", fill_value: Optional[float] = None):
    """
    Filter or clamp values by [min_val, max_val].

    Args:
        values: torch.Tensor | np.ndarray | list
        min_val: lower bound (inclusive)
        max_val: upper bound (inclusive)
        behavior: 'filter' to drop out-of-range; 'clamp' to clip into range; 'mask' to only return mask
        fill_value: when behavior='filter' and input is tensor/np with fixed shape not desired, ignored;
                    when behavior='clamp' and value is NaN after processing, replaced by fill_value if provided

    Returns:
        If behavior='mask': (mask: np.ndarray[bool])
        Else: (values_same_type, mask: np.ndarray[bool])
    """
    arr = _to_numpy(values)
    mask = np.isfinite(arr) & (arr >= min_val) & (arr <= max_val)

    if behavior == "mask":
        return mask

    if behavior == "filter":
        filtered = arr[mask]
        return _from_numpy(filtered, values), mask

    # clamp
    clamped = np.clip(arr, min_val, max_val)
    if fill_value is not None:
        clamped = np.nan_to_num(clamped, nan=fill_value)
    return _from_numpy(clamped, values), mask


def physiological_filter(values, task: str, behavior: str = "filter", fill_value: Optional[float] = None, custom_ranges: Optional[Dict[str, Tuple[float, float]]] = None):
    """
    Apply physiological range filtering by task.

    Args:
        values: torch.Tensor | np.ndarray | list
        task: task key, e.g., 'hr', 'resp_rr', 'spo2', 'BP_sys', 'BP_dia', 'samsung_hr', 'oura_hr'
        behavior: 'filter' | 'clamp' | 'mask'
        fill_value: used when behavior='clamp' for NaN replacement
        custom_ranges: optional override of default ranges

    Returns:
        behavior='mask' -> mask
        otherwise -> (filtered_or_clamped_values, mask)
    """
    ranges = get_physiological_ranges()
    if custom_ranges:
        ranges.update(custom_ranges)
    if task not in ranges:
        logging.warning(f"physiological_filter: task {task} not in predefined ranges; returning input unchanged.")
        if behavior == "mask":
            return np.ones_like(_to_numpy(values), dtype=bool)
        return values, np.ones_like(_to_numpy(values), dtype=bool)

    min_v, max_v = ranges[task]
    return filter_values_by_range(values, min_v, max_v, behavior=behavior, fill_value=fill_value)


def collate_fn_with_metadata(batch):
    """
    Custom collate function for DataLoader to handle data with metadata.
    
    Args:
        batch: List of samples where each sample is either:
               - (data, label) tuple for normal data
               - ((data, label), metadata) tuple for data with metadata
    
    Returns:
        If metadata present: ((batch_data, batch_labels), batch_metadata)
        Otherwise: (batch_data, batch_labels)
    """
    # Check if first sample has metadata
    if isinstance(batch[0], tuple) and len(batch[0]) == 2 and isinstance(batch[0][1], dict):
        # Has metadata: ((data, label), metadata)
        data_label_pairs = [item[0] for item in batch]
        metadata_list = [item[1] for item in batch]
        
        # Separate data and labels
        data = torch.stack([item[0] for item in data_label_pairs])
        labels = torch.stack([item[1] for item in data_label_pairs])
        
        return (data, labels), metadata_list
    else:
        # No metadata: (data, label)
        data = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        
        return data, labels


def save_prediction_pairs_detailed(
    preds: torch.Tensor,
    targets: torch.Tensor,
    save_path: str,
    metadata: Optional[List[Dict]] = None,
    task: Optional[str] = None,
    fold: Optional[str] = None,
    exp_name: Optional[str] = None
) -> bool:
    """
    Save detailed prediction pairs to a CSV file with metadata.
    
    Args:
        preds: Prediction tensor
        targets: Target tensor
        save_path: Path to save the CSV file
        metadata: Optional list of metadata dictionaries containing subject_id, scenario, timestamp, etc.
        task: Task name
        fold: Fold name
        exp_name: Experiment name
    
    Returns:
        bool: True if save was successful, False otherwise
    """
    try:
        from datetime import datetime
        
        # Convert tensors to numpy
        preds_np = preds.detach().cpu().numpy().flatten()
        targets_np = targets.detach().cpu().numpy().flatten()
        
        # Create base dataframe
        data_dict = {
            'prediction': preds_np,
            'target': targets_np,
        }
        
        # Add metadata if available
        if metadata is not None and len(metadata) == len(preds_np):
            # Extract metadata fields
            subject_ids = [m.get('subject_id', 'unknown') for m in metadata]
            scenarios = [m.get('scenario', 'unknown') for m in metadata]
            start_times = [m.get('start_time', None) for m in metadata]
            end_times = [m.get('end_time', None) for m in metadata]
            tasks = [m.get('task', task if task else 'unknown') for m in metadata]
            
            data_dict.update({
                'subject_id': subject_ids,
                'scenario': scenarios,
                'start_time': start_times,
                'end_time': end_times,
                'task': tasks,
            })
        else:
            # If no metadata, add basic info
            if task:
                data_dict['task'] = [task] * len(preds_np)
            if fold:
                data_dict['fold'] = [fold] * len(preds_np)
        
        if exp_name:
            data_dict['exp_name'] = [exp_name] * len(preds_np)
        
        # Create DataFrame
        df = pd.DataFrame(data_dict)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(save_path, index=False)
        logging.info(f"Saved {len(df)} detailed prediction pairs to: {save_path}")
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to save detailed prediction pairs to {save_path}: {e}")
        return False
