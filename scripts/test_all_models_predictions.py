"""
Test ALL models in 'models' folder and generate detailed prediction pairs
Supports: BP, HR, RR, SPO2 tasks
"""
import os
import sys
import json
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

sys.path.insert(0, '/root/RingTool')

from nets.load_model import load_model
from utils.utils import calculate_metrics, save_prediction_pairs_detailed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = Path('/root/RingTool')


class DetailedDataset(torch.utils.data.Dataset):
    def __init__(self, config, raw_data, channels, task, scenarios, subject_id):
        self.config = config
        self.raw_data = raw_data
        self.channels = channels
        self.task = task
        self.scenarios = scenarios
        self.subject_id = subject_id
        self.data = []
        self.metadata = []
        self._load_data()
    
    def _load_data(self):
        import numpy as np
        target_fs = self.config.get("dataset", {}).get("target_fs", 100)
        window_duration = self.config.get("dataset", {}).get("window_duration", 30)
        target_length = target_fs * window_duration
        min_length = int(target_length * 0.95)
        quality_th = self.config.get("dataset", {}).get("quality_assessment", {}).get("th", 0)
        
        for i in range(len(self.raw_data)):
            if self.raw_data.iloc[i]['ir-quality'] < quality_th or self.raw_data.iloc[i]['red-quality'] < quality_th:
                continue
            if self.scenarios is not None and self.raw_data.iloc[i]["Label"] not in self.scenarios:
                continue
            
            channel_tensors = []
            skip_sample = False
            
            for channel in self.channels:
                channel_data = self.raw_data.iloc[i][channel]
                if not isinstance(channel_data, np.ndarray):
                    skip_sample = True
                    break
                if len(channel_data) < min_length:
                    skip_sample = True
                    break
                if len(channel_data) > target_length:
                    channel_data = channel_data[:target_length]
                elif len(channel_data) < target_length:
                    padding = np.zeros(target_length - len(channel_data))
                    channel_data = np.concatenate([channel_data, padding])
                channel_tensor = torch.tensor(channel_data, dtype=torch.float32).unsqueeze(1)
                channel_tensors.append(channel_tensor)
            
            if skip_sample or len(channel_tensors) == 0:
                continue
            
            label = self.raw_data.iloc[i][self.task]
            if label is None:
                continue
            
            signal_tensor = torch.cat(channel_tensors, dim=1)
            label_tensor = torch.tensor(float(label), dtype=torch.float32)
            
            self.data.append((signal_tensor, label_tensor))
            self.metadata.append({
                'subject_id': self.subject_id,
                'scenario': self.raw_data.iloc[i]['Label'],
                'start_time': self.raw_data.iloc[i]['start'],
                'end_time': self.raw_data.iloc[i]['end']
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.metadata[idx]


def test_model_with_metadata(model, test_loader, checkpoint_path, device):
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            # Try loading with strict=False for incompatible models
            logging.warning(f"State dict mismatch, trying strict=False: {e}")
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    model.eval()
    model.to(device)
    
    all_preds, all_targets, all_metadata = [], [], []
    
    with torch.no_grad():
        for batch_data, batch_metadata in test_loader:
            inputs, labels = batch_data
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs, _ = model(inputs)
            
            all_preds.append(outputs.cpu())
            all_targets.append(labels.cpu())
            all_metadata.extend(batch_metadata)
    
    return torch.cat(all_preds), torch.cat(all_targets), all_metadata


def save_predictions(predictions, targets, metadata_list, exp_name, task, fold):
    """Save predictions using unified function from utils.py"""
    csv_path = BASE_DIR / "predictions" / exp_name / f"{fold}.csv"
    save_prediction_pairs_detailed(
        preds=predictions,
        targets=targets,
        save_path=str(csv_path),
        metadata=metadata_list,
        task=task,
        fold=fold,
        exp_name=exp_name
    )
    logging.info(f"✓ Saved {len(predictions)} {task} predictions to {csv_path.name}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='生成模型predictions')
    parser.add_argument('--models', nargs='+', help='指定模型名称，不指定则生成所有模型')
    args = parser.parse_args()
    
    data_path = '/root/autodl-fs/RingDatasetV2'
    
    # Find all model directories in models folder
    models_dir = BASE_DIR / 'models'
    if args.models:
        all_model_dirs = [models_dir / name for name in args.models if (models_dir / name).exists() and 'ring' in name]
    else:
        all_model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and 'ring' in d.name]
    
    logging.info(f"Found {len(all_model_dirs)} models in models folder")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    success_count = 0
    
    for model_dir in sorted(all_model_dirs):
        exp_name = model_dir.name
        
        # Find config file: 1) from model folder, 2) from config/test
        config_path = None
        
        # First, try to find config in model folder itself (saved during training)
        for config_file in model_dir.rglob('*.json'):
            if exp_name in config_file.name:
                config_path = config_file
                logging.info(f"Using saved config from model folder")
                break
        
        # If not found, search in config/test folder
        if not config_path:
            search_names = [exp_name]
            if 'inceptiontime' in exp_name:
                search_names.append(exp_name.replace('inceptiontime', 'inception-time'))
            
            for search_name in search_names:
                for config_dir in BASE_DIR.glob(f'config/test/**/{search_name}.json'):
                    config_path = config_dir
                    break
                if config_path:
                    break
        
        if not config_path or not config_path.exists():
            logging.warning(f"⚠ No config found for {exp_name}, skipping")
            continue
        
        logging.info(f"\n{'='*70}")
        logging.info(f"Testing: {exp_name}")
        logging.info(f"{'='*70}")
        
        try:
            # Load config
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Determine ring type
            ring_type = config["dataset"]["ring_type"]
            
            # Load all data
            all_data = {}
            for filename in os.listdir(data_path):
                if filename.endswith('.pkl') and ring_type in filename:
                    subject_id = filename.split('_')[0]
                    all_data[subject_id] = pd.read_pickle(os.path.join(data_path, filename))
            
            # Generate splits
            split_configs = []
            for i in range(5):
                test_fold = i + 1
                test_p = config['split']['5-Fold'][f'Fold-{test_fold}']
                split_configs.append({
                    "test": [p for p in test_p if p in all_data],
                    "fold": f"Fold-{test_fold}"
                })
            
            tasks = config["dataset"]["label_type"]
            if not isinstance(tasks, list):
                tasks = [tasks]
            
            channels = config["dataset"]["input_type"]
            
            # Use all scenarios for testing (full scenario test)
            scenarios = ["sitting", "spo2", "deepsquat", "talking", "shaking_head", "standing", "striding"]
            logging.info(f"Using all scenarios for testing: {scenarios}")
            
            for task in tasks:
                logging.info(f"\nTask: {task}")
                
                for split_config in split_configs:
                    fold = split_config["fold"]
                    
                    # Find checkpoint
                    checkpoint_subdir = "hr" if task in ["samsung_hr", "oura_hr"] else task
                    if task == "spo2":
                        checkpoint_subdir = "spo2"
                    elif task == "resp_rr":
                        checkpoint_subdir = "resp_rr"
                    
                    checkpoint_path = model_dir / checkpoint_subdir / fold / f"{exp_name}_{checkpoint_subdir}_{fold}_best.pt"
                    
                    if not checkpoint_path.exists():
                        logging.warning(f"  ⚠ {fold}: checkpoint not found")
                        continue
                    
                    # Test integrity
                    try:
                        torch.load(str(checkpoint_path), map_location='cpu')
                    except Exception as e:
                        logging.error(f"  ✗ {fold}: corrupted - {e}")
                        continue
                    
                    # Load model (filter out incompatible params)
                    method_config = config['method'].copy()
                    if 'params' in method_config and 'channels_first' in method_config['params']:
                        del method_config['params']['channels_first']
                        logging.info(f"  Removed 'channels_first' param from config")
                    
                    model = load_model(method_config)
                    
                    # Prepare dataset
                    test_subjects = split_config["test"]
                    all_test_data, all_test_metadata = [], []
                    
                    for subject_id in test_subjects:
                        if subject_id not in all_data:
                            continue
                        dataset = DetailedDataset(config, all_data[subject_id], channels, task, scenarios, subject_id)
                        for i in range(len(dataset)):
                            data, metadata = dataset[i]
                            all_test_data.append(data)
                            all_test_metadata.append(metadata)
                    
                    if len(all_test_data) == 0:
                        logging.warning(f"  ⚠ {fold}: no test data")
                        continue
                    
                    # Create dataloader
                    class MetadataDataset(torch.utils.data.Dataset):
                        def __init__(self, data, metadata):
                            self.data, self.metadata = data, metadata
                        def __len__(self):
                            return len(self.data)
                        def __getitem__(self, idx):
                            return self.data[idx], self.metadata[idx]
                    
                    def custom_collate(batch):
                        data_list = [item[0] for item in batch]
                        metadata_list = [item[1] for item in batch]
                        inputs = torch.stack([d[0] for d in data_list])
                        labels = torch.stack([d[1] for d in data_list])
                        return (inputs, labels), metadata_list
                    
                    test_dataset = MetadataDataset(all_test_data, all_test_metadata)
                    batch_size = config.get("test", {}).get("batch_size", 128)
                    if 'mamba' in exp_name.lower():
                        batch_size = 32  # Use smaller batch for mamba
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
                    
                    # Test
                    predictions, targets, metadata = test_model_with_metadata(model, test_loader, str(checkpoint_path), device)
                    save_predictions(predictions, targets, metadata, exp_name, task, fold)
                    
                    metrics = calculate_metrics(predictions, targets)
                    logging.info(f"  ✓ {fold}: {len(predictions)}样本, MAE={metrics['mae']:.2f}")
            
            success_count += 1
            logging.info(f"✓ Completed: {exp_name}")
            
        except Exception as e:
            logging.error(f"✗ Error: {exp_name} - {e}", exc_info=False)
    
    logging.info(f"\n{'='*70}")
    logging.info(f"Summary: {success_count}/{len(all_model_dirs)} models tested")
    logging.info(f"Predictions saved in: /root/RingTool/predictions/")
    logging.info(f"{'='*70}")

