import logging
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from constants.dataset import (
    ACCEL_CHANNELS,
    ALL_SCENARIOS,
    AVAILABLE_TASKS,
    DatasetType,
)
from utils.accel_features import extract_accel_features_cuda
from utils.utils import calculate_metrics, save_metrics_to_csv, physiological_filter


def load_dataset(config: Dict, raw_data: pd.DataFrame, task: str="hr", channels: List=None, dataset_type=DatasetType.TRAIN, scenarios: List[str]=None):
    """
    Load the dataset for the specified task.
    Args:
        config (Dict): Configuration dictionary.
        raw_data (pd.DataFrame): Raw data DataFrame.
        task (str): Task name. Default is "hr".
        channels (List): List of channels to load. Default is None.
        dataset_type (DatasetType): Type of dataset (train, valid, test). Default is DatasetType.TRAIN.
        scenarios (List[str]): List of scenarios to be loaded. If None, **all scenarios** will be loaded. We recommend set scenarios to None during training and validation, and per demand for testing.
    Returns:
        RingToolDataset: Loaded dataset.
    """
    if task not in AVAILABLE_TASKS:
        logging.error(f"Invalid task: {task}. Choose from {AVAILABLE_TASKS}.")
        raise ValueError(f"Invalid task. Choose from {AVAILABLE_TASKS}.")
    # Load Data
    return RingToolDataset(config, raw_data, task=task, channels=channels, dataset_type=dataset_type, scenarios=scenarios)


class RingToolDataset(Dataset):
    def __init__(self, config: Dict, raw_data: pd.DataFrame, channels: List, task="hr", dataset_type=DatasetType.TRAIN, scenarios: List[str]=None):
        self.task = task
        self.dataset_type = dataset_type
        self.scenarios = scenarios
        self.raw_data = raw_data
        self.config = config
        self.channels = channels
        self._load_data()

        
    def _load_data(self):
        self.data = []
        # Get target_length from config or use default of 3000
        target_fs = self.config.get("dataset", {}).get("target_fs", 100)
        window_duration = self.config.get("dataset", {}).get("window_duration", 30)
        # dataset_scenario_list = self.config.get("dataset", {}).get("task", ALL_SCENARIOS)

        if self.scenarios is not None:
            logging.info(f"Scenario mode on. Loading {self.dataset_type} dataset from scenarios: {self.scenarios}.")
        else:
            logging.info(f"Scenario mode off. Loading {self.dataset_type} dataset from all scenarios: {ALL_SCENARIOS}.")

        # Calculate target length or use default
        if target_fs and window_duration:
            target_length = target_fs * window_duration
        else:
            target_length = 3000
        min_length = int(target_length * 0.95)  # 95% of target length
        # Properly access nested quality threshold
        quality_th = self.config.get("dataset", {}).get("quality_assessment", {}).get("th", 0)  # Default quality threshold is 0
        commercial_hr_label = []
        accel_combined = self.config.get("dataset", {}).get("accel_combined", False)
        combined_method = self.config.get("dataset", {}).get("accel_combined_method", "magnitude")
        if accel_combined:
            logging.info(f"Using combined accels with metric {combined_method}.")

        for i in tqdm(range(len(self.raw_data))):
            # Load the data in channels
            channel_tensors = []
            skip_sample = False
            if self.raw_data.iloc[i]['ir-quality'] < quality_th or self.raw_data.iloc[i]['red-quality'] < quality_th:
                continue
            if self.scenarios is not None and self.raw_data.iloc[i]["Label"] not in self.scenarios:  # Filter out data not in scenarios.
                continue

            accels_data = {}  # handle accels data separately
            # Process each channel separately
            for channel in self.channels:
                # Get the numpy array for this channel
                channel_data = self.raw_data.iloc[i][channel]
                # check if the channel is np.array
                if not isinstance(channel_data, np.ndarray):
                    skip_sample = True
                    break
                # Check if length is less than minimum required (95% of target)
                if len(channel_data) < min_length:
                    skip_sample = True
                    break
                # Handle length:
                if len(channel_data) > target_length:
                    # If longer than target, truncate
                    channel_data = channel_data[:target_length]
                elif len(channel_data) < target_length:
                    # If shorter than target, pad with zeros
                    padding = np.zeros(target_length - len(channel_data))
                    channel_data = np.concatenate([channel_data, padding])
                
                # if config: accel_combined set to True, handle accels separately
                if accel_combined and channel in ACCEL_CHANNELS:
                    accels_data[channel] = channel_data
                    continue

                # Convert to tensor (adding a channel dimension)
                channel_tensor = torch.tensor(channel_data, dtype=torch.float32).unsqueeze(1)
                channel_tensors.append(channel_tensor)

            # process accels_data
            if accel_combined:
                try:
                    assert self.config.get("method").get("params").get("in_channels") == len(self.config.get("dataset").get("input_type")) - 2
                except (AssertionError, AttributeError, KeyError) as e:
                    logging.error(f"Configuration error: {e}. Mismatch between 'in_channels' and the number of input types minus 2. Since accel_combined is applied, channels for accels should be altered from 3 to 1.")
                    exit(1)

                if len(accels_data) > 3:
                    logging.error("Only support one set of accelerometers. Try use ax/ay/az-standardized in input_type.")
                    exit(1)
                elif len(accels_data) == 3:
                    ax = ay = az = None

                    # Loop through the keys and assign values accordingly
                    for key, value in accels_data.items():
                        if key.startswith('ax-'):
                            ax = value
                        elif key.startswith('ay-'):
                            ay = value
                        elif key.startswith('az-'):
                            az = value
                    
                    # accel_features = extract_accel_features(ax, ay, az)
                    accel_features = extract_accel_features_cuda(ax, ay, az)  # GPU

                    channel_tensor = accel_features[combined_method].unsqueeze(1)
                    channel_tensors.append(channel_tensor.cpu())


            # Skip this sample if any channel was too short
            if skip_sample:
                continue
            # Concatenate all channel tensors along dimension 1
            signal_tensor = torch.cat(channel_tensors, dim=1)
            # Load the label
            label = self.raw_data.iloc[i][self.task]
            # if task is "oura" or "samsung", compare the hr and "oura" or "samsung", print the metrics, return the data contains hr and "oura" or "samsung"
            # Skip if label is None
            if label is None or (self.task in ["oura_hr", "samsung_hr"] and self.raw_data.iloc[i]['hr'] is None):
                continue
            if (self.task == "oura_hr" or self.task == "samsung_hr") and self.raw_data.iloc[i]['hr'] is not None:
                # Compare the hr and "oura" or "samsung"
                hr = self.raw_data.iloc[i]['hr']
                # Calculate metrics
                hr_tensor = torch.tensor(float(hr), dtype=torch.float32)
                label_tensor = torch.tensor(float(label), dtype=torch.float32)
                commercial_hr_label.append((label_tensor, hr_tensor))
                # print(commercial_hr_label)
                # Use hr as the label for training
                label_tensor = hr_tensor
            else:
                label_tensor = torch.tensor(float(label), dtype=torch.float32)
            # Add the data to the list
            self.data.append((signal_tensor, label_tensor))
            
        if self.task == "oura_hr" or self.task == "samsung_hr":
            if commercial_hr_label:  # Check if we have any data to process
                
                # Convert lists of tensors to tensors
                predictions_list, targets_list = zip(*commercial_hr_label)
                predictions = torch.stack(predictions_list)
                targets = torch.stack(targets_list)
                metrics = calculate_metrics(predictions, targets)
                logging.info(f"calculate metrics for {self.task} : {metrics}")
                
                save_metrics_to_csv(metrics, self.config, self.task+'_com')
            else:
                metrics = {"note": "No data available for metrics calculation"}
            
            
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Assuming each item in the dataset is a tuple (input, label)
        return self.data[idx]