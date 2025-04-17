import logging
from typing import Dict, List
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import os
import pickle
import numpy as np
import random
from utils.utils import calculate_metrics, save_metrics_to_csv, extract_accel_features_cuda


accel_channels = {
    "ax-raw","ax-filtered","ax-standardized","ax-difference","ax-welch","ax-filtered-rr","ax-welch-rr",
    "ay-raw","ay-filtered","ay-standardized","ay-difference","ay-welch","ay-filtered-rr","ay-welch-rr",
    "az-raw","az-filtered","az-standardized","az-difference","az-welch","az-filtered-rr","az-welch-rr"
}


def load_dataset(config: Dict, raw_data: pd.DataFrame, task: str="hr", channels: List=None):
    """
    Load the dataset for the specified task.
    Args:
        config (Dict): Configuration dictionary.
        raw_data (pd.DataFrame): Raw data DataFrame.
        task (str): Task name. Default is "hr".
        channels (List): List of channels to load. Default is None.
    Returns:
        List: List of tuples containing signal tensors and labels.
    """
    if task not in ['hr', 'bvp_hr', 'bvp_sdnn', 'bvp_rmssd',
       'bvp_nn50', 'bvp_pnn50', 'resp_rr', 'spo2', 'samsung_hr', 'oura_hr',
       'BP_sys', 'BP_dia']:
        raise ValueError("Invalid task. Choose from 'hr', 'bvp_hr', 'bvp_sdnn', 'bvp_rmssd', 'bvp_nn50', 'bvp_pnn50', 'resp_rr', 'spo2', 'samsung_hr', 'oura_hr', 'BP_sys', 'BP_dia'")
        return [(torch.randn(201, 1), torch.randn(1))]
    # Load Data
    return LoadDataset(config, raw_data, task=task, channels=channels)
    

class LoadDataset(Dataset):
    def __init__(self, config: Dict, raw_data: pd.DataFrame, channels: List, task: str="hr"):
        self.task = task
        self.raw_data = raw_data
        self.config = config
        self.channels = channels
        self.load_data()
        # import ipdb; ipdb.set_trace()
    
        
    def load_data(self):
        self.data = []
        # Get target_length from config or use default of 3000
        target_fs = self.config.get("dataset", {}).get("target_fs", 100)
        window_duration = self.config.get("dataset", {}).get("window_duration", 30)
        # Calculate target length or use default
        if target_fs and window_duration:
            target_length = target_fs * window_duration
        else:
            target_length = 3000
        min_length = int(target_length * 0.95)  # 95% of target length
        # Properly access nested quality threshold
        quality_th = self.config.get("quality_assessment", {}).get("th", 0.8)  # Default quality threshold is 0.8
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
                if accel_combined and channel in accel_channels:
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
            if label is None:
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
            
            
        logging.info(f"Loaded {len(self)} samples for task {self.task} with channels {self.channels}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Assuming each item in the dataset is a tuple (input, label)
        return self.data[idx]


if __name__ == "__main__":
    # Example usage
    '''
    -rings
        -subject_ring1_processed.pkl    :DataFrame
        -subject_ring2_processed.pkl    :DataFrame
    
            -id,start,end,fs,ir-raw,ir-standardized,ir-filtered,ir-difference,ir-welch,red-raw,red-standardized,red-filtered,red-difference,red-welch,ax-raw,ax-standardized,ay-raw,ay-standardized,az-raw,az-standardized,ir-quality,red-quality,hr,bvp_hr,bvp_sdnn,bvp_rmssd,bvp_nn50,bvp_pnn50,resp_rr,spo2,samsung_hr,oura_hr,BP_sys,BP_dia,Experiment,Label
    
            np.array:ir-raw,ir-standardized,ir-filtered,ir-difference,ir-welch,red-raw,red-standardized,red-filtered,red-difference,red-welch,ax-raw,ax-standardized,ay-raw,ay-standardized,az-raw,az-standardized
            str:Experiment,Label
            float:id,start,end,fs,ir-quality,red-quality,hr,bvp_hr,bvp_sdnn,bvp_rmssd,bvp_nn50,bvp_pnn50,resp_rr,spo2,samsung_hr,oura_hr,BP_sys,BP_dia
    '''
    data_path = "/home/disk2/disk/3/tjk/RingData/Preprocessed_test/rings"
    # check sample data first
    split_config = {'train': ['00021', '00030', '00010', '00028', '00019', '00014', '00017', '00025', '00022', '00013', '00007', '00009', '00031', '00026', '00032', '00020', '00040', '00016', '00027', '00008', '00011'], 'test': ['00000', '00001', '00002', '00003', '00004', '00005', '00006'], 'valid': ['00023', '00024', '00018', '00029', '00015', '00012']}
    sample_data = pd.read_pickle(os.path.join(data_path, "00000"+"_ring1_processed.pkl"))
    sample_data = sample_data.iloc[:10]
    print(sample_data)
    print(sample_data.columns)
    print(sample_data['ir-raw'].iloc[0].shape)
    print(sample_data['samsung_hr'])
    
    # # Usage example
    # print(LoadDataset(
    #     config={},
    #     raw_data=sample_data,
    #     channels=['ir-raw', 'red-raw', 'ax-raw', 'ay-raw', 'az-raw'],
    #     task='hr'
    # ))
    