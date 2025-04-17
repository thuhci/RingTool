import datetime
import json
import os
import numpy as np
import pandas as pd
import logging
import warnings
from scipy.signal import welch, get_window, butter, filtfilt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import argparse
from nets.load_model import load_model
from dataset.load_dataset import load_dataset
from trainer.load_trainer import load_trainer
from typing import List, Dict, Optional, Union
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random


# TODO: 
'''
1. Unspuervised methods for hr spo2 rr BP
2. multi-channels optmize with IMU -dyk
3. task -finetune -dyk
4. Support InceptionTime/Mamba/transformer -wzy
'''

DATA_PATH = "/home/disk2/disk/3/tjk/RingData/Preprocessed/rings"


def generate_split_config(mode: str, config: Dict):
    split_subject_list = config['split']
    split_config = []
    if mode == "5fold":
        for i in range(5):
            test_fold = i + 1  # Folds are 1-indexed
            valid_fold = (i + 1) % 5 + 1  # Wraps around to fold 1 after fold 5

            valid_p = split_subject_list['5-Fold'][f'Fold-{valid_fold}']
            test_p = split_subject_list['5-Fold'][f'Fold-{test_fold}']
            
            # Train participants are from the remaining folds
            train_p = []
            for j in range(1, 6):  # Folds are 1-indexed
                if j != valid_fold and j != test_fold:
                    train_p.extend(split_subject_list['5-Fold'][f'Fold-{j}'])
            
            split_config.append({"train": train_p, "valid": valid_p, "test": test_p, "fold": f"Fold-{test_fold}"})
    elif mode == "train":
        # split into train, valid, test
        split_config.append({"train": split_subject_list['train'], "valid": split_subject_list['valid'], "test": split_subject_list['test'], "fold": "Fold-1"})
    
    else:
        raise ValueError("Invalid mode. Choose '5fold' or 'train'.")
    return split_config


    
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def find_all_data(path, ring_type) -> Dict[str, pd.DataFrame]:
    # load all subject data from a folder, subject_ring1_processed.pkl
    all_data = {}  # subject_id -> pd.DF
    for filename in os.listdir(path):
        if filename.endswith('.pkl') and ring_type in filename:
            # load data
            file_path = os.path.join(path, filename)
            try:
                data = pd.read_pickle(file_path)
                # get subject id from filename
                subject_id = filename.split('_')[0]
                # add data to dictionary
                all_data[subject_id] = data
            except Exception as e:
                logging.error(f"Error loading {filename}: {e}")
                continue
    return all_data

def set_seed(seed: int):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def unsupervised(config_path):   
    config = load_config(config_path)
    # load all data
    all_data = find_all_data(DATA_PATH, config["dataset"]["ring_type"])
    subject_list = list(all_data.keys())
    all_data = pd.concat(all_data.values())
    logging.info(f"Found {len(subject_list)} subjects in the data folder.") 
    # set seed
    set_seed(config["seed"])
    # only test on the whole dataset without split, unsupervised methods
    if config["mode"] not in ["train", "test", "5fold"]:
        raise ValueError("Invalid mode. Choose 'train' or 'test', '5fold'.")
    if config["mode"] == "test" and config["method"]["type"]== "unsupervised":
        # load dataset
        channels = config["dataset"]["input_type"]
        tasks = config["dataset"]["label_type"]
        logging.info(f"Channels: {channels}, Task: {tasks}")
        tester = load_trainer(config['method'], config['method']['name'], config)
        for task in tasks:
            all_dataset = load_dataset(
                config=config,
                raw_data=all_data,
                channels=channels,
                task=task
            )
            all_loader = DataLoader(all_dataset, batch_size=config["dataset"]["batch_size"], shuffle=False)
            test_results = tester.test(all_loader, None, task)
            # print test results
            logging.info(f"Test results for task {task}: {test_results}")


def main(config_path):
    config = load_config(config_path)
    # load all data
    all_data = find_all_data(DATA_PATH, config["dataset"]["ring_type"])
    subject_list = list(all_data.keys())
    logging.info(f"Found {len(subject_list)} subjects in the data folder.")
    # set seed
    set_seed(config["seed"])
    # training 
    if config["mode"] not in ["train", "test", "5fold"]:
        raise ValueError("Invalid mode. Choose 'train' or 'test', '5fold'.")
    # check if the key in split_config is in the subject list, save the cross into split_config
    # Correct the call to generate_split_config
    split_configs = generate_split_config(config["mode"], config)
    
    # Check if all subjects in split_configs exist in available data
    for split_config in split_configs:
        for split_type in ["train", "valid", "test"]:
            if split_type in split_config:
                # Filter out subjects that don't exist in available data
                split_config[split_type] = [subj for subj in split_config[split_type] if subj in subject_list]
    
    logging.info(f"Generated {len(split_configs)} split configurations.")
 
    for split_config in split_configs:
        config['fold'] = split_config["fold"]
        # Extract channels and task from config
        channels = config["dataset"]["input_type"]
        tasks = config["dataset"]["label_type"]
        logging.info(f"Channels: {channels}, Task: {tasks}")
        for task in tasks:
            # load model
            model = load_model(config['method'])
            logging.info(f"Successfully loaded model {config['method']}")
            logging.info(f"Running experiment with split config: {split_config}")

            trainer = load_trainer(model, config['method']['name'], config)
            
            if task == "oura_hr" or "samsung_hr":
                train_task = "hr"
            else: 
                train_task = task

            if "train" in split_config:
                # prepare training dataset
                train_data = pd.concat([all_data[p] for p in split_config["train"]])
                train_dataset = load_dataset(
                    config=config,
                    raw_data=train_data,
                    channels=channels,
                    task=train_task
                )
                train_loader = DataLoader(train_dataset, batch_size=config["dataset"]["batch_size"], shuffle=True)
                
                valid_data = pd.concat([all_data[p] for p in split_config["valid"]])
                valid_dataset = load_dataset(
                    config=config,
                    raw_data=valid_data,
                    channels=channels,
                    task=task
                )
                valid_loader = DataLoader(valid_dataset, batch_size=config["dataset"]["batch_size"], shuffle=False)
                
                # Train the model
                checkpoint_path = trainer.fit(train_loader, valid_loader, task)
        
            # test model 
            test_data = pd.concat([all_data[p] for p in split_config["test"]])
            test_dataset = load_dataset(
                config=config,
                raw_data=test_data,
                channels=channels,
                task=task
            )
            test_loader = DataLoader(test_dataset, batch_size=config["dataset"]["batch_size"], shuffle=False)
            test_results = trainer.test(test_loader,checkpoint_path,task)


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn')

    parser = argparse.ArgumentParser(description='Process ring PPG data using FFT.')
    # parser.add_argument('--config', type=str, default="./config/Resnet.json", help='Path to the configuration JSON file.')
    # parser.add_argument('--config', type=str, default="./config/Transformer.json", help='Path to the configuration JSON file.')
    parser.add_argument('--config', type=str, default="./config/InceptionTime.json", help='Path to the configuration JSON file.')
    args = parser.parse_args()
    # Load the configuration
    config = load_config(args.config)
    exp_name = config.get("exp_name")

    # Set up logging
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Get current timestamp for the log filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/ringtool_{exp_name}_{timestamp}.log"

    # Set up file and console logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )    
    logging.info(f"Logging to: {log_filename}")


    if config["method"]["type"] == "unsupervised":
        # unsupervised methods
        unsupervised(args.config)
    else:
        # supervised methods
        main(args.config)
