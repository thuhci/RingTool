import logging
from typing import Dict
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
# from utils import *
from utils.utils import calculate_metrics, plot_and_save_metrics,save_metrics_to_csv
from unsupervised.hr.hr import get_hr
from unsupervised.rr.rr import get_rr
from unsupervised.spo2.spo2 import get_spo2
from torch.cuda.amp import autocast, GradScaler

class BaseTrainer:
    def __init__(self, model, config: Dict):
        self.config = config
        if config["method"]["name"] in ["resnet", "transformer", "inception_time", "mamba2"]:
            self.model = model
            self.device = torch.device("cuda:" + str(config["train"]["device"]) 
                                    if torch.cuda.is_available() and config["train"]["device"] != "cpu" 
                                    else "cpu")
            self.model.to(self.device)


    def load_optimizer(self):
        """加载优化器"""
        raise NotImplementedError("子类需要实现 load_optimizer 方法")

    def load_criterion(self):
        """加载损失函数"""
        raise NotImplementedError("子类需要实现 load_criterion 方法")

    def fit(self, train_loader, valid_loader, task=None):
        """训练模型"""
        raise NotImplementedError("子类需要实现 fit 方法")

    def test(self, test_loader, checkpoint_path=None, task=None):
        """测试模型"""
        raise NotImplementedError("子类需要实现 test 方法")
    
# -------------------------------
# 非监督测试器
# -------------------------------
class UnsupervisedTester(BaseTrainer):
    def __init__(self, model, config: Dict):
        super().__init__(model, config)
        
    def load_optimizer(self):
        # Not needed for unsupervised testing
        pass

    def load_criterion(self):
        # Not needed for unsupervised testing
        pass

    def fit(self, train_loader, valid_loader, task=None):
        # Not used in unsupervised approach
        logging.info("Unsupervised methods do not require fitting/training")
        return None

    def test(self, test_loader, checkpoint_path=None, task="hr"):
        if self.config["method"]["name"] not in ["peak", "fft", "ratio"]:
            raise ValueError("This tester is only for unsupervised methods, choose from 'peak', 'fft', 'ratio'")

        logging.info(f"Running unsupervised testing for {task}...")
        all_predictions = []
        all_targets = []
        
        algorithm = self.config["method"].get("name", "peak")
        logging.info(f"Using algorithm: {algorithm} for task: {task}")

        for inputs, labels in tqdm(test_loader, desc=f"Testing {task}"):
            # Convert tensors to numpy for processing
            inputs_np = inputs.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            batch_size = inputs_np.shape[0]
            batch_predictions = []
            
            # Process each sample in the batch
            for i in range(batch_size):
                signal = inputs_np[i]
                
                # Call the appropriate unsupervised method based on task
                if task in ["hr", "bvp_hr", "samsung_hr", "oura_hr"]:
                    prediction = get_hr(signal, method=algorithm)
                elif task == "resp_rr":
                    prediction = get_rr(signal, method=algorithm)
                elif task == "spo2":
                    ppg_ir = signal[:, 0]
                    ppg_red = signal[:, 1]
                    prediction = get_spo2(ppg_ir, ppg_red, ring_type=self.config["dataset"].get("ring_type", "ring1"),method=algorithm)
                else:
                    raise ValueError(f"Unsupported task: {task}")
                
                batch_predictions.append(prediction)
            
            all_predictions.extend(batch_predictions)
            all_targets.extend(labels_np.reshape(-1).tolist())

        # Convert to tensors for metrics calculation
        all_predictions = torch.tensor(all_predictions).reshape(-1, 1)
        all_targets = torch.tensor(all_targets).reshape(-1, 1)
        
        # Calculate metrics
        metrics = calculate_metrics(all_predictions, all_targets)
        logging.debug(f"Task: {task} - "
              f"MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, "
              f"MAPE: {metrics['mape']:.2f}%, Pearson: {metrics['pearson']:.4f}")
        
        # Save metrics to CSV
        save_metrics_to_csv(metrics, self.config, task)
        
        # Plot and save metrics
        plot_and_save_metrics(predictions=all_predictions, targets=all_targets, config=self.config, task=task)
        
        return {
            "loss": 0,  # No loss computation in unsupervised methods
            **metrics
        }

# -------------------------------
# 监督训练器
# -------------------------------

class SupervisedTrainer(BaseTrainer):
    def __init__(self, model, config: Dict, eval_func=None):
        super().__init__(model, config)
        self.eval_func = eval_func
        self.load_optimizer()
        self.load_criterion()
        self.gradient_accum = config["train"].get("gradient_accum", 1)

    def load_criterion(self):
        criterion_type = self.config["train"]["criterion"]
        if criterion_type == "cross entropy":
            self.criterion = nn.CrossEntropyLoss()
        elif criterion_type == "mse":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported criterion type: {criterion_type}")

    def load_optimizer(self):
        optimizer_type = self.config["train"]["optimizer"]
        if optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.get("lr", 0.001))
        elif optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.get("lr", 0.001))
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    def fit(self, train_loader, valid_loader, task=None):
        scaler = GradScaler(enabled=True)
        epochs = self.config.get("train", {}).get("epochs", 200)
        best_loss = float('inf')  # For metrics like loss where lower is better
        
        # Early stopping setup
        early_stopping = self.config.get("train", {}).get("early_stopping", {})
        early_stop = False
        early_stopping_patience = early_stopping.get("patience", 40)  # Default patience is 40 epochs
        monitor = early_stopping.get("monitor", "val_loss")  # Metric to monitor
        mode = early_stopping.get("mode", "min")  # 'min' for loss, 'max' for metrics like accuracy
        counter = 0  # Counter for early stopping
        best_score = float('inf') if mode == "min" else float('-inf')

        scheduler = None
        if self.config.get("train", {}).get("scheduler", {}).get("type", None) == "reduce_on_plateau":
            factor = self.config.get("train", {}).get("scheduler", {}).get("factor", 0.5)
            patience = self.config.get("train", {}).get("scheduler", {}).get("patience", 10)
            min_lr = self.config.get("train", {}).get("scheduler", {}).get("min_lr", 1e-6)
            threshold = self.config.get("train", {}).get("scheduler", {}).get("threshold", 1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',              # min for loss, max for accuracy
                factor=factor,           # multiply LR by this factor
                patience=patience,       # wait N epochs before reducing
                threshold=threshold,     # min change to count as an improvement
                cooldown=0,              # wait time after LR is reduced
                min_lr=min_lr,           # lower bound on LR
                verbose=True             # log messages
            )
        checkpoint_dir = os.path.join("models", self.config.get("exp_name"),task)
        model_name = self.config.get("exp_name")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_best.pt")
        if self.gradient_accum > 1:
            logging.info(f"Training with gradient accumulation: {self.gradient_accum} steps")
        
        # Log early stopping config if enabled
        if early_stopping:
            logging.info(f"Early stopping enabled with patience={early_stopping_patience}, monitoring={monitor}, mode={mode}")
            
        progress_bar = tqdm(range(epochs), desc="Training Progress")
        for epoch in progress_bar:
            # 训练阶段
            self.model.train()
            train_loss = 0
            self.optimizer.zero_grad()
            for idx, (inputs, labels) in enumerate(train_loader):
                # Gradient accumulation

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                with autocast():
                    outputs, _ = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                scaled_loss = scaler.scale(loss)
                scaled_loss = scaled_loss / self.gradient_accum
                scaled_loss.backward()
                if (idx + 1) % self.gradient_accum == 0:
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                train_loss += loss.item() / self.gradient_accum

            if len(train_loader) % self.gradient_accum != 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # 验证阶段
            self.model.eval()
            valid_loss = 0
            all_preds, all_targets = [], []
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs, _ = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    valid_loss += loss.item()
                    all_preds.append(outputs.cpu())
                    all_targets.append(labels.cpu())
            
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            # Update progress bar with metrics
            train_loss_avg = train_loss / len(train_loader)
            valid_loss_avg = valid_loss / len(valid_loader)
            metrics = calculate_metrics(all_preds, all_targets)
            
            # Get current score for early stopping
            current_score = valid_loss_avg if monitor == "val_loss" else metrics.get(monitor, valid_loss_avg)
            
            # Save checkpoint if current model is better
            if valid_loss_avg < best_loss:
                best_loss = valid_loss_avg
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'valid_loss': valid_loss_avg,
                    'metrics': metrics
                }, best_checkpoint_path)
                # message = f"Epoch {epoch+1}: Saved best model with validation loss={valid_loss_avg:.4f}"
                # logging.info(message)
                progress_bar.set_description(f"Training Progress (saved best model, val_loss={valid_loss_avg:.4f})")
            
            # Check early stopping conditions
            if early_stopping:
                improved = (mode == "min" and current_score < best_score) or (mode == "max" and current_score > best_score)
                if improved:
                    best_score = current_score
                    counter = 0
                else:
                    counter += 1
                    if counter >= early_stopping_patience:
                        logging.info(f"Early stopping triggered after {epoch+1} epochs! No improvement in {monitor} for {early_stopping_patience} epochs.")
                        early_stop = True

            if scheduler:
                prev_lr = self.optimizer.param_groups[0]['lr']
                scheduler.step(valid_loss_avg)
                current_lr = self.optimizer.param_groups[0]['lr']
                # if current_lr < prev_lr:
                #     logging.info(f"Epoch {epoch+1}: Reduced learning rate from {prev_lr:.8f} to {current_lr:.8f} due to plateau.")
                # else:
                #     logging.info(f"Epoch {epoch+1}: Learning rate unchanged at {current_lr:.8f}.")

            progress_bar.set_postfix(
                epoch=f"{epoch+1}/{epochs}",
                task=task,
                train_loss=f"{train_loss_avg:.4f}",
                val_loss=f"{valid_loss_avg:.4f}",
                mae=f"{metrics['mae']:.4f}",
                learning_rate=f"{self.optimizer.param_groups[0]['lr']:.8f}",
                early_stop_count=f"{counter}/{early_stopping_patience}" if early_stopping else "N/A"
            )

            # Print detailed metrics every 10 epochs
            if epoch % 10 == 0 or epoch == epochs-1:
                logging.info(f"\nEpoch {epoch+1}/{epochs}:  Task: {task}")
                logging.debug(f"  Training Loss: {train_loss_avg:.4f}")
                logging.debug(f"  Validation Loss: {valid_loss_avg:.4f}, "
                    f"MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, "
                    f"MAPE: {metrics['mape']:.2f}%, Pearson: {metrics['pearson']:.4f}")
                
                if self.eval_func is not None:
                    score = self.eval_func(all_preds, all_targets)
                    logging.debug(f"Custom evaluation score: {score}")
            
            # Break the loop if early stopping is triggered
            if early_stop:
                logging.info("Training stopped early due to early stopping criteria.")
                break

        
        return best_checkpoint_path

    def test(self, test_loader, checkpoint_path,task):
        # Load the best checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            logging.info(f"Loading best model checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            total_params = sum(p.numel() for p in self.model.state_dict().values())
            logging.debug(f"Model parameters: {total_params}")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logging.debug(f"Loaded model from epoch {checkpoint['epoch']+1} with "
                  f"validation loss: {checkpoint['valid_loss']:.4f}, "
                  f"MAE: {checkpoint['metrics']['mae']:.4f}")
        
        self.model.eval()
        test_loss = 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Testing"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs, _ = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                all_preds.append(outputs.cpu())
                all_targets.append(labels.cpu())
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = calculate_metrics(all_preds, all_targets)
        logging.critical(f"Task:{task} Test Loss: {test_loss / len(test_loader):.4f}, "
              f"MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, "
              f"MAPE: {metrics['mape']:.2f}%, Pearson: {metrics['pearson']:.4f}")
        
        # Save metrics to CSV
        save_metrics_to_csv(metrics, self.config, task)
        # Plot and save metrics
        plot_and_save_metrics(predictions=all_preds, targets=all_targets, config=self.config, task=task)
       
        if self.eval_func is not None:
            score = self.eval_func(all_preds, all_targets)
            logging.debug(f"Custom evaluation score: {score}")

        return {
            "loss": test_loss / len(test_loader),
            **metrics
        }

# -------------------------------
# 训练器选择加载
# -------------------------------

def load_trainer(model, model_name: str, config: Dict):
    """根据模型名称加载对应的训练器"""
    if model_name in ["resnet", "transformer", "inception_time", "mamba2"]:
        return SupervisedTrainer(model, config)
    elif model_name in ["peak", "fft", "ratio"]:
        return UnsupervisedTester(model, config)
    return BaseTrainer(model, config)


if __name__ == '__main__':
    import argparse
    import json

    def load_config(config_path):
        with open(config_path, 'r') as file:
            config = json.load(file)
        return config

    config = load_config("/home/disk2/disk/3/tjk/RingTool/config/Resnet.json")
    print(config)
    print(config.get("img_path"))
