import datetime as dt
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datareader import MTHSDataset
from model.residual import ResidualModel

# Set random seed for reproducibility
seed_number = 2024
random.seed(seed_number)
np.random.seed(seed_number)
torch.manual_seed(seed_number)


def run_train(
    num_epochs: int = 50,
    lr: float = 0.001,
    fold_no: int = 1,
    batch_size: int = 8,
    signal_len: int = 10, # in seconds
    exclude_subjects: list = [],
    scheduler_step: int = 20,
    norm_type: str = "individual",
    norm_mode: str = "min-max",
):
    
    # Initiate device
    device = check_set_gpu()
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load Model
    model = ResidualModel(in_channels=3)
    model.to(device)
    
    # Setup Some Hyperparameters
    num_workers = os.cpu_count() - 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=0.5)
    
    # Criterion and Metric
    mse_fn = nn.MSELoss()
    log_cosh_fn = log_cosh_loss_torch
    mae_fn = nn.L1Loss()
    
    all_val_mae = []
    all_val_rmse = []
    
    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss, train_mae, train_rmse = 0, 0, 0
        val_mae, val_rmse = 0, 0
        train_list_of_errors = []
        train_list_of_losses = []
        val_list_of_errors = []
        
        # DataLoader
        train_dataset = MTHSDataset(
            split="train",
            used_fold=fold_no,
            norm_type=norm_type,
            norm_mode=norm_mode,
            signal_length=signal_len,            
        )
        
        val_dataset = MTHSDataset(
            split="val",
            used_fold=fold_no,
            norm_type=norm_type,
            norm_mode=norm_mode,
            signal_length=signal_len,
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        
        sub13_dataset = MTHSDataset(
            split="val",
            used_fold=fold_no,
            norm_type=norm_type,
            norm_mode=norm_mode,
            signal_length=signal_len,
            white_list_subjects=[13],
            override_start_idx=0
        )
        
        for i, batch in enumerate(train_loader):
            x = batch["signal"].to(device)
            y = batch["spo2"]
            y = y.mean(dim=1).to(device)
            
            # Forward Pass
            output = model(x)
            output = output.squeeze()
                        
            # Loss Calculation
            # loss = log_cosh_fn(y, output)
            loss = mse_fn(y, output)
            
            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Get Error and Update Metrics
            error = abs(output - y)
            error = error.detach().cpu().numpy()
            train_list_of_errors.extend(error)
            train_list_of_losses.append(loss.item())
            
        scheduler.step()
        
        # Calculate Metrics
        train_mae = np.mean(train_list_of_errors)
        train_rmse = np.sqrt(np.mean(np.square(train_list_of_errors)))
        train_loss = np.mean(train_list_of_losses)
        
        print(f"Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.4f} | Train RMSE: {train_rmse:.4f}")
        
        # Evaluation
        model.eval()
        with torch.inference_mode():
            for i, batch in enumerate(val_loader):
                x = batch["signal"].to(device)
                y = batch["spo2"]
                y = y.mean(dim=1).to(device)
                
                # Forward Pass
                output = model(x)
                # print(f"Output: {output}")
                
                # Get Error and Update Metrics
                error = abs(output - y)
                error = error.detach().cpu().numpy()
                val_list_of_errors.extend(error)
                
            val_mae = np.mean(val_list_of_errors)
            val_rmse = np.sqrt(np.mean(np.square(val_list_of_errors)))
            
            print(f"Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f}")
            
            all_val_mae.append(val_mae)
            all_val_rmse.append(val_rmse)
            
            # Try with Subject 13 (Spo2 below 90)
            data = sub13_dataset[0]
            x = data["signal"].to(device)
            y = data["spo2"].to(device)
            output = model(x)
            error = abs(output - y)
            error = error.detach().cpu().numpy()
            print(f"Subject 13 MAE: {np.mean(error):.4f}")
            
    print(f"Lowest Val MAE: {min(all_val_mae)}")
    print(f"RMSE on the lowest Val MAE: {all_val_rmse[np.argmin(all_val_mae)]}")

def check_set_gpu(override=None):
    if override == None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print(f"Using MPS: {torch.backends.mps.is_available()}")
        else:
            device = torch.device('cpu')
            print(f"Using CPU: {torch.device('cpu')}")
    else:
        device = torch.device(override)
    return device

def log_cosh_loss_torch(y_true, y_pred):
    diff = y_pred - y_true
    # Clipping the difference might help avoid overflow, but it's a workaround
    diff_clipped = torch.clamp(diff, min=-10, max=10)
    cosh_diff = torch.cosh(diff_clipped)
    log_cosh_diff = torch.log(cosh_diff)
    loss = torch.sum(log_cosh_diff)
    return loss


if __name__ == "__main__":

    for i in range(1, 2):
        run_train(
            num_epochs=100,
            lr=1e-3,
            fold_no=i,
            batch_size=8,
            signal_len=10,
            exclude_subjects=[13],
            scheduler_step=30,
            norm_type="joint",
            norm_mode="z-score",
        )