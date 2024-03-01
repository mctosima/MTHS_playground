import json
import os
import random
from glob import glob

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset


class MTHSDataset(Dataset):
    def __init__(
        self,
        root_path: str = "MTHS/Data",
        kfold_split_json: str = "kfold_split.json",
        split: str = "train",
        used_fold: int = 1,
        signal_length: int = 10, # in seconds
        norm_type: str = "joint",
        norm_mode: str = "z-score",
        exclude_subjects: list = [],
        white_list_subjects: list = [],
        override_start_idx: int = -1,
    ):
        self.root_path = root_path
        self.signal_length = signal_length
        self.norm_type = norm_type
        self.norm_mode = norm_mode
        self.override_start_idx = override_start_idx

        if not os.path.exists(kfold_split_json):
            print(f"Creating kfold split json file: {kfold_split_json}")

            # populate all files with format: labels_{i}.npy
            labels_files = glob(os.path.join(root_path, "label_*.npy"))
            subject_ids = sorted(
                [int(f.split("_")[-1].split(".")[0]) for f in labels_files]
            )

            # make sure that every label has a corresponding signal files
            signal_files = glob(os.path.join(root_path, "signal_*.npy"))
            signal_ids = sorted(
                [int(f.split("_")[-1].split(".")[0]) for f in signal_files]
            )
            assert subject_ids == signal_ids, "Mismatch between labels and signal files"

            # create kfold split
            self.create_fold(subject_ids)

        with open(kfold_split_json, "r") as f:
            kfold_split = json.load(f)

        if split == "train":
            self.subject_ids = kfold_split[str(used_fold)]["train"]
        else:
            self.subject_ids = kfold_split[str(used_fold)]["val"]
            
        if len(exclude_subjects) > 0:
            self.subject_ids = [s for s in self.subject_ids if s not in exclude_subjects]
            
        if len(white_list_subjects) > 0:
            self.subject_ids = white_list_subjects

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        subject_id = self.subject_ids[idx]

        # load signal and labels
        signal = np.load(os.path.join(self.root_path, f"signal_{subject_id}.npy"))
        labels = np.load(os.path.join(self.root_path, f"label_{subject_id}.npy"))
        
        # get a random start index
        signal_sampling_rate = 30
        labels_len = labels.shape[0]
        lbl_rnd_start_idx = random.randint(0, labels_len - self.signal_length)
        
        if self.override_start_idx > -1:
            lbl_rnd_start_idx = self.override_start_idx
        
        signal_rnd_start_idx = lbl_rnd_start_idx * signal_sampling_rate
        
        # trim the signal and labels
        signal = signal[signal_rnd_start_idx:signal_rnd_start_idx + self.signal_length * signal_sampling_rate]
        labels = labels[lbl_rnd_start_idx:lbl_rnd_start_idx + self.signal_length]
        
        # normalize the signal
        signal = self.normalize_signal(signal, normalization_type=self.norm_type, method=self.norm_mode)
        
        # reshape from (length, 3) to (3, length)
        signal = np.transpose(signal, (1, 0))
        
        # return as dictionary. Before returning, convert to tensor
        return_data = {
            "subject_id": subject_id,
            "signal": torch.tensor(signal, dtype=torch.float32),
            "hr": torch.tensor(labels[:,0], dtype=torch.float32),
            "spo2": torch.tensor(labels[:,1], dtype=torch.float32),
            "start_idx": lbl_rnd_start_idx,
        }
        
        return return_data

    def create_fold(self, subject_ids):
        kf = KFold(n_splits=5, shuffle=True, random_state=2023)
        splits = {}

        for i, (train_idx, val_idx) in enumerate(kf.split(subject_ids)):
            train_sub = [subject_ids[i] for i in train_idx]
            val_sub = [subject_ids[i] for i in val_idx]

            splits[i + 1] = {"train": train_sub, "val": val_sub}

        with open("kfold_split.json", "w") as f:
            json.dump(splits, f)


    def normalize_signal(self, signal, normalization_type='joint', method='z-score'):
        """
        Normalize the RGB signal.

        Parameters:
        - signal (numpy array): The signal to be normalized, with shape (length, 3).
        - normalization_type (str): The type of normalization, 'joint' or 'individual'.
        - method (str): The normalization method, 'z-score' for standardization or 'min-max' for scaling to [0, 1].

        Returns:
        - numpy array: The normalized signal, preserving the shape and dimension.
        """
        if method == 'z-score':
            if normalization_type == 'joint':
                mean = np.mean(signal, axis=0)
                std = np.std(signal, axis=0)
                normalized_signal = (signal - mean) / std
            elif normalization_type == 'individual':
                # Normalize each channel individually
                mean = np.mean(signal, axis=0)
                std = np.std(signal, axis=0)
                normalized_signal = (signal - mean.reshape(1, -1)) / std.reshape(1, -1)
        elif method == 'min-max':
            if normalization_type == 'joint':
                min_val = np.min(signal)
                max_val = np.max(signal)
                normalized_signal = (signal - min_val) / (max_val - min_val)
            elif normalization_type == 'individual':
                min_val = np.min(signal, axis=0)
                max_val = np.max(signal, axis=0)
                normalized_signal = (signal - min_val.reshape(1, -1)) / (max_val - min_val).reshape(1, -1)
        else:
            raise ValueError("Normalization method must be 'z-score' or 'min-max'")
        
        return normalized_signal




if __name__ == "__main__":
    dataset = MTHSDataset(
        split="train",
        used_fold=1,
        norm_type="individual",
        norm_mode="min-max",
        white_list_subjects=[13],
        override_start_idx=0
    )
    single_data = dataset[0]
    print(f"Signal shape: {single_data['signal'].shape} | HR shape: {single_data['hr'].shape} | SpO2 shape: {single_data['spo2'].shape}")
    print(f"Subject ID: {single_data['subject_id']} | Start index: {single_data['start_idx']}")
    
    # print the type of the data
    print(f"Signal type: {single_data['signal'].dtype} | HR type: {single_data['hr'].dtype} | SpO2 type: {single_data['spo2'].dtype}")
        
    # import matplotlib.pyplot as plt
    # signal = single_data['signal']
    # hr = single_data['hr']
    # spo2 = single_data['spo2']
    
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # ax1.plot(signal)
    # ax1.set_title('Signal')
    
    # ax2.plot(hr, label='HR')
    # ax2.plot(spo2, label='SpO2')
    # ax2.set_title('HR and SpO2')
    # ax2.legend()
    
    # plt.show()
