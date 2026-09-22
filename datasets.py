import torch
import glob
import os
import numpy as np
from torch.utils.data import Dataset


class ShipsEarDataset(Dataset):
    def __init__(self, data_dir):
        self.target_classes = {
            'Dredger': 0,
            'Fishboat': 1,
            'Motorboat': 2,
            'Musselboat': 3,
            'Naturalambientnoise': 4,
            'Oceanliner': 5,
            'Passengers': 6,
            'RORO': 7,
            'Sailboat': 8
        }

        all_files = glob.glob(os.path.join(data_dir, '*.npy'))
        self.data = []
        self.labels = []

        for file_path in all_files:
            class_name = os.path.basename(file_path).split('_')[0]
            if class_name in self.target_classes:
                self.data.append(file_path)
                self.labels.append(self.target_classes[class_name])

        print(f"ShipsEar dataset: {len(all_files)} files found, {len(self.data)} files selected")
        print(f"Classes: {list(self.target_classes.keys())}")

        class_counts = {name: 0 for name in self.target_classes}
        for file_path in self.data:
            class_name = os.path.basename(file_path).split('_')[0]
            class_counts[class_name] += 1

        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = np.load(self.data[idx]).T
        feature = torch.from_numpy(feature).float().unsqueeze(0)
        label = self.labels[idx]

        return feature, label


class DeepShipDataset(Dataset):
    def __init__(self, data_dir):
        self.target_classes = {
            'Cargo': 0,
            'Passengers': 1,
            'Tanker': 2,
            'Tug': 3}

        all_files = glob.glob(os.path.join(data_dir, '*.npy'))
        self.data = []
        self.labels = []

        for file_path in all_files:
            class_name = os.path.basename(file_path).split('_')[0]
            if class_name in self.target_classes:
                self.data.append(file_path)
                self.labels.append(self.target_classes[class_name])

        print(f"DeepShip dataset: {len(all_files)} files found, {len(self.data)} files selected")
        print(f"Classes: {list(self.target_classes.keys())}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = np.load(self.data[idx]).T
        feature = torch.from_numpy(feature).float().unsqueeze(0)
        label = self.labels[idx]

        return feature, label