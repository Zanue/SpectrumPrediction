import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
import scipy.io as scio
import warnings

warnings.filterwarnings('ignore')

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean


class Dataset_Spectrum(Dataset):
    def __init__(self, data_path, flag='train', threshold=-106.5, size=None, scale=True):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.data_path = data_path
        self.threshold = threshold
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data = np.array(scio.loadmat(self.data_path)['psd_matrix']).transpose(1,0) # [L, C]
        # occupancy = np.array(scio.loadmat(self.data_path)['occupancy_matrix'], dtype=np.bool_).transpose(1,0) # [L, C]
        # data = torch.from_numpy(data)
        # occupancy = torch.from_numpy(occupancy)

        data_len = data.shape[0]
        border1s = [0, int(data_len * 0.7) - self.seq_len, int(data_len * 0.8) - self.seq_len]
        border2s = [int(data_len * 0.7), int(data_len * 0.8), data_len]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)
            self.threshold = (self.threshold - self.scaler.mean) / self.scaler.std
        else:
            data = data

        self.data_x = data[border1:border2]

        print("dataset shape: ", self.data_x.shape)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_x[r_begin:r_end]

        return seq_x, seq_y, self.threshold

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

