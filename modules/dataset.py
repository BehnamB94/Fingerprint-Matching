import numpy as np
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, feature_arr, label_arr, pre_process=True):
        self.feature_arr = feature_arr
        self.label_arr = label_arr
        self.pre_process = pre_process

    def __len__(self):
        return self.feature_arr.shape[0]

    def __getitem__(self, idx):
        if self.pre_process:
            feature = self.feature_arr[idx].astype(np.float)
            feature[0] -= feature[0].min()
            feature[0] /= feature[0].max()
            feature[1] -= feature[1].min()
            feature[1] /= feature[1].max()
        else:
            feature = self.feature_arr[idx]
        return feature, self.label_arr[idx]
