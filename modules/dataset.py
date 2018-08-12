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
            for i in range(2):
                feature[i] -= feature[i].mean()
                feature[i][feature[i] > 0] /= feature[i].max()
                feature[i][feature[i] < 0] /= -feature[i].min()
        else:
            feature = self.feature_arr[idx]
        return feature, self.label_arr[idx]
