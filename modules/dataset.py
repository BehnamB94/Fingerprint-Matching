from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, feature_arr, label_arr):
        self.feature_arr = feature_arr
        self.label_arr = label_arr

    def __len__(self):
        return self.feature_arr.shape[0]

    def __getitem__(self, idx):
        return self.feature_arr[idx], self.label_arr[idx]
