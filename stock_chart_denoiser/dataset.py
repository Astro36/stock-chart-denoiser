import lightning as L
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


class YfinanceDataModule(L.LightningDataModule):
    def __init__(self, data, seq_len: int = 60, batch_size: int = 32):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.batch_size = batch_size

    def setup(self, stage: str):
        raw = self.data[["Close", "Volume"]].values
        timeseries = []
        for i in range(len(raw) - (self.seq_len + 1)):
            scaler = MinMaxScaler()
            timeseries.append(scaler.fit_transform(raw[i : i + self.seq_len]))
        x = np.array(timeseries)
        x = torch.tensor(x, dtype=torch.float32)
        dataset = TensorDataset(x, x)
        self.dataset_train, self.dataset_val, self.dataset_test = random_split(dataset, [0.8, 0.1, 0.1])

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size)
