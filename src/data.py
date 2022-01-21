# -*- coding: utf-8 -*-

from tkinter.tix import Tree
from typing import List, Optional, cast
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from sklearn import preprocessing


class Datagen(Dataset):
    def __init__(self, dataframe: pd.DataFrame, window: int = 5, future: int = 1):
        self.df = dataframe
        self.window = window
        self.future = future - 1

        self.x = []
        self.y = []

        self._Preprocess()

    def _Preprocess(self):
        cols = list(self.df)

        print(f"selected columns {cols}")

        data = self.df[cols].to_numpy()

        for i in range(len(data)):
            if i + 1 + self.window + self.future > len(data):
                break
            if i + 1 + self.window > len(data):
                break
            self.x.append(data[i : i + self.window])
            self.y.append(data[i + self.window + self.future, -2])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx].astype(float), self.y[idx].astype(float))


class StockDataModule(pl.LightningDataModule):
    def __init__(
        self,
        file: str,
        window: int = 5,
        future: int = 1,
        batchSize: int = 32,
        workers: int = 2,
        scaler: str = "MinMax",
    ):  # sourcery skip: remove-redundant-if
        super().__init__()

        self.file = file
        self.window = window
        self.future = future
        self.batchSize = batchSize
        self.workers = workers
        self._features: int = 0

        if scaler == "Standard":
            self.scaler = preprocessing.StandardScaler()
        else:
            self.scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

        self.columns = None
        self.preprocessing = None

    @property
    def features(self) -> int:
        # assert self._features != 0
        return self._features

    def setup(self, stage: Optional[str] = None) -> None:

        dataframe = pd.read_csv(self.file)
        toDrop: List[str] = []

        for col in dataframe.columns:
            try:
                dataframe[col] = pd.DataFrame(
                    self.scaler.fit_transform(pd.DataFrame(cast(pd.DataFrame, dataframe[col]))), columns=[col]
                )
            except Exception as e:
                print(f"unable to scale column {col}")
                toDrop.append(col)

        print(f">> dropping columns {toDrop}")
        dataframe.drop(columns=toDrop, inplace=True)

        fcols = list(dataframe)
        self._features = len(fcols)
        print(f">> feature columns {fcols}")

        self.train, self.test = train_test_split(dataframe, test_size=0.25, shuffle=False)
        self.train, self.validate = train_test_split(self.train, test_size=0.2, shuffle=False)

        print(f">> sample train: {len(self.train)}")
        print(f">> samples validate: {len(self.validate)}")
        print(f">> samples test: {len(self.test)}")

    def train_dataloader(self):
        return DataLoader(
            Datagen(cast(pd.DataFrame, self.train), window=self.window, future=self.future),
            batch_size=self.batchSize,
            num_workers=self.workers,
        )

    def val_dataloader(self):
        return DataLoader(
            Datagen(cast(pd.DataFrame, self.validate), window=self.window, future=self.future),
            batch_size=self.batchSize,
            num_workers=self.workers,
        )

    def test_dataloader(self):
        return DataLoader(
            Datagen(cast(pd.DataFrame, self.test), window=self.window, future=self.future),
            batch_size=self.batchSize,
            num_workers=self.workers,
        )

