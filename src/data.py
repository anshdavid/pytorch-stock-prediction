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
    def __init__(
        self, dataframe: pd.DataFrame, window: int = 5, future: int = 1, targetColumn: str = "Close"
    ):
        self.df = dataframe
        self.window = window
        self.future = future - 1
        self.targetColumn = targetColumn

        self.x = []
        self.y = []

        self._Preprocess()

    def _Preprocess(self):
        print(f">> feature columns {list(self.df)}")

        y = pd.DataFrame(self.df, columns=[self.targetColumn]).to_numpy()
        x = self.df.drop(columns=[self.targetColumn]).to_numpy()

        for i in range(len(x)):
            if i + 1 + self.window + self.future > len(x):
                break
            if i + 1 + self.window > len(x):
                break
            self.x.append(x[i : i + self.window])
            self.y.append(y[i + self.window + self.future])

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
        targetColumn: str = "Close",
    ):  # sourcery skip: remove-redundant-if
        super().__init__()

        self.file = file
        self.window = window
        self.future = future
        self.batchSize = batchSize
        self.workers = workers
        self.targetColumn = targetColumn
        self._features: int = 0

        if scaler == "Standard":
            self.scaler = preprocessing.StandardScaler()
        else:
            self.scaler = preprocessing.MinMaxScaler(feature_range=(0, 10))

        self.columns = None
        self.preprocessing = None

    @property
    def features(self) -> int:
        # assert self._features != 0
        return self._features

    def PreProcess(self) -> pd.DataFrame:

        dataframe = pd.read_csv(self.file)
        toDrop: List[str] = []

        for col in dataframe.columns:
            try:
                dataframe[col] = pd.DataFrame(
                    self.scaler.fit_transform(pd.DataFrame(cast(pd.DataFrame, dataframe[col]))), columns=[col]
                )
            except Exception as e:
                print(f">> unable to scale column {col}")
                toDrop.append(col)

        dataframe.drop(columns=toDrop, inplace=True)

        fcols = list(dataframe)
        self._features = len(fcols)

        return dataframe

    def setup(self, stage: Optional[str] = None) -> None:

        dataframe = self.PreProcess()

        self.train, self.test = train_test_split(dataframe, test_size=0.25, shuffle=False)
        self.train, self.validate = train_test_split(self.train, test_size=0.2, shuffle=False)

        print(f">> sample train: {len(self.train)}")
        print(f">> samples validate: {len(self.validate)}")
        print(f">> samples test: {len(self.test)}")

    def train_dataloader(self):
        return DataLoader(
            Datagen(
                cast(pd.DataFrame, self.train),
                window=self.window,
                future=self.future,
                targetColumn=self.targetColumn,
            ),
            batch_size=self.batchSize,
            num_workers=self.workers,
        )

    def val_dataloader(self):
        return DataLoader(
            Datagen(
                cast(pd.DataFrame, self.validate),
                window=self.window,
                future=self.future,
                targetColumn=self.targetColumn,
            ),
            batch_size=self.batchSize,
            num_workers=self.workers,
        )

    def test_dataloader(self):
        return DataLoader(
            Datagen(
                cast(pd.DataFrame, self.test),
                window=self.window,
                future=self.future,
                targetColumn=self.targetColumn,
            ),
            batch_size=self.batchSize,
            num_workers=self.workers,
        )

