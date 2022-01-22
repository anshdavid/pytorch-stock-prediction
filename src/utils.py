# -*- coding: utf-8 -*-
from typing import List, Optional, Sequence, Tuple, cast
import numpy as np
import pandas as pd
from sklearn import preprocessing


def DataSplit(dataLen: int, splitValTrain=0.3, shuffle=False) -> Tuple[Sequence[int], Sequence[int]]:
    indicesList = list(range(dataLen))
    split = int(np.ceil((1 - splitValTrain) * dataLen))

    if shuffle:
        np.random.shuffle(indicesList)

    # trainIdx, valIdx = indicesList[:split], indicesList[split:]
    return indicesList[:split], indicesList[split:]


def ScaleMinMax(df: pd.DataFrame):

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

    for col in df.columns:
        try:
            df[col] = pd.DataFrame(
                min_max_scaler.fit_transform(pd.DataFrame(cast(pd.DataFrame, df[col]))), columns=[col]
            )
        except Exception as e:
            print(f"column {col} - {e}")
    return df


def ScaleStandard(df: pd.DataFrame):

    standard_scaler = preprocessing.StandardScaler()

    for col in df.columns:
        try:
            df[col] = pd.DataFrame(
                standard_scaler.fit_transform(pd.DataFrame(cast(pd.DataFrame, df[col]))), columns=[col]
            )
        except Exception as e:
            print(f"column {col} - {e}")
    return df

