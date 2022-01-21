# -*- coding: utf-8 -*-
# import torch
# print(torch.cuda.current_device())
# print(torch.cuda.get_device_name(0))
# print(torch.cuda.is_available())


import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger


from src.data import StockDataModule
from src.lightning import Regressor, Multivariate

import psutil

if __name__ == "__main__":

    tensorboard = TensorBoardLogger("./reports/", name="logs")

    criterion = nn.MSELoss()

    stockData = StockDataModule(r"data/raw/GOOG.csv", workers=psutil.cpu_count())
    model = Multivariate(features=6, layers=1, dropout=0.4)
    regressor = Regressor(model, criterion, 0.002)

    trainer = Trainer(
        gpus=1,
        accelerator="gpu",
        max_epochs=10,
        logger=tensorboard,
        profiler="pytorch",
        # progress_bar_refresh_rate=2 //pytorch_lightning.callbacks.progress.TQDMProgressBar
    )
    trainer.fit(regressor, stockData)

