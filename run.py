# -*- coding: utf-8 -*-
# import torch
# print(torch.cuda.current_device())
# print(torch.cuda.get_device_name(0))
# print(torch.cuda.is_available())

from __future__ import annotations

import configparser
import json
from pprint import pprint
from typing import Dict, Union

import psutil
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from src.data import StockDataModule
from src.lightning import Multivariate, Regressor


def run(config: Dict):

    workers = (
        psutil.cpu_count() if config["module.data"]["workers"] == -1 else config["module.data"]["workers"]
    )

    stockData = StockDataModule(
        file=config["module.data"]["file"],
        window=config["module.data"]["window"],
        future=config["module.data"]["future"],
        batchSize=config["module.data"]["batchsize"],
        workers=workers,
        scaler=config["module.data"]["scaler"],
        targetColumn=config["module.data"]["targetcolumn"],
    )

    model = Multivariate(
        features=len(list(stockData.PreProcess())) - 1,
        hidden=config["module.model"]["hidden"],
        layers=config["module.model"]["layers"],
        dropout=config["module.model"]["dropout"],
    )

    criterion = None
    if config["module.lightning"]["criterion"] == "MSELoss":
        criterion = nn.MSELoss()
    assert criterion
    regressor = Regressor(model, criterion, config["module.lightning"]["learningrate"], params=config)

    callbacks = []
    if config["module.earlyStopping"]["enable"]:
        callbacks.append(
            EarlyStopping(
                monitor=config["module.earlyStopping"]["monitor"],
                stopping_threshold=config["module.earlyStopping"]["stopping_threshold"],
                divergence_threshold=config["module.earlyStopping"]["divergence_threshold"],
                check_finite=config["module.earlyStopping"]["check_finite"],
            )
        )
    tensorboard = TensorBoardLogger(
        config["module.tensorboard"]["dir"], name=config["module.tensorboard"]["name"]
    )
    trainer = Trainer(
        gpus=config["module.lightning"]["gpus"],
        accelerator=config["module.lightning"]["accelerator"],
        max_epochs=config["module.lightning"]["epochs"],
        logger=tensorboard,
        callbacks=callbacks
        # profiler="pytorch",
        # progress_bar_refresh_rate=2 //pytorch_lightning.callbacks.progress.TQDMProgressBar
    )

    trainer.fit(regressor, stockData)


if __name__ == "__main__":

    config: Dict[str, Union[str, int, float, bool]] = {}
    with open(r"config.json", "r") as f:
        config = json.load(f)

    pprint(config)

    run(config)
