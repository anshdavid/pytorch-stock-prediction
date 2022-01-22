# -*- coding: utf-8 -*-

from tkinter.tix import Tree
from typing import Dict
import torch.nn as nn
import torch
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT


class Multivariate(nn.Module):
    def __init__(self, features: int, output: int = 1, hidden: int = 20, layers: int = 2, dropout=0.2):
        super().__init__()
        self.features = features
        self.output = output
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout

        # print("###############", self.features)

        # if batch_first = True => batch_size,seq_len, num_directions * hidden_size
        self.lstm: nn.LSTM = nn.LSTM(
            input_size=features,
            hidden_size=self.hidden,
            num_layers=self.layers,
            dropout=self.dropout,
            batch_first=True,
        )

        self.linear = nn.Linear(self.hidden, self.output)

    def forward(self, x):

        h0 = torch.zeros(self.layers, x.size(0), self.hidden, device=x.device).requires_grad_()
        c0 = torch.zeros(self.layers, x.size(0), self.hidden, device=x.device).requires_grad_()

        # truncated backpropagation through time (BPTT)
        output, (h0, c0) = self.lstm(x.float(), (h0.float().detach(), c0.float().detach()))
        # output, (h0, c0) = self.lstm(x, (h0.detach(), c0.detach()))

        return self.linear(output[:, -1, :])


class Regressor(pl.LightningModule):
    def __init__(
        self, model: nn.Module, criterion: nn.modules.loss._Loss, learningRate: float, params: Dict
    ) -> None:
        super().__init__()

        self.model = model
        self.learningRate = learningRate
        self.criterion = criterion
        self.FM_accuracy = torchmetrics.Accuracy()

        self.save_hyperparameters()

    def _histogram_logger(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)  # type:ignore

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learningRate)

    # def on_train_start(self) -> None:
    #     self.logger.log_hyperparams(self.hparams, {"hp/val_loss": 0})  # type:ignore

    def training_step(self, batch, batch_idx):
        # sourcery skip: class-extract-method
        loss, acc = self._shared_eval_step(batch, batch_idx)
        self.logger.log_metrics({"train_loss": loss}, self.global_step)  # type:ignore
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        self.log("val_loss", loss)
        self.logger.log_metrics({"val_loss": loss}, self.global_step)  # type:ignore
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        self.logger.log_metrics({"test_loss": loss}, self.global_step)  # type:ignore
        return {"test_loss": loss}

    def training_epoch_end(self, outputs: EPOCH_OUTPUT):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()  # type:ignore
        self.logger.log_metrics({"avg_train_loss": avg_loss}, self.current_epoch)  # type:ignore
        self._histogram_logger()

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()  # type:ignore
        self.logger.log_metrics({"avg_val_loss": avg_loss}, self.current_epoch)  # type:ignore

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        y = y.flatten()
        y_hat = y_hat.flatten()

        # return self.criterion(y_hat.float(), y.float())
        # return self.criterion(y_hat.float(), y.float()), self.FM_accuracy(y_hat.float(), y.float())

        return self.criterion(y_hat.float(), y.float()), 0.0

