from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy


class MlpClassifier(pl.LightningModule):

    def __init__(self, hparams):
        super(MlpClassifier, self).__init__()
        self.save_hyperparameters(hparams)
        layers = [
            # TODO: define model layers here
            # Input self.hparams.num_features
            # Output self.hparams.num_classes
            nn.Linear(self.hparams.num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(32, self.hparams.num_classes)
        ]
        self.model = nn.Sequential(*layers)
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log_dict({'val_loss': loss, 'val_acc': acc},
                      on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self(x)
        pred = y_hat.argmax(dim=-1)
        return pred

    def configure_optimizers(self):
        # TODO: define optimizer and optionally learning rate scheduler
        # The simplest form would be `return torch.optim.Adam(...)`
        # For more advanced usages, see https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers

        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-5)

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--num_features', type=int, default=512)
        parser.add_argument('--num_classes', type=int, default=15)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--scheduler_factor', type=float, default=0.3)
        parser.add_argument('--scheduler_patience', type=int, default=5)
        return parser
