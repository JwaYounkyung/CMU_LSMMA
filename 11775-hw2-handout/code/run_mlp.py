import os.path as osp
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from modules import FeatureDataModule, MlpClassifier

import wandb


def parse_args(argv=None):
    parser = ArgumentParser(__file__, add_help=False)
    parser.add_argument('--name', default='cnn3d')
    parser = FeatureDataModule.add_argparse_args(parser)
    parser = MlpClassifier.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--earlystop_patience', type=int, default=15)
    parser = ArgumentParser(parents=[parser])
    parser.set_defaults(accelerator='gpu', devices=1,
                        default_root_dir=osp.abspath(
                            osp.join(osp.dirname(__file__), '../data/mlp')))
    args = parser.parse_args(argv)
    return args


def main(args):
    data_module = FeatureDataModule(args)
    model = MlpClassifier(args)

    logger = TensorBoardLogger(args.default_root_dir, args.name)
    wandb.login(key="0699a3c4c17f76e3d85a803c4d7039edb8c3a3d9")    
    # Create your wandb run
    wandb.init(
        name = "run", ### Wandb creates random run names if you skip this field, we recommend you give useful names
        reinit=True, ### Allows reinitalizing runs when you re-run this cell
        project="lsmma-hw2", ### Project should be created in your wandb account 
        sync_tensorboard=True
        #config=config ### Wandb Config for your run
    )

    #wandb_logger = pl.loggers.WandbLogger()

    checkpoint_callback = ModelCheckpoint(
        filename='{epoch}-{step}-{val_acc:.4f}', monitor='val_acc',
        mode='max', save_top_k=-1)
    early_stop_callback = EarlyStopping(
        'val_acc', patience=args.earlystop_patience, mode='max', verbose=True)
    trainer = pl.Trainer.from_argparse_args(
        args, logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback])
    trainer.fit(model, data_module)
    predictions = trainer.predict(datamodule=data_module, ckpt_path='best')
    df = data_module.test_df.copy()
    df['Category'] = torch.concat(predictions).numpy()
    prediction_path = osp.join(logger.log_dir, 'test_prediction.csv')
    df.to_csv(prediction_path, index=False)
    print('Output file:', prediction_path)

    wandb.finish()

if __name__ == '__main__':
    main(parse_args())
