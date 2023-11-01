from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss

from data_processing.parse import Parse
from utils import config
from utils.model_torch import ModelTorchBase


class Model(ModelTorchBase):
    def train(self):
        device = self._describe_env()

        dataset_dir_path = Path(self.model_hp.dataset_dir)
        print("Dataset directory and files:")
        for path in dataset_dir_path.glob('**/*'):
            print(path)

        data = self._get_df_from_dataset_file(dataset_dir_path)

        # define dataset
        max_encoder_length = self.model_hp.context_length
        max_prediction_length = self.model_hp.prediction_length
        training_cutoff = data["time_idx"].max() - max_prediction_length

        training = TimeSeriesDataSet(
            data[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="close",
            # weight="weight",
            group_ids=["pair"],
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            time_varying_unknown_reals=["open", "high", "low", "volume"],
        )

        # create validation and training dataset
        validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=training.index.time.max() + 1, stop_randomization=True)
        train_dataloader = training.to_dataloader(train=True, batch_size=self.model_hp.batch_size, num_workers=2)
        val_dataloader = validation.to_dataloader(train=False, batch_size=self.model_hp.batch_size, num_workers=2)

        # define trainer with early stopping
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min")
        lr_logger = LearningRateMonitor()
        trainer = pl.Trainer(
            max_epochs=self.model_hp.epochs,
            gpus=0,
            gradient_clip_val=0.1,
            limit_train_batches=30,
            callbacks=[lr_logger, early_stop_callback],
        )

        # create the model
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=0.03,
            hidden_size=self.model_hp.hidden_dim,
            attention_head_size=self.model_hp.num_heads,
            dropout=0.1,
            hidden_continuous_size=self.model_hp.hidden_dim / 2,
            output_size=7,
            loss=QuantileLoss(),
            log_interval=2,
            reduce_on_plateau_patience=4
        )
        print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

        # find optimal learning rate (set limit_train_batches to 1.0 and log_interval = -1)
        res = trainer.tuner.lr_find(
            tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, early_stop_threshold=1000.0, max_lr=0.3,
        )

        print(f"suggested learning rate: {res.suggestion()}")
        fig = res.plot(show=True, suggest=True)
        fig.show()

        # fit the model
        trainer.fit(
            tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
        )