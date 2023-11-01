from pathlib import Path

import pandas as pd
from darts import TimeSeries
from darts.metrics import mape
from darts.models import TFTModel
from darts.utils import torch

from utils.model_torch import ModelTorchBase


class Model(ModelTorchBase):
    def train(self):
        device = self._describe_env()

        dataset_dir_path = Path(self.model_hp.dataset_dir)
        print("Dataset directory and files:")
        for path in dataset_dir_path.glob('**/*'):
            print(path)

        data = self._get_df_from_dataset_file(dataset_dir_path)
        data["date"] = pd.to_datetime(data['date'])
        data = data.set_index('date', drop=True)
        data.index = data.index.tz_localize(None)

        ts = TimeSeries.from_dataframe(
            df=data,
            # time_col="date",
            value_cols=["open", "close", "high", "low"],
        )
        # train, val = ts.split_after(0.75)
        train = ts.head(4000)
        val = ts.tail(1000)

        model = TFTModel(
            input_chunk_length=self.model_hp.context_length,
            output_chunk_length=self.model_hp.prediction_length,
            add_relative_index=True,

            # likelihood=LaplaceLikelihood(),
            batch_size=self.model_hp.batch_size,

            loss_fn=torch.nn.MSELoss(),
            likelihood=None,

            model_name=self._get_model_id(),
            work_dir=self.model_hp.model_dir,
            log_tensorboard=True,
        )
        model.fit(train, epochs=self.model_hp.epochs, num_loader_workers=8, verbose=True)
        model.save((Path(model.work_dir) / model.model_name / "model.pt").as_posix())

        # model = TFTModel.load((Path(self.model_hp.model_dir) / self._get_model_id() / "model.pt").as_posix())

        # Evaluate on test data
        val_predictions = model.predict(len(val), num_loader_workers=8)
        print(val_predictions.to_json())
        print("model {} obtains MAPE: {:.2f}%".format(model, mape(val, val_predictions)))

        print("Completed training")
        return model
