from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
from darts import TimeSeries
from darts.metrics import mase
from darts.models import TCNModel
from darts.utils.likelihood_models import LaplaceLikelihood

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
            # value_cols=["open", "close", "high", "low", "volume"],
        )
        train, val = ts.split_after(0.75)

        model = TCNModel(
            input_chunk_length=self.model_hp.context_length,
            output_chunk_length=self.model_hp.prediction_length,

            kernel_size=1,
            num_filters=1,
            num_layers=1,
            dilation_base=1,

            # likelihood=LaplaceLikelihood(),
            batch_size=self.model_hp.batch_size,
            log_tensorboard=True,
            work_dir=self.model_hp.model_dir,
        )
        model.fit(train, epochs=self.model_hp.epochs, verbose=True)

        # model = TCNModel.load((Path(self.model_hp.model_dir) / "2023-03-28_16.23.59.748213_torch_model_run_24273" / "model.pt").as_posix())

        # Save model
        model.save((Path(model.work_dir) / model.model_name / "model.pt").as_posix())

        # Evaluate on test data
        val_predictions = model.predict(len(val))
        print("model {} obtains MAPE: {:.2f}%".format(model, mase(val, val_predictions)))

        print("Completed training")
        return model
