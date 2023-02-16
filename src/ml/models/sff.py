#
# NOTE: This file must stay at the root of the `./src` directory due to sagemaker-local stripping the path from the
#  entry_point. Follow this issue for new developments https://github.com/aws/sagemaker-python-sdk/issues/1597.
#
# GluonTS provides an example module for serving models. This module is based on that example, and includes support
#  Lambda functions.
#

from pathlib import Path

from gluonts.dataset.common import load_datasets
from gluonts.dataset.stat import calculate_dataset_statistics
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import backtest_metrics
from gluonts.mx.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.mx.trainer import Trainer

from utils import config
from utils.model import ModelBase


class Model(ModelBase):
    def train(self):
        self._describe_model()

        dataset_dir_path = Path(self.model_hp.dataset_dir)
        print("Dataset directory and files:")
        for path in dataset_dir_path.glob('**/*'):
            print(path)

        datasets = load_datasets(
            metadata=(dataset_dir_path / config.METADATA_DATASET_FILENAME).parent,
            train=(dataset_dir_path / config.TRAIN_DATASET_FILENAME).parent,
            test=(dataset_dir_path / config.TEST_DATASET_FILENAME).parent,
            cache=True
        )
        train_stats = calculate_dataset_statistics(datasets.train)
        print(f"Train dataset stats: {train_stats}")
        test_stats = calculate_dataset_statistics(datasets.test)
        print(f"Test dataset stats: {test_stats}")

        if not self.model_hp.num_batches_per_epoch:
            self.model_hp.num_batches_per_epoch = train_stats.max_target_length // self.model_hp.batch_size
            print(f"Defaulting num_batches_per_epoch to: [{self.model_hp.num_batches_per_epoch}] "
                  f"= (length of train dataset [{train_stats.max_target_length}]) "
                  f"/ (batch size [{self.model_hp.batch_size}])")
        print(f"Note that the max number of samples from the training dataset that can be used is "
              f"[{self.model_hp.num_batches_per_epoch * self.model_hp.batch_size * self.model_hp.epochs}]. This should "
              f"be larger than the number of samples in the training dataset [{train_stats.max_target_length}] or you "
              f"risk not using the full dataset.")

        ctx = self._get_ctx()
        distr_output = self._get_distr_output()
        num_hidden_dimensions = self._get_hidden_dimensions()

        estimator = SimpleFeedForwardEstimator(
            context_length=self.model_hp.context_length,
            prediction_length=self.model_hp.prediction_length,

            num_hidden_dimensions=num_hidden_dimensions,
            distr_output=distr_output,
            batch_size=self.model_hp.batch_size,

            trainer=Trainer(
                ctx=ctx,
                epochs=self.model_hp.epochs,
                num_batches_per_epoch=self.model_hp.num_batches_per_epoch,
                # learning_rate=self.model_hp.learning_rate
            )
        )

        # Train the model
        predictor = estimator.train(
            training_data=datasets.train,
            validation_data=datasets.test
        )

        net_name = type(predictor.prediction_net).__name__
        num_model_param = estimator.trainer.count_model_params(predictor.prediction_net)
        print(f"Number of parameters in {net_name}: {num_model_param}")

        # Evaluate trained model on test data. This will serialize each of the agg_metrics into a well formatted log.
        # We use this to capture the metrics needed for hyperparameter tuning
        agg_metrics, item_metrics = backtest_metrics(
            test_dataset=datasets.test,
            predictor=predictor,
            evaluator=Evaluator(
                quantiles=[0.1, 0.5, 0.9]
            ),
            num_samples=100,  # number of samples used in probabilistic evaluation
        )

        self._print_metrics(agg_metrics, item_metrics, datasets.metadata)

        # Save the model
        predictor.serialize(Path(self.model_hp.model_dir))

        print("Completed training")
        return predictor
