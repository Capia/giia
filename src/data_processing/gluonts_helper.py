import numpy as np

from gluonts.dataset.common import ListDataset, TrainDatasets, CategoricalFeatureInfo, MetaData
from gluonts.dataset.field_names import FieldName

from utils import config


def df_to_covariate_dataset(df, feature_columns, freq=config.DATASET_FREQ):
    return ListDataset(
        [
            {
                FieldName.START: df.index[0],
                FieldName.TARGET: df[column_name][:].values,
                FieldName.ITEM_ID: column_name,
                FieldName.FEAT_STATIC_CAT: np.array([idx]),
            } for idx, column_name in enumerate(feature_columns)
        ],
        freq=freq
    )


def df_to_multi_feature_dataset(df, feature_columns, freq=config.DATASET_FREQ):
    return ListDataset(
        [
            {
                FieldName.START: df.index[0],
                FieldName.TARGET: df["close"][:].values,
                FieldName.ITEM_ID: "close",
                FieldName.FEAT_DYNAMIC_REAL: [
                    df[column_name][:].values for column_name in feature_columns
                ],
            }
        ],
        freq=freq
    )


def df_to_univariate_dataset(df, freq=config.DATASET_FREQ):
    return ListDataset(
        [
            {
                FieldName.START: df.index[0],
                FieldName.TARGET: df["close"][:].values,
                FieldName.ITEM_ID: "close"
            }
        ],
        freq=freq
    )


def df_to_multivariate_target_dataset(df, feature_columns, freq=config.DATASET_FREQ):
    return ListDataset(
        [
            {
                FieldName.START: df.index[0],
                FieldName.TARGET: [df[column_name][:].values for column_name in feature_columns],
                FieldName.ITEM_ID: "ETH/USD",
            }
        ],
        freq=freq,
        one_dim_target=False
    )


def build_train_datasets(train_dataset, test_dataset):
    return TrainDatasets(
        metadata=MetaData(
            freq=config.DATASET_FREQ,
            # Purposely leave out prediction_length as it will couple the hyper parameter to the dataset
        ),
        train=train_dataset,
        test=test_dataset
    )


def get_feature_columns(df, exclude_close=True):
    covariate_blacklist = ["volume"]
    if exclude_close:
        covariate_blacklist.append("close")

    print(f"covariate_blacklist = [{covariate_blacklist}]")
    feature_columns = []
    for column_name in df.columns:
        if column_name not in covariate_blacklist:
            feature_columns.append(column_name)

    return feature_columns
