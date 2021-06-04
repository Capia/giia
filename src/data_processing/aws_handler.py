from pathlib import Path

import boto3
import sagemaker
import tarfile

from utils import config
from utils.logger_util import LoggerUtil


class AWSHandler:
    logger = None
    sagemaker_session = None
    s3_bucket = None
    s3_bucket_resource = None
    s3_dataset_dir_uri = None
    s3_model_output_uri = None

    def __init__(self, logger: LoggerUtil, model_id: str):
        self.logger = logger
        self.sagemaker_session = sagemaker.Session()
        self.s3_bucket = self.sagemaker_session.default_bucket()
        self.s3_bucket_resource = boto3.resource('s3').Bucket(self.s3_bucket)
        self.s3_dataset_dir_uri = f"s3://{self.s3_bucket}/{model_id}/datasets"
        self.s3_model_output_uri = f"s3://{self.s3_bucket}/{model_id}/models"

    def upload_to_sagemaker_s3_bucket(self, dataset_dir_path: Path, dataset_channel_file: str, override=True):
        self.logger.log(f"Data will be uploaded to [{self.s3_bucket}]")

        local_file = dataset_dir_path / dataset_channel_file
        s3_path = f"{self.s3_dataset_dir_uri}/{dataset_channel_file}"

        assert s3_path.startswith('s3://')
        split = s3_path.split('/')
        path = '/'.join(split[3:])

        if len(list(self.s3_bucket_resource.objects.filter(Prefix=path))) > 0:
            if not override:
                self.logger.log('File s3://{}/{} already exists.\n'
                                'Set override to `True` to upload anyway.\n'.format(self.s3_bucket, s3_path), 'warning')
                return
            else:
                self.logger.log('Overwriting existing file')

        with open(local_file, 'rb') as data:
            self.s3_bucket_resource.put_object(Key=path, Body=data)
            self.logger.log(f"Uploaded {dataset_channel_file} to {self.s3_dataset_dir_uri}")

    def upload_train_datasets(self, dataset_dir_path: Path, override=True, filedataset_based=True):
        if filedataset_based:
            train_dataset_files = [
                config.METADATA_DATASET_FILENAME,
                config.TRAIN_DATASET_FILENAME,
                config.TEST_DATASET_FILENAME
            ]
        else:
            train_dataset_files = [
                config.TRAIN_CSV_FILENAME,
                config.TEST_CSV_FILENAME
            ]

        for file in train_dataset_files:
            self.upload_to_sagemaker_s3_bucket(dataset_dir_path, file, override)

    def download_model_from_s3(self, model_data_zip_path: str, local_artifact_dir: Path):
        # First download the compressed model
        model_data_zip_path = model_data_zip_path[1:]
        local_zip_path = local_artifact_dir / model_data_zip_path
        local_zip_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger.log(f"Downloading [{model_data_zip_path}] from s3 to [{str(local_zip_path)}]")
        self.s3_bucket_resource.download_file(model_data_zip_path, str(local_zip_path))
        self.logger.log(f"Download complete")

        # Then extract the compressed model
        local_model_path = local_zip_path.parent.parent / "model"
        local_model_path.mkdir(parents=True, exist_ok=True)

        self.logger.log(f"Extracting [{str(local_zip_path)}] to [{str(local_model_path)}]")
        tar = tarfile.open(str(local_zip_path), "r:gz")
        tar.extractall(path=str(local_model_path))
        tar.close()
        self.logger.log(f"Extract complete")

        return local_model_path

    def upload_model_to_s3(self, model_dir_path: Path):
        model_tar_filename = "model.tar.gz"
        model_tar_path = model_dir_path.parent / model_tar_filename

        # First compress the model
        self.logger.log(f"Compressing [{str(model_dir_path)}] to [{str(model_tar_path)}]")
        with tarfile.open(str(model_tar_path), "w:gz") as tar:
            tar.add(model_dir_path, arcname=model_dir_path.name)
        self.logger.log(f"Compress complete")

        # Then upload the compressed model
        s3_path = f"{model_dir_path.parent.stem}/{model_tar_filename}"
        fully_qualified_s3_path = f"s3://{self.s3_bucket}/{s3_path}"
        self.logger.log(f"Uploading [{model_tar_path}] to [{fully_qualified_s3_path}]")
        self.s3_bucket_resource.upload_file(str(model_tar_path), s3_path)
        self.logger.log(f"Upload complete")

        return fully_qualified_s3_path
