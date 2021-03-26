import boto3
import sagemaker
from utils.logger_util import LoggerUtil


class Upload:
    logger = None
    s3_dataset_dir_uri = None
    s3_model_output_uri = None

    sagemaker_session = None
    s3_bucket = None
    s3_bucket_resource = None

    def __init__(self, logger: LoggerUtil, model_id: str):
        self.logger = logger

        # Initialize static variables
        if not Upload.sagemaker_session:
            Upload.sagemaker_session = sagemaker.Session()
        if not Upload.s3_bucket:
            Upload.s3_bucket = Upload.sagemaker_session.default_bucket()
        if not Upload.s3_bucket_resource:
            Upload.s3_bucket_resource = boto3.resource('s3').Bucket(Upload.s3_bucket)

        self.s3_dataset_dir_uri = f"s3://{self.s3_bucket}/{model_id}"
        self.s3_model_output_uri = f"s3://{self.s3_bucket}/{model_id}/model"

    def upload_to_sagemaker_s3_bucket(self, dataset_dir_path, dataset_channel_file, override=True):
        # TODO: s3_bucket is None?
        self.logger.log("Data will be uploaded to: ", self.s3_bucket)

        local_file = dataset_dir_path / f"{dataset_channel_file}"
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
            self.logger.log('Uploading file to {}'.format(s3_path))
            self.s3_bucket_resource.put_object(Key=path, Body=data)
