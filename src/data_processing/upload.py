import boto3
import s3fs
import sagemaker
from utils.logging import LoggerUtil


class Upload:
    sagemaker_session = None
    s3_bucket = None
    s3_bucket_resource = None
    logger = None

    def __init__(self, logger: LoggerUtil):
        self.logger = logger
        # Initialize static variables
        if not Upload.sagemaker_session:
            Upload.sagemaker_session = sagemaker.Session()
        if not Upload.s3_bucket:
            Upload.s3_bucket = Upload.sagemaker_session.default_bucket()
        if not Upload.s3_bucket_resource:
            Upload.s3_bucket_resource = boto3.resource('s3').Bucket(Upload.s3_bucket)

    def upload_to_sagemaker_s3_bucket(self, model_name: str, train_dataset_path: str, test_dataset_path: str):
        self.logger.log("Data will be uploaded to: ", self.s3_bucket)

        s3_train_dataset_path = "s3://{}/{}/train".format(self.s3_bucket, model_name)
        s3_test_dataset_path = "s3://{}/{}/test".format(self.s3_bucket, model_name)

        self.copy_to_s3(train_dataset_path, s3_train_dataset_path + "/train.csv")
        self.copy_to_s3(test_dataset_path, s3_test_dataset_path + "/test.csv")

        # Check if the data was uploaded correctly
        s3filesystem = s3fs.S3FileSystem()
        with s3filesystem.open(s3_train_dataset_path + "/train.csv", 'rb') as fp:
            self.logger.log(fp.readline().decode("utf-8")[:100] + "...")

        return s3_train_dataset_path, s3_test_dataset_path

    def copy_to_s3(self, local_file, s3_path, override=True):
        assert s3_path.startswith('s3://')
        split = s3_path.split('/')
        path = '/'.join(split[3:])

        if len(list(self.s3_bucket_resource.objects.filter(Prefix=path))) > 0:
            if not override:
                self.logger.log('File s3://{}/{} already exists.\n'
                                'Set override to `True` to upload anyway.\n'.format(self.s3_bucket, s3_path))
                return
            else:
                self.logger.log('Overwriting existing file')
        with open(local_file, 'rb') as data:
            self.logger.log('Uploading file to {}'.format(s3_path))
            self.s3_bucket_resource.put_object(Key=path, Body=data)
