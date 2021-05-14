import sagemaker
import boto3
import botocore

from utils import config


class DeployEnv(object):
    def __init__(self):
        self._client = None
        self._runtime_client = None

    def is_deployed(self):
        """
        Checks if the model is deployed.
        IMPORTANT: always returns `False` for local endpoints as LocalSagemakerClient.describe_endpoint()
        seems to always throw:
        botocore.exceptions.ClientError: An error occurred (ValidationException) when calling the describe_endpoint operation: Could not find local endpoint
        """
        _is_deployed = False
        try:
            self.client().describe_endpoint(EndpointName=config.MODEL_NAME)
            _is_deployed = True
        except botocore.exceptions.ClientError as e:
            pass

        return _is_deployed

    def runtime_client(self):
        if self._runtime_client:
            return self._runtime_client

        if self.is_local():
            self._runtime_client = sagemaker.local.LocalSagemakerRuntimeClient()
        else:
            self._runtime_client = boto3.client('sagemaker-runtime')

        return self._runtime_client

    def client(self):
        if self._client:
            return self._client

        if self.is_local():
            self._client = sagemaker.local.LocalSagemakerClient()
        else:
            self._client = boto3.client('sagemaker')

        return self._client

    def is_local(self):
        return self.current_env() == 'local'

    def is_production(self):
        return self.current_env() == 'production'
