# Notebooks

## Using an AWS Trained Model Locally
You may want to download an AWS trained model to inspect and test it locally. If you are using the same notebook kernel 
as the one used to train the model, then the notebook will handle downloading the model in the "Load Model" step. 
Otherwise, you can download the model manually with the following steps:
```bash
# Download the model from S3. You can find this in the notebook output after training.
export MODEL_PATH=/giia-sff-1.0.3/models/mxnet-training-2023-02-11-14-31-11-050/output/model.tar.gz
aws s3 cp s3://${MODEL_PATH} ./out/local_cli/aws_model/model.tar.gz

#untar the model
tar -xvf ./out/local_cli/aws_model/model.tar.gz -C ./out/local_cli/model
```


