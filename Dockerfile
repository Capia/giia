#FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-inference:1.8.0-cpu-py37-ubuntu16.04
FROM public.ecr.aws/lambda/python:3.7

# Copy inference code
## LAMBDA_TASK_ROOT is provided by the base image
COPY src/ ${LAMBDA_TASK_ROOT}/src

WORKDIR ${LAMBDA_TASK_ROOT}/src
ENV PYTHONPATH="${LAMBDA_TASK_ROOT}/src"

# Install image dependencies
RUN yum groupinstall -y "Development Tools"

# Install python dependencies
RUN pip3 install -r ${LAMBDA_TASK_ROOT}/src/requirements.txt --no-cache

# Copy model
## For consistency we use the same environment variable name (SM_MODEL_DIR) that sagemaker used during training
ENV SM_MODEL_DIR=${LAMBDA_TASK_ROOT}/giia_probabilistic_distro_inference
## MODEL_PATH is provided by the docker build command
ARG MODEL_PATH
ADD ${MODEL_PATH} ${SM_MODEL_DIR}/model/
#COPY ${MODEL_PATH} ${SM_MODEL_DIR}/
#RUN tar xfz "${SM_MODEL_DIR}/model.tar.gz" -C ${SM_MODEL_DIR}/model

CMD ["sff.handler"]
