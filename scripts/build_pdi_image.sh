#!/usr/bin/env bash

#
# This script expects to be ran from the root of the repository
#

tag=$1
model_path=$2

ACCOUNT_ID=$(aws sts get-caller-identity --query Account | tr -d '"')
AWS_REGION=$(aws configure get region)
IMAGE="giia_probabilistic_distro_inference"

image="${IMAGE}"
echo "Building image [${image}]"

docker build \
  -t ${image} \
  --build-arg MODEL_PATH=${model_path} \
  -f Dockerfile .

ecr_repo_prefix="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
docker tag ${image} "${ecr_repo_prefix}/${image}:${tag}"
