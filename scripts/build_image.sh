#!/usr/bin/env bash

#
# This script expects to be ran from the root of the repository
#

image=$1
tag=$2
model_path=$3

ACCOUNT_ID=$(aws sts get-caller-identity --query Account | tr -d '"')
AWS_REGION=$(aws configure get region)


echo "Building image [${image}]"
docker build \
  -t ${image} \
  --build-arg MODEL_PATH=${model_path} \
  -f Dockerfile .

ecr_repo_prefix="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${image}"
docker tag ${image} "${ecr_repo_prefix}_staging:${tag}"
docker tag ${image} "${ecr_repo_prefix}_prod:${tag}"
