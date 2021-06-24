#!/usr/bin/env bash

image=$1
tag=$2

ACCOUNT_ID=$(aws sts get-caller-identity --query Account | tr -d '"')
AWS_REGION=$(aws configure get region)
ecr_repo_prefix="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${image}"

aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin "${ecr_repo_prefix}_staging"
docker push "${ecr_repo_prefix}_staging:${tag}"

aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin "${ecr_repo_prefix}_prod"
docker push "${ecr_repo_prefix}_prod:${tag}"
