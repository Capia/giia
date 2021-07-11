#!/usr/bin/env bash

tag=$1

ACCOUNT_ID=$(aws sts get-caller-identity --query Account | tr -d '"')
AWS_REGION=$(aws configure get region)

IMAGE="giia_probabilistic_distro_inference"
ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${IMAGE}"

aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin "${ECR_REPO}"
docker push "${ECR_REPO}:${tag}"
