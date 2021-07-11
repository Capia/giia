#!/usr/bin/env bash

ACCOUNT_ID=$(aws sts get-caller-identity --query Account | tr -d '"')
AWS_REGION=$(aws configure get region)
ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/giia_freqtrade"

TAG="2021.5"
docker pull "freqtradeorg/freqtrade:${TAG}"
docker tag "freqtradeorg/freqtrade:${TAG}" "${ECR_REPO}:${TAG}"

aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin "${ECR_REPO}"
docker push "${ECR_REPO}:${TAG}"
