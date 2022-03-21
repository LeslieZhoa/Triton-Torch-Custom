#!/bin/bash

set -x

BASE_IMAGE=nvcr.io/nvidia/tritonserver:20.06-py3
REPO=machine_learning/face_ai/icpm

IMAGE_NAME=${REPO}/content/selfop/models/pytorch_serving_gpu_pytorch_example:latest

DIR="$(dirname "$(readlink -f "$0")")"
#docker pull ${BASE_IMAGE}
docker build -t ${IMAGE_NAME} \
    --build-arg BASE_IMAGE=${BASE_IMAGE} \
    -f ${DIR}/Dockerfile ${DIR}
# docker push ${IMAGE_NAME}
