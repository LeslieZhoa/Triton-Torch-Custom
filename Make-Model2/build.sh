#!/bin/bash

set -x

BASE_IMAGE=nvcr.io/nvidia/pytorch:20.06-py3
REPO=machine_learning/face_ai/icpm

IMAGE_NAME=${REPO}/content/pytorch_selfop_serving:latest

DIR="$(dirname "$(readlink -f "$0")")"
#docker pull ${BASE_IMAGE}
docker build -t ${IMAGE_NAME} \
    --build-arg BASE_IMAGE=${BASE_IMAGE} \
    -f ${DIR}/Dockerfile ${DIR}
# docker push ${IMAGE_NAME}
