nvidia-docker run --gpus '"device=0"'  -p 8079:8001 -it --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -t machine_learning/face_ai/icpm/content/pytorch_selfop_serving:latest
