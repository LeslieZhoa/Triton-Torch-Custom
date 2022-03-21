nvidia-docker run --gpus '"device=1"'  -p 8011:8001 -it --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -t machine_learning/face_ai/icpm/content/selfop/models/pytorch_serving_gpu_pytorch_example:latest
