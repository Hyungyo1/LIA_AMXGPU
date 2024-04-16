# IPEX Environment with CUDA support
IPEX_CUDA builds IPEX (Intel Extension for Pytorch) with CUDA-supported Pytorch.

# Docker-based environment setup
```
# Download the Git Repository
git clone https://github.com/Hyungyo1/IPEX_CUDA.git
cd intel-extension-for-pytorch
git submodule sync
git submodule update --init --recursive

# Build an image with the provided Dockerfile
DOCKER_BUILDKIT=1 docker build -f examples/cpu/inference/python/llm/Dockerfile --build-arg COMPILE=ON -t ipex-llm:cpu-gpu .

# Run the container with GPU
docker run --rm -it --gpus all --privileged ipex-llm:cpu-gpu bash

# Activate environment variables
cd llm
source ./tools/env_activate.sh
```
