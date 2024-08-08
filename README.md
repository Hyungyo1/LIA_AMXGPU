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
DOCKER_BUILDKIT=1 docker build -f examples/cpu/inference/python/llm/Dockerfile --build-arg COMPILE=ON -t lia-amxgpu:main .

# Run the container with GPU
docker run --rm -it --gpus all --privileged lia-amxgpu:main bash

# Activate environment variables
cd llm
source ./tools/env_activate.sh
cp lia/cxl/* /home/ubuntu/miniconda3/envs/py310/lib/python3.10/site-packages/transformers/models/opt/
cp lia/generation_utils.py ~/miniconda3/envs/py310/lib/python3.10/site-packages/transformers/generation/utils.py
cp lia/modeling_opt.py ~/miniconda3/envs/py310/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py
```

# Run Inference
Example Code:
```
OMP_NUM_THREADS=40 numactl -m 1 -C 40-79 python run.py --benchmark -m /home/storage/hyungyo2/opt-model/opt-30b/ --dtype bfloat16 --ipex --input-tokens 32 --max-new-tokens 32 --batch-size 2 --token-latency --num-iter 2 --num-warmup 1 --greedy --prefill-policy 0 --decoding-policy 1 --gpu-percentage 0 --num-minibatch 2 --gpu-percentage 0 --pin-weight
```
