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

# Run the container with GPU (please mount a directory to save files onto)
docker run --rm -it --gpus all -v /$(mount_dir):/home/storage --privileged lia-amxgpu:main bash

# Activate environment variables (need to do it every time you create a docker container)
cd llm
source ./tools/env_activate.sh
cp lia/generation_utils.py ~/miniconda3/envs/py310/lib/python3.10/site-packages/transformers/generation/utils.py
cp lia/modeling_opt.py ~/miniconda3/envs/py310/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py
```

# Generating Dummy Model Weights (Just need to do it once, will save the dummy model weights to your mounted directory)
```
bash opt-dummy-weight.sh
```

# Run Inference
Bash script for Online Inference Profiling:
```
bash profile.sh
```
