---
layout: default
title: "Compilation FlashAttention for CUDA 12.8"
date: 2025-12-26
categories: [llm, software]

images:
  - /assets/articles/general/flashattention_logo.png
  - /assets/articles/general/Logo_and_CUDA.png
---
> Date: {{ page.date | date: "%d.%m.%Y" }}  

# Compilation FlashAttention for CUDA 12.8 

{% include gallery.html images=page.images gallery_id=page.title %}

## Hot links
If not exist check archive.org  
- [CUDA toolkit](https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=runfile_local)
- [CUDA toolkit installer](https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run)
- [FlashAttention source code](https://github.com/Dao-AILab/flash-attention)

## Requirments
- Ubuntu 24.04
- PyTorch 2.7.1
- Python 3.12
- NVIDIA driver 570
- CUDA toolkit 12.8.0

## Test environment 
- Workstation 40 GB RAM, 500GB SSD, 750W Power supply 
- Ubuntu 24.04 LTS HWE Kernel
- Install python 3.12

> My test environment: HP Z440 + NVIDIA RTX 3090

## Ubuntu preparation

```bash
sudo apt-get install --install-recommends linux-generic-hwe-24.04
hwe-support-status --verbose
sudo apt dist-upgrade
sudo reboot
```

## Driver setup
- Install drivers **nvidia-driver-570**

```bash
sudo apt install nvidia-driver-570 clinfo
sudo reboot
```
- Install CUDA

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
sudo sh cuda_12.8.0_570.86.10_linux.run --toolkit --samples

echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' | sudo tee /etc/profile.d/cuda.sh
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' | sudo tee -a /etc/profile.d/cuda.sh
echo 'export CUDA_HOME=/usr/local/cuda-12.8' | sudo tee -a /etc/profile.d/cuda.sh
source /etc/profile.d/cuda.sh
```
- Check installation

```bash
nvidia-smi
clinfo
nvcc --version
```

## Install PyTorch
- Preapre python environment for CUDA:

```bash
mkdir -p ~/llm && cd ~/llm
python3 -m venv .venv_llm
source ./.venv_llm/bin/activate
python -m pip install --upgrade pip
pip install "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1" --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate
```
- Check PyTorch installation

```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available());print(torch.cuda.get_device_name(0));"
```

## Build FlashAttention
- Install build dependancies

```bash
pip install setuptools wheel
pip install packaging ninja
```
- Compile FlashAttention and install to virtalenv

```bash
MAX_JOBS=4 pip install "flash-attn==2.6.3" --no-build-isolation
```
- Check BitsAndBytes installation

```bash
python -m bitsandbytes
```

## It works!
