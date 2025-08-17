---
layout: default
title: "NVIDIA Tesla K80 GPU"
date: 2025-08-02
categories: [gpu, hardware]

images:
  - /assets/articles/nvidia-tesla-k80/1.jpg
  - /assets/articles/nvidia-tesla-k80/2.jpg
  - /assets/articles/nvidia-tesla-k80/3.jpg
---

# NVIDIA Tesla K80 GPU 

{% include gallery.html images=page.images gallery_id=page.title %}

## Hot links
- [CUDA Toolkit](https://developer.nvidia.com/cuda-11-4-4-download-archive)
- [PyTorch 2.2.0](https://github.com/pytorch/pytorch/tree/release/2.2?tab=readme-ov-file#from-source)

## Limitations
- Ubuntu 20.04 only
- This GPU is outdated; most tools must be built from source
- Required external fun

## Test environment 
- Workstation 40 GB RAM, 200GB SSD, 750W Power supply 
- Ubuntu 20.04 
- Install python 3.8

> My test environment: HP Z440

## Ubuntu preparation

```bash
sudo apt dist-upgrade
sudo reboot
```

## Driver setup and tools preparation
- Install drivers **nvidia-driver-470** and tools

```bash
sudo apt install nvidia-driver-470 clinfo cmake-mozilla python3.8-venv python3.8-dev git
sudo reboot
```
- Install CUDA

```bash
wget https://developer.download.nvidia.com/compute/cuda/11.4.4/local_installers/cuda_11.4.4_470.82.01_linux.run
sudo sh cuda_11.4.4_470.82.01_linux.run --toolkit --samples

echo 'export PATH=/usr/local/cuda-11.4/bin:$PATH' | sudo tee /etc/profile.d/cuda.sh
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH' | sudo tee -a /etc/profile.d/cuda.sh
echo 'export CUDA_HOME=/usr/local/cuda-11.4' | sudo tee -a /etc/profile.d/cuda.sh
source /etc/profile.d/cuda.sh
nvcc --version
```

- Check nvidia driver installation

```bash
nvidia-smi
clinfo
```

## Build PyTorch
- Prepare python environment

```bash
mkdir -p ~/llm && cd ~/llm
python3 -m venv .venv_llm
source ./.venv_llm/bin/activate
python -m pip install --upgrade pip
```
- Get **PyTorch** sources

```bash
git clone -b release/2.2 https://github.com/pytorch/pytorch.git
cd ./pytorch
```
- Compile PyTorch and install to virtalenv

```bash
pip install -r requirements.txt
USE_CUDA=1 python setup.py install
```
- Check PyTorch installation

```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available());print(torch.cuda.get_device_name(0));"
```
