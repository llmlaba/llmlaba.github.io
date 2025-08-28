---
layout: default
title: "Compilation PyTorch BitsAndBytes for CUDA 11.4"
date: 2025-08-28
categories: [llm, software]

images:
  - /assets/articles/cuda-kepler-bitsandbytes/1.jpg
---

# Compilation PyTorch BitsAndBytes for CUDA 11.4 (Kepler)

{% include gallery.html images=page.images gallery_id=page.title %}

## Hot links
- [CUDA Toolkit](https://developer.nvidia.com/cuda-11-4-4-download-archive)
- [PyTorch 2.2.0](https://github.com/pytorch/pytorch/tree/release/2.2?tab=readme-ov-file#from-source)
- [BitsAndBytes 0.44.1](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.45.1/docs/source/installation.mdx)

## Requirments
- Ubuntu 20.04
- This GPU is outdated; most tools must be built from source
- PyTorch 2.2.0
- Python 3.8

## Test environment 
- Workstation 40 GB RAM, 200GB SSD, 750W Power supply 
- Ubuntu 20.04 
- Install python 3.8
- NVIDIA Tesla K80

> My test environment: HP Z440 + NVIDIA Tesla K80

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
- Install CUDA 11.4

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
- Get **PyTorch** sources 2.2.0

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
cd ~/llm
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available());print(torch.cuda.get_device_name(0));"
```

## Build BitsAndBytes

- Get **BitsAndBytes** sources 0.44.1

```bash
git clone -b 0.44.1 https://github.com/bitsandbytes-foundation/bitsandbytes.git
cd ./bitsandbytes
```

- Create custom `requirements-cus.txt`

```bash
cat <<EOF >> requirements-cus.txt
# Requirements used for local development
setuptools>=63
pytest~=8.3.3
einops~=0.8.0
wheel~=0.44.0
lion-pytorch~=0.2.2
scipy~=1.10.1
pandas~=2.0.2
matplotlib~=3.7.5
EOF
```

- Compile BitsAndBytes and install to virtalenv

```bash
pip install -r requirements-cus.txt
cmake -DNO_CUBLASLT=true -DCOMPUTE_BACKEND=cuda -S .
make
pip install .
```
- Check PyTorch installation

```bash
cd ~/llm
python -m bitsandbytes
```

## It works!
