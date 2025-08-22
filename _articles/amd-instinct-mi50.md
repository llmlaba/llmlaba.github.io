---
layout: default
title: "AMD Instinct Mi50 GPU"
date: 2025-08-07
categories: [gpu, hardware]

images:
  - /assets/articles/amd-instinct-mi50/1.jpg
  - /assets/articles/amd-instinct-mi50/2.jpg
  - /assets/articles/amd-instinct-mi50/3.jpg
---

# AMD Instinct Mi50 GPU 

{% include gallery.html images=page.images gallery_id=page.title %}

## Hot links
If not exist check archive.org  
- [Download driver](https://www.amd.com/en/support/downloads/drivers.html/accelerators/instinct/instinct-mi-series/instinct-mi50-32gb.html)
- [Install installer](https://amdgpu-install.readthedocs.io/en/latest/install-prereq.html#installing-the-installer-package)
- [Install driver option 1](https://amdgpu-install.readthedocs.io/en/latest/install-script.html)
- [Install driver option 2](https://amdgpu-install.readthedocs.io/en/latest/install-installing.html#installing-the-all-open-use-case)

## Limitations
- Linux only, there is no driver for windows
- This GPU is considered outdated; future versions of ROCm may drop support for it
- Required external fun

## Test environment 
- Workstation 40 GB RAM, 200GB SSD, 750W Power supply 
- Ubuntu 24.04 LTS HWE Kernel
- Install python 3.12

> My test environment: HP Z440

## Ubuntu preparation
```bash
sudo apt-get install --install-recommends linux-generic-hwe-24.04
hwe-support-status --verbose
sudo apt dist-upgrade
sudo reboot
```

## Driver setup
- Install drivers
```bash
gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv 9386B48A1A693C5C
gpg --export --armor 9386B48A1A693C5C | sudo apt-key add -
cd /tmp
wget https://repo.radeon.com/amdgpu-install/6.4.2.1/ubuntu/noble/amdgpu-install_6.4.60402-1_all.deb
sudo apt install ./amdgpu-install_6.4.60402-1_all.deb
sudo apt install rocm-smi clinfo rocminfo
sudo amdgpu-install --usecase=workstation --vulkan=pro --opencl=rocr
```
- Add your user to the required groups to enable access to ROCm drivers and GPU hardware
```bash
sudo groupadd kfd
sudo usermod -aG video $USER
sudo usermod -aG render $USER
sudo usermod -aG kfd $USER
```
- Check installation
```bash
clinfo
sudo /opt/rocm/bin/rocminfo
sudo rocm-smi
```

## Check ROCm in python
- Priparing PyTorch
```bash
mkdir -p ~/llm && cd ~/llm
python3 -m venv .venv_llm
source ./.venv_llm/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.hip);print(torch.cuda.get_device_name(0));"
```
- Expected responce
```
2.4.1+rocm6.0
True
6.0.32830-d62f6a171
AMD Instinct MI50/MI60
```

## Done!
> Your workstation ready to test most common models from huggingface.co 

> For example: [Mistral 7b ROCm PyTorch Test](/articles/pytorch-rocm-mistral-test.html)
