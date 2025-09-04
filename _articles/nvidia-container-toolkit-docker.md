---
layout: default
title: "NVIDIA Container Toolkit for Docker"
date: 2025-09-03
categories: [general]

images:
  - /assets/articles/nvidia-container-toolkit-docker/1.jpg
  - /assets/articles/nvidia-container-toolkit-docker/2.jpg
  - /assets/articles/general/nvidia-container-toolkit.png
---
> Date: {{ page.date | date: "%d.%m.%Y" }}  

# NVIDIA Container Toolkit for Docker
> In this articale detailed described how to install NVIDIA Container Toolkit for Docker 

{% include gallery.html images=page.images gallery_id=page.title %}

## Hot links
- [Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Supported platforms](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/supported-platforms.html)
- [Package for Ubuntu 18.04 *(just in case)*](https://nvidia.github.io/libnvidia-container/stable/ubuntu18.04/nvidia-container-toolkit.list)

## Supported platforms

| OS |
|-|
| Ubuntu 24.04 |
| Ubuntu 22.04 |
| Ubuntu 20.04 |

## Requirments

- This package requires the NVIDIA driver (>= 340.29) to be installed separately.

## Test environment 
- NVIDIA Tesla V100
- Workstation 40 GB RAM, 500GB SSD, 750W Power supply 
- Ubuntu 24.04 LTS
- Docker CE

> My test environment: HP Z440 + NVIDIA Tesla V100

## Preparation

- Prepare your workstation with following instruction
[NVIDIA Tesla V100 GPU SXM2](/articles/nvidia-tesla-v100-sxm2.html)

- Check installation
```bash
nvidia-smi
clinfo
```

## Install NVIDIA Container Toolkit
- Install GPG key for ubuntu nvidia repo

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
```
- Add nvidia toolkit repo

```bash
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```
- Update apt repo data

```bash
sudo apt update
```
- Install toolkit

```bash
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
sudo apt install -y \
    nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
```

- Configure docker for nvidia plugin

```bash
sudo nvidia-ctk runtime configure --runtime=docker
cat /etc/docker/daemon.json
sudo reboot
```

- Check nvidia container toolkit configuration
> Docker image with cuda not required for testing, nvidia plugin will mount nvidia-smi executable directly to container

```bash
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
```

## What docker run `--gpus all` actually do?
Something like this:

```bash
docker run --rm \
  --device=/dev/nvidia0 \
  --device=/dev/nvidiactl \
  --device=/dev/nvidia-uvm \
  -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro \
  -v /usr/lib/x86_64-linux-gnu/libnvidia-ml.so:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so:ro \
  -e LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so \
   ubuntu:24.04 nvidia-smi
```
> Can we use nvidia GPU without nvidia-container-toolkit in docker?  
> We can, but it may cause some unexpected behaviour
