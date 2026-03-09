---
layout: default
title: "sd.cpp - run Stable Diffuision everywhere"
date: 2026-03-08
categories: [software, llm]

images:
  - /assets/articles/general/sd-cpp-logo.png
  - /assets/articles/sd-cpp-simple/1.jpg
  - /assets/articles/sd-cpp-simple/2.jpg
  - /assets/articles/sd-cpp-simple/3.jpg
  - /assets/articles/sd-cpp-simple/4.jpg
  - /assets/articles/sd-cpp-simple/5.jpg
---
> Date: {{ page.date | date: "%d.%m.%Y" }}  

# sd.cpp - run Stable Diffuision everywhere

> sd.cpp is a universal solution for running quantized stable diffusion models on a wide range of hardware.

{% include gallery.html images=page.images gallery_id=page.title %}

## RUN Stable diffusion XL from scratch with sd.cpp runtime

> My test environment: Mac Pro 7.1 + NVIDIA CMP 170HX

### Requirments 
- PC 16 GB RAM, 500GB SSD
- Ubuntu 24.04 LTS
- NVIDIA GPU 10GB VRAM or higher

### Driver setup:
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
- Developer packages

```bash
sudo apt install -y python3-venv python3-dev git git-lfs
```

### Preapre sd.cpp environment:

```bash
mkdir -p ~/sdcpp && cd ~/sdcpp
git clone https://github.com/leejet/stable-diffusion.cpp
cd stable-diffusion.cpp
git submodule init
git submodule update

mkdir build && cd build
cmake .. -DSD_CUDA=ON
cmake --build . --config Release
```

### Get the Stable Diffusion XL:

```bash
mkdir -p ~/llm && cd ~/llm
git lfs install
git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 sdxl
```

### Conversion the original weights to GGUF format:

```bash
cd ~/sdcpp
./stable-diffusion.cpp/build/bin/sd-cli -M convert \
  -m ~/llm/sdxl/sd_xl_base_1.0_0.9vae.safetensors \
  -o ./sd_xl_base_1.0_0_Q8_0.gguf -v --type q8_0
```

### Run sd.cpp server:
```bash
cd ~/sdcpp
export CUDA_VISIBLE_DEVICES=0
./stable-diffusion.cpp/build/bin/sd-server \
  -m ./sd_xl_base_1.0_0_Q8_0.gguf \
  --vae-on-cpu \
  --listen-ip 0.0.0.0 \
  --listen-port 8081 \
  --seed -1
```

### Call sd.cpp api to generate picture:

```bash
curl -s http://127.0.0.1:8081/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sdxl",
    "prompt": "A lovely cat<sd_cpp_extra_args>{\"seed\": 357925}</sd_cpp_extra_args>",
    "n": 1,
    "size": "1024x1024",
    "response_format": "b64_json"
  }' | jq -r '.data[0].b64_json' | base64 --decode > out.png
```

### Enjoy the result!
