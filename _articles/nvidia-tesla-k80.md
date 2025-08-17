---
layout: default
title: "NVIDIA Tesla K80 GPU"
date: 2025-08-02
categories: [gpu, hardware]

images:
  - /assets/articles/nvidia-tesla-k80/1.jpg
  - /assets/articles/nvidia-tesla-k80/2.jpg
  - /assets/articles/nvidia-tesla-k80/3.jpg
  - /assets/articles/nvidia-tesla-k80/4.jpg
  - /assets/articles/nvidia-tesla-k80/5.jpg
  - /assets/articles/nvidia-tesla-k80/6.jpg
  - /assets/articles/nvidia-tesla-k80/7.jpg
  - /assets/articles/nvidia-tesla-k80/8.jpg
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
- PyTorch 2.2.0

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
cd ~/llm
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available());print(torch.cuda.get_device_name(0));"
```
- Install LLM dependancies

```bash
pip install "transformers==4.46.3" "accelerate==0.34.2" "tokenizers<0.21" "safetensors<0.5" "diffusers==0.34.0"
```

## Dry-run!

### Mistral 7b

- Get the Mistral:

```bash
cd ~/llm
git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-v0.1 mistral
```
- Create script test_bad_cuda_mistral.py:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import time

print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))

model_path = "/home/sysadmin/llm/mistral"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model     = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    offload_folder="offload",
    torch_dtype=torch.float16
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

print(generator("Tell the story about sun.", max_new_tokens=120)[0]["generated_text"])
```

### Stable Diffusion v1.5

- Get the StableDiffusion 1.5

```bash
git lfs install
git clone https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5 sd1.5
```
- Create script test_bad_cuda_sd1.5.py:

```python
from diffusers import StableDiffusionPipeline
import torch

print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))

model_path = "/home/sysadmin/llm/sd1.5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="balanced",
    offload_folder="offload",
    safety_checker=None,
    feature_extractor=None,
    use_safetensors=True,
    local_files_only=True
)


out = pipe(
    prompt= "cat sitting on a chair",
    height=512, width=512, guidance_scale=9, num_inference_steps=80)
image = out.images[0]

image.save("test.png", format="PNG")
```
## It works!
> I guess that's why it worked — my heroic win!
