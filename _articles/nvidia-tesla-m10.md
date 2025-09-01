---
layout: default
title: "NVIDIA Tesla M10 GPU"
date: 2025-07-05
categories: [gpu, hardware]

images:
  - /assets/articles/nvidia-tesla-m10/1.jpg
  - /assets/articles/nvidia-tesla-m10/2.jpg
  - /assets/articles/nvidia-tesla-m10/3.jpg
  - /assets/articles/nvidia-tesla-m10/4.jpg
  - /assets/articles/nvidia-tesla-m10/5.jpg
  - /assets/articles/nvidia-tesla-m10/6.jpg
  - /assets/articles/nvidia-tesla-m10/7.jpg
  - /assets/articles/nvidia-tesla-m10/8.jpg
---
> Date: {{ page.date | date: "%d.%m.%Y" }}  

# NVIDIA Tesla M10 GPU 

{% include gallery.html images=page.images gallery_id=page.title %}

## Limitations
- Linux only, there is no driver for windows
- This GPU is considered outdated; future versions of nvidia drivers may drop support for it
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
- Install drivers **nvidia-driver-570**
```bash
sudo apt install nvidia-driver-570 clinfo
sudo reboot
```

- Check installation
```bash
nvidia-smi
clinfo
```

## Check CUDA in python
- Priparing PyTorch
```bash
mkdir -p ~/llm && cd ~/llm
python3 -m venv .venv_llm
source ./.venv_llm/bin/activate
python -m pip install --upgrade pip
pip install "torch==2.5.0" "torchvision==0.20.0" "torchaudio==2.5.0" --index-url https://download.pytorch.org/whl/cu124
pip install "bitsandbytes==0.44.1"
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available());print(torch.cuda.get_device_name(0));"
```
- Expected responce
```
2.5.0+cu124
True
Tesla M10
```

## Dry-run!

### Mistral 7b

- Get the Mistral:
```bash
git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-v0.1 mistral
```
- Preapre python environment for CUDA 12:
```bash
mkdir -p ~/llm && cd ~/llm
python3 -m venv .venv_llm_mistral
source ./.venv_llm_mistral/bin/activate
python -m pip install --upgrade pip
pip install "torch==2.5.0" "torchvision==0.20.0" "torchaudio==2.5.0" --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate
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
    torch_dtype=torch.bfloat16
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
- Preapre python environment for CUDA:

```bash
mkdir -p ~/llm && cd ~/llm
python3 -m venv .venv_llm_sd1.5
source ./.venv_llm_sd1.5/bin/activate
python -m pip install --upgrade pip
pip install "torch==2.5.0" "torchvision==0.20.0" "torchaudio==2.5.0" --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate diffusers safetensors
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
    torch_dtype=torch.bfloat16,
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
> But looks ugly.
