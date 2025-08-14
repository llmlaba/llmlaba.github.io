---
layout: default
title: "Stable Diffusion v1.5 CUDA low memory GPU PyTorch Test"
date: 2025-06-25
categories: [llm, software]

images:
  - /assets/articles/pytorch-bad-cuda-sd1-5-test/1.jpg
---

# Stable Diffusion v1.5 Bad CUDA PyTorch Test 

{% include gallery.html images=page.images gallery_id=page.title %}

## Requirments 
- NVIDIA RTX 4070 8GB
- Workstation 64 GB RAM, 200GB SSD
- Windows 11
- Install python 3.11
- NVIDIA Driver 577

> Memory on my gaming laptop is not sufficient for running Stable Diffusion v1.5, so I will try to use `device_map="balanced"` and specify `offload_folder="offload"` as the directory for model offloading.

## Steps

### Get the StableDiffusion 1.5
```bash
git lfs install
git clone https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5 sd1.5
```
### Preapre python environment for CUDA:
```bash
python -m venv .venv_llm_sd1.5
.\.venv_llm_sd1.5\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate diffusers safetensors
python .\test_cuda_sd1.5.py
```
### Create script test_cuda_sd1.5.py:
```python
from diffusers import StableDiffusionPipeline
import torch

print(torch.cuda.is_available())

pipe = StableDiffusionPipeline.from_pretrained(
    "C:\\Users\\admin\\llm\\sd1.5",
    torch_dtype=torch.float16,
    device_map="balanced",
    offload_folder="offload",
    safety_checker=None,
    feature_extractor=None,
    use_safetensors=True
)

out = pipe(
    prompt= "vodka in the moon", 
    height=512, width=512, guidance_scale=9, num_inference_steps=80)
image = out.images[0]

image.save("test.png", format="PNG")
```
### Open test.png and enjoy the result!
> The model will generate image very slow due to limited GPU memory and offloading.
