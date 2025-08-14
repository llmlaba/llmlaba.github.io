---
layout: default
title: "Stable Diffusion v1.5 ROCm PyTorch Test"
date: 2025-08-09
categories: [llm, software]

images:
  - /assets/articles/pytorch-rocm-sd1-5-test/1.jpg
  - /assets/articles/pytorch-rocm-sd1-5-test/2.jpg
---

# Stable Diffusion v1.5 ROCm PyTorch Test 

{% include gallery.html images=page.images gallery_id=page.title %}

## Requirments 
- AMD Mi50/MI100 32Gb VRAM
- Workstation 40 GB RAM, 200GB SSD, 750W Power supply 
- Ubuntu 24.04 LTS HWE Kernel
- Install python 3.11 or 3.12

## Steps

### Get the StableDiffusion 1.5
```bash
git lfs install
git clone https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5 sd1.5
```
### Preapre python environment for ROCm:
```bash
python3 -m venv .venv_llm_sd1.5
source ./.venv_llm_sd1.5/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
pip install transformers accelerate diffusers safetensors
python .\test_rocm_sd1.5.py
```
### Create script test_rocm_sd1.5.py:
```python
from diffusers import StableDiffusionPipeline
import torch

print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))

model_path = "/home/sysadmin/llm/sd1.5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    safety_checker=None,
    feature_extractor=None,
    use_safetensors=True,
    local_files_only=True
)

pipe = pipe.to("cuda")

out = pipe(
    prompt= "cat sitting on a chair",
    height=512, width=512, guidance_scale=9, num_inference_steps=80)
image = out.images[0]

image.save("test.png", format="PNG")
```
### Open test.png and enjoy the result!
