---
layout: default
title: "WAN 2.1 1.3b diffusers ROCm PyTorch Test"
date: 2025-08-11
categories: [draft]
---

# WAN 2.1 1.3b diffusers ROCm PyTorch Test 

## Requirments 
- AMD MI200 64Gb VRAM
- Workstation 40 GB RAM, 200GB SSD, 750W Power supply 
- Ubuntu 24.04 LTS HWE Kernel
- Install python 3.11 or 3.12

## Steps

### Get the WAN 2.1
```bash
git lfs install
git clone git clone https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers wan2.1d
```
### Preapre python environment for ROCm:
```bash
python3 -m venv .venv_llm_wan2.1d
source ./.venv_llm_wan2.1d/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
pip install transformers accelerate diffusers safetensors ftfy
python .\test_rocm_wan2.1d.py
```
### Create script test_rocm_wan2.1d.py:
```python
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from diffusers import AutoModel, WanVACEPipeline
from diffusers.utils import load_image, export_to_video

print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))

model_id = "/home/sysadmin/llm/wan2.1d"

vae = AutoModel.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanVACEPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)

pipe.to("cuda")

# 4) утилиты под размер (VACE любит кратность patch_size и масштабу VAE)
def fit_hw_for_pipe(width, height, pipe, max_area=832*480):
    aspect = height / width
    mod = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    H = round(np.sqrt(max_area * aspect)) // mod * mod
    W = round(np.sqrt(max_area / aspect)) // mod * mod
    return int(W), int(H)

# 5) входы: одна или несколько референс-картинок
ref = load_image("test.png")  # или URL
W, H = fit_hw_for_pipe(ref.width, ref.height, pipe, max_area=832*480)  # целимся в 480p (832×480)

# (опционально) подгоним картинку под размер, это безопасно для conditioning
ref = ref.resize((W, H), Image.LANCZOS)

prompt = "red cat sits on a windowsill"
negative = "text on screen, watermarks, blurry, distortion, low quality"

# 6) запуск: reference_images + prompt
result = pipe(
    prompt=prompt,
    negative_prompt=negative,
    reference_images=[ref],  # можно список из нескольких референсов
    height=H,
    width=W,
    num_frames=81,           # ~5 сек при 16 fps
    num_inference_steps=40,  # шаги диффузии
    guidance_scale=5.0
).frames[0]

export_to_video(result, "test.mp4", fps=16)
```
### Open test.mp4 and enjoy the result!
> Currently it not works I down't have mi200 
