---
layout: default
title: "WAN 2.1 1.3b diffusers CUDA PyTorch Test"
date: 2025-12-30
categories: [llm, software]

images:
  - /assets/articles/pytorch-cuda-wan2-1-test-prompt/1.jpg
  - /assets/articles/pytorch-cuda-wan2-1-test-prompt/2.jpg
---
> Date: {{ page.date | date: "%d.%m.%Y" }}  

# WAN 2.1 1.3b diffusers CUDA PyTorch Test 

{% include gallery.html images=page.images gallery_id=page.title %}

{% include video.html 
    autoplay=false
    src="/assets/articles/pytorch-cuda-wan2-1-test-prompt/1.mp4" 
    src_webm="/assets/articles/pytorch-cuda-wan2-1-test-prompt/1.webm" 
%}

## Test environment 
- Workstation 40 GB RAM, 500GB SSD, 750W Power supply 
- Ubuntu 24.04 LTS HWE Kernel
- Install python 3.12

> My test environment: HP Z440 + NVIDIA RTX 3090

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
python3 -m venv .venv_llm_wan
source ./.venv_llm_wan/bin/activate
python -m pip install --upgrade pip
pip install "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1" --index-url https://download.pytorch.org/whl/cu128
pip install "bitsandbytes==0.46.1"
pip install transformers accelerate diffusers safetensors
pip install ftfy opencv-python imageio imageio-ffmpeg
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available());print(torch.cuda.get_device_name(0));"
```
- Expected responce

```
2.7.0+cu128
True
NVIDIA GeForce RTX 3090
```
- Check BitsAndBytes installation

```bash
python -m bitsandbytes
```

## Steps

### Get the WAN 2.1
```bash
git lfs install
git clone https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers wan2.1d
```

### Create script test_rocm_wan2.1d.py:
```python
import torch
from diffusers import AutoencoderKLWan, WanVACEPipeline, WanVACETransformer3DModel
from transformers import UMT5EncoderModel, AutoTokenizer

from diffusers.utils import load_image, export_to_video

model_dir = "/home/sysadmin/llm/wan2.1d"

vae = AutoencoderKLWan.from_pretrained(model_dir, subfolder="vae", torch_dtype=torch.float16)

pipe = WanVACEPipeline.from_pretrained(
    model_dir, 
    vae=vae, 
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")


prompt = "ginger cat sits on a chair"
negative = "text on screen, watermarks, blurry, distortion, low quality"

seed = torch.seed()
generator = torch.Generator("cuda").manual_seed(seed)

result = pipe(
    prompt=prompt,
    negative_prompt=negative,
    #reference_images=[ref], 
    height=512,
    width=512,
    num_frames=40, 
    num_inference_steps=20,
    guidance_scale=5.0,
    generator=generator
)

export_to_video(result.frames[0], f"test_image_wan2.1_{seed}.mp4", fps=16)
```

### Open test.mp4 and enjoy the result!
