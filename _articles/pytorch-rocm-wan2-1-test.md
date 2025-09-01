---
layout: default
title: "WAN 2.1 1.3b diffusers ROCm PyTorch Test"
date: 2025-08-11
categories: [draft]
---
> Date: {{ page.date | date: "%d.%m.%Y" }}  

# WAN 2.1 1.3b diffusers ROCm PyTorch Test 

## Requirments 
- AMD MI200 64Gb VRAM
- Workstation 40 GB RAM, 200GB SSD, 750W Power supply 
- Ubuntu 24.04 LTS HWE Kernel
- Install python 3.11 or 3.12

## Steps

### Preapre python environment for ROCm + PyTorch + BitsAndBytes:
> Prepare 
[Compilation BitsAndBytes for ROCm 6.2](/articles/rocm-bitsandbytes.html)

- Check PyTorch installation

```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.hip);print(torch.cuda.get_device_name(0));"
```
- Check BitsAndBytes installation

```bash
python -m bitsandbytes
```

### Get the WAN 2.1
```bash
git lfs install
git clone git clone https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers wan2.1d
```

### Create script test_rocm_wan2.1d.py:
```python
import torch
from diffusers import AutoencoderKLWan, WanVACEPipeline, WanVACETransformer3DModel
from diffusers import BitsAndBytesConfig as DF_BNB
from transformers import UMT5EncoderModel, AutoTokenizer
from transformers import BitsAndBytesConfig as TF_BNB
import time

import numpy as np
from PIL import Image
from diffusers.utils import load_image, export_to_video

model_dir = "/home/sysadmin/llm/wan2.1d"

bnb_tx = DF_BNB(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
transformer = WanVACETransformer3DModel.from_pretrained(
    model_dir, 
    subfolder="transformer",
    quantization_config=bnb_tx, 
    torch_dtype=torch.float16
)

bnb_te = TF_BNB(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
text_encoder = UMT5EncoderModel.from_pretrained(
    model_dir, 
    subfolder="text_encoder",
    quantization_config=bnb_te, 
    torch_dtype=torch.float16
)

vae = AutoencoderKLWan.from_pretrained(model_dir, subfolder="vae", torch_dtype=torch.float32)

pipe = WanVACEPipeline.from_pretrained(
    model_dir, 
    vae=vae, 
    transformer=transformer, 
    text_encoder=text_encoder,
    torch_dtype=torch.float16
)
#).to("cuda")

pipe.enable_attention_slicing("max")
pipe.enable_model_cpu_offload() 

#time.sleep(30)

####

def fit_hw_for_pipe(width, height, pipe, max_area=832*480):
    aspect = height / width
    mod = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    H = round(np.sqrt(max_area * aspect)) // mod * mod
    W = round(np.sqrt(max_area / aspect)) // mod * mod
    return int(W), int(H)

ref = load_image("test_bnb.png") 
W, H = fit_hw_for_pipe(ref.width, ref.height, pipe, max_area=832*480)

ref = ref.resize((W, H), Image.LANCZOS)

prompt = "ginger cat sits on a chair"
negative = "text on screen, watermarks, blurry, distortion, low quality"

result = pipe(
    prompt=prompt,
    negative_prompt=negative,
    reference_images=[ref], 
    height=H,
    width=W,
    num_frames=40, 
    num_inference_steps=20,
    guidance_scale=5.0
).frames[0]

export_to_video(result, "test.mp4", fps=16)
```
### Open test.mp4 and enjoy the result!
> Currently it not works I down't have mi200 
