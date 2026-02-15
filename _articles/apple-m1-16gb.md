---
layout: default
title: "Apple M1 16Gb Unified RAM"
date: 2025-09-09
categories: [hardware]

images:
  - /assets/articles/apple-m1-16gb/1.jpg
  - /assets/articles/apple-m1-16gb/2.jpg
  - /assets/articles/apple-m1-16gb/3.jpg
---
> Date: {{ page.date | date: "%d.%m.%Y" }}  

# Apple M1 16Gb Unified RAM

{% include gallery.html images=page.images gallery_id=page.title %}

## Limitations
- Apple MPS still about in beta in PyTorch

## Test environment 
- MBP 16 inch 2021
- Mac OS 15.5
- Python 3.11

## MacOS preparation
```bash
brew install python@3.11
```

## Dry-run!

### Stable Diffusion v1.5

- Prepare Python environment for MPS:

```bash
mkdir -p ~/llm && cd ~/llm
python3.11 -m venv _venv_llm
source ./_venv_llm/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers accelerate diffusers safetensors
python3 -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available());print(torch.backends.mps.is_built());"
```
- Get the StableDiffusion 1.5

```bash
git lfs install
git clone https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5 sd1.5
```
- Create script test_mps_sd1.5.py:

```python
from diffusers import StableDiffusionPipeline
import torch

print("MPS available:", torch.backends.mps.is_available())
print("MPS support:", torch.backends.mps.is_built())

model_path = "/Users/mattcosta/llm/sd1.5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    safety_checker=None,
    feature_extractor=None,
    use_safetensors=True,
    local_files_only=True
).to("mps")

out = pipe(
    prompt= "cat sitting on a chair",
    height=512, width=512, guidance_scale=9, num_inference_steps=80)
image = out.images[0]

image.save("test.png", format="PNG")
```

### Run test
> Check `powermetrics` during each test `while true; do powermetrics --samplers gpu_power -i500 -n1; sleep 1; done`

- Run PyTorch

```bash
python ./test_mps_sd1.5.py
```

## It works!
