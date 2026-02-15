---
layout: default
title: "NVIDIA Tesla V100 GPU SXM2"
date: 2025-09-03
categories: [gpu, hardware]

images:
  - /assets/articles/nvidia-tesla-v100-sxm2/1.jpg
  - /assets/articles/nvidia-tesla-v100-sxm2/2.jpg
  - /assets/articles/nvidia-tesla-v100-sxm2/3.jpg
  - /assets/articles/nvidia-tesla-v100-sxm2/4.jpg
  - /assets/articles/nvidia-tesla-v100-sxm2/5.jpg
  - /assets/articles/nvidia-tesla-v100-sxm2/6.jpg
  - /assets/articles/nvidia-tesla-v100-sxm2/7.jpg
  - /assets/articles/nvidia-tesla-v100-sxm2/8.jpg
---
> Date: {{ page.date | date: "%d.%m.%Y" }}  

# NVIDIA Tesla V100 GPU SXM2

{% include gallery.html images=page.images gallery_id=page.title %}

## Limitations
- Requires an SXM2-to-PCIe adapter that looks ugly and is not reliable

## Test environment 
- Workstation 40 GB RAM, 500GB SSD, 750W Power supply 
- Ubuntu 24.04 LTS HWE Kernel
- Python 3.12

> My test environment: HP Z440 + NVIDIA Tesla V100 SXM2

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

## Check CUDA in Python
- Preparing PyTorch
```bash
mkdir -p ~/llm && cd ~/llm
python3 -m venv .venv_llm
source ./.venv_llm/bin/activate
python -m pip install --upgrade pip
pip install "torch==2.5.0" "torchvision==0.20.0" "torchaudio==2.5.0" --index-url https://download.pytorch.org/whl/cu124
pip install "bitsandbytes==0.44.1"
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available());print(torch.cuda.get_device_name(0));"
```
- Expected response
```
2.5.0+cu124
True
Tesla V100-SXM2-16GB
```

## Dry-run!

### Mistral 7b

- Prepare Python environment for CUDA 12:

```bash
mkdir -p ~/llm && cd ~/llm
python3 -m venv .venv_llm_mistral
source ./.venv_llm_mistral/bin/activate
python -m pip install --upgrade pip
pip install "torch==2.5.0" "torchvision==0.20.0" "torchaudio==2.5.0" --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate
```
- Get the Mistral:

```bash
git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-v0.1 mistral
```

- Create script test_cuda_mistral.py:

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
    dtype=torch.bfloat16
).to("cuda")

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

print(generator("Tell the story about sun.", max_new_tokens=120)[0]["generated_text"])
```

### Stable Diffusion v1.5

- Prepare Python environment for CUDA:

```bash
mkdir -p ~/llm && cd ~/llm
python3 -m venv .venv_llm_sd1.5
source ./.venv_llm_sd1.5/bin/activate
python -m pip install --upgrade pip
pip install "torch==2.5.0" "torchvision==0.20.0" "torchaudio==2.5.0" --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate diffusers safetensors
```
- Get the StableDiffusion 1.5

```bash
git lfs install
git clone https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5 sd1.5
```
- Create script test_cuda_sd1.5.py:

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
).to("cuda")

out = pipe(
    prompt= "cat sitting on a chair",
    height=512, width=512, guidance_scale=9, num_inference_steps=80)
image = out.images[0]

image.save("test.png", format="PNG")
```

## Benchmark 

> Prepare benchmark pp512 tg128
[PyTorch pp512 and tg128 LLM Benchmark](/articles/pytorch-pp512-tg128-bench.html)

### Run test 
> Check `nvidia-smi` during each test `while true; do nvidia-smi; sleep 1; done`

- With quantization

```bash
python ./app.py -m ../mistral --tests pp512 tg128 --dtype fp16 --batch 1 --attn sdpa --ubatch 128 --warmup 3 --iters 10 --quant 4bit
```

## It works!
