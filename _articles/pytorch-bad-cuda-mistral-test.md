---
layout: default
title: "Mistral 7b CUDA low memory GPU PyTorch Test"
date: 2025-06-14
categories: [llm, software]
---

# Mistral 7b ROCm PyTorch Test 

{% include gallery.html images=page.images gallery_id=page.title %}

## Requirments 
- NVIDIA RTX 4070 8GB
- Workstation 64 GB RAM, 200GB SSD
- Windows 11
- Install python 3.11
- NVIDIA Driver 577

> Memory on my gaming laptop is not sufficient for running Mistral 7b (requires 15GB VRAM), so I will try to use `device_map="auto"` and specify `offload_folder="offload"` as the directory for model offloading.

## Steps

### Get the Mistral
```bash
git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-v0.1 mistral
```
### Preapre python environment for ROCm:
```bash
python -m venv .venv_llm_mistral
.\.venv_llm_mistral\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate
python .\test_cuda_mistral.py
```
### Create script test_rocm_mistral.py:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

print(torch.cuda.is_available())
tokenizer = AutoTokenizer.from_pretrained("C:\\Users\\admin\\llm\\mistral")
model     = AutoModelForCausalLM.from_pretrained(
    "C:\\Users\\admin\\llm\\mistral",
    device_map="auto",
    offload_folder="offload",
    torch_dtype=torch.bfloat16
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

print(generator("what you know about Sun?", max_new_tokens=60)[0]["generated_text"])
```
### Result
> The model will generate text slowly due to limited GPU memory and offloading, but it should produce coherent output.  
> Output:
> ```
> what you know about Sun? The Sun is the star at the center of the Solar System. It is a nearly perfect sphere of hot plasma, and it provides the energy necessary for life on Earth. The Sun is composed primarily of hydrogen and helium...
> ```
