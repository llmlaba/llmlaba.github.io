---
layout: default
title: "Pixtral 12b CUDA low memory GPU PyTorch Test"
date: 2026-01-25
categories: [llm, software]

images:
  - /assets/articles/general/mistral-ai-logo.png
  - /assets/articles/pytorch-bad-cuda-pixtral-test/1.jpg
  - /assets/articles/pytorch-bad-cuda-pixtral-test/2.jpg
---
> Date: {{ page.date | date: "%d.%m.%Y" }}  

# Pixtral 12b CUDA low memory GPU PyTorch Test

{% include gallery.html images=page.images gallery_id=page.title %}

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
- Install dev tools

```bash
sudo apt install -y python3-venv python3-dev git git-lfs
```

## Dry-run Pixtral 12b

- Preapre python environment for CUDA:

```bash
mkdir -p ~/llm && cd ~/llm
python3 -m venv .venv_llm_pixtral
source ./.venv_llm_pixtral/bin/activate
python -m pip install --upgrade pip
pip install "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1" --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate
```
- Get the Pixtral 12b model:

```bash
git lfs install
git clone https://huggingface.co/mistral-community/pixtral-12b pixtral
```
- Put test picture to `/home/sysadmin/llm/pixtral/2.jpeg`

- Create script test_bad_cuda_pixtral.py:

```python
from transformers import AutoProcessor, LlavaForConditionalGeneration, set_seed
import torch
from PIL import Image

print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))

model_path = "/home/sysadmin/llm/pixtral"

seed = torch.seed() % (2**32)
print(f"Using seed: {seed}")
set_seed(seed)

processor = AutoProcessor.from_pretrained(model_path)
model = LlavaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)

img_links = [
"/home/sysadmin/llm/pixtral/2.jpeg", 
]
prompt = "<s>[INST]Describe the image.\n[IMG][/INST]"

inputs = processor(text=prompt, 
    images=img_links, 
    return_tensors="pt"
)

inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

generate_ids = model.generate(
    **inputs,     
    max_new_tokens=1000, #32768,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    # repetition loop prevention
    repetition_penalty=1.1,
    no_repeat_ngram_size=4)

print(processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
```
- Run test

```bash
python test_bad_cuda_pixtral.py
```

## Enjoy the result!
