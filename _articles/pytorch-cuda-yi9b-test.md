---
layout: default
title: "Yi 9b CUDA PyTorch Test"
date: 2026-01-11
categories: [llm, software]

images:
  - /assets/articles/general/Yi_logo_icon_light.png
  - /assets/articles/pytorch-cuda-yi9b-test/1.jpg
  - /assets/articles/pytorch-cuda-yi9b-test/2.jpg
---
> Date: {{ page.date | date: "%d.%m.%Y" }}  

# Yi 9b CUDA PyTorch Test 

{% include gallery.html images=page.images gallery_id=page.title %}

## Notes:
- Model likely will have repitition loop with `max_new_tokens=3000`

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

## Dry-run Yi 9b

- Preapre python environment for CUDA:

```bash
mkdir -p ~/llm && cd ~/llm
python3 -m venv .venv_llm_yi9b
source ./.venv_llm_yi9b/bin/activate
python -m pip install --upgrade pip
pip install "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1" --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate
```
- Get the Falcon model:

```bash
git lfs install
git clone https://huggingface.co/01-ai/Yi-9B yi9b
```

- Create script test_cuda_yi9b.py:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import torch

print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))

model_path = "/home/sysadmin/llm/yi9b"

seed = torch.seed() % (2**32)
print(f"Using seed: {seed}")
set_seed(seed)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model     = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
).to("cuda")

prompt = """People love to know how fast (or slow) \
their computers are. We have always \
been interested in speed; it is human \
nature. To help us with this quest, \
we can use various benchmark test \
programs to measure aspects of processor \
and system performance. Although no \
single numerical measurement can \
completely describe the performance \
of a complex device such as a processor \
or a complete PC, benchmarks can be \
useful tools for comparing different \
components and systems."""

inputs = tokenizer(
    prompt, 
    return_tensors="pt").to("cuda")

output_ids = model.generate(
    **inputs,
    max_new_tokens= 3000, #4096,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    # repetition loop prevention
    repetition_penalty=1.1,
    no_repeat_ngram_size=4
)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```
- Run test

```bash
python test_cuda_yi9b.py
```

## Enjoy the result!
