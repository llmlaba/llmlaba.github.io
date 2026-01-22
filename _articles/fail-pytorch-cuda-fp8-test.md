---
layout: default
title: "Failed PyTorch CUDA FP8 Test"
date: 2026-01-21
categories: [fail]

images:
  - /assets/articles/general/fp8.jpeg
  - /assets/articles/fail-pytorch-cuda-fp8-test/1.jpg
---
> Date: {{ page.date | date: "%d.%m.%Y" }}  

# Failed PyTorch CUDA FP8 Test

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
sudo apt install -y python3-venv python3-dev git 
```

## Run Mistral 7b with FP8

- Preapre python environment for CUDA:

```bash
mkdir -p ~/llm && cd ~/llm
python3 -m venv .venv_llm_torchao
source ./.venv_llm_torchao/bin/activate
python -m pip install --upgrade pip
pip install "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1" "torchao==0.12.0" --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate
```
- Get the Mistral 7b model:

```bash
git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-v0.1 mistral
```
- Create script test_cuda_fp8_mistral.py:

```python
from transformers import TorchAoConfig, AutoTokenizer, AutoModelForCausalLM, set_seed
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, Float8WeightOnlyConfig
import torch

print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))

model_path = "/home/sysadmin/llm/mistral"

seed = torch.seed() % (2**32)
print(f"Using seed: {seed}")
set_seed(seed)

quant_config = Float8DynamicActivationFloat8WeightConfig()
# or only weights:
# quant_config = Float8WeightOnlyConfig()

quantization_config = TorchAoConfig(quant_type=quant_config)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model     = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype="auto",
    quantization_config=quantization_config
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

inputs = tokenizer(prompt, 
    return_tensors="pt").to("cuda")

output_ids = model.generate(
    **inputs,
    max_new_tokens=3000, #32768,
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
python test_cuda_fp8_mistral.py
```

## Test Failed
- Why
> FP8 supported only in CUDA compute capability 8.9, so min RTX 4090 (maybe) or better RTX 5090
