---
layout: default
title: "CUDA PyTorch BitsAndBytes Mistral Test"
date: 2025-09-21
categories: [draft]

images:
  - /assets/articles/cuda-bitsandbytes-mistral-test/1.jpg

---
> Date: {{ page.date | date: "%d.%m.%Y" }}  

# CUDA PyTorch BitsAndBytes Mistral Test

{% include gallery.html images=page.images gallery_id=page.title %}

## Requirments 
- NVIDIA Tesla P100 16Gb VRAM
- Workstation 40 GB RAM, 500GB SSD, 750W Power supply 
- Ubuntu 24.04 LTS
- Python 3.12

## TESTs

### Preapre python environment for cuda + PyTorch + BitsAndBytes:
- Priparing PyTorch

```bash
mkdir -p ~/llm && cd ~/llm
python3 -m venv .venv_llm_cuda_bnb_mistral
source ./.venv_llm_cuda_bnb_mistral/bin/activate
python -m pip install --upgrade pip
pip install "torch==2.5.0" "torchvision==0.20.0" "torchaudio==2.5.0" --index-url https://download.pytorch.org/whl/cu124
pip install "bitsandbytes==0.44.1"
pip install transformers accelerate
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available());print(torch.cuda.get_device_name(0));"
```
- Expected responce

```
2.5.0+cu124
True
Tesla P100-PCIE-16GB
```
- Check BitsAndBytes installation

```bash
python -m bitsandbytes
```

### Get the Mistral

```bash
git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-v0.1 mistral
```

### Create script test_cuda_mistral.py:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))

model_path = "/home/sysadmin/llm/mistral"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model     = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16
).to("cuda")

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0  # Use GPU
)

print(generator("What you know about sun?", max_new_tokens=60)[0]["generated_text"])
```

### Create script test_cuda_bnb4_mistral.py:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import BitsAndBytesConfig
import torch

print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))

model_path = "/home/sysadmin/llm/mistral"

qconf = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type='nf4', 
    bnb_4bit_use_double_quant=False, 
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model     = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=qconf,
    torch_dtype=torch.float16
).to("cuda")

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

print(generator("What you know about sun?", max_new_tokens=60)[0]["generated_text"])
```

### Create script test_cuda_bnb8_mistral.py:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import BitsAndBytesConfig
import torch

print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))

model_path = "/home/sysadmin/llm/mistral"

qconf = BitsAndBytesConfig(
    load_in_8bit=True
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model     = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=qconf,
    torch_dtype=torch.float16
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

print(generator("What you know about Sun?", max_new_tokens=160)[0]["generated_text"])
```

### Run test 
> Check `nvidia-smi` during each test `while true; do nvidia-smi; sleep 1; done`

- Without quantization

```bash
python ./test_cuda_mistral.py
```

- With quantization 4bit

```bash
python ./test_cuda_bnb4_mistral.py
```

- With quantization 8bit

```bash
python ./test_cuda_bnb8_mistral.py
```
