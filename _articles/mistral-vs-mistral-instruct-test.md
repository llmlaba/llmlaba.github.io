---
layout: default
title: "Mistral 7b General VS Instruct Test"
date: 2025-09-29
categories: [llm, software]

images:
  - /assets/articles/mistral-vs-mistral-instruct-test/1.jpg
  - /assets/articles/mistral-vs-mistral-instruct-test/2.jpg
  - /assets/articles/mistral-vs-mistral-instruct-test/3.jpg
  - /assets/articles/mistral-vs-mistral-instruct-test/4.jpg
  - /assets/articles/mistral-vs-mistral-instruct-test/5.jpg
  - /assets/articles/mistral-vs-mistral-instruct-test/6.jpg
  - /assets/articles/mistral-vs-mistral-instruct-test/7.jpg
---
> Date: {{ page.date | date: "%d.%m.%Y" }}  

# Mistral 7b Bad CUDA PyTorch Test 

{% include gallery.html images=page.images gallery_id=page.title %}

## Test environment 
- Workstation 40 GB RAM, 500GB SSD, 750W Power supply 
- Ubuntu 24.04 LTS HWE Kernel
- Install python 3.12

> My test environment: HP Z440 + NVIDIA Tesla P100

## Steps

### Preapre python environment for CUDA:

```bash
mkdir -p ~/llm && cd ~/llm
python3 -m venv .venv_llm_mistral
source ./.venv_llm_mistral/bin/activate
python -m pip install --upgrade pip
pip install "torch==2.5.0" "torchvision==0.20.0" "torchaudio==2.5.0" --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate
```

### Get the Mistral

- Get general mistral

```bash
git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-v0.1 mistral
```
- Get mistral instruct

```bash
git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1 mistral-instruct
```

### Create script test_cuda_mistral.py:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))

model_path = "/home/sysadmin/llm/mistral"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model     = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16
).to("cuda")

inputs = tokenizer("Hello!", return_tensors="pt").to("cuda")

output_ids = model.generate(
    **inputs,
    max_new_tokens=160,
)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```
- Run test

```bash
python test_cuda_mistral.py
```

### Create script test_cuda_mistral_chat_mode.py

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))

model_path = "/home/sysadmin/llm/mistral"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model     = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16
).to("cuda")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
]

parts = []
for m in messages:
    role = m.get("role", "user")
    content = m.get("content", "")
    parts.append(f"{role}: {content}")
prompt = "\n".join(parts) + "\nassistant:"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

output_ids = model.generate(
    **inputs,
    max_new_tokens=160,
)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```
- Run test

```bash
python test_cuda_mistral_chat_mode.py
```

### Create script test_cuda_mistral_instruct.py:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))

model_path = "/home/sysadmin/llm/mistral-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model     = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16
).to("cuda")

inputs = tokenizer("Hello!", return_tensors="pt").to("cuda")

output_ids = model.generate(
    **inputs,
    max_new_tokens=160,
)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```
- Run test

```bash
python test_cuda_mistral_instruct.py
```

### Create script test_cuda_mistral_instruct_chat_mode.py:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))

model_path = "/home/sysadmin/llm/mistral-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model     = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16
).to("cuda")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

output_ids = model.generate(
    **inputs,
    max_new_tokens=160,
)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```
- Run test

```bash
python test_cuda_mistral_instruct_chat_mode.py
```

### Enjoy the result!
