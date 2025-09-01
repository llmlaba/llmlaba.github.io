---
layout: default
title: "TensorFlow vs PyTorch GPT2"
date: 2025-08-18
categories: [llm, software]

images:
  - /assets/articles/tensorflow-pytorch-gpt2/1.jpg
  - /assets/articles/tensorflow-pytorch-gpt2/2.jpg
  - /assets/articles/tensorflow-pytorch-gpt2/3.jpg
  - /assets/articles/tensorflow-pytorch-gpt2/4.jpg
---
> Date: {{ page.date | date: "%d.%m.%Y" }}  

# TensorFlow vs PyTorch GPT2

{% include gallery.html images=page.images gallery_id=page.title %}

## Test environment 
- NVIDIA GPU 8GB, Maxwell or higher
- Workstation 40 GB RAM, 200GB SSD
- Ubuntu 24.04
- Install python 3.12
- NVIDIA Driver 570

> My test environment: HP Z440

## Steps

### Get the GPT2

```bash
mkdir -p ~/llm && cd ~/llm
git lfs install
git clone https://huggingface.co/openai-community/gpt2 gpt2
```

### Prepare TensorFlow test

- Prepare python environment for TensorFlow with CUDA support

```bash
mkdir -p ~/llm && cd ~/llm
python3 -m venv .venv_llm_tf
source ./.venv_llm_tf/bin/activate
python -m pip install --upgrade pip
pip install "tensorflow[and-cuda]==2.16.*" "tf-keras~=2.16" "transformers<5"
```
- Check if TensorFlow can access the GPU

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

- Create a test script **test_tf_cuda_gpt2.py** with the following content:

```python
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM, pipeline

print("TF GPUs:", tf.config.list_physical_devices("GPU"))

model_path = "/home/sysadmin/llm/gpt2"

tok = AutoTokenizer.from_pretrained(model_path)
tok.pad_token = tok.eos_token 
model = TFAutoModelForCausalLM.from_pretrained(model_path)

inputs = tok("The space stars is?", return_tensors="tf")
out = model.generate(**inputs, max_new_tokens=160)
print(tok.decode(out[0], skip_special_tokens=True))

```
- Test TensorFlow with a simple script

```bash
rm ./gpt2/*.bin ./gpt2/*.safetensors
export TF_CPP_MIN_LOG_LEVEL=2
export TF_USE_LEGACY_KERAS=1
python ./test_tf_cuda_gpt2.py
```

### Prepare PyTorch test

- Restore gpt2 lfs files

```bash
cd gpt2
git lfs pull
cd ..
```

- Prepare python environment for PyTorch with CUDA support

```bash
mkdir -p ~/llm && cd ~/llm
python3 -m venv .venv_llm_torch
source ./.venv_llm_torch/bin/activate
python -m pip install --upgrade pip
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
pip install "transformers<5"
```
- Check if PyTorch can access the GPU

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available());print(torch.cuda.get_device_name(0));"
```
- Create a test script **test_torch_cuda_gpt2.py** with the following content:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))

model_path = "/home/sysadmin/llm/gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model     = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32
)

model.to("cuda")

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

print(generator("The space stars is?", max_new_tokens=160)[0]["generated_text"])
```
## Enjoy the result!
