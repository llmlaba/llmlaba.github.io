---
layout: default
title: "Stable Diffusion 1.5 LoRA Tranning Test"
date: 2025-09-30
categories: [llm, software]

images:
  - /assets/articles/sd1-5-lora-traning/1.jpg
  - /assets/articles/sd1-5-lora-traning/2.jpg
  - /assets/articles/sd1-5-lora-traning/3.jpg
  - /assets/articles/sd1-5-lora-traning/4.jpg
---
> Date: {{ page.date | date: "%d.%m.%Y" }}  

# Stable Diffusion 1.5 LoRA Tranning Test

{% include gallery.html images=page.images gallery_id=page.title %}

## Test environment 
- Workstation 40 GB RAM, 500GB SSD, 750W Power supply 
- Ubuntu 24.04 LTS HWE Kernel
- Install python 3.12

> My test environment: HP Z440 + NVIDIA Tesla P100

## Steps

### Preapre python environment for CUDA:

```bash
mkdir -p ~/lora && cd ~/lora
python3 -m venv .venv
source ./.venv/bin/activate
python -m pip install --upgrade pip
pip install "torch==2.5.0" "torchvision==0.20.0" "torchaudio==2.5.0" --index-url https://download.pytorch.org/whl/cu124
```
- Installing diffusers from source

```bash
git clone -b v0.34.0-release https://github.com/huggingface/diffusers
cd diffusers
pip install .
cd examples/text_to_image
pip install -r requirements.txt
accelerate config default
pip install "peft>=0.15.0"
```
- Check Python installation

```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available());print(torch.cuda.get_device_name(0));"
```

### Get the StableDiffusion 1.5

```bash
git lfs install
git clone https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5 sd1.5
```

### Prepare dataset

- Create png list from favorite movie

```bash
ffmpeg -i movie.mkv -vf "fps=1" f2/%06d.png
```

- Create jsonl with metadata

```jsonl
{"file_name":"000134.png","text":"alpopo, cartoon hero, full body, looking left"}
{"file_name":"000135.png","text":"alpopo, cartoon hero, full body, looking down"}
```

### Run Stable diffusion LoRA traning

```bash
export MODEL_DIR="/home/sysadmin/llm/sd1.5"
export DATA_DIR="/home/sysadmin/lora/f2"
export OUT_DIR="/home/sysadmin/lora/alpopo-lora"

accelerate launch train_text_to_image_lora.py \
  --pretrained_model_name_or_path="$MODEL_DIR" \
  --train_data_dir="$DATA_DIR" \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --max_train_steps=15000 --learning_rate=1e-4 \
  --lr_scheduler=cosine --lr_warmup_steps=0 \
  --output_dir="$OUT_DIR" \
  --checkpointing_steps=500 \
  --validation_prompt="alpopo, cartoon hero, portrait" \
  --validation_epochs=1 \
  --seed=1337 \
  --mixed_precision="no" # mean load model in float32
```

### Run tensorboard for training monitoring

```bash
tensorboard --bind_all --logdir /home/sysadmin/lora/alpopo-lora/logs/text2image-fine-tune/
```

## Enjoy the result
