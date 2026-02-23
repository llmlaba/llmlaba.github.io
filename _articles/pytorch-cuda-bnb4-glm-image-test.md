---
layout: default
title: "GLM Image CUDA PyTorch BNB4 Test"
date: 2026-02-22
categories: [llm, software]

images:
  - /assets/articles/pytorch-cuda-bnb4-glm-image-test/1.jpg
  - /assets/articles/pytorch-cuda-bnb4-glm-image-test/2.jpg
  - /assets/articles/pytorch-cuda-bnb4-glm-image-test/3.jpg
---
> Date: {{ page.date | date: "%d.%m.%Y" }}  

# GLM Image CUDA PyTorch BNB4 Test

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

## Check CUDA in Python
- Preparing PyTorch

```bash
mkdir -p ~/llm && cd ~/llm
python3 -m venv .venv_llm_gimage
source ./.venv_llm_gimage/bin/activate
python -m pip install --upgrade pip
pip install "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1" --index-url https://download.pytorch.org/whl/cu128
pip install "bitsandbytes==0.46.1"
pip install "transformers==5.2.0" accelerate safetensors
pip install git+https://github.com/huggingface/diffusers.git
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available());print(torch.cuda.get_device_name(0));"
```
- Expected response

```
2.7.0+cu128
True
NVIDIA GeForce RTX 3090
```
- Check BitsAndBytes installation

```bash
python -m bitsandbytes
```

## Dry run

### Get the glm Image
```bash
git lfs install
git clone https://huggingface.co/zai-org/GLM-Image glm-image
```

### Create script test_cuda_bnb4_glm_image.py:

```python
from diffusers.pipelines.glm_image import GlmImagePipeline
from diffusers import PipelineQuantizationConfig
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
import torch

print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))

model_path = "/home/sysadmin/llm/glm-image"

seed = torch.seed()
print(f"Using seed: {seed}")

tf_bnb4 = TransformersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

df_bnb = DiffusersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

quant_config = PipelineQuantizationConfig(
    quant_mapping={
        "transformer": df_bnb,
        "text_encoder": tf_bnb4,
        "vision_language_encoder": tf_bnb4,
    }
)

pipe = GlmImagePipeline.from_pretrained(
    model_path,
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
    local_files_only=True,
).to("cuda")

generator = torch.Generator("cuda").manual_seed(seed)

prompt = "ginger cat sits on a chair"

out = pipe(
    prompt=prompt,
    height=32 * 32, 
    width=32 * 32,
    num_inference_steps=50,
    guidance_scale=1.5,
    generator=generator
)

image = out.images[0]

image.save(f"test_image_glm_{seed}.png", format="PNG")
```

### Run test 
> Check `nvidia-smi` during the test `watch -n 1 nvidia-smi`

- With quantization 4bit

```bash
python ./test_cuda_bnb4_glm_image.py
```

## Enjoy the result!
- It pretty good!
