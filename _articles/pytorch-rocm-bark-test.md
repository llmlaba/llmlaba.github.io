---
layout: default
title: "SUNO Bark ROCm PyTorch Test"
date: 2025-08-30
categories: [llm, software]
---

# SUNO Bark ROCm PyTorch Test

![test1](/assets/articles/pytorch-rocm-bark-test/1.mp4)

## Requirments 
- AMD Mi50/MI100 32Gb VRAM
- Workstation 40 GB RAM, 500GB SSD, 750W Power supply 
- Ubuntu 22.04 LTS
- Python 3.10

> My test environment: HP Z440 + AMD Mi50

## Steps

### Preapre python environment for ROCm:

```bash
mkdir -p ~/llm && cd ~/llm
python3 -m venv .venv_llm_bark
source ./.venv_llm_bark/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
pip install "transformers<4.48" accelerate scipy
python ./test_rocm_bark.py
```

### Get the SUNO Bark

```bash
git lfs install
git clone git clone https://huggingface.co/suno/bark bark
```

### Create script test_rocm_bark.py:

```python
import torch
import os, numpy as np
from transformers import AutoProcessor, BarkModel
from scipy.io.wavfile import write as write_wav

print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))

MODEL_PATH = "/home/sysadmin/llm/bark"  # or "bark-small"

def load_voice_history(model_dir: str, voice: str, device: str):
    name = voice.split("/")[-1]
    root = os.path.join(model_dir, "speaker_embeddings", "v2")
    sem = np.load(os.path.join(root, f"{name}_semantic_prompt.npy"))
    crs = np.load(os.path.join(root, f"{name}_coarse_prompt.npy"))
    fin = np.load(os.path.join(root, f"{name}_fine_prompt.npy"))

    def to_long_dev(x):
        t = torch.as_tensor(x, dtype=torch.long, device=device)
        return t.squeeze(0) if t.ndim == 2 and t.size(0) == 1 else t

    return {
        "semantic_prompt": to_long_dev(sem),
        "coarse_prompt":   to_long_dev(crs),
        "fine_prompt":     to_long_dev(fin),
    }

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

model = BarkModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    local_files_only=True,
).to("cuda")

inputs = processor(
    text=["Hi! Am am a dummy robot that speaks with a human voice."], 
    return_tensors="pt",
).to("cuda")

if "attention_mask" not in inputs:
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

eos_id = model.generation_config.eos_token_id or model.config.eos_token_id
model.generation_config.pad_token_id = eos_id
model.config.pad_token_id = eos_id

history_prompt = load_voice_history(
    MODEL_PATH, 
    "v2/en_speaker_6", 
    "cuda"
)

audio = model.generate(
    **inputs,
    history_prompt=history_prompt,
    do_sample=True,
    fine_temperature=0.4,
    coarse_temperature=0.8,
)

write_wav(
    "bark_test.wav", 
    model.generation_config.sample_rate, 
    audio[0].detach().cpu().numpy()
)
```

### Open bark_test.wav and enjoy the result!
