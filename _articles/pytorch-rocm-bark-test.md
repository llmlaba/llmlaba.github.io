---
layout: default
title: "SUNO Bark ROCm PyTorch Test"
date: 2025-08-31
categories: [llm, software]
---
> Date: {{ page.date | date: "%d.%m.%Y" }}
  
# SUNO Bark ROCm PyTorch Test

{% include video.html 
    autoplay=false
    src="/assets/articles/pytorch-rocm-bark-test/1.mp4" 
    src_webm="/assets/articles/pytorch-rocm-bark-test/1.webm" 
%}

## Requirments 
- AMD Mi50/MI100 32Gb VRAM
- Workstation 40 GB RAM, 500GB SSD, 750W Power supply 
- Ubuntu 22.04 LTS
- Python 3.10

> My test environment: HP Z440 + AMD Mi50

## Test steps

### Preapre python environment for ROCm:

```bash
mkdir -p ~/llm && cd ~/llm
python3 -m venv .venv_llm_bark
source ./.venv_llm_bark/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
pip install transformers accelerate scipy
```

### Get the SUNO Bark

```bash
git lfs install
git clone git clone https://huggingface.co/suno/bark bark
```

### Convert bark `pytorch_model.bin` to `model.safetensors`
> More here [PyTorch binary weights from bin to safetensors](/articles/issue-bin-to-safetensors.html)

- Create script convert.py:

```python
import torch
from safetensors.torch import save_file

src = "/home/sysadmin/llm/bark/pytorch_model.bin"
dst = "/home/sysadmin/llm/bark/model.safetensors"

# важно: weights_only=True (см. предупреждение torch)
sd = torch.load(src, map_location="cpu", weights_only=True)

new_sd = {}
seen = {}  # ключ: (data_ptr, size, dtype, shape, stride)
for k, t in sd.items():
    if not isinstance(t, torch.Tensor):
        continue
    stg = t.untyped_storage()
    sig = (stg.data_ptr(), stg.size(), t.dtype, tuple(t.size()), tuple(t.stride()))
    if sig in seen:
        # if tensor uses shared memory copy it
        new_sd[k] = t.clone()  # or t.contiguous().clone()
    else:
        seen[sig] = k
        new_sd[k] = t  # original one keep without changes

# safe safetensors without shared memory with metadata 
save_file(new_sd, dst, metadata={"format": "pt"})
print("Saved:", dst)
```
- Convert bark `pytorch_model.bin` to `model.safetensors`

```bash
python3 ./convert.py
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

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

model = BarkModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    local_files_only=True,
    use_safetensors=True,
).to("cuda")

inputs = processor(
    text=["Hi! Am am a dummy robot that speaks with a human voice."], 
    voice_preset="v2/en_speaker_6",
    return_tensors="pt",
).to("cuda")

audio = model.generate(
    **inputs,
    do_sample=True,
    fine_temperature=0.4,
    coarse_temperature=0.8,
    pad_token_id = processor.tokenizer.pad_token_id,
)

write_wav(
    "bark_test.wav", 
    model.generation_config.sample_rate, 
    audio[0].detach().cpu().numpy()
)
```

### Run bark llm

```bash
python ./test_rocm_bark.py
```

### Open bark_test.wav and enjoy the result!
