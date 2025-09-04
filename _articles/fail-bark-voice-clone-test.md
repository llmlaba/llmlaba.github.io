---
layout: default
title: "SUNO Bark voice clone ROCm PyTorch Test"
date: 2025-08-31
categories: [fail]
---
> Date: {{ page.date | date: "%d.%m.%Y" }}
  
# SUNO Bark voice clone ROCm PyTorch Test

{% include video.html 
    autoplay=false
    src="/assets/articles/fail-bark-voice-clone-test/1.mp4" 
    src_webm="/assets/articles/fail-bark-voice-clone-test/1.webm" 
%}

## NOTES
- It works only with English language

## Requirments 
- AMD Mi50/MI100 32Gb VRAM
- Workstation 40 GB RAM, 500GB SSD, 750W Power supply 
- Ubuntu 22.04 LTS
- Python 3.10

> My test environment: HP Z440 + AMD Mi50

## Test steps

### Preapre python environment for ROCm:

- Prepare venv

```bash
mkdir -p ~/llm && cd ~/llm
python3 -m venv .venv_llm_hubert
source ./.venv_llm_hubert/bin/activate
python -m pip install "pip<24.1"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
pip install transformers accelerate scipy
```
- Install bark preset generator

```bash
pip install encodec soundfile 
git clone https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer.git
pip install -e bark-voice-cloning-HuBERT-quantizer
```

- Check python torchaudio backends

```bash
python -c 'import torchaudio; print("backends:", torchaudio.list_audio_backends())'
```

### Get the SUNO Bark

```bash
git lfs install
git clone https://huggingface.co/suno/bark bark
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
seen = {}  # key: (data_ptr, size, dtype, shape, stride)
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

### Generate voice preset for bark
- Get LLM to generate the preset

```bash
mkdir ./hubert && cd ./hubert
wget https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt
wget https://huggingface.co/GitMylo/bark-voice-cloning/resolve/main/quantifier_V1_hubert_base_ls960_23.pth
```

- Record your own voice and convert it to **WAV**

```bash
ffmpeg -i 2.m4a -ac 1 -ar 24000 -c:a pcm_s16le out2.wav
```
- Create a file with converter to convert sample of voice to bark preset `bark_voice_to_preset.py`

```python
import torch, torchaudio, numpy as np
from encodec import EncodecModel
from encodec.utils import convert_audio
from bark_hubert_quantizer.pre_kmeans_hubert import CustomHubert
from bark_hubert_quantizer.customtokenizer   import CustomTokenizer

WAV_PATH = "/home/sysadmin/llm/hubert/out2.wav"
HUBERT_CKPT = "/home/sysadmin/llm/hubert/hubert_base_ls960.pt"
TOKENIZER_CKPT = "/home/sysadmin/llm/hubert/quantifier_V1_hubert_base_ls960_23.pth"

# 1) load wav & resample mono 24k
wav, sr = torchaudio.load(WAV_PATH)
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)
wav = convert_audio(wav, sr, model.sample_rate, model.channels).unsqueeze(0)

# 2) fine/coarse via EnCodec
with torch.no_grad():
    frames = model.encode(wav)
codes = torch.cat([f[0] for f in frames], dim=-1)  # [1, n_q, T]
fine_prompt = codes[0].cpu().numpy().astype(np.int64)
coarse_prompt = fine_prompt[:2, :]

# 3) semantic via HuBERT + quantizer
hubert = CustomHubert(checkpoint_path=HUBERT_CKPT)
tokenizer = CustomTokenizer.load_from_checkpoint(TOKENIZER_CKPT)
semantic_vectors = hubert.forward(wav.squeeze(0), input_sample_hz=model.sample_rate)
semantic_prompt = tokenizer.get_token(semantic_vectors).squeeze(0).cpu().numpy().astype(np.int64)

# 4) pack npz
np.savez("my_voice.npz",
         semantic_prompt=semantic_prompt,
         fine_prompt=fine_prompt,
         coarse_prompt=coarse_prompt)
print("Saved my_voice.npz")
```
- Run voice converter

```bash
python ./bark_voice_to_preset.py
```

### Create script test_rocm_bark.py:
> Script will use our own voice from voice preset

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
    text=["Hi! This is a test of Bark. I can speak your voice!"], 
    voice_preset="/home/sysadmin/llm/my_voice.npz",
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
