---
layout: default
title: "PyTorch convert binary weights from bin to safetensors"
date: 2025-09-01
categories: [issue]
---

# Instruction how to convert PyTorch binary weights from bin to safetensors

## Why
It required by PyTorch community, because safetensors will not allow to execute any code inside weights, but it is possible in old binary format

## How

### General instruction
> Convert mannualy pytorch_model.bin to model.safetensors with **torch.load()** and **safetensors.torch.save_file()** 

```python
import torch
from safetensors.torch import save_file

src = "/home/sysadmin/llm/bark/pytorch_model.bin"
dst = "/home/sysadmin/llm/bark/model.safetensors"

sd = torch.load(src, map_location="cpu", weights_only=True)

new_sd = {}
seen = {}  # key: (data_ptr, size, dtype, shape, stride)
for k, t in sd.items():
    if not isinstance(t, torch.Tensor):
        continue
    stg = t.untyped_storage()
    sig = (stg.data_ptr(), stg.size(), t.dtype, tuple(t.size()), tuple(t.stride()))
    if sig in seen:
        # этот тензор делит память — делаем независимую копию
        new_sd[k] = t.clone()  # или t.contiguous().clone()
    else:
        seen[sig] = k
        new_sd[k] = t  # первый экземпляр оставляем как есть

# сохраняем уже без шеринга
save_file(new_sd, dst, metadata={"format": "pt"})
print("Saved:", dst)
```
