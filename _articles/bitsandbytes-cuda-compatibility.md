---
layout: default
title: "BitsAndBytes CUDA Compatibility"
date: 2025-08-28
categories: [general]

images:
  - /assets/articles/general/Logo_and_CUDA.png
  - /assets/articles/general/BNB_Logo.png
---

# BitsAndBytes CUDA Compatibility — quick cheat sheet

{% include gallery.html images=page.images gallery_id=page.title %}

> Quick reference for **BNB - CUDA Toolkit - Compute Capability - PyTorch**.  
> PyTorch versions are **approximate** and reflect practical compatibility.
> It’s often simpler to **build from source** especially for old GPU.

| BNB Version | CUDA Toolkit | Min CUDA CC | PyTorch Version |
|---|---|---|---|
| **0.43.3** | ≥11.1 | **Kepler** CC = 3.7 | **2.2.0 +** |
| **0.44.1** | ≥11.1 | **Kepler** ≈ CC ≥3.5; `LLM.int8()` **7.5+** | **2.2.0 +** |
| **0.45.5** | ≥11.7 | **Maxwell** **5.0+** (Kepler dropped since 0.45.0); `LLM.int8()` **7.5+** | **2.4 – 2.5** |
| **0.46.1** | **12 +** | **Turing 7.5+** (Maxwell dropped since 0.46.0) | **2.5 +** |

## Notes
- **Kepler (CC ~3.5–3.7)**: prebuilt wheels are long gone; use **source builds** and Kepler presets. Possible on `0.43.x–0.44.x`; starting with `0.45.0` Kepler is **removed**.
- **Maxwell (CC ≥5.0)**: supported for 8‑bit optimizers and 4‑bit NF4/FP4; marked *deprecated* in newer branches. `LLM.int8()` still needs **CC ≥7.5** (Turing+).
- **PyTorch 2.6**: `bitsandbytes ≥ 0.45.2` includes a compatibility fix for Triton 3.2.
- **Mismatched CUDA between PyTorch and BNB**: you can override BNB’s toolkit with `BNB_CUDA_VERSION=12x` and extend `LD_LIBRARY_PATH` accordingly.

## Quick environment check
```bash
python3 -m bitsandbytes
```
