---
layout: default
title: "CUDA & PyTorch Compatibility by Compute Capability"
date: 2025-08-14
categories: [general]

images:
  - /assets/articles/general/Logo_and_CUDA.png
  - /assets/articles/general/PyTorch_logo.jpeg
---

# CUDA & PyTorch Compatibility by Compute Capability (CC)

{% include gallery.html images=page.images gallery_id=page.title %}

This cheat sheet maps **Compute Capability (CC)** → **newest usable CUDA Toolkit** → **a recent PyTorch version with official wheels** → **ready-to-copy pip command**.

> Notes
> - **Maxwell/Pascal/Volta (CC 5.x/6.x/7.0–7.2)** should stay on **CUDA 12.x**. CUDA 13.x drops support for these architectures in the toolkit.
> - PyTorch wheels (cuXXX) **bundle the CUDA runtime**. You only need the system CUDA Toolkit if you compile custom CUDA extensions.
> - Choose the CUDA flavor (cu121 / cu124 / cu126 / cu128) that matches your environment and driver capabilities.

| Compute Capability (CC) | Typical generations | Newest usable CUDA Toolkit | Recommended PyTorch (latest with such wheels) | Example pip command |
|---:|---|---|---|---|
| **3.5 / 3.7** | Kepler (e.g., Tesla K80) | **CUDA 10.2** | **PyTorch 1.12.1** (cu102) | `pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102` |
| **3.5 / 3.7** | Kepler (e.g., Tesla K80) | **CUDA 11.4** | **PyTorch 2.2.0** | **manual build** [Telsa K80 + PyTorch 2.2.0](/articles/nvidia-tesla-k80.html) |
| **5.0 / 5.2 / 5.3** | **Maxwell** (Tesla M10/M60, GTX 9xx) | **CUDA 12.4** (12.1 also fine) | **PyTorch 2.5.0** (cu124 or cu121) | cu124: `pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124`  •  cu121: `... --index-url https://download.pytorch.org/whl/cu121` |
| **6.0 / 6.1 / 6.2** | **Pascal** (P100/P40, GTX 10xx) | **CUDA 12.4** | **PyTorch 2.5.0** (cu124 or cu121) | same as above (cu124/cu121) |
| **7.0 / 7.2** | **Volta** (V100/Jetson Xavier) | **CUDA 12.4** | **PyTorch 2.5.0** (cu124 or cu121) | same as above (cu124/cu121) |
| **7.5** | **Turing** (RTX 20xx/Quadro RTX) | **CUDA 12.8 / 13.x** | **PyTorch 2.7.1** (cu128; cu126 also available) | cu128: `pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128`  •  cu126: `... --index-url https://download.pytorch.org/whl/cu126` |
| **8.0 / 8.6 / 8.7** | **Ampere** (A100/RTX 30xx/Orin) | **CUDA 12.8 / 13.x** | **PyTorch 2.7.1** (cu128; cu126 also available) | use the 2.7.1 cu128/cu126 commands above |
| **8.9** | **Ada Lovelace** (RTX 40xx/L40) | **CUDA 12.8 / 13.x** | **PyTorch 2.7.1** (cu128; cu126 also available) | use the 2.7.1 cu128/cu126 commands above |
| **9.0** | **Hopper** (H100/GH200) | **CUDA 12.8 / 13.x** | **PyTorch 2.7.1** (cu128; cu126 also available) | use the 2.7.1 cu128/cu126 commands above |

### Quick guidance
- If you're on **Tesla M10** (Maxwell, CC 5.0): pick **PyTorch 2.5.0** with **cu124** (or cu121) wheels. Ensure your NVIDIA driver is from the **R570/R580** line.
- For **Kepler** (CC 3.x): modern PyTorch 2.x wheels don't target it; use **PyTorch 1.12.1 + cu102** or build from source with `TORCH_CUDA_ARCH_LIST="3.5;3.7"`.

