---
layout: default
title: "NVIDIA CUDA Compatibility"
date: 2025-08-14
categories: [general]

images:
  - /assets/articles/general/Logo_and_CUDA.png
---
> Date: {{ page.date | date: "%d.%m.%Y" }}  

# NVIDIA GPU architectures, drivers, and CUDA — quick cheat sheet

{% include gallery.html images=page.images gallery_id=page.title %}

The table below shows, for each NVIDIA architecture generation: its **Compute Capability (SM)**, the **last Linux driver branch** that still supports it, and the **newest compatible CUDA Toolkit** you can use.

> For your **Tesla M10** (Maxwell, SM **5.0**): the last driver branch is **R580**, and the newest CUDA that still supports it is **12.x** (e.g., 12.8).

| Architecture (examples) | Compute Capability (SM) | Last Linux driver branch that supports the architecture | Newest compatible CUDA Toolkit |
|---|---:|---|---|
| **Kepler** (GTX 780, Tesla K80) | 3.0 / 3.5 / 3.7 | **R470 (Legacy)** — support removed after this branch | **11.4** for SM 3.5/3.7; for SM 3.0 use **10.2** |
| **Maxwell** (Tesla M10/M60, GTX 9xx) | 5.0 / 5.2 / 5.3 | **R580** — last branch for Maxwell | **12.x** |
| **Pascal** (P100/P40, GTX 10xx) | 6.0 / 6.1 / 6.2 | **R580** — last branch for Pascal | **12.x** |
| **Volta** (V100/Jetson Xavier) | 7.0 / 7.2 | **R580** — last branch for Volta | **12.x** |
| **Turing** (RTX 20xx/Quadro RTX) | 7.5 | Supported by current branches (R580 and newer) | **13.x** and newer |
| **Ampere** (A100/RTX 30xx/Orin) | 8.0 / 8.6 / 8.7 | Supported by current branches | **13.x** and newer |
| **Ada Lovelace** (RTX 40xx/L40) | 8.9 | Supported by current branches | **13.x** and newer |
| **Hopper** (H100/GH200) | 9.0 | Supported by current branches | **13.x** and newer |
| **Blackwell** (B200/GB200) | 10.0 | Supported by current branches | **13.x** and newer |

### Tips
- For **Maxwell / Pascal / Volta** stay on **R580** (or earlier) and use **CUDA 12.x**. CUDA 13.0+ no longer targets SM 5.x / 6.x / 7.0–7.2.
- On Ubuntu 24.04, prefer installing **cuda-toolkit-12-x** from NVIDIA’s APT repo (and avoid the `cuda`/`cuda-drivers` metapackages if you already manage the driver separately).
