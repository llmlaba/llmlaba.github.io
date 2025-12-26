---
layout: default
title: "FlashAttention compatibility"
date: 2025-12-25
categories: [general]

images:
  - /assets/articles/general/flashattention_logo.png
  - /assets/articles/general/Logo_and_CUDA.png
---
> Date: {{ page.date | date: "%d.%m.%Y" }}  

# FlashAttention, NVIDIA GPU architectures, and CUDA — quick cheat sheet

{% include gallery.html images=page.images gallery_id=page.title %}

The table below shows compatibility FlashAttention intermidiate versions for each NVIDIA architecture generation.

FlashAttention focused on NVIDIA Ampere and higher architecture:
- **Turing** (SM75) — RTX 20xx, T4  
- **Ampere** (SM80/SM86) — A100, RTX 30xx 
- **Ada** (SM89) — RTX 40xx  
- **Hopper** (SM90) — H100/H800  

| FA generation | FA version | CUDA | GPU architecture | PyTorch |
|---|---:|---:|---|---:|
| FA1 | 1.0.9 | ≥ 11.4 | Turing, Ampere | ≥ 1.12 |
| FA2 | 2.0.9 | ≥ 11.4 | Ampere, Ada, Hopper | ≥ 1.12 |
| FA2 | 2.1.2.post3 | ≥ 11.4 | Ampere, Ada, Hopper | ≥ 1.12 |
| FA2 | 2.2.5 | ≥ 11.4 | Ampere, Ada, Hopper | ≥ 1.12 |
| FA2 | 2.3.6 | ≥ 11.4 | Ampere, Ada, Hopper | ≥ 1.12 |
| FA2 | 2.4.3.post1 | ≥ 11.6 | Ampere, Ada, Hopper | ≥ 1.12 |
| FA2 | 2.5.9.post1 | ≥ 11.6 | Ampere, Ada, Hopper | ≥ 1.12 |
| FA2 | 2.6.3 | ≥ 11.6 | Ampere, Ada, Hopper | ≥ 1.12 |
| FA3 | 2.7.4.post1 | ≥ 12.0 | Hopper | ≥ 2.2 |
| FA3 | 2.8.3 | ≥ 12.0 | Hopper| ≥ 2.2 |
