---
layout: default
title: "BitsAndBytes ROCm Compatibility"
date: 2025-08-28
categories: [general]

images:
  - /assets/articles/general/ROCm_logo.png
  - /assets/articles/general/BNB_Logo.png
---

# BitsAndBytes ROCm Compatibility — quick cheat sheet

{% include gallery.html images=page.images gallery_id=page.title %}

> Quick reference for **BNB - ROCm Toolkit - PyTorch**.  
> Focus: what you can realistically run or build from source. PyTorch versions are **approximate** and reflect practical compatibility.

## Matrix for ROCm backend

| BNB Version | ROCm Toolkit | Min AMD GFX | PyTorch |
|---|---|---|---|
| **0.43.3** | **6.1** (preview) | Targets commonly used in preview builds: **gfx90a / gfx942 / gfx1100** (CDNA2/3, RDNA3) | **2.2 – 2.4** |
| **0.44.1** | **6.1 – 6.2** | Same primary targets: **gfx90a / gfx942 / gfx1100** | **2.2 – 2.5** |
| **0.45.5** | **6.1 – 6.3** | Focused on **gfx908/90a/942 (CDNA/2/3)** and **gfx1100/1101 (RDNA3)**; | **2.2 – 2.6** |
| **0.46.1** | **6.1 – 6.4** | Main targets: **gfx908/90a/942/1100**. | **2.3 – 2.6+** |

### Important notes
- **Status:** ROCm support in bnb is a **multi-backend/preview** story. Prebuilt wheels exist only for *specific* ROCm versions and *specific* GFX targets; everything else → **build from source**.
- **Architectures:** Fast kernels rely on matrix/WMMA stacks available on **gfx90x/942** (CDNA2/3) and **gfx11xx** (RDNA3). Older archs (e.g., **gfx906**) typically fall back to slower code paths — when they run at all.
- **OS:** ROCm backend is **Linux only**.
