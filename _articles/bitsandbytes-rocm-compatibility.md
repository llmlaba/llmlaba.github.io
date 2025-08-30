---
layout: default
title: "BitsAndBytes ROCm Compatibility"
date: 2025-08-28
categories: [general]

images:
  - /assets/articles/general/ROCm_logo.png
  - /assets/articles/general/BNB_Logo.png
---

# BitsAndBytes ROCm Compatibility

{% include gallery.html images=page.images gallery_id=page.title %}

> The ROCm support still in **beta** and exist only in one specific branch  
> Few latest BitsAndBytes releases **0.43.3 0.44.1, 0.45.5, 0.46.0** - **NOT INCLUDES** **ROCM** support.  
> To build and install BitsAndBytes for ROCm please use AMD BitsAndBytes fork [AMD/ROCm/bitsandbytes](https://github.com/ROCm/bitsandbytes/tree/rocm_enabled_multi_backend)

## Matrix for ROCm backend

| BNB Version | ROCm Toolkit | Min AMD GFX | PyTorch |
|---|---|---|---|
| **rocm_enabled_multi_backend** *(start from version 0.43.3)* | **6.1 +** | **gfx906** | **2.2 +** |
