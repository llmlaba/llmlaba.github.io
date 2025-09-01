---
layout: default
title: "NVIDIA CUDA Docker Base Images"
date: 2025-08-23
categories: [general]

images:
  - /assets/articles/general/Logo_and_CUDA.png
---
> Date: {{ page.date | date: "%d.%m.%Y" }}  

# NVIDIA CUDA Docker Base Images 

{% include gallery.html images=page.images gallery_id=page.title %}

A curated snapshot of the latest **minor/patch** releases for major CUDA lines starting from CUDA 11.3.  
Images are NVIDIA's official **dev** bases (no PyTorch / TensorFlow included).  
All tags are explicit.

| CUDA version | Base OS     | Image (Docker Hub link) |
|---|---|---|
| 11.3 | Ubuntu 20.04 | [nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04](https://hub.docker.com/layers/nvidia/cuda/11.3.1-cudnn8-devel-ubuntu20.04/images/sha256-bba65f869979ef6b4157263c0f96b622b03373846a22edf21704cb9d0b1bbab7) |
| 11.4 | Ubuntu 20.04 | [nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04](https://hub.docker.com/layers/nvidia/cuda/11.4.3-cudnn8-devel-ubuntu20.04/images/sha256-9b1d7c5e6d8b13b7193517cd79de1bfdcd915c5a7c162bdebd03b3e7c07f1b6f) |
| 11.5 | Ubuntu 20.04 | [nvidia/cuda:11.5.2-cudnn8-devel-ubuntu20.04](https://hub.docker.com/layers/nvidia/cuda/11.5.2-cudnn8-devel-ubuntu20.04/images/sha256-d29799715646a30c652487078ec85d9f681a361204d2fd934a416df37512ae79) |
| 11.6 | Ubuntu 20.04 | [nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04](https://hub.docker.com/layers/nvidia/cuda/11.6.2-cudnn8-devel-ubuntu20.04/images/sha256-77532dc0c82a1914809de4afde61840b94549c133583c3312a91e01d3942e1cd) |
| 11.7 | Ubuntu 20.04 | [nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04](https://hub.docker.com/layers/nvidia/cuda/11.7.1-cudnn8-devel-ubuntu20.04/images/sha256-fb2afb86d2ad20c40e1daff83fcb8e33f88c29878535e602f8f752136a6b9db2) |
| 11.8 | Ubuntu 22.04 | [nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04](https://hub.docker.com/layers/nvidia/cuda/11.8.0-cudnn8-devel-ubuntu22.04/images/sha256-bd746eb3b9953805ebe644847a227e218b5da775f47007c69930569a75c9ad7d) |
| 12.0 | Ubuntu 22.04 | [nvidia/cuda:12.0.1-cudnn8-devel-ubuntu22.04](https://hub.docker.com/layers/nvidia/cuda/12.0.1-cudnn8-devel-ubuntu22.04/images/sha256-b51c8d3d0d2116b5d8b3edd3ee6eb62ef950510ab0aa5a6e719580d0a07d28c2) |
| 12.1 | Ubuntu 22.04 | [nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04](https://hub.docker.com/layers/nvidia/cuda/12.1.1-cudnn8-devel-ubuntu22.04/images/sha256-eef662e5e56afc7bd9ac1bb9f3c1cb8959d4a6002e7a305374dc14237ca9e73f) |
| 12.2 | Ubuntu 22.04 | [nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04](https://hub.docker.com/layers/nvidia/cuda/12.2.2-cudnn8-devel-ubuntu22.04/images/sha256-59754b393b23dabe825aa0f474f83f2bdaa418c5e9dc778afb4a980536e28196) |
| 12.3 | Ubuntu 22.04 | [nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04](https://hub.docker.com/layers/nvidia/cuda/12.3.2-cudnn9-devel-ubuntu22.04/images/sha256-4f00d5116a3679bab6bc13318c8555d7207206de2318e77348a9a93f66e73e21) |
| 12.4 | Ubuntu 22.04 | [nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04](https://hub.docker.com/layers/nvidia/cuda/12.4.1-cudnn-devel-ubuntu22.04/images/sha256-0a1cb6e7bd047a1067efe14efdf0276352d5ca643dfd77963dab1a4f05a003a4) |
| 12.5 | Ubuntu 22.04 | [nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04](https://hub.docker.com/layers/nvidia/cuda/12.5.1-cudnn-devel-ubuntu22.04/images/sha256-6f0ad658b45e0468d4de5b3d464cec34b2173343061a667b10877a584c229b77) |
| 12.6 | Ubuntu 24.04 | [nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04](https://hub.docker.com/layers/nvidia/cuda/12.6.3-cudnn-devel-ubuntu24.04/images/sha256-c51bfc8bcd4febe3e26952615496b4347767f61f9079f08ffc914b42905e510d) |
| 12.8 | Ubuntu 22.04 | [nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04](https://hub.docker.com/layers/nvidia/cuda/12.8.1-cudnn-devel-ubuntu22.04/images/sha256-799b227be38cc134465aa75feb94ad6cd7c6f9ebf8a014de4485818d11bbbfa8) |
| 12.9 | Ubuntu 22.04 | [nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04](https://hub.docker.com/layers/nvidia/cuda/12.9.1-cudnn-devel-ubuntu22.04/images/sha256-9afe8629681745513026de134b8e25d57339ee59add5b554f675d33cdebbe3ab) |
| 13.0 | Ubuntu 24.04 | [nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04](https://hub.docker.com/layers/nvidia/cuda/13.0.0-cudnn-devel-ubuntu24.04/images/sha256-a40069bdefcb252e760d5d061ee31a8b4b041343848ab2b9b910bc0a0c026e78) |
