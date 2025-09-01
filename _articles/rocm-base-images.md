---
layout: default
title: "AMD ROCm Docker Base Images"
date: 2025-08-23
categories: [general]

images:
  - /assets/articles/general/ROCm_logo.png
---
> Date: {{ page.date | date: "%d.%m.%Y" }}  

# AMD ROCm Docker Base Images 

{% include gallery.html images=page.images gallery_id=page.title %}

A curated snapshot of the latest **minor/patch** releases for major ROCm lines starting from ROCm 5.  
Images are AMD's official **dev** bases (no PyTorch / TensorFlow included).  
All tags are explicit.

| ROCm version | Base OS     | Image (Docker Hub link) |
|---|---|---|
| 6.4.3 | Ubuntu 24.04 | [rocm/dev-ubuntu-24.04:6.4.3-complete](https://hub.docker.com/r/rocm/dev-ubuntu-24.04/tags?name=6.4.3-complete) |
| 6.4.3 | Ubuntu 22.04 | [rocm/dev-ubuntu-22.04:6.4.3-complete](https://hub.docker.com/r/rocm/dev-ubuntu-22.04/tags?name=6.4.3-complete) |
| 6.3.4 | Ubuntu 22.04 | [rocm/dev-ubuntu-22.04:6.3.4-complete](https://hub.docker.com/r/rocm/dev-ubuntu-22.04/tags?name=6.3.4-complete) |
| 6.2.4 | Ubuntu 22.04 | [rocm/dev-ubuntu-22.04:6.2.4-complete](https://hub.docker.com/r/rocm/dev-ubuntu-22.04/tags?name=6.2.4-complete) |
| 6.1.2 | Ubuntu 22.04 | [rocm/dev-ubuntu-22.04:6.1.2-complete](https://hub.docker.com/r/rocm/dev-ubuntu-22.04/tags?name=6.1.2-complete) |
| 6.0.2 | Ubuntu 22.04 | [rocm/dev-ubuntu-22.04:6.0.2-complete](https://hub.docker.com/r/rocm/dev-ubuntu-22.04/tags?name=6.0.2-complete) |
| 5.7.1 | Ubuntu 20.04 | [rocm/dev-ubuntu-20.04:5.7.1-complete](https://hub.docker.com/r/rocm/dev-ubuntu-20.04/tags?name=5.7.1-complete) |
