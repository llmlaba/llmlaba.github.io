---
layout: default
title: "NVIDIA CUDA Docker Images for PyTorch / TensorFlow"
date: 2025-08-23
categories: [general]

images:
  - /assets/articles/general/Logo_and_CUDA.png
  - /assets/articles/general/PyTorch_logo.jpeg
  - /assets/articles/general/TF_logo.png
---
> Date: {{ page.date | date: "%d.%m.%Y" }}  

# NVIDIA CUDA Docker Images for PyTorch / TensorFlow

{% include gallery.html images=page.images gallery_id=page.title %}

A concise snapshot of the latest **minor/patch** releases per major CUDA line (starting from CUDA 11), mapped to corresponding **PyTorch**/**TensorFlow** images.
- Only fixed tags (no `latest`).
- Images are the official `pytorch/pytorch` and `tensorflow/tensorflow` images.

| CUDA | Ubuntu | Framework version| Docker image |
|---|---:|---:|---|
| 11.2 | Ubuntu 20.04 | TensorFlow 2.10.0 | [tensorflow/tensorflow:2.10.0-gpu](https://hub.docker.com/r/tensorflow/tensorflow/tags?name=2.10.0-gpu) |
| 11.3 | Ubuntu 20.04 | PyTorch 1.12.1 | [pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel](https://hub.docker.com/r/pytorch/pytorch/tags?name=1.12.1-cuda11.3-cudnn8-devel) |
| 11.6 | Ubuntu 20.04 | PyTorch 1.13.1 | [pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel](https://hub.docker.com/r/pytorch/pytorch/tags?name=1.13.1-cuda11.6-cudnn8-devel) |
| 11.7 | Ubuntu 20.04 | PyTorch 2.0.1 | [pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel](https://hub.docker.com/r/pytorch/pytorch/tags?name=2.0.1-cuda11.7-cudnn8-devel) |
| 11.8 | Ubuntu 22.04 | PyTorch 2.7.1 | [pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel](https://hub.docker.com/r/pytorch/pytorch/tags?name=2.7.1-cuda11.8-cudnn9-devel) |
| 11.8 | Ubuntu 20.04 | TensorFlow 2.14.0 | [tensorflow/tensorflow:2.14.0-gpu](https://hub.docker.com/r/tensorflow/tensorflow/tags?name=2.14.0-gpu) |
| 12.1 | Ubuntu 22.04 | PyTorch 2.5.1 | [pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel](https://hub.docker.com/r/pytorch/pytorch/tags?name=2.5.1-cuda12.1-cudnn9-devel) |
| 12.2 | Ubuntu 22.04 | TensorFlow 2.15.0 | [tensorflow/tensorflow:2.15.0-gpu](https://hub.docker.com/r/tensorflow/tensorflow/tags?name=2.15.0-gpu) |
| 12.3 | Ubuntu 22.04 | TensorFlow 2.16.1 | [tensorflow/tensorflow:2.16.1-gpu](https://hub.docker.com/r/tensorflow/tensorflow/tags?name=2.16.1-gpu) |
| 12.3 | Ubuntu 22.04 | TensorFlow 2.17.0 | [tensorflow/tensorflow:2.17.0-gpu](https://hub.docker.com/r/tensorflow/tensorflow/tags?name=2.17.0-gpu) |
| 12.4 | Ubuntu 22.04 | PyTorch 2.6.0 | [pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel](https://hub.docker.com/r/pytorch/pytorch/tags?name=2.6.0-cuda12.4-cudnn9-devel) |
| 12.5 | Ubuntu 22.04 | TensorFlow 2.19.0 | [tensorflow/tensorflow:2.19.0-gpu](https://hub.docker.com/r/tensorflow/tensorflow/tags?name=2.19.0-gpu) |
| 12.6 | Ubuntu 24.04 | PyTorch 2.8.0 | [pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel](https://hub.docker.com/r/pytorch/pytorch/tags?name=2.8.0-cuda12.6-cudnn9-devel) |
| 12.8 | Ubuntu 22.04 | PyTorch 2.8.0 | [pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel](https://hub.docker.com/r/pytorch/pytorch/tags?name=2.8.0-cuda12.8-cudnn9-devel) |
| 12.9 | Ubuntu 22.04 | PyTorch 2.8.0 | [pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel](https://hub.docker.com/r/pytorch/pytorch/tags?name=2.8.0-cuda12.9-cudnn9-devel) |

### NOTES
- Avoid using the proprietary NVIDIA repository `nvcr.io` as most usage scenarios may violate NVIDIA's license.
