---
layout: default
title: "AMD ROCm PyTorch in Docker Test"
date: 2025-08-24
categories: [software, llm]

images:
  - /assets/articles/docker-rocm-pytorch-test/1.jpg
  - /assets/articles/docker-rocm-pytorch-test/2.jpg
  - /assets/articles/docker-rocm-pytorch-test/3.jpg
---

# AMD ROCm PyTorch in Docker Test

> In this articale detailed described how to run PyTorch with AMD ROCm in docker container.  
> Tested LLM Mistral 7b.  

{% include gallery.html images=page.images gallery_id=page.title %}

## Requirments 
- AMD Mi50/MI100 32Gb VRAM
- Workstation 40 GB RAM, 500GB SSD, 750W Power supply 
- Ubuntu 24.04 LTS
- Docker CE

> My test environment: HP Z440 + AMD Mi50 32gb

## Steps

### Get Mistral 7b for test

```bash
git lfs install
git clone https://huggingface.co/mistralai/Mistral-7B-v0.1 mistral
```

### Prepare `Dockerfile` to run mistral

- Dockerfile 
> There are few important steps that we need to complete in Dockerfile.  
> - Create application user
> - Install tini to avoid zombie processes
> - Install all necessary libraries for Mistral like `transformers`, etc...
> - Put simple web server to docker image, just for tests

```dockerfile
FROM docker.io/rocm/pytorch:rocm6.2.2_ubuntu22.04_py3.10_pytorch_release_2.3.0

USER root

RUN groupadd -g 4001 appuser && \
    useradd -m -u 4001 -g 4001 appuser && \
    mkdir /{app,llm} && \
    chown appuser:appuser /{app,llm}

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tini && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
COPY environment.yml ./environment.yml

RUN conda env update -n py_3.10 -f environment.yml

COPY run_mistral.py ./run_mistral.py

USER appuser
ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["python3", "/app/run_mistral.py"]
```

- Web server `run_mistral.py`
> Web server implementation description:  
