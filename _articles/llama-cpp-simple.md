---
layout: default
title: "llama.cpp it is simple"
date: 2025-08-13
categories: [software, llm]

images:
  - /assets/articles/llama-cpp-simple/1.jpg
  - /assets/articles/llama-cpp-simple/2.jpg
  - /assets/articles/llama-cpp-simple/1.gif
---

# llama.cpp - run LLM everywhere 

> llama.cpp is a universal solution for running quantized LLM models on a wide range of hardware.

{% include gallery.html images=page.images gallery_id=page.title %}

## RUN LLM Mistral from scratch with llama.cpp runtime

> My test environment: HP ProDesk 405 G5

### Requirments 
- PC 16 GB RAM, 200GB SSD
- Ubuntu 24.04 LTS
- CPU that support AVX/AVX2/AVX512
- Docker CE for containerised run

### Get the most popular LLM Mistral
```bash
mkdir llm && cd llm
git lfs install
git clone https://huggingface.co/lmstudio-community/mathstral-7B-v0.1-GGUF mistral
```

## In OS
### Preapre llama.cpp environment:
```bash
sudo apt update
sudo apt install -y git build-essential cmake \
  libopenblas-dev libcurl4-openssl-dev pkg-config
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
cmake --build build -j
cd ..
```
### Run llama.cpp benchmark 
```bash
export cores=$(nproc)
./llama.cpp/build/bin/llama-bench -m \
  ./mistral/mathstral-7B-v0.1-Q4_K_M.gguf \
  -t $cores
```
### Run llama.cpp generator:
```bash
export cores=$(nproc)
./llama.cpp/build/bin/llama-cli \
  -m ./mistral/mathstral-7B-v0.1-Q4_K_M.gguf \
  -p "What you know about sun?" \
  -n 2000 -t $cores -c 2048
```
### Run llama.cpp server:
```bash
./llama.cpp/build/bin/llama-server \
  -m ./mistral/mathstral-7B-v0.1-Q4_K_M.gguf \
  --chat-template llama2 --port 8080 --host 0.0.0.0
```

## In Docker
### Run llama.cpp benchmark
```bash
docker run --rm -it -v ~/llm/mistral:/models \
  --entrypoint /bin/bash ghcr.io/ggml-org/llama.cpp:full \
  -c './llama-bench -m /models/mathstral-7B-v0.1-Q4_K_M.gguf \
  -t $(nproc)'
```
### Run llama.cpp generator:
```bash
docker run --rm -it -v ~/llm/mistral:/models \
  --entrypoint /bin/bash ghcr.io/ggml-org/llama.cpp:full \
  -c './llama-cli  -m /models/mathstral-7B-v0.1-Q4_K_M.gguf \
  -p "What you know about sun?" -n 2000 -t $(nproc) -c 2048'
```
### Run llama.cpp server:
```bash
docker run --rm -it -p 8080:8080 -v ~/llm/mistral:/models \
  --entrypoint /bin/bash ghcr.io/ggml-org/llama.cpp:full \
  -c './llama-server -m /models/mathstral-7B-v0.1-Q4_K_M.gguf \
  --chat-template llama2 --port 8080 --host 0.0.0.0'
```
### Enjoy the result!