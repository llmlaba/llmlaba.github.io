---
layout: default
title: "Begin your journey in LLM"
date: 2026-01-26
categories: [general]
---
> Date: {{ page.date | date: "%d.%m.%Y" }}  

# Begin your journey in LLM
> ...

## Hugging Face
There is only one reliable hub and commuinty for llm models - [Hugging Face](https://huggingface.co)  
It makes sense to create an account there as soon as you start working with LLMs.  
It is also helpful to have external storage for LLMs, as each model may require several dozen GB of disk space. Make sure you have sufficient storage.

## Recomended models
- Only apache2, mit, openrail licence

### LLM

#### Text completion/chat
- [Mistral 7b v1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [Falcon 7b](https://huggingface.co/tiiuae/falcon-7b)
- [GLM4 9b](https://huggingface.co/zai-org/GLM-4-9B-0414)
- [Olmo3 7b](https://huggingface.co/allenai/Olmo-3-1025-7B)
- [Yi 9b](https://huggingface.co/01-ai/Yi-9B)
- [Qwen3 8b](https://huggingface.co/Qwen/Qwen3-8B-Base)
- [Internlm3 8B](https://huggingface.co/internlm/internlm3-8b-instruct)
- [PHI4](https://huggingface.co/microsoft/phi-4)
- [Granite3.3 8B](https://huggingface.co/ibm-granite/granite-3.3-8b-base)

#### Multimodal LLM
- [Pixtral 12b](https://huggingface.co/mistral-community/pixtral-12b)
- [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)

#### Embedding
- [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- [Qwen3 Embedding 0.6b](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

#### Rerank
- [ms-marco-MiniLM-L12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2)
- [Qwen3 Reranker 0.6b](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)

### No LLM

#### Picture generation
- [Stable Diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) - picture generation model
- [Stable Diffusion 2.0](https://huggingface.co/sd2-community/stable-diffusion-2)
- [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

#### Video generation
- [WAN 2.1 VACE Diffusers](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers)

#### TTS
- [SUNO Bark](https://huggingface.co/suno/bark)

## Civitai
[civitai](https://civitai.com) - very nice example cloud image generation solution, where you can create your own LoRA and Generate pictures.

## Software
- PyTorch - LLM running and training solution
- llama.cpp - C++ runtime implementation for running quantized LLMs locally in gguf format
- CUDA - NVIDIA GPU API to use gpu for different calculations
- ROCm - AMD GPU API to use gpu for different calculations

## Hardware
In order to work with LLM you need a very powerful GPU, I recommend a minimum of 32GB.  
- Nvidia - RTX 4090/5090
- AMD - Instinct 100/200/300
