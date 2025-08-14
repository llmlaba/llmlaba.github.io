---
layout: default
title: "Begin your journey in LLM"
date: 2025-08-14
categories: [general]
---

# Begin your journey in LLM
> ...

## Hugging Face
There is only one reliable hub and commuinty for llm models - [Hugging Face](https://huggingface.co)  
It makes sense to create an account there as soon as you start working with LLMs.  
It is also helpful to have external storage for LLMs, as each model may require several dozen GB of disk space. Make sure you have sufficient storage.

## Recomended models
- [Mistral 7b v1](https://huggingface.co/mistralai/Mistral-7B-v0.1) - good text generation model
- [Stable Diffusion v1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) - picture generation model

## Civitai
[civitai](https://civitai.com) - very nice example cloud image generation solution, where you can create your own LoRA and Generate pictures.

## Software
- PyTorch - LLM running and training solution
- llama.cpp - C++ runtime implementation for running quantized LLMs locally in gguf format
- CUDA - NVIDIA GPU API to use gpu for different calculations
- ROCm - AMD GPU API to use gpu for different calculations

## Hardware
In order to work with LLM you need a very powerful GPU, I recommend a minimum of 32GB.  
- Nvidia - RTX 4090/5090/Quadro RTX 8000/Tesla A100 
- AMD - Instinct Mi50/100/200
