---
layout: default
title: "TensorFlow-friendly causal LMs in Transformers 4.x"
date: 2025-08-18
categories: [general]

images:
  - /assets/articles/general/TF_logo.png
---

# TensorFlow-friendly causal LMs in 🤗 Transformers 4.x

{% include gallery.html images=page.images gallery_id=page.title %}

This table lists decoder-only (causal LM) model families that have **TensorFlow classes in Transformers 4.x** and publish **TensorFlow weights (`tf_model.h5`)** on the Hugging Face Hub.  
If `tf_model.h5` is present for the checkpoint, you can load with `TFAutoModelForCausalLM.from_pretrained(...)` without converting from PyTorch.

| Model family | Hugging Face repo | TF weights present? (Conversion needed?) |
|---|---|---|
| OpenAI GPT (GPT‑1) | [openai-community/openai-gpt](https://huggingface.co/openai-community/openai-gpt) | **Yes** — `tf_model.h5` available → **No conversion needed** |
| GPT‑2 (family) | [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) | **Yes** — `tf_model.h5` available → **No conversion needed** |
| DistilGPT‑2 | [distilbert/distilgpt2](https://huggingface.co/distilbert/distilgpt2) | **Yes** — `tf_model.h5` available → **No conversion needed** |
| DialoGPT (GPT‑2 based) | [microsoft/DialoGPT-medium](https://huggingface.co/microsoft/DialoGPT-medium) | **Yes** — `tf_model.h5` available → **No conversion needed** |
| CTRL | [Salesforce/ctrl](https://huggingface.co/Salesforce/ctrl) | **Yes** — `tf_model.h5` available → **No conversion needed** |
| Transformer‑XL | [transfo-xl/transfo-xl-wt103](https://huggingface.co/transfo-xl/transfo-xl-wt103) | **Yes** — `tf_model.h5` available → **No conversion needed** |
| XLNet | [xlnet/xlnet-base-cased](https://huggingface.co/xlnet/xlnet-base-cased) | **Yes** — `tf_model.h5` available → **No conversion needed** |
| XLM (CLM variant) | [FacebookAI/xlm-clm-ende-1024](https://huggingface.co/FacebookAI/xlm-clm-ende-1024) | **Yes** — `tf_model.h5` available → **No conversion needed** |
| OPT (small) | [facebook/opt-125m](https://huggingface.co/facebook/opt-125m) | **Yes** — `tf_model.h5` available → **No conversion needed** |
| OPT (2.7B) | [facebook/opt-2.7b](https://huggingface.co/facebook/opt-2.7b) | **Yes** — `tf_model.h5` available → **No conversion needed** |
| GPT‑J‑6B | [EleutherAI/gpt-j-6b](https://huggingface.co/EleutherAI/gpt-j-6b) | **Yes** — `tf_model.h5` available → **No conversion needed** |

## Notes

- **Transformers v4.x only.** TensorFlow classes (e.g., `TFAutoModelForCausalLM`, `TFGPT2LMHeadModel`) are part of the 4.x line. In **Transformers v5**, TF support has been removed. Pin `transformers<5` for TF usage.
- **Per‑checkpoint variation.** Some orgs publish TF weights for certain sizes but not others. Always check the target checkpoint’s *Files & versions* tab.
  - GPT‑2 `medium`/`large`/`xl` also provide TF weights (`tf_model.h5`).
- **If there is no `tf_model.h5` but the family has a TF class:** you can often load from PyTorch with `from_pt=True` and then `save_pretrained(...)` to create a local `tf_model.h5`.
- **Modern LLMs (e.g., LLaMA, Mistral, Mixtral, Gemma)** generally **do not** ship TF classes in Transformers; they are PyTorch‑only in the library.
- **Keras 3 compatibility:** with TF ≥ 2.16 use `tf-keras~=2.16` and set `TF_USE_LEGACY_KERAS=1` before importing TensorFlow so `transformers` 4.x works smoothly.
