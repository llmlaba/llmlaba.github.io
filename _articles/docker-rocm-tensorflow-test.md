---
layout: default
title: "AMD ROCm TensorFlow in Docker Test"
date: 2025-08-25
categories: [software, llm]

images:
  - /assets/articles/docker-rocm-tensorflow-test/1.jpg
  - /assets/articles/docker-rocm-tensorflow-test/2.jpg
  - /assets/articles/docker-rocm-tensorflow-test/3.jpg
---
> Date: {{ page.date | date: "%d.%m.%Y" }}  

# AMD ROCm TensorFlow in Docker Test

> In this article detailed described how to run TensorFlow with AMD ROCm in docker container.  
> Tested LLM GPT2.  

{% include gallery.html images=page.images gallery_id=page.title %}

## Requirements 
- AMD Mi50/MI100 32Gb VRAM
- Workstation 40 GB RAM, 500GB SSD, 750W Power supply 
- Ubuntu 24.04 LTS
- Docker CE

> My test environment: HP Z440 + AMD Mi50 32gb

## Steps

### Get GPT2 for test

```bash
git lfs install
git clone https://huggingface.co/openai-community/gpt2 gpt2
```

### Prepare `Dockerfile` to run GPT2

#### Dockerfile 
There are a few important steps that we need to complete in Dockerfile.  
- Build metapackage for tensorflow 2.16.2
- Create application user
- Install tini to avoid zombie processes
- Install the TensorFlow metapackage to prevent tensorflow-rocm from being replaced by the default TensorFlow package
- Install all necessary libraries for GPT2 like `transformers`, etc...
- Put simple web server to docker image, just for tests

```dockerfile
FROM docker.io/rocm/tensorflow:rocm6.2.2-py3.9-tf2.16.2-dev AS builder

USER root

WORKDIR /build

COPY meta-tensorflow.toml /build/tensorflow/pyproject.toml

RUN pip3 install --upgrade pip wheel setuptools build && \
    python3 -m build --wheel /build/tensorflow

FROM docker.io/rocm/tensorflow:rocm6.2.2-py3.9-tf2.16.2-dev

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

COPY --from=builder /build/tensorflow/dist/tensorflow-2.16.2.post1-py3-none-any.whl ./tensorflow-2.16.2.post1-py3-none-any.whl

RUN pip3 install --upgrade pip && \
    pip3 install ./tensorflow-2.16.2.post1-py3-none-any.whl && \
    pip3 install -r requirements.txt

COPY run_gpt2.py ./run_gpt2.py

USER appuser
ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["python3", "/app/run_gpt2.py"]
```

#### Web server `run_gpt2.py`
Web server implementation description:  
- Run GPT2
- Run web server with one endpoint for testing `/v1/completion` - a legacy-style text completion endpoint

```python
from flask import Flask, request, jsonify, Response
from transformers import AutoTokenizer, TFAutoModelForCausalLM
import tensorflow as tf
import time, uuid

print("TF GPUs:", tf.config.list_physical_devices("GPU"))

MODEL_PATH = "/llm/gpt2"

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
tok.pad_token = tok.eos_token
model = TFAutoModelForCausalLM.from_pretrained(MODEL_PATH)

print("Model loaded.")

inputs = tok("The space stars is?", return_tensors="tf")
out = model.generate(**inputs, max_new_tokens=20)
print(tok.decode(out[0], skip_special_tokens=True))

app = Flask(__name__)

# -------- helpers --------
def _truncate_at_stop(text, stops):
    if not stops:
        return text, None
    cut_idx = None
    for s in stops:
        if not s:
            continue
        i = text.find(s)
        if i == 0:  
            continue
        if i != -1 and (cut_idx is None or i < cut_idx):
            cut_idx = i
    if cut_idx is not None:
        return text[:cut_idx], "stop"
    return text, None

def _tok_count(s: str) -> int:
    return len(tok.encode(s, add_special_tokens=False))

# -------- endpoints --------
@app.get("/health")
def health():
    return Response("ok", mimetype="text/plain")

@app.post("/v1/completion")
def completion():
    """
    JSON:
      {
        "prompt": "string",            # required
        "max_tokens": 128,             # optional
        "temperature": 0.7,            # optional
        "top_p": 0.95,                 # optional
        "stop": "\n\n" or ["###"]      # optional
      }
    """
    data = request.get_json(force=True) or {}
    prompt = data.get("prompt")
    if not isinstance(prompt, str):
        return jsonify({"error": {"message": "Field 'prompt' (string) is required"}}), 400

    max_tokens  = int(data.get("max_tokens", 128))
    temperature = float(data.get("temperature", 0.7))
    top_p       = float(data.get("top_p", 0.95))
    stop        = data.get("stop")
    stops = [stop] if isinstance(stop, str) else [s for s in (stop or []) if isinstance(s, str)]

    do_sample = temperature > 0.0

    compl_id = f"cmpl-{uuid.uuid4().hex}"
    t0 = time.time()

    inputs = tok(prompt, return_tensors="tf")
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=do_sample,
        temperature=max(temperature, 1e-8),
        top_p=top_p,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    app.logger.info(f"[{compl_id}] {time.time()-t0:.2f}s for {max_tokens} tokens")

    prompt_len = int(inputs["input_ids"].shape[1])
    gen_ids = output_ids[0][prompt_len:]
    text = tok.decode(gen_ids, skip_special_tokens=True)

    text, finish_reason = _truncate_at_stop(text.lstrip(), stops)
    if finish_reason is None:
        finish_reason = "length" if _tok_count(text) >= max_tokens else "stop"

    usage = {
        "prompt_tokens": _tok_count(prompt),
        "completion_tokens": _tok_count(text),
        "total_tokens": _tok_count(prompt) + _tok_count(text),
    }

    resp = {
        "id": compl_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": "gpt2-tf-local",
        "choices": [{
            "index": 0,
            "text": text,
            "finish_reason": finish_reason
        }],
        "usage": usage
    }
    return jsonify(resp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, threaded=True)
```

#### PIP install GPT2 dependencies
To run an LLM inside the Docker container provided by AMD on Docker Hub, we need to install several additional libraries. The tricky part is that AMD compiled TensorFlow with a `-rocm` suffix, which breaks pip dependencies that expect the standard `tensorflow` package name. To resolve this, we create a metapackage for pip with the correct name and version (`tensorflow` `2.16.2.post1`). This package is built in a build container and must be installed before installing `requirements.txt`.
- File `meta-tensorflow.toml` with tensorflow metapackage build settings

```toml
[build-system]
requires = ["setuptools>=68", "wheel", "build"]
build-backend = "setuptools.build_meta"

[project]
name = "tensorflow"
version = "2.16.2.post1"
description = "Alias: tensorflow -> tensorflow-rocm"
requires-python = ">=3.9"
dependencies = ["tensorflow-rocm==2.16.2"]

[tool.setuptools]
py-modules = []
```

- File `requirements.txt` required to install pip dependancies

```
Flask==3.0.3
transformers==4.41.2
tokenizers==0.19.1
safetensors==0.4.3
huggingface-hub==0.23.4
sentencepiece==0.2.0
tf-keras~=2.16
```

### Run TensorFlow with ROCm in Docker Compose

#### Prepare `docker-compose.yaml` for AMD ROCm
To run AMD ROCm in docker we will use docker-compose orchestration to make deploy more clear.  
Main docker compose orchestration steps
- Build new image for LLM, bake libraries and application scripts inside
- Enable port forwarding for application to docker host
- Set environment variables to run tensorflow properly
- Mount AMD driver devices to container
- Add AMD ROCm groups to container user
- Mount folder with LLM GPT2
- Create local network just in case

```yaml
version: "3.3"

services:
  tensorflow-rocm.local:
    image: tensorflow-rocm:latest
    build:
      context: ./
      dockerfile: Dockerfile
    ports:
      - "8080:8080"
    environment:
      TZ: "Etc/GMT"
      LANG: "C.UTF-8"
      TF_CPP_MIN_LOG_LEVEL: "2"
      TF_USE_LEGACY_KERAS: "1"
    ipc: host
    devices:
      - /dev/kfd
      - /dev/dri
    group_add:
      - "${RENDER_GID}"
      - "${VIDEO_GID}"
    volumes:
      - ../gpt2:/llm/gpt2
    networks:
      - docker-compose-network

networks:
  docker-compose-network:
    ipam:
      config:
        - subnet: 172.24.24.0/24
```

#### Run GPT2 in Docker and make a test request
- Deploy docker compose 
```bash
echo "RENDER_GID=$(getent group render | cut -d: -f3)" > .env
echo "VIDEO_GID=$(getent group video  | cut -d: -f3)" >> .env
docker-compose up
```
- Check logs
```bash
docker container logs tensorflow-rocm_tensorflow-rocm.local_1
```

- Test request 
```bash
curl -s http://localhost:8080/v1/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What you know about sun?",
    "max_tokens": 60,
    "temperature": 0.7,
    "top_p": 0.95,
    "stop": "eof"
  }' | jq
```

- Stop docker container
```bash
docker-compose down
```

## Enjoy the result
All project avalible on [github](https://github.com/llmlaba/llm-in-docker)
 