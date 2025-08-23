---
layout: default
title: "AMD ROCm Docker Images for PyTorch / TensorFlow"
date: 2025-08-23
categories: [general]

images:
  - /assets/articles/general/ROCm_logo.png
  - /assets/articles/general/PyTorch_logo.jpeg
  - /assets/articles/general/TF_logo.png
---

# AMD ROCm Docker Images for PyTorch / TensorFlow

{% include gallery.html images=page.images gallery_id=page.title %}


A concise snapshot of the latest **minor/patch** releases per major ROCm line (starting from ROCm 5), mapped to corresponding **PyTorch**/**TensorFlow** images.

- Only fixed tags (no `latest`).
- Images are the official `rocm/pytorch` and `rocm/tensorflow` images.
- Base OS is included when the tag or AMD docs explicitly indicate it; in some TF tags it is inferred from the Python version (e.g., `py3.12` → Ubuntu 24.04; `py3.10` → Ubuntu 22.04).

| ROCm version | Base OS | Image | Framework version |
|---|---|---|---|
| 6.4.3 | Ubuntu 24.04 | [rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0](https://hub.docker.com/r/rocm/pytorch/tags?name=rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0) | PyTorch 2.6.0 |
| 6.4.3 | Ubuntu 22.04 | [rocm/pytorch:rocm6.4.3_ubuntu22.04_py3.10_pytorch_release_2.5.1](https://hub.docker.com/r/rocm/pytorch/tags?name=rocm6.4.3_ubuntu22.04_py3.10_pytorch_release_2.5.1) | PyTorch 2.5.1 |
| 6.4.2 | Ubuntu 24.04 | [rocm/tensorflow:rocm6.4.2-py3.12-tf2.18-dev](https://hub.docker.com/r/rocm/tensorflow/tags?name=rocm6.4.2-py3.12-tf2.18-dev) | TensorFlow 2.18.1 |
| 6.4.2 | Ubuntu 22.04 | [rocm/tensorflow:rocm6.4.2-py3.10-tf2.18-dev](https://hub.docker.com/r/rocm/tensorflow/tags?name=rocm6.4.2-py3.10-tf2.18-dev) | TensorFlow 2.18.1 |
| 6.3.4 | Ubuntu 22.04 | [rocm/pytorch:rocm6.3.4_ubuntu22.04_py3.10_pytorch_release_2.4.0](https://hub.docker.com/r/rocm/pytorch/tags?name=rocm6.3.4_ubuntu22.04_py3.10_pytorch_release_2.4.0) | PyTorch 2.4.0 |
| 6.3.1 | Ubuntu 24.04 | [rocm/tensorflow:rocm6.3.1-py3.12-tf2.17.0-dev](https://hub.docker.com/layers/rocm/tensorflow/rocm6.3.1-py3.12-tf2.17.0-dev/images/sha256-804121ee4985718277ba7dcec53c57bdade130a1ef42f544b6c48090ad379c17) | TensorFlow 2.17.0 |
| 6.3.1 | Ubuntu 22.04 | [rocm/tensorflow:rocm6.3.1-py3.10-tf2.17.0-dev](https://hub.docker.com/layers/rocm/tensorflow/rocm6.3.1-py3.10-tf2.17.0-dev/images/sha256-776837ffa945913f6c466bfe477810a11453d21d5b6afb200be1c36e48fbc08e) | TensorFlow 2.17.0 |
| 6.2.2 | Ubuntu 22.04 | [rocm/pytorch:rocm6.2.2_ubuntu22.04_py3.10_pytorch_release_2.3.0](https://hub.docker.com/r/rocm/pytorch/tags?name=rocm6.2.2_ubuntu22.04_py3.10_pytorch_release_2.3.0) | PyTorch 2.3.0 |
| 6.2.2 | Ubuntu 22.04* | [rocm/tensorflow:rocm6.2.2-py3.9-tf2.16.2-dev](https://hub.docker.com/layers/rocm/tensorflow/rocm6.2.2-py3.9-tf2.16.2-dev/images/sha256-393b65c1b3b58d3f91fd2bf0fddbbe42e97da6d15f0a94140f984c542287cc81) | TensorFlow 2.16.2 |
| 6.1.2 | Ubuntu 22.04 | [rocm/pytorch:rocm6.1.2_ubuntu22.04_py3.10_pytorch_release-2.1.2](https://hub.docker.com/layers/rocm/pytorch/rocm6.1.2_ubuntu22.04_py3.10_pytorch_release-2.1.2/images/sha256-c8b4e8dfcc64e9bf68bf1b38a16fbc5d65b653ec600f98d3290f66e16c8b6078) | PyTorch 2.1.2 |
| 6.1.x | Ubuntu 22.04* | [rocm/tensorflow:rocm6.1-py3.9-tf2.14-runtime](https://hub.docker.com/layers/rocm/tensorflow/rocm6.1-py3.9-tf2.14-runtime/images/sha256-013ed06ba92330596c3e1771916fe2489567aab94306893eeaaad4585bda4d48) | TensorFlow 2.14.0 |
| 6.0.2 | Ubuntu 22.04 | [rocm/pytorch:rocm6.0.2_ubuntu22.04_py3.10_pytorch_2.1.2](https://hub.docker.com/r/rocm/pytorch/tags?name=rocm6.0.2_ubuntu22.04_py3.10_pytorch_2.1.2) | PyTorch 2.1.2 |
| 6.0.x | Ubuntu 22.04* | [rocm/tensorflow:rocm6.0-tf2.13-dev](https://hub.docker.com/layers/rocm/tensorflow/rocm6.0-tf2.13-dev/images/sha256-c4b215474f048467b48283535b9ca4561a77bbe8f6f7cc877ebc2b124e75d6b5) | TensorFlow 2.13.x |
| 5.7 | Ubuntu 22.04 | [rocm/pytorch:rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1](https://hub.docker.com/layers/rocm/pytorch/rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1/images/sha256-21df283b1712f3d73884b9bc4733919374344ceacb694e8fbc2c50bdd3e767ee) | PyTorch 2.0.1 |
| 5.5 | Ubuntu 20.04* | [rocm/tensorflow:rocm5.5-tf2.10-dev](https://hub.docker.com/layers/rocm/tensorflow/rocm5.5-tf2.10-dev/images/sha256-dd60d71143782b1f73f3e7579717b87fa13a48b9e5eafe981dcd69803f5fddf1) | TensorFlow 2.10 |

---

### Notes
- **OS inference**: In TF tags where the OS is not encoded, OS is inferred from Python version or time period; entries marked with `*` are best-effort.
- For TF 6.4.x, AMD documentation explicitly lists Ubuntu 24.04 for `py3.12` and Ubuntu 22.04 for `py3.10` tags.
