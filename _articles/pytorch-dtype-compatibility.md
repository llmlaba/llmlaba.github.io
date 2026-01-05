---
layout: default
title: "PyTorch dtype compatibility"
date: 2026-01-05
categories: [general]

images:
  - /assets/articles/general/PyTorch_logo.jpeg
---
> Date: {{ page.date | date: "%d.%m.%Y" }}  

# PyTorch dtype compatibility — quick cheat sheet

{% include gallery.html images=page.images gallery_id=page.title %}

The table below shows, **dtype in API** that we can use in `torch_dtype=...` for PyTorch versions

## Floating-point dtype

| PyTorch version | Supported dtype |
|---|---|
| ≥ 1.1 | `torch.float16` (`torch.half`)<br>`torch.float32` (`torch.float`)<br>`torch.float64` (`torch.double`) |
| ≥ 1.6 | `torch.float16` (`torch.half`)<br>`torch.float32` (`torch.float`)<br>`torch.float64` (`torch.double`)<br>`torch.bfloat16` |
| ≥ 2.2 | `torch.float16` (`torch.half`)<br>`torch.float32` (`torch.float`)<br>`torch.float64` (`torch.double`)<br>`torch.bfloat16`<br>`torch.float8_e4m3fn`<br>`torch.float8_e5m2` |
| ≥ 2.9 | `torch.float16` (`torch.half`)<br>`torch.float32` (`torch.float`)<br>`torch.float64` (`torch.double`)<br>`torch.bfloat16`<br>`torch.float8_e4m3fn`<br>`torch.float8_e5m2`<br>`torch.float8_e4m3fnuz`<br>`torch.float8_e5m2fnuz`<br>`torch.float8_e8m0fnu`<br>`torch.float4_e2m1fn_x2` |
