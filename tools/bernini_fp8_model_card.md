---
license: apache-2.0
base_model: Wan-AI/Wan2.2-T2V-A14B
pipeline_tag: text-to-video
library_name: diffusers
tags:
  - comfyui
  - bernini-r
  - wan2.2
  - text-to-video
  - video-editing
  - reference-to-video
  - fp8
---

# Bernini-R fp8 (e4m3) — for ComfyUI-BerniniR

fp8 (`float8_e4m3fn`, **weight-only**) build of **[ByteDance/Bernini-R](https://huggingface.co/ByteDance/Bernini-R)**
(which is Wan2.2-T2V-A14B inside), **self-contained** (2 transformers + VAE + UMT5 + tokenizer +
scheduler), packaged for the **[ComfyUI-BerniniR](https://github.com/neuregex/ComfyUI-BerniniR)**
custom node. Runs the **full pipeline in 24 GB**.

The Linear weights are stored in `float8_e4m3fn` and upcast to bf16 on every forward;
norms / embeddings / time-/text-embedders / patch-embed stay in bf16/fp32. These weights are
**bit-identical** to the node's on-the-fly fp8 quantization — validated end-to-end on an i2i edit:
same seed, same GPU → **0 pixel difference**. Loading this pre-quantized bundle is also faster
(no bf16 → fp8 cast at load).

## VRAM (measured: `torch.cuda.max_memory_allocated`, NVIDIA A10 24 GB, fp8 + sequential offload)

| Task | Frames / resolution | Peak VRAM | Fits 24 GB |
|---|---|---|---|
| i2i / t2i (image edit / image) | 1 frame, 848×848 | **~16.7 GB** | ✅ |
| t2v / v2v / rv2v (video / video edit) | **81 frames** (full length), 480p | **~18.8 GB** | ✅ |

The UMT5 text encoder is freed before the experts load, and offload keeps a single ~14 GB expert
resident — that is what makes full-length 480p video fit in 24 GB.

## Tasks

`t2v` · `t2i` · `i2i` (image edit) · `v2v` (video edit) · `rv2v` (video edit + reference) ·
`r2v` (reference-to-video). Edits preserve the source content/motion via Bernini's source-id RoPE
(validated qualitatively).

## Usage

Install the **[ComfyUI-BerniniR](https://github.com/neuregex/ComfyUI-BerniniR)** node, then in
`BerniniR · Load Model`:

- set **source = `neuregex/Bernini-R-fp8 (auto)`** with `auto_download = True` — downloads ~40 GB to
  `download_dir` on first run (with a free-space check and progress bar), or
- `hf download neuregex/Bernini-R-fp8 --local-dir models/bernini/Bernini-R-fp8` and use `source = local`.

For the full **bf16** weights instead, point the node at `ByteDance/Bernini-R-Diffusers`
(`source = ... (full bf16)`), which needs more VRAM (A100-class) or on-the-fly fp8.

## Credits & license

- Algorithm & model: **Bernini: Latent Semantic Planning for Video Diffusion**, ByteDance
  ([arXiv:2605.22344](https://arxiv.org/abs/2605.22344) · [code](https://github.com/bytedance/Bernini)) — Apache-2.0.
- Base: [Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B).
- fp8 build by **neuregex**. **Apache-2.0**.
