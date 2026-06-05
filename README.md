# ComfyUI-BerniniR

[![Comfy Registry](https://img.shields.io/badge/Comfy_Registry-comfyui--berninir-1971c2)](https://registry.comfy.org/nodes/comfyui-berninir)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Support for **[ByteDance/Bernini-R](https://huggingface.co/ByteDance/Bernini-R)** in ComfyUI — text-to-video / text-to-image, image and video **editing**, and **reference-to-video**, with Bernini's own logic (source-id RoPE + multi-condition APG guidance) faithfully reimplemented.

> **The key finding:** Bernini-R **is** Wan2.2-T2V-A14B under the hood. Its `transformer/config.json` declares `"_class_name": "WanTransformer3DModel"` with the exact A14B config (40 layers, 40 heads, ffn 13824, 16 channels), an `AutoencoderKLWan` VAE and a UMT5 text encoder. The **weight keys are 100% standard Wan** — there are no extra tensors. Everything that makes Bernini distinctive lives in the **inference code**, not in new parameters. This package reimplements that code on top of the already battle-tested `diffusers` modules.

---

## What exactly is the "Bernini" part (and where this repo reproduces it)?

| Bernini mechanism | What it does | Where, in this repo |
|---|---|---|
| **source-id RoPE** | Every visual "stream" (target, videos, references) carries an integer `source_id`; its RoPE grid is multiplied by a constant complex phase `visual_id_freqs[source_id]`. This distinguishes the same spatial position across streams without offsets or extra channels. | `bernini/rope.py` |
| **Stream concatenation** (not channel concat) | The model stays at 16 channels. Each condition is *patch-embedded* into its own block of tokens and concatenated **before** the target along the sequence axis; a mask keeps only the target's output. | `bernini/model.py` |
| **Dual-expert switch** | high-noise (`transformer`) if `t ≥ 875`, low-noise (`transformer_2`) if `t < 875` — by **timestep value**. On switch, the omegas are scaled ×0.8 once. | `bernini/model.py`, `bernini/sampler.py` |
| **7 guidance modes** | `rv2v` (4 fwd), `v2v` (2), `v2v_chain` (3), `t2v` (2), `r2v_apg` (3), `v2v_apg` (2), `t2v_apg` (2). The `*_apg` modes run **Adaptive Projected Guidance** in x-space. | `bernini/sampler.py`, `bernini/guidance.py` |
| **APG** | Orthogonal/parallel projection of the guidance diff, reduced over `{C,H,W}` **per frame** (not over T), in float64, with momentum persisted across steps. | `bernini/guidance.py` |
| **Scheduler** | UniPCMultistepScheduler with `flow_shift = 3.0` (the CLI's `flow_shift=5.0` is dead code on the default UniPC path!). | `bernini/sampler.py`, `bernini/constants.py` |
| **Text** | UMT5, per-task system-prompt prefix concatenated to the positive prompt, padded to 512. | `bernini/loader.py`, `bernini/constants.py` |

**Validation anchor:** with a single stream and `source_id=0`, `visual_id_freqs[0]=1` (identity phase) ⇒ the forward matches **standard Wan2.2 exactly**. That's why `t2v`/`t2i` is the safest path to verify first.

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/neuregex/ComfyUI-BerniniR
pip install -r ComfyUI-BerniniR/requirements.txt
# (torch is provided by ComfyUI; don't reinstall it)
```

## Weights

The **BerniniR · Load Model** node can fetch the weights for you — no manual download. It exposes:

- **`source`** — which weights to use:
  - **`neuregex/Bernini-R-fp8 (auto)`** *(default)* — [fp8 (e4m3) self-contained bundle](https://huggingface.co/neuregex/Bernini-R-fp8), **~40 GB**, runs the full pipeline in **24 GB**. The fp8 weights are bit-identical to the node's on-the-fly quantization.
  - **`ByteDance/Bernini-R-Diffusers (full bf16)`** — original bf16 weights (~126 GB; A100-class, or use on-the-fly `fp8`).
  - **`local`** — use the `model_dir` path directly.
- **`auto_download`** *(default on)* — if the chosen repo's weights are missing, downloads them (with a free-space check, a `~40 GB first run` notice, and a progress bar). Turn it off to require a manual download.
- **`download_dir`** *(default `models/bernini`)* — where HF repos are downloaded (relative to ComfyUI, or absolute).

Manual download (optional), then set `source = local` and `model_dir` to the folder:

```bash
pip install -U huggingface_hub
hf download neuregex/Bernini-R-fp8 --local-dir models/bernini/Bernini-R-fp8      # fp8, ~40GB, 24GB-ready
# or the full bf16:
hf download ByteDance/Bernini-R-Diffusers --local-dir models/bernini/Bernini-R-Diffusers
```

Each repo is self-contained: VAE + UMT5 + tokenizer + scheduler + both transformers.

---

## Usage (nodes)

Minimal **t2v** pipeline:

```
BerniniR · Load Model ─┐
BerniniR · Load VAE ───┤
BerniniR · Text Encode ┴─► BerniniR · Sampler ─► BerniniR · VAE Decode ─► SaveAnimatedWEBP
```

For editing/reference, add **BerniniR · Encode Source/Reference** (takes `source_video` and/or `reference_images`) and wire it into the sampler's `src` input. The Text Encode `task_type` auto-selects the `guidance_mode` (you can override it).

Example workflows (in **API format**, ready for `/prompt` or Modal) live in [`workflows/`](workflows/):
`bernini_t2v`, `bernini_t2i`, `bernini_i2i`, `bernini_v2v`, `bernini_rv2v`, `bernini_r2v`.

> The video workflows (`v2v`, `rv2v`) load frames with the built-in **BerniniR · Load Video** node (webp/gif via PIL — no extra dependency; put the file in `ComfyUI/input`). For `mp4`/`avi`, use `VHS_LoadVideo` from **ComfyUI-VideoHelperSuite** and wire it into the same `source_video` input.

---

## VRAM and quantization (≤24GB and below)

It's **2×14B** (~56GB in bf16 for the transformers alone). Strategies, from most to least VRAM:

| Config | Peak VRAM | How |
|---|---|---|
| bf16 + sequential offload | A100-class | `dtype=bf16`, `offload_experts=True`. One expert on GPU at a time (each ~28GB bf16). |
| **fp8 + offload, video 480p** | **~18.8 GB** (81 frames, measured on A10 24GB) | `fp8=True`. Each expert ~14GB; the inactive one is moved to CPU. **This is the 24GB target.** |
| **fp8 + offload, t2i/i2i** (1 frame) | **~16.7 GB** (measured) | Images: the sequence is much shorter. Comfortable on 24GB, fits on 16GB. |
| GGUF Q4/Q5 (advanced) | ~8–12 GB | Only the `t2v`/`t2i` path via native ComfyUI, see below. |

Numbers above are real peaks (`torch.cuda.max_memory_allocated`) measured end-to-end on an **NVIDIA A10 (24GB)**: full pipeline (UMT5 → sampling over both experts → VAE decode). The binding peak is the sampling stage; the text encoder is freed before the experts are loaded, and offload keeps a single expert resident — that combination is what makes **full-length 480p video (81 frames) fit in 24GB**.

Notes:
- `fp8` here stores the linear weights in `float8_e4m3fn` and upcasts to bf16 on every forward (slower, half the VRAM). It's the main lever for 24GB and, in practice, does not visibly degrade quality (an anchored i2i edit is ~41 dB PSNR vs bf16; free-running t2v yields a different but equally sharp sample).
- `offload_experts=True` mirrors Bernini's behavior (moves the high-noise expert to CPU when switching to the low-noise one).
- Lower `num_frames` and resolution to cut activation memory (attention grows with sequence length, and here the sequence includes the condition tokens).

### GGUF / native-ComfyUI path (sub-16GB, t2v/t2i only for now)

Because `t2v`/`t2i` with `source_id=0` is **identical to standard Wan2.2**, you can run it with ComfyUI's native nodes (which already support fp8 and GGUF) after converting the weights to native keys:

```bash
python tools/convert_bernini_to_comfy.py --repo Bernini-R-Diffusers --out-dir comfy_out --dtype fp8_e4m3fn
# -> comfy_out/bernini_r_high_noise_14B_fp8_e4m3fn.safetensors  (+ low_noise)
# copy to ComfyUI/models/diffusion_models/ and load with two "Load Diffusion Model" nodes
```

For GGUF, pass the native `.safetensors` through [`city96/ComfyUI-GGUF`](https://github.com/city96/ComfyUI-GGUF) `tools/convert.py` (it only accepts the native format: that's why we convert first) and load them with `UnetLoaderGGUF`. The converter's key mapping is verified against 🤗 diffusers' official `convert_wan_to_diffusers.py` (20/20 cases in the unit test). *Editing/reference tasks require this package's backend (fp8+offload), not the GGUF path.*

---

## Validate on Modal (serverless GPU)

The weights are ~160GB and the 28B model wants an A100-80GB-class GPU for the full bf16 path, so the end-to-end run is validated in the cloud. Harness included in [`modal/app.py`](modal/app.py):

```bash
pip install modal && modal token new          # once

# 1) download the weights to a Modal Volume (~126GB, once)
modal run modal/app.py::download_weights

# 2) cheap numeric check: load BOTH real experts (bf16) and run 1 multi-stream
#    forward (shapes + NaN). Add --fp8 to also exercise the quantized path.
modal run modal/app.py::smoke

# 3) run a full workflow headless and save the result to the Volume.
#    Overrides (to make the first run cheap): --num-frames, --steps, --width,
#    --height, --fp8, --gen-input <name.png> (generates a test image for i2i).
modal run modal/app.py::run --workflow workflows/bernini_t2v.json --num-frames 25 --steps 20 --fp8

# 4) the 24GB target: pick the GPU via env var, then run i2i/t2v with fp8 + offload.
#    nodes.py reports peak VRAM per stage.
#    PowerShell:  $env:BERNINIR_GPU="A10G"; modal run modal/app.py::run --workflow workflows/bernini_i2i.json --steps 20 --gen-input input.png --fp8
#    bash:        BERNINIR_GPU=A10G modal run modal/app.py::run --workflow workflows/bernini_i2i.json --steps 20 --gen-input input.png --fp8

# 5) download results
modal volume get berninir-out / ./outputs
```

The harness builds a CUDA 12.4 + torch 2.5.1 image (Bernini's stack), clones ComfyUI, mounts this custom node, starts the headless server and posts the workflow to the `/prompt` endpoint. `gpu` and CPU `memory` are parameterizable via `BERNINIR_GPU` / `BERNINIR_CPU_MEM` (read when the app is defined).

---

## Validation status

This port is built against Bernini's **verbatim source** (github.com/bytedance/Bernini) and the **real code** of `diffusers==0.35.2` and current ComfyUI. Verified so far:

**Locally (CPU, tiny random-weight models — `tests/`):**
- End-to-end wiring of `t2v` / `rv2v` / `r2v_apg` runs without exceptions, returns `[1,16,T,H,W]`, no NaNs.
- **Fidelity anchor (C1):** with a single `source_id=0` target stream, `BerniniExpert.forward_streams` matches the **stock diffusers `WanTransformer3DModel.forward`** to `torch.allclose` (max abs diff ~6e-7) — confirming the complex source-id RoPE is numerically equivalent to diffusers' cos/sin RoPE at the identity phase.

**On Modal (real weights, A100-80GB and A10):**
- **Smoke (bf16):** both real experts load and the multi-stream forward (2 steps straddling the t=875 boundary, exercising both experts) runs with no NaNs.
- **t2v / i2i render** produce coherent output. For **i2i**, the result preserves the source image's content/composition **and** applies the prompt edit — i.e., the **cross-stream source-id mechanism works** (the target stream attends to the reference stream under distinct id phases).
- **fp8 quantization (e4m3 weight-only + upcast) does not visibly degrade quality** and is faster than bf16; the monkeypatch quantizes 401 linear layers per expert and runs cleanly.
- **24GB target met:** `t2v` at **81 frames / 480p / fp8 + offload** fits on an NVIDIA A10 (24GB) with **no OOM** — peak 18.8 GB allocated / 20.6 GB reserved; `i2i` peaks at ~16.7 GB.

**Not yet done:** a numerical comparison against ByteDance's official reference outputs (no public ground-truth tensors), so this is "faithful, stable, produces correct-looking results" rather than "bit-exact validated against the original."

Design note worth knowing: diffusers 0.35.2 applies RoPE in **real cos/sin** form, but Bernini uses it **complex** (required for the source-id phase); that's why we replace each block's self-attention *processor* with a complex one (`bernini/model.py::_BerniniSelfAttnProcessor`), leaving cross-attn / FFN / modulation stock. Loading the experts in bf16 also keeps several modules in fp32, so the I/O dtype is taken from the `patch_embedding` weight rather than from `next(parameters())`.

---

## Credits and license

- Model and algorithm: **Bernini: Latent Semantic Planning for Video Diffusion**, ByteDance ([arXiv:2605.22344](https://arxiv.org/abs/2605.22344), [code](https://github.com/bytedance/Bernini)). Apache-2.0.
- Base: [Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B).
- Key mapping derived from 🤗 diffusers' `convert_wan_to_diffusers.py`.

This package: **Apache-2.0** ([LICENSE](LICENSE)). Does not include weights.
