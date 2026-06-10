# ComfyUI-BerniniR

[![Comfy Registry](https://img.shields.io/badge/Comfy_Registry-comfyui--berninir-1971c2)](https://registry.comfy.org/nodes/comfyui-berninir)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/badge/Discord-join_the_community-5865F2?logo=discord&logoColor=white)](https://discord.gg/HxfP9TnctJ)
[![GGUF on HF](https://img.shields.io/badge/🤗_GGUF-neuregex%2FBernini--R--GGUF-ffce1c)](https://huggingface.co/neuregex/Bernini-R-GGUF)

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

## Models — download what fits your VRAM

Two ready-to-use repos, each bundling everything ComfyUI needs (renderer + VAE + UMT5):

| Your VRAM | Use this | Repo |
|---|---|---|
| **24 GB** | **14B dual-expert** — full quality, `Q4`–`Q8` GGUF | **[neuregex/Bernini-R-GGUF](https://huggingface.co/neuregex/Bernini-R-GGUF)** |
| **4–12 GB** | **1.3B** single-expert — tiny & fast, `bf16` + `Q4`–`Q8` GGUF, self-contained | **[neuregex/Bernini-1.3B-ComfyUI](https://huggingface.co/neuregex/Bernini-1.3B-ComfyUI)** |

Install [`city96/ComfyUI-GGUF`](https://github.com/city96/ComfyUI-GGUF) for the `.gguf` files. Ready-made graphs in [`workflows/ui/`](workflows/ui/): `bernini_i2i_gguf_dual` / `bernini_v2v_gguf_dual` / `bernini_rv2v_gguf_dual` (14B), `bernini_i2i_1.3B` (1.3B).

> **Wiring:** `UnetLoaderGGUF` → **BerniniR · Apply Patches** → **BerniniR · Source Stream** → **BerniniR · Guider**. For the **14B**, load both experts and set `model` = high-noise, `model_low` = low-noise; the **1.3B** is single-expert, so leave `model_low` empty. GGUF carries no fp8 tensors, so both 14B experts coexist in 24 GB with no offload crash.

## Weights (alternative: diffusers backend, auto-download)

The **BerniniR · Load Model** node can fetch the weights for you — no manual download. It exposes:

- **`source`** — which weights to use:
  - **`neuregex/Bernini-R-fp8 (auto)`** *(default)* — [fp8 (e4m3) self-contained bundle](https://huggingface.co/neuregex/Bernini-R-fp8), **~40 GB**, runs the full pipeline in **24 GB**. The fp8 weights are bit-identical to the node's on-the-fly quantization.
  - **`ByteDance/Bernini-R-Diffusers (full bf16)`** — original bf16 weights (~126 GB; A100-class, or use on-the-fly `fp8`).
  - **`local`** — use the `model_dir` path directly.
- **`auto_download`** *(default on)* — if the chosen repo's weights are missing, downloads them. The first run pulls **~40 GB** (fp8 bundle), so it takes a while; the console shows **real byte progress** (`12.3%  4.95/40.30GB  47.2MB/s  ETA 12.6min`), not just a file count. The download is **resumable** — if you cancel (Ctrl-C / Cancel), the next Run picks up where it left off. A free-space check runs first. Turn it off to require a manual download.
- **`download_dir`** *(default `models/bernini`)* — where HF repos are downloaded (relative to ComfyUI, or absolute).

> **The download stalled at "Fetching N files: 5%"?** That old file-count bar (≤0.3.3) looked frozen while a 14 GB shard streamed. 0.3.4+ shows true byte progress instead. The node also disables HF's **xet** transport by default (`HF_HUB_DISABLE_XET=1`), which can hang on Windows portable; it uses plain HTTPS (LFS), which is slower but rock-solid. To re-enable xet, set `HF_HUB_DISABLE_XET=0` before launching ComfyUI.

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

Example workflows come in two formats:

- **UI format** (drag-and-drop onto the ComfyUI canvas) — [`workflows/ui/`](workflows/ui/): `bernini_t2v.json`, `bernini_i2i.json`. Use these in the app: just drop the file on the canvas (or *Workflow → Open*). Validated to load and run in a real ComfyUI (the `BR_PATH` Load Model → VAE/Text Encode wiring is drawn for you).
- **API format** (for the `/prompt` endpoint or *Workflow → Open (API)*) — [`workflows/`](workflows/): `bernini_t2v`, `bernini_t2i`, `bernini_i2i`, `bernini_v2v`, `bernini_rv2v`, `bernini_r2v`.

> The video workflows (`v2v`, `rv2v`) load frames with the built-in **BerniniR · Load Video** node (webp/gif via PIL — no extra dependency; put the file in `ComfyUI/input`). For `mp4`/`avi`, use `VHS_LoadVideo` from **ComfyUI-VideoHelperSuite** and wire it into the same `source_video` input.

---

## VRAM and quantization (≤24GB and below)

It's **2×14B** (~56GB in bf16 for the transformers alone). Strategies, from most to least VRAM:

| Config | Peak VRAM | How |
|---|---|---|
| bf16 + sequential offload | A100-class | `dtype=bf16`, `offload_experts=True`. One expert on GPU at a time (each ~28GB bf16). |
| **fp8 + offload, video 480p** | **~18.8 GB** (81 frames, measured on A10 24GB) | `fp8=True`. Each expert ~14GB; the inactive one is moved to CPU. **This is the 24GB target.** |
| **fp8 + offload, t2i/i2i** (1 frame) | **~16.7 GB** (measured) | Images: the sequence is much shorter. Comfortable on 24GB, fits on 16GB. |
| **fp8 + offload + block-swap** | **down to ~9.4 GB** (video 81f) / **~5 GB** (i2i) sampling | `blocks_to_swap=20…40`. Streams the last N of the 40 transformer blocks CPU↔GPU per block. See below. |
| GGUF Q4/Q5 (advanced) | ~8–12 GB | Only the `t2v`/`t2i` path via native ComfyUI, see below. |

Numbers above are real peaks (`torch.cuda.max_memory_allocated`) measured end-to-end on an **NVIDIA A10 (24GB)**: full pipeline (UMT5 → sampling over both experts → VAE decode). The binding peak is the sampling stage; the text encoder is freed before the experts are loaded, and offload keeps a single expert resident — that combination is what makes **full-length 480p video (81 frames) fit in 24GB**.

Notes:
- `fp8` here stores the linear weights in `float8_e4m3fn` and upcasts to bf16 on every forward (slower, half the VRAM). It's the main lever for 24GB and, in practice, does not visibly degrade quality (an anchored i2i edit is ~41 dB PSNR vs bf16; free-running t2v yields a different but equally sharp sample).
- `offload_experts=True` mirrors Bernini's behavior (moves the high-noise expert to CPU when switching to the low-noise one).
- Lower `num_frames` and resolution to cut activation memory (attention grows with sequence length, and here the sequence includes the condition tokens).

### Block-swap (`blocks_to_swap`, lower VRAM keeping editing)

Each expert is 40 `WanTransformerBlock`s. `blocks_to_swap=N` keeps the **last N** blocks on CPU and streams them to the GPU one at a time during their own forward (the rest stay resident). It cuts the **weight** footprint of the active expert — orthogonal to `fp8` and `offload_experts`, and it works for **every** task (editing included), unlike the GGUF path. It does **not** reduce activation memory (driven by resolution/frames), so combine it with lower resolution if you're activation-bound.

It does not change the math: moving weights CPU↔GPU is numerically a no-op. Verified — the full i2i pipeline with `blocks_to_swap=0` vs `30` produces a **bit-identical** image on the same GPU (same seed, real fp8 weights).

Measured sampling peak (fp8, NVIDIA A10), with the speed cost:

| Task | `blocks_to_swap=0` | `=20` | `=30` | speed (0 → 30) |
|---|---|---|---|---|
| **i2i** 848² | 14.84 GB | 8.45 GB | **5.08 GB** | +14% |
| **t2v** 81f / 480p | 19.18 GB | 12.78 GB | **9.42 GB** | +4% |

With block-swap the experts stop being the bottleneck; the binding peak becomes the **UMT5 text encoder (~10.8 GB, transient — freed before sampling)**. On a 16 GB NVIDIA T4, `i2i` at 640² **OOMs with `blocks_to_swap=0`** (the 14 GB expert won't fit) but runs at **6.28 GB sampling with `blocks_to_swap=40`** — i.e. the pipeline fits in **~12 GB**. (The T4 is pre-Ampere with no flash-attention, so it's slow and activation-heavy at high resolution; an Ampere 12 GB card such as an RTX 3060 fares much better.)

Rule of thumb: `0` if you have ≥24 GB, `20` for ~16 GB, `30–40` for 12 GB. Higher N = lower VRAM, more CPU↔GPU traffic.

### GGUF / native-ComfyUI path (sub-16GB)

> **Pre-quantized GGUF — no conversion needed:** both experts (high & low noise) are hosted at **[neuregex/Bernini-R-GGUF](https://huggingface.co/neuregex/Bernini-R-GGUF)** in `Q4_K_M`, `Q5_K_M` and `Q8_0`. Drop them in `ComfyUI/models/unet/` and load with `UnetLoaderGGUF` (from [`city96/ComfyUI-GGUF`](https://github.com/city96/ComfyUI-GGUF)).

- **t2v / t2i** — `source_id=0` is **identical to standard Wan2.2**, so the native GGUF path just works: one `UnetLoaderGGUF` → sampler.
- **Editing — i2i / v2v / rv2v, dual-expert** — load each GGUF with `UnetLoaderGGUF` → **BerniniR · Apply Patches** → **BerniniR · Source Stream** → **BerniniR · Guider** (`model` = high-noise, `model_low` = low-noise). The guider switches expert by timestep (t=875) and runs Bernini's APG guidance. GGUF has no fp8 tensors, so both experts coexist in 24 GB with no offload crash. Graphs: [`bernini_i2i_gguf_dual`](workflows/ui/bernini_i2i_gguf_dual.json) · [`bernini_v2v_gguf_dual`](workflows/ui/bernini_v2v_gguf_dual.json) · [`bernini_rv2v_gguf_dual`](workflows/ui/bernini_rv2v_gguf_dual.json).
- **Even less VRAM → the 1.3B** — the single-expert [neuregex/Bernini-1.3B-ComfyUI](https://huggingface.co/neuregex/Bernini-1.3B-ComfyUI) bundle (fine-tuned from Wan2.1-1.3B, ~2.6 GB) runs on 4–12 GB cards. Same nodes, leave `model_low` empty; graph [`bernini_i2i_1.3B`](workflows/ui/bernini_i2i_1.3B.json).

Prefer to quantize yourself? Convert to native keys, then GGUF:

```bash
python tools/convert_bernini_to_comfy.py --repo Bernini-R-Diffusers --out-dir comfy_out --dtype bf16
# then city96/ComfyUI-GGUF tools/convert.py + llama-quantize
# (Wan needs the patched llama.cpp + a fix_5d_tensors.py pass for the 5D patch_embedding)
```

The converter's key mapping is verified against 🤗 diffusers' official `convert_wan_to_diffusers.py` (20/20 cases in the unit test).

---

## Validation status

This port is built against Bernini's **verbatim source** (github.com/bytedance/Bernini) and the **real code** of `diffusers==0.35.2` and current ComfyUI. Verified so far:

**Locally (CPU, tiny random-weight models — `tests/`):**
- End-to-end wiring of `t2v` / `rv2v` / `r2v_apg` runs without exceptions, returns `[1,16,T,H,W]`, no NaNs.
- **Fidelity anchor (C1):** with a single `source_id=0` target stream, `BerniniExpert.forward_streams` matches the **stock diffusers `WanTransformer3DModel.forward`** to `torch.allclose` (max abs diff ~6e-7) — confirming the complex source-id RoPE is numerically equivalent to diffusers' cos/sin RoPE at the identity phase.

**With the real weights (A100-80GB / A10):**
- **Smoke (bf16):** both real experts load and the multi-stream forward (2 steps straddling the t=875 boundary, exercising both experts) runs with no NaNs.
- **t2v / i2i render** produce coherent output. For **i2i**, the result preserves the source image's content/composition **and** applies the prompt edit — i.e., the **cross-stream source-id mechanism works** (the target stream attends to the reference stream under distinct id phases).
- **fp8 quantization (e4m3 weight-only + upcast) does not visibly degrade quality** and is faster than bf16; the monkeypatch quantizes 401 linear layers per expert and runs cleanly.
- **24GB target met:** `t2v` at **81 frames / 480p / fp8 + offload** fits on an NVIDIA A10 (24GB) with **no OOM** — peak 18.8 GB allocated / 20.6 GB reserved; `i2i` peaks at ~16.7 GB.

**Not yet done:** a numerical comparison against ByteDance's official reference outputs (no public ground-truth tensors), so this is "faithful, stable, produces correct-looking results" rather than "bit-exact validated against the original."

Design note worth knowing: diffusers 0.35.2 applies RoPE in **real cos/sin** form, but Bernini uses it **complex** (required for the source-id phase); that's why we replace each block's self-attention *processor* with a complex one (`bernini/model.py::_BerniniSelfAttnProcessor`), leaving cross-attn / FFN / modulation stock. Loading the experts in bf16 also keeps several modules in fp32, so the I/O dtype is taken from the `patch_embedding` weight rather than from `next(parameters())`.

---

## Community

Join the Discord for updates, roadmaps, projects, or just to chat: **[discord.gg/HxfP9TnctJ](https://discord.gg/HxfP9TnctJ)**

---

## Credits and license

- Model and algorithm: **Bernini: Latent Semantic Planning for Video Diffusion**, ByteDance ([arXiv:2605.22344](https://arxiv.org/abs/2605.22344), [code](https://github.com/bytedance/Bernini)). Apache-2.0.
- Base: [Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B).
- Key mapping derived from 🤗 diffusers' `convert_wan_to_diffusers.py`.

This package: **Apache-2.0** ([LICENSE](LICENSE)). Does not include weights.
