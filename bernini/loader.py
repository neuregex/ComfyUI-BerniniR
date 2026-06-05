# Copyright (c) 2026 — ComfyUI-BerniniR. Apache-2.0.
"""Carga de pesos de Bernini-R-Diffusers + cuantización fp8 ligera + UMT5."""
import html
import os
import re

import torch
import torch.nn.functional as Fnn

_DT = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


# --------------------------------------------------------------------------
# fp8 ligero: pesos en float8_e4m3fn (almacenamiento), upcast en cada forward.
# Halva la VRAM de cada experto (~28GB bf16 -> ~14GB).
#
# Separado en dos pasos para soportar pesos PRE-cuantizados en disco:
#   cast_fp8_(m)          -> castea los Linear (salvo skip) a e4m3. Solo pesos.
#   apply_fp8_forward_(m) -> parchea el forward de los Linear que YA están en fp8
#                            (upcast a x.dtype en cada forward). NO re-castea.
# quantize_fp8_ = cast + apply (cuantización on-the-fly del bf16).
# Skip-list: norm/embed/head/modulation/time_/text_emb (y patch_embedding NO está
# en la lista pero su nombre tampoco matchea, así que queda en bf16 por defecto...
# OJO: patch_embedding es Conv3d, no Linear, así que nunca se castea aquí).
# --------------------------------------------------------------------------
FP8_SKIP = ("norm", "embed", "head", "modulation", "time_", "text_emb")


def _fp8_dtype():
    return getattr(torch, "float8_e4m3fn", None)


def cast_fp8_(module, skip=FP8_SKIP):
    """Castea a float8_e4m3fn los pesos de los `nn.Linear` cuyo nombre NO matchea
    `skip`. NO toca el forward ni los bias. Devuelve el módulo (in-place)."""
    fp8 = _fp8_dtype()
    if fp8 is None:
        print("[BerniniR] aviso: torch sin float8_e4m3fn; se omite cast fp8")
        return module
    n = 0
    for name, lin in module.named_modules():
        if isinstance(lin, torch.nn.Linear) and not any(s in name for s in skip):
            lin.weight.data = lin.weight.data.to(fp8)
            n += 1
    print(f"[BerniniR] fp8 cast: {n} Linear -> e4m3 en {module.__class__.__name__}")
    return module


def apply_fp8_forward_(module):
    """Parchea el forward de los `nn.Linear` que YA están en fp8 (upcast del peso a
    x.dtype en cada forward). Sirve tras `cast_fp8_` o tras cargar un checkpoint
    pre-fp8; NO re-castea (preserva el fp8 del disco)."""
    fp8 = _fp8_dtype()
    if fp8 is None:
        return module
    n = 0
    for lin in module.modules():
        if isinstance(lin, torch.nn.Linear) and lin.weight.dtype == fp8:
            def make_fwd(l):
                def fwd(x):
                    return Fnn.linear(x, l.weight.to(x.dtype), l.bias)
                return fwd
            lin.forward = make_fwd(lin)
            n += 1
    print(f"[BerniniR] fp8 forward patch: {n} Linear")
    return module


def quantize_fp8_(module, skip=FP8_SKIP):
    """On-the-fly: castea a fp8 y parchea el forward (cast_fp8_ + apply_fp8_forward_)."""
    cast_fp8_(module, skip=skip)
    return apply_fp8_forward_(module)


def _is_fp8_repo(model_dir, subfolder):
    """True si los safetensors de model_dir/subfolder tienen algún tensor en e4m3
    (es decir, es un bundle PRE-cuantizado tipo Bernini-R-fp8)."""
    import glob
    shards = sorted(glob.glob(os.path.join(model_dir, subfolder, "*.safetensors")))
    if not shards or _fp8_dtype() is None:
        return False
    try:
        from safetensors import safe_open
        with safe_open(shards[0], framework="pt") as f:
            for k in f.keys():
                if "F8_E4M3" in f.get_slice(k).get_dtype():
                    return True
    except Exception:
        return False
    return False


def _load_prefp8(model_dir, subfolder):
    """Carga un WanTransformer3DModel PRE-fp8 preservando el dtype e4m3 del disco:
    init_empty_weights (sin tocar buffers no persistentes como el rope) +
    load_state_dict(assign=True) -> los params toman el dtype EXACTO del checkpoint;
    luego apply_fp8_forward_ parchea solo los Linear fp8."""
    import glob
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from diffusers import WanTransformer3DModel

    config = WanTransformer3DModel.load_config(model_dir, subfolder=subfolder)
    with init_empty_weights(include_buffers=False):
        m = WanTransformer3DModel.from_config(config)
    sd = {}
    for shard in sorted(glob.glob(os.path.join(model_dir, subfolder, "*.safetensors"))):
        sd.update(load_file(shard))
    missing, unexpected = m.load_state_dict(sd, strict=False, assign=True)
    if missing:
        print(f"[BerniniR] fp8 load {subfolder}: {len(missing)} missing (ej: {missing[:2]})")
    if unexpected:
        print(f"[BerniniR] fp8 load {subfolder}: {len(unexpected)} unexpected (ej: {unexpected[:2]})")
    apply_fp8_forward_(m)
    print(f"[BerniniR] {subfolder}: cargado PRE-fp8 (e4m3 del disco preservado)")
    return m


# --------------------------------------------------------------------------
# Expertos (transformer / transformer_2)
# --------------------------------------------------------------------------
def load_experts(model_dir, dtype="bf16", fp8=False, device="cpu"):
    """Carga los dos WanTransformer3DModel de un repo Bernini-R(-fp8).

    Detecta automáticamente si el repo ya está en fp8 (e4m3 en disco): en ese caso
    PRESERVA el fp8 (no re-castea). Si no, camino bf16 y, con fp8=True, cuantiza
    on-the-fly. Con `device` != "cpu" mueve cada experto al cargarse (pico de RAM de
    CPU ~1 experto). dtype solo aplica al camino bf16.
    """
    from diffusers import WanTransformer3DModel
    td = _DT[dtype]

    def _load(subfolder):
        if _is_fp8_repo(model_dir, subfolder):
            m = _load_prefp8(model_dir, subfolder)          # preserva e4m3
        else:
            m = WanTransformer3DModel.from_pretrained(model_dir, subfolder=subfolder, torch_dtype=td)
            if fp8:
                quantize_fp8_(m)                            # cuantiza on-the-fly
        m.eval().requires_grad_(False)
        if str(device) != "cpu":
            m.to(device)
        return m

    hi = _load("transformer")
    lo = _load("transformer_2") if os.path.isdir(os.path.join(model_dir, "transformer_2")) else None
    # En offload (device="cpu"), ambos arrancan en CPU; el sampler sube el activo a GPU.
    return hi, lo


# --------------------------------------------------------------------------
# VAE (AutoencoderKLWan)
# --------------------------------------------------------------------------
def load_vae(model_dir, dtype="fp16", device="cpu"):
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(model_dir, subfolder="vae", torch_dtype=_DT.get(dtype, torch.float16))
    vae.eval().requires_grad_(False)
    return vae.to(device)


# --------------------------------------------------------------------------
# Text encoder (UMT5) + tokenizer
# --------------------------------------------------------------------------
def _clean(text: str) -> str:
    try:
        import ftfy
        text = ftfy.fix_text(text)
    except Exception:
        pass
    text = html.unescape(html.unescape(text))
    return re.sub(r"\s+", " ", text).strip()


class TextEncoder:
    def __init__(self, model_dir, dtype="bf16", device="cpu"):
        from transformers import UMT5EncoderModel, AutoTokenizer
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
        self.te = UMT5EncoderModel.from_pretrained(
            model_dir, subfolder="text_encoder", torch_dtype=_DT[dtype]).eval().requires_grad_(False).to(device)
        self.max_len = 512

    @torch.no_grad()
    def encode(self, prompt: str, system_prefix: str = "") -> torch.Tensor:
        text = (system_prefix or "") + _clean(prompt)
        ids = self.tok(text, padding="max_length", max_length=self.max_len, truncation=True,
                       add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
        mask = ids.attention_mask.to(self.device)
        emb = self.te(ids.input_ids.to(self.device), mask).last_hidden_state[0]   # [512, 4096]
        seq_len = int(mask.gt(0).sum())
        emb = emb[: min(seq_len, self.max_len)]
        pad = emb.new_zeros(self.max_len - emb.size(0), emb.size(1))
        return torch.cat([emb, pad], dim=0).unsqueeze(0)   # [1, 512, 4096]
