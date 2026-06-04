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
# Halva la VRAM de cada experto (~28GB bf16 -> ~14GB). Experimental.
# --------------------------------------------------------------------------
def quantize_fp8_(module, skip=("norm", "embed", "head", "modulation", "time_", "text_emb")):
    fp8 = getattr(torch, "float8_e4m3fn", None)
    if fp8 is None:
        print("[BerniniR] aviso: torch sin float8_e4m3fn; se omite fp8")
        return module
    n = 0
    for name, lin in module.named_modules():
        if isinstance(lin, torch.nn.Linear) and not any(s in name for s in skip):
            lin.weight.data = lin.weight.data.to(fp8)

            def make_fwd(l):
                def fwd(x):
                    return Fnn.linear(x, l.weight.to(x.dtype), l.bias)
                return fwd
            lin.forward = make_fwd(lin)
            n += 1
    print(f"[BerniniR] fp8: {n} capas lineales cuantizadas en {module.__class__.__name__}")
    return module


# --------------------------------------------------------------------------
# Expertos (transformer / transformer_2)
# --------------------------------------------------------------------------
def load_experts(model_dir, dtype="bf16", fp8=False, device="cpu"):
    """Carga los dos WanTransformer3DModel de un repo Bernini-R-Diffusers."""
    from diffusers import WanTransformer3DModel
    td = _DT[dtype]
    hi = WanTransformer3DModel.from_pretrained(model_dir, subfolder="transformer", torch_dtype=td)
    lo = None
    lo_dir = os.path.join(model_dir, "transformer_2")
    if os.path.isdir(lo_dir):
        lo = WanTransformer3DModel.from_pretrained(model_dir, subfolder="transformer_2", torch_dtype=td)
    if fp8:
        quantize_fp8_(hi)
        if lo is not None:
            quantize_fp8_(lo)
    hi.eval().requires_grad_(False)
    if lo is not None:
        lo.eval().requires_grad_(False)
    # En offload, ambos arrancan en CPU; el sampler sube el activo a GPU.
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
