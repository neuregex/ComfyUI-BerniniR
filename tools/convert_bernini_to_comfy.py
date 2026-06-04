#!/usr/bin/env python3
# Copyright (c) 2026 — ComfyUI-BerniniR
# Licensed under the Apache License, Version 2.0.
"""
Convierte los transformers de **Bernini-R** (formato *diffusers*,
`WanTransformer3DModel`) al formato de claves **nativo de ComfyUI/Wan**
("reference"), produciendo un único `.safetensors` por experto, listo para
`UNETLoader` (o, tras `gguf` convert, para `UnetLoaderGGUF`).

¿Por qué hace falta?  ComfyUI detecta y carga Wan **por nombres de tensor
nativos** (`patch_embedding.weight`, `head.modulation`, `blocks.0.ffn.0.weight`,
`blocks.0.self_attn.q.weight`, ...). Los checkpoints de Bernini-R vienen en
formato *diffusers* (`blocks.0.attn1.to_q.weight`, `condition_embedder.*`,
`scale_shift_table`, ...). Este script aplica el renombrado inverso al del
`convert_wan_to_diffusers.py` de 🤗 diffusers — es la fuente de verdad del
mapeo, no una conjetura.

NO es necesario para el camino por defecto del paquete (que carga los pesos
*diffusers* directamente). Úsalo solo para el camino **GGUF / ComfyUI-nativo**
pensado para <16 GB de VRAM.

Uso
----
    # Un experto (carpeta diffusers con *.safetensors + index.json):
    python convert_bernini_to_comfy.py \
        --in  Bernini-R-Diffusers/transformer \
        --out bernini_r_high_noise_14B.safetensors --dtype bf16

    python convert_bernini_to_comfy.py \
        --in  Bernini-R-Diffusers/transformer_2 \
        --out bernini_r_low_noise_14B.safetensors --dtype fp8_e4m3fn

    # Atajo: convierte ambos expertos de un repo Bernini-R-Diffusers de una vez
    python convert_bernini_to_comfy.py --repo Bernini-R-Diffusers --out-dir comfy_out --dtype bf16

Después, copia los .safetensors a  ComfyUI/models/diffusion_models/  y cárgalos
con dos nodos "Load Diffusion Model" (UNETLoader). Para GGUF, pásalos por
`city96/ComfyUI-GGUF/tools/convert.py` (que SOLO acepta formato nativo: por eso
hay que convertir antes).
"""
import argparse
import glob
import json
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

# --------------------------------------------------------------------------
# Mapeo nativo(ComfyUI) <- diffusers, derivado de
# huggingface/diffusers/scripts/convert_wan_to_diffusers.py
# (TRANSFORMER_KEYS_RENAME_DICT) **invertido**.
# Lo aplicamos como reemplazos de subcadena sobre cada clave diffusers.
# --------------------------------------------------------------------------

# Reemplazos simples diffusers -> nativo (orden importa poco; son disjuntos).
_SUBSTR = [
    # condition_embedder -> time/text embeddings + time projection
    ("condition_embedder.time_embedder.linear_1", "time_embedding.0"),
    ("condition_embedder.time_embedder.linear_2", "time_embedding.2"),
    ("condition_embedder.text_embedder.linear_1", "text_embedding.0"),
    ("condition_embedder.text_embedder.linear_2", "text_embedding.2"),
    ("condition_embedder.time_proj", "time_projection.1"),
    # I2V (no presente en Bernini-R T2V, pero lo soportamos por completitud)
    ("condition_embedder.image_embedder.norm1", "img_emb.proj.0"),
    ("condition_embedder.image_embedder.ff.net.0.proj", "img_emb.proj.1"),
    ("condition_embedder.image_embedder.ff.net.2", "img_emb.proj.3"),
    ("condition_embedder.image_embedder.norm2", "img_emb.proj.4"),
    # FFN
    ("ffn.net.0.proj", "ffn.0"),
    ("ffn.net.2", "ffn.2"),
    # Self-attention (attn1 -> self_attn)
    ("attn1.to_q", "self_attn.q"),
    ("attn1.to_k", "self_attn.k"),
    ("attn1.to_v", "self_attn.v"),
    ("attn1.to_out.0", "self_attn.o"),
    ("attn1.norm_q", "self_attn.norm_q"),
    ("attn1.norm_k", "self_attn.norm_k"),
    # Cross-attention (attn2 -> cross_attn)
    ("attn2.to_q", "cross_attn.q"),
    ("attn2.to_k_img", "cross_attn.k_img"),
    ("attn2.to_v_img", "cross_attn.v_img"),
    ("attn2.norm_k_img", "cross_attn.norm_k_img"),
    ("attn2.to_k", "cross_attn.k"),
    ("attn2.to_v", "cross_attn.v"),
    ("attn2.to_out.0", "cross_attn.o"),
    ("attn2.norm_q", "cross_attn.norm_q"),
    ("attn2.norm_k", "cross_attn.norm_k"),
]


def _rename_key(key: str) -> str:
    """diffusers -> nativo para UNA clave de tensor."""
    k = key

    # 1) Cabecera (top-level) — deben ir antes que el swap de norm y son exactos.
    if k == "scale_shift_table":
        return "head.modulation"
    if k == "proj_out.weight":
        return "head.head.weight"
    if k == "proj_out.bias":
        return "head.head.bias"

    # 2) modulation por-bloque:  blocks.N.scale_shift_table -> blocks.N.modulation
    k = re.sub(r"^(blocks\.\d+\.)scale_shift_table$", r"\1modulation", k)

    # 3) Reemplazos de subcadena (attn, ffn, embeddings).
    for src, dst in _SUBSTR:
        if src in k:
            k = k.replace(src, dst)

    # 4) Swap de LayerNorms:  diffusers norm2 <-> norm3 (ver comentario del
    #    convert oficial: el orden original es norm1, norm3, norm2).
    #    Solo afecta a las norms de bloque (no a norm_q/norm_k ni norm_out).
    k = re.sub(r"(\.)norm2(\.|$)", r"\1__NORM_TMP__\2", k)
    k = re.sub(r"(\.)norm3(\.|$)", r"\1norm2\2", k)
    k = re.sub(r"(\.)__NORM_TMP__(\.|$)", r"\1norm3\2", k)

    return k


_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp8_e4m3fn": getattr(torch, "float8_e4m3fn", None),
    "fp8_e5m2": getattr(torch, "float8_e5m2", None),
}

# Capas que NO conviene cuantizar a fp8 (sensibles: normalización, modulación,
# embeddings y la cabeza). Se mantienen en bf16 aunque --dtype sea fp8.
_FP8_SKIP = (
    "norm", "modulation", "time_embedding", "text_embedding",
    "time_projection", "patch_embedding", "head.", "img_emb",
)


def _load_diffusers_dir(path: str) -> dict:
    """Carga y fusiona todos los *.safetensors de una carpeta diffusers."""
    shards = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not shards:
        sys.exit(f"[error] no se encontraron *.safetensors en {path}")
    sd = {}
    for shard in shards:
        part = load_file(shard)
        sd.update(part)
    return sd


def _cast(t: torch.Tensor, key: str, dtype_name: str) -> torch.Tensor:
    if not t.is_floating_point():
        return t
    dt = _DTYPES[dtype_name]
    if dt is None:
        sys.exit(f"[error] tu PyTorch no soporta {dtype_name}; actualiza torch>=2.1")
    if dtype_name.startswith("fp8"):
        # Solo cuantiza pesos 2D de capas lineales (no sesgos, no capas skip).
        if t.ndim < 2 or any(s in key for s in _FP8_SKIP):
            return t.to(torch.bfloat16)
        return t.to(dt)
    return t.to(dt)


def convert_one(in_dir: str, out_path: str, dtype: str) -> None:
    print(f"[*] cargando diffusers desde {in_dir} ...")
    sd = _load_diffusers_dir(in_dir)
    print(f"    {len(sd)} tensores")

    out = {}
    seen = set()
    for k, v in sd.items():
        nk = _rename_key(k)
        if nk in seen:
            sys.exit(f"[error] colisión de claves tras renombrar: {nk} (origen {k})")
        seen.add(nk)
        out[nk] = _cast(v.contiguous(), nk, dtype)

    # Sanity-check: claves que ComfyUI usa para detectar Wan.
    needed = ["patch_embedding.weight", "head.modulation", "head.head.weight",
              "blocks.0.ffn.0.weight", "blocks.0.self_attn.q.weight"]
    missing = [n for n in needed if n not in out]
    if missing:
        print(f"[aviso] faltan claves esperadas por ComfyUI: {missing}\n"
              f"        (¿la carpeta de entrada es realmente un transformer Wan diffusers?)")

    n_layers = len({int(m.group(1)) for k in out
                    for m in [re.match(r"blocks\.(\d+)\.", k)] if m})
    meta = {
        "format": "pt",
        "modelspec.architecture": "wan2.1-t2v-14B",
        "comfyui_berninir.converted_from": "diffusers/WanTransformer3DModel",
        "comfyui_berninir.dtype": dtype,
        "comfyui_berninir.num_layers": str(n_layers),
        # Pista opcional para model_detection (merge sobre config inferida):
        "config": json.dumps({"transformer": {"num_layers": n_layers}}),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    print(f"[*] guardando {out_path}  ({dtype}, {n_layers} capas) ...")
    save_file(out, out_path, metadata=meta)
    sz = os.path.getsize(out_path) / 1e9
    print(f"    OK — {sz:.1f} GB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_dir", help="carpeta diffusers de UN experto (transformer/ o transformer_2/)")
    ap.add_argument("--out", help="ruta .safetensors de salida (modo un experto)")
    ap.add_argument("--repo", help="carpeta Bernini-R-Diffusers (convierte ambos expertos)")
    ap.add_argument("--out-dir", default="comfy_out", help="carpeta de salida en modo --repo")
    ap.add_argument("--dtype", default="bf16", choices=list(_DTYPES.keys()),
                    help="precisión de salida (fp8_e4m3fn recomendado para <16GB con UNETLoader)")
    args = ap.parse_args()

    if args.repo:
        hi = os.path.join(args.repo, "transformer")
        lo = os.path.join(args.repo, "transformer_2")
        os.makedirs(args.out_dir, exist_ok=True)
        convert_one(hi, os.path.join(args.out_dir, f"bernini_r_high_noise_14B_{args.dtype}.safetensors"), args.dtype)
        convert_one(lo, os.path.join(args.out_dir, f"bernini_r_low_noise_14B_{args.dtype}.safetensors"), args.dtype)
    elif args.in_dir and args.out:
        convert_one(args.in_dir, args.out, args.dtype)
    else:
        ap.error("usa --repo <dir>  o  (--in <dir> --out <archivo>)")


if __name__ == "__main__":
    main()
