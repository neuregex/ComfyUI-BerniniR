#!/usr/bin/env python3
# Copyright (c) 2026 — ComfyUI-BerniniR. Apache-2.0.
"""Upcast fp8 -> bf16 de un .safetensors **NATIVO** (claves Wan ya remapeadas),
en STREAMING (tensor por tensor a disco). NO renombra claves: es solo un cambio
de precisión sobre los expertos que ya produjo `convert_bernini_to_comfy.py`.

Por qué existe:
  * Los cuantizadores GGUF (city96/ComfyUI-GGUF -> llama.cpp) NO entienden fp8
    e4m3; necesitan f16/bf16 como fuente. Este script da ese bf16 intermedio.
  * Lo hace en streaming: lee cada tensor por mmap, lo castea, y escribe sus
    bytes al destino, liberándolo enseguida. Pico de RAM = 1 tensor (~MBs), NO
    el modelo entero (~28 GB) -> esquiva el muro de memoria que ya sufrimos.
  * Escribe el header safetensors a mano (no usa safetensors.torch.load_file,
    que SEGFAULTEA con fp8 en torch 2.8/Windows).

Nota de calidad: si la fuente ya es fp8, el bf16 resultante tiene precisión de
fp8 (no recupera bits). Da igual para el objetivo: el GGUF resultante desbloquea
el dual-expert (low-noise) y con eso desaparece el grano; la precisión queda al
nivel fp8, que es justo el target de lanzamiento.

Uso:
    python fp8_to_bf16.py --in  bernini_r_high_noise_14B_fp8_e4m3fn.safetensors \
                          --out bernini_r_high_noise_14B_bf16.safetensors
    # ...y repetir para el low-noise.
"""
import argparse
import json
import mmap as _mmap
import os
import struct
import sys

import torch

# safetensors dtype string <-> (torch dtype, bytes por elemento)
_ST = {
    "F64": (torch.float64, 8), "F32": (torch.float32, 4), "F16": (torch.float16, 2),
    "BF16": (torch.bfloat16, 2),
    "F8_E4M3": (getattr(torch, "float8_e4m3fn", None), 1),
    "F8_E5M2": (getattr(torch, "float8_e5m2", None), 1),
    "I64": (torch.int64, 8), "I32": (torch.int32, 4), "I16": (torch.int16, 2),
    "I8": (torch.int8, 1), "U8": (torch.uint8, 1), "BOOL": (torch.bool, 1),
}
_FLOAT_ST = {"F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2"}


def _prod(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def _cast_to_bf16_bytes(raw: bytes, src_st: str, shape) -> bytes:
    """Reinterpreta `raw` como el dtype fuente, castea a bf16 y devuelve sus bytes."""
    src_dt = _ST[src_st][0]
    if src_dt is None:
        sys.exit(f"[error] tu torch no soporta {src_st}; actualiza torch>=2.1")
    t = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(src_dt)
    try:
        out = t.to(torch.bfloat16)
    except Exception:                      # algunos builds no castean fp8 en CPU
        out = t.to("cuda").to(torch.bfloat16).cpu()
    return out.contiguous().flatten().view(torch.uint8).numpy().tobytes()


def upcast(in_path: str, out_path: str) -> None:
    fsz = os.path.getsize(in_path)
    with open(in_path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    base = 8 + n

    # 1) Planificar el header de salida (bf16 para floats; resto igual) + offsets.
    src_meta = header.get("__metadata__", {})
    plan = []                              # (name, src_st, src_s, src_e, out_st, out_s, out_e, shape)
    cursor = 0
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        src_st = meta["dtype"]
        if src_st not in _ST:
            sys.exit(f"[error] dtype no soportado: {src_st} ({name})")
        shape = meta["shape"]
        s, e = meta["data_offsets"]
        out_st = "BF16" if src_st in _FLOAT_ST else src_st
        out_nbytes = _prod(shape) * _ST[out_st][1]
        plan.append((name, src_st, s, e, out_st, cursor, cursor + out_nbytes, shape))
        cursor += out_nbytes

    out_header = {"__metadata__": {**{k: str(v) for k, v in src_meta.items()},
                                   "comfyui_berninir.upcast": "fp8->bf16 (streaming)"}}
    for name, src_st, s, e, out_st, os_, oe_, shape in plan:
        out_header[name] = {"dtype": out_st, "shape": shape, "data_offsets": [os_, oe_]}
    hb = json.dumps(out_header, separators=(",", ":")).encode("utf-8")

    total_out = 8 + len(hb) + cursor
    print(f"[*] {os.path.basename(in_path)}  {fsz/1e9:.1f} GB fp8 -> ~{total_out/1e9:.1f} GB bf16")
    print(f"    {len(plan)} tensores; escribiendo en streaming (pico RAM = 1 tensor)")

    # 2) Escribir: 8 bytes len + header + datos (en el MISMO orden de los offsets).
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    written = 0
    last_pct = -1
    with open(in_path, "rb") as fin, open(out_path, "wb") as fout:
        mm = _mmap.mmap(fin.fileno(), 0, access=_mmap.ACCESS_READ)
        try:
            fout.write(struct.pack("<Q", len(hb)))
            fout.write(hb)
            for i, (name, src_st, s, e, out_st, os_, oe_, shape) in enumerate(plan):
                raw = mm[base + s: base + e]
                if out_st == src_st:                      # no-float: copia directa
                    fout.write(raw)
                else:
                    fout.write(_cast_to_bf16_bytes(raw, src_st, shape))
                written += (oe_ - os_)
                pct = int(100 * written / max(cursor, 1))
                if pct != last_pct and pct % 5 == 0:
                    print(f"    {pct:3d}%  ({written/1e9:5.1f} / {cursor/1e9:.1f} GB)", flush=True)
                    last_pct = pct
        finally:
            mm.close()
    sz = os.path.getsize(out_path) / 1e9
    print(f"[OK] {out_path}  ({sz:.1f} GB bf16)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", required=True, help=".safetensors NATIVO fp8 de entrada")
    ap.add_argument("--out", required=True, help=".safetensors bf16 de salida")
    args = ap.parse_args()
    if not os.path.isfile(args.in_path):
        sys.exit(f"[error] no existe: {args.in_path}")
    upcast(args.in_path, args.out)


if __name__ == "__main__":
    main()
