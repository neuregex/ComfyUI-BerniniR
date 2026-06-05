#!/usr/bin/env python
# Copyright (c) 2026 — ComfyUI-BerniniR. Apache-2.0.
"""Construye un bundle **Bernini-R fp8 (e4m3) self-contained** desde Bernini-R-Diffusers.

Carga los dos `WanTransformer3DModel` en bf16, castea sus `nn.Linear` a
`float8_e4m3fn` con la MISMA skip-list que el runtime (norm/embed/head/modulation/
time_/text_emb; la `patch_embedding` es Conv3d → queda en bf16), y los guarda con
`save_pretrained`. Copia vae/ text_encoder/ tokenizer/ scheduler/ y los archivos de
raíz (model_index.json, etc.) sin tocar, para que el repo fp8 sea self-contained.
Al final VERIFICA que en disco los pesos quedaron en float8_e4m3fn.

Uso:
    python tools/save_fp8_diffusers.py --in Bernini-R-Diffusers --out Bernini-R-fp8

(Pensado para correr donde están los pesos — p.ej. el Volume de Modal, ~126GB.)
"""
import argparse
import glob
import os
import shutil
import sys


def _verify_fp8(dst):
    """Comprueba en disco que los transformers tienen pesos en e4m3."""
    import torch
    from safetensors import safe_open
    for sub in ("transformer", "transformer_2"):
        d = os.path.join(dst, sub)
        if not os.path.isdir(d):
            continue
        shards = sorted(glob.glob(os.path.join(d, "*.safetensors")))
        n_fp8 = n_bf16 = n_f32 = n_other = 0
        confirmed = None
        for sh in shards:
            with safe_open(sh, framework="pt") as f:
                for k in f.keys():
                    dt = f.get_slice(k).get_dtype()      # "F8_E4M3" / "BF16" / "F32" ...
                    if "F8_E4M3" in dt:
                        n_fp8 += 1
                        if confirmed is None:
                            confirmed = (k, f.get_tensor(k).dtype)
                    elif "BF16" in dt:
                        n_bf16 += 1
                    elif dt in ("F32", "F16"):
                        n_f32 += 1
                    else:
                        n_other += 1
        print(f"[verify] {sub}: fp8={n_fp8} bf16={n_bf16} f32/16={n_f32} otros={n_other}")
        print(f"[verify] {sub}: ejemplo fp8 -> {confirmed}")
        assert n_fp8 > 0, f"{sub}: NINGÚN tensor quedó en fp8 — algo falló"
        assert confirmed and confirmed[1] == torch.float8_e4m3fn, \
            f"{sub}: el dtype torch del tensor no es float8_e4m3fn ({confirmed})"
    print("[ok] verificación fp8 en disco: OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", required=True, help="repo Bernini-R-Diffusers (bf16)")
    ap.add_argument("--out", dest="dst", required=True, help="destino del bundle fp8")
    args = ap.parse_args()

    import torch
    from diffusers import WanTransformer3DModel
    # importa cast_fp8_/FP8_SKIP del paquete (este script vive en tools/)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from bernini.loader import cast_fp8_, FP8_SKIP

    assert getattr(torch, "float8_e4m3fn", None) is not None, \
        "este torch no tiene float8_e4m3fn (necesitas torch>=2.1)"

    os.makedirs(args.dst, exist_ok=True)

    for sub in ("transformer", "transformer_2"):
        if not os.path.isdir(os.path.join(args.src, sub)):
            print(f"[skip] {sub} no existe en {args.src}")
            continue
        print(f"[*] cargando {sub} (bf16)...")
        m = WanTransformer3DModel.from_pretrained(args.src, subfolder=sub, torch_dtype=torch.bfloat16)
        cast_fp8_(m, skip=FP8_SKIP)
        out_sub = os.path.join(args.dst, sub)
        print(f"[*] save_pretrained {sub} (fp8) -> {out_sub}")
        m.save_pretrained(out_sub)
        del m

    # resto self-contained (sin tocar)
    for sub in ("vae", "text_encoder", "tokenizer", "scheduler"):
        s = os.path.join(args.src, sub)
        if os.path.isdir(s):
            print(f"[*] copiando {sub}/ ...")
            shutil.copytree(s, os.path.join(args.dst, sub), dirs_exist_ok=True)
    for f in os.listdir(args.src):
        fp = os.path.join(args.src, f)
        if os.path.isfile(fp):
            shutil.copy(fp, os.path.join(args.dst, f))

    _verify_fp8(args.dst)
    print(f"[ok] bundle fp8 self-contained en {args.dst}")


if __name__ == "__main__":
    main()
