# Copyright (c) 2026 — ComfyUI-BerniniR. Apache-2.0.
"""Backend NATIVO (v0.4) — loaders que entregan el modelo como `MODEL` de ComfyUI.

Por qué existe: `comfy.utils.load_torch_file` (lo que usa el "Load Diffusion Model"
estándar) llama a `safetensors.torch.load_file`, que SEGFAULTEA ('access violation')
materializando tensores **fp8 e4m3** en **torch 2.8 / Windows** (mismo bug que sufrió
nuestra carga diffusers). Aquí materializamos los pesos A MANO (mmap + `mm[a:b]` copia +
`uint8 -> view(dtype)`) y se los pasamos a `comfy.sd.load_diffusion_model_state_dict`,
que NO vuelve a leer del disco -> esquiva el crash. El `MODEL` resultante lo gestiona
ComfyUI igual que cualquier otro (mmap/offload/lowvram nativos)."""
import json
import mmap as _mmap
import struct
import warnings

import torch

# safetensors dtype string -> nombre de dtype de torch
_ST2TORCH = {
    "F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16",
    "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2",
    "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8",
    "U8": "uint8", "BOOL": "bool",
}


def safe_read_safetensors(path):
    """Lee un .safetensors -> {clave: tensor} SIN safetensors.torch.load_file.
    Copia cada tensor con `mm[a:b]` (bytes, NO vista del mmap -> el mmap se cierra
    limpio) y reinterpreta `uint8 -> view(dtype) -> reshape`. Soporta fp8 e4m3 sin
    el access violation de torch 2.8/Windows."""
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        mm = _mmap.mmap(f.fileno(), 0, access=_mmap.ACCESS_READ)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for name, meta in header.items():
                    if name == "__metadata__":
                        continue
                    dtn = _ST2TORCH.get(meta["dtype"])
                    if dtn is None:
                        raise RuntimeError(f"[BerniniR] dtype safetensors no soportado: {meta['dtype']}")
                    dt = getattr(torch, dtn)
                    s, e = meta["data_offsets"]
                    shape = meta["shape"]
                    if e > s:
                        raw = bytearray(mm[base + s: base + e])   # COPIA -> mm cerrable
                        t = torch.frombuffer(raw, dtype=torch.uint8).view(dt)
                        out[name] = t.reshape(shape) if shape else t.reshape(())
                    else:
                        out[name] = torch.empty(shape, dtype=dt)
        finally:
            mm.close()
    return out


class BerniniRLoadModelNative:
    """Carga un experto Wan (formato ComfyUI nativo, p.ej. el .safetensors producido por
    `tools/convert_bernini_to_comfy.py`) como `MODEL` de ComfyUI, esquivando el crash de
    `load_torch_file` con fp8 e4m3 en torch 2.8/Windows. Salida `MODEL` -> se conecta a
    KSampler/SamplerCustom como cualquier modelo nativo."""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {"required": {
            "unet_name": (folder_paths.get_filename_list("diffusion_models"),
                          {"tooltip": "Experto Wan en models/diffusion_models "
                                      "(convertido con convert_bernini_to_comfy.py)."}),
        }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "BerniniR"

    def load(self, unet_name):
        import comfy.sd
        import folder_paths
        path = folder_paths.get_full_path("diffusion_models", unet_name)
        if path is None:
            raise FileNotFoundError(
                f"[BerniniR] no encuentro '{unet_name}' en models/diffusion_models")
        print(f"[BerniniR] cargando (safe fp8, sin load_torch_file) '{unet_name}' ...", flush=True)
        sd = safe_read_safetensors(path)
        model = comfy.sd.load_diffusion_model_state_dict(sd, model_options={})
        if model is None:
            raise RuntimeError(
                f"[BerniniR] ComfyUI no detectó un modelo de difusión en '{unet_name}'. "
                f"¿Es un experto Wan convertido a formato nativo (convert_bernini_to_comfy.py)?")
        print(f"[BerniniR] MODEL nativo listo: '{unet_name}' (gestión de memoria por ComfyUI)", flush=True)
        return (model,)


NODE_CLASS_MAPPINGS = {
    "BerniniRLoadModelNative": BerniniRLoadModelNative,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BerniniRLoadModelNative": "BerniniR · Load Model (native, safe fp8)",
}
