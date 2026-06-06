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
        import torch
        import comfy.model_detection
        import comfy.model_management
        import comfy.model_patcher
        import folder_paths

        path = folder_paths.get_full_path("diffusion_models", unet_name)
        if path is None:
            raise FileNotFoundError(
                f"[BerniniR] no encuentro '{unet_name}' en models/diffusion_models")
        print(f"[BerniniR] cargando (safe fp8 + esqueleto META, 0-RAM) '{unet_name}' ...", flush=True)

        # 1) Leer los pesos fp8 a mano (sin load_torch_file -> sin access violation).
        sd = safe_read_safetensors(path)

        # 2) Detectar el modelo desde las claves.
        model_config = comfy.model_detection.model_config_from_unet(sd, "")
        if model_config is None:
            raise RuntimeError(
                f"[BerniniR] ComfyUI no detectó un modelo de difusión en '{unet_name}' "
                f"(¿experto Wan convertido con convert_bernini_to_comfy.py?).")

        # 2b) Fijar dtype de cómputo / manual-cast ANTES de construir. Si no, las ops NO
        #     castean y el forward peta con "Input type (float) vs bias (BFloat16)" (lo que
        #     vimos: 'manual cast: None'). El checkpoint es MIXTO (fp8 + bf16 + f32);
        #     manual_cast=bf16 hace que cada op convierta su peso al vuelo: fp8->bf16 en los
        #     Linear, y bf16->float32 en el patch_embedding (que el WanModel corre en .float()).
        has_fp8 = any(getattr(t, "dtype", None) == torch.float8_e4m3fn for t in sd.values())
        unet_dtype = torch.float8_e4m3fn if has_fp8 else torch.bfloat16
        manual_cast_dtype = torch.bfloat16
        try:
            model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)
        except Exception as e:
            print(f"[BerniniR] aviso: set_inference_dtype falló ({e}); fijo manual_cast directo", flush=True)
            try:
                model_config.manual_cast_dtype = manual_cast_dtype
            except Exception:
                pass

        # 3) Construir el esqueleto en META: 0 RAM y, clave en torch 2.8/Windows, NO se
        #    ejecuta ningún torch.empty(dtype=fp8) real (que segfaultea). operations.Linear
        #    respeta device -> con device=meta los pesos no se materializan aquí.
        model = model_config.get_model(sd, "", device=torch.device("meta"))

        # 4) Asignar nuestros pesos fp8 sobre el esqueleto (assign=True: reemplaza los
        #    params meta por nuestros tensores, SIN copiar -> RAM baja). El WanModel nativo
        #    no tiene buffers persistentes (RoPE se calcula al vuelo), así que todo lo que
        #    necesita valor está en el checkpoint.
        diff = model.diffusion_model
        missing, unexpected = diff.load_state_dict(sd, strict=False, assign=True)

        meta_left = [n for n, p in diff.named_parameters() if getattr(p, "is_meta", False)]
        meta_left += [n for n, b in diff.named_buffers() if getattr(b, "is_meta", False)]
        if meta_left:
            raise RuntimeError(
                f"[BerniniR] {len(meta_left)} tensores quedaron en META (no estaban en el "
                f"checkpoint): {meta_left[:6]}. Avísame con esta lista y los materializo.")
        if unexpected:
            print(f"[BerniniR] aviso: {len(unexpected)} claves inesperadas (ej: {unexpected[:3]})", flush=True)

        # 5) Envolver en ModelPatcher -> a partir de aquí lo gestiona ComfyUI (offload/lowvram).
        model_patcher = comfy.model_patcher.ModelPatcher(
            model,
            load_device=comfy.model_management.get_torch_device(),
            offload_device=comfy.model_management.unet_offload_device(),
        )
        print(f"[BerniniR] MODEL nativo listo (meta+assign, sin alloc fp8): '{unet_name}'", flush=True)
        # M2: instala los patches de Bernini (src-id RoPE + stream-concat). No-op para t2v
        # (sin streams en transformer_options); con el guider de M3 alimentando streams,
        # habilita la edición. Tolerante a fallos: si algo va mal, corre como Wan nativo.
        try:
            from .native_patches import apply_bernini_patches
            apply_bernini_patches(model_patcher)
            print("[BerniniR] patches Bernini instalados (src-id RoPE + stream-concat)", flush=True)
        except Exception as e:
            print(f"[BerniniR] aviso: patches Bernini NO instalados ({e}); corre como Wan nativo", flush=True)
        return (model_patcher,)


NODE_CLASS_MAPPINGS = {
    "BerniniRLoadModelNative": BerniniRLoadModelNative,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BerniniRLoadModelNative": "BerniniR · Load Model (native, safe fp8)",
}
