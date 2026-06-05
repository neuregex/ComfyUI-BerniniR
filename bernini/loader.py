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
    """True si los safetensors de model_dir/subfolder tienen algún tensor F8_E4M3
    (bundle PRE-cuantizado tipo Bernini-R-fp8). Lee SOLO el header JSON del primer shard
    (unos KB; sin safe_open ni mmap) -> determinista y robusto. La versión anterior usaba
    safe_open y, bajo presión de RAM tras cargar el 1er experto, fallaba en el 2º -> caía
    al camino bf16 de diffusers (que hace f.read() del archivo entero) y reventaba con
    MemoryError. Leer el header evita eso por completo."""
    import glob
    import json
    import struct
    if _fp8_dtype() is None:
        return False
    shards = sorted(glob.glob(os.path.join(model_dir, subfolder, "*.safetensors")))
    if not shards:
        return False
    try:
        with open(shards[0], "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        return any(isinstance(v, dict) and v.get("dtype") == "F8_E4M3"
                   for k, v in header.items() if k != "__metadata__")
    except Exception:
        return False


def _st_expected_size(path):
    """Tamaño en bytes que IMPLICA el header del safetensors: 8 (u64 LE = N) + N
    (JSON del header) + fin del último tensor (max data_offsets[1]). Se lee SOLO el
    header (KB), no los 14GB ni red. Permite detectar archivos TRUNCADOS antes de
    mapearlos."""
    import json
    import struct
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) < 8:
            raise RuntimeError(f"safetensors inválido (8 bytes de cabecera ausentes): {path}")
        n = struct.unpack("<Q", head)[0]
        meta = json.loads(f.read(n))
    end = max((v["data_offsets"][1] for k, v in meta.items()
               if k != "__metadata__" and isinstance(v, dict) and "data_offsets" in v),
              default=0)
    return 8 + n + end


# Mapa de dtypes de safetensors -> torch, para leer los tensores a mano.
_ST_DTYPE = {
    "F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16",
    "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2",
    "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8",
    "U8": "uint8", "BOOL": "bool",
}


def _manual_load(path, keep):
    """Lee un .safetensors a mano vía mmap PEREZOSO y reinterpreta cada tensor
    (uint8 -> view(dtype) -> reshape). Logra dos cosas a la vez:
      - Esquiva la construcción fp8 de safetensors.torch.load_file, que en torch 2.8 +
        safetensors 0.7 (Windows) segfaultea ('access violation') con los F8_E4M3.
      - NO copia el archivo a RAM: los tensores COMPARTEN el mmap (perezoso), igual que
        hacía load_file. Clonar 14GB por experto hacía MemoryError en equipos con poca RAM.
    `keep` mantiene vivos el mmap y el file mientras existan los tensores (los respaldan)."""
    import json
    import mmap as _mmap
    import struct
    import warnings
    import numpy as np
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
    base = 8 + n
    # SOLO-LECTURA a propósito: NO reserva 'commit charge'. ACCESS_COPY (copy-on-write)
    # en Windows reserva commit por TODO el tamaño del mapeo -> con 2 expertos de 14.6GB
    # agotaba el pagefile ('archivo de paginación demasiado pequeño', os error 1455) al
    # cargar UMT5 después. ACCESS_READ es file-backed: las páginas son caché reclamable.
    mm = _mmap.mmap(f.fileno(), 0, access=_mmap.ACCESS_READ)
    keep.append(f)
    keep.append(mm)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # frombuffer puede avisar de buffer no-escribible
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dtname = _ST_DTYPE.get(meta["dtype"])
            if dtname is None:
                raise RuntimeError(f"[BerniniR] dtype safetensors no soportado: {meta['dtype']}")
            dt = getattr(torch, dtname)
            s, e = meta["data_offsets"]
            shape = meta["shape"]
            if e > s:
                # np.frombuffer acepta el buffer de solo-lectura sin quejas; from_numpy
                # comparte su memoria (perezoso). El tensor queda de solo-lectura: no
                # escribimos los pesos (el forward hace weight.to(dtype), que copia).
                arr = np.frombuffer(mm, dtype=np.uint8, count=e - s, offset=base + s)
                t = torch.from_numpy(arr).view(dt)
                out[name] = t.reshape(shape) if shape else t.reshape(())
            else:
                out[name] = torch.empty(shape, dtype=dt)          # tensor vacío
    return out


def _safe_load(shard, keep):
    """Verifica que el .safetensors no esté truncado (error claro en vez de segfault) y
    materializa los tensores A MANO (_manual_load) para esquivar el bug de carga fp8 de
    safetensors. `load_file` mapearía el archivo y, en torch 2.8 + safetensors 0.7
    (Windows), revienta con access violation al construir los F8_E4M3."""
    actual = os.path.getsize(shard)
    try:
        expected = _st_expected_size(shard)
    except Exception as e:
        raise RuntimeError(
            f"[BerniniR] no pude leer el header de '{shard}' ({e}). Probablemente está "
            f"corrupto: bórralo y re-descarga.")
    if actual < expected:
        raise RuntimeError(
            f"[BerniniR] modelo CORRUPTO/INCOMPLETO: '{os.path.basename(shard)}' pesa "
            f"{actual:,} bytes pero su header espera {expected:,} ({100*actual/expected:.1f}%). "
            f"La descarga quedó a medias. Borra ese archivo (o toda la carpeta del repo) y "
            f"re-ejecuta con auto_download (es resumible); idealmente con HF_HUB_DISABLE_XET=1.")
    return _manual_load(shard, keep)


def _load_prefp8(model_dir, subfolder):
    """Carga un WanTransformer3DModel PRE-fp8 preservando el dtype e4m3 del disco:
    init_empty_weights (sin tocar buffers no persistentes como el rope) +
    load_state_dict(assign=True) -> los params toman el dtype EXACTO del checkpoint;
    luego apply_fp8_forward_ parchea solo los Linear fp8."""
    import glob
    from accelerate import init_empty_weights
    from diffusers import WanTransformer3DModel

    config = WanTransformer3DModel.load_config(model_dir, subfolder=subfolder)
    with init_empty_weights(include_buffers=False):
        m = WanTransformer3DModel.from_config(config)
    sd = {}
    keep = []
    for shard in sorted(glob.glob(os.path.join(model_dir, subfolder, "*.safetensors"))):
        sd.update(_safe_load(shard, keep))
    missing, unexpected = m.load_state_dict(sd, strict=False, assign=True)
    m._bernini_keep = keep      # mantiene vivos mmap+file mientras viva el modelo
    m._bernini_cpu_sd = sd      # pesos read-only mmap (CPU, perezosos): para volver a
    #                             "aparcar" el experto en disco al hacer offload (to_idle)
    #                             sin copiar 14.6GB a RAM.
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
