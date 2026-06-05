# Copyright (c) 2026 — ComfyUI-BerniniR. Apache-2.0.
"""Nodos de ComfyUI para Bernini-R (ByteDance) — Wan2.2-T2V-A14B + src-id RoPE + APG."""
import os

import torch

from .bernini import (
    constants as C,
    BerniniRenderer, BerniniExpert, BerniniSampler,
    load_experts, load_vae, TextEncoder, latents as L,
)

try:
    import comfy.model_management as mm
    def _device():
        return mm.get_torch_device()
    def _offload_device():
        return mm.unet_offload_device()
except Exception:  # fuera de ComfyUI (tests)
    def _device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def _offload_device():
        return torch.device("cpu")


def _report_vram(tag):
    """Reporte de VRAM gated por env (BERNINIR_REPORT_VRAM); silencioso por
    defecto. max_memory_allocated()/reserved() son picos desde el inicio del
    proceso -> el último reporte da el pico global del pipeline (objetivo ≤24GB)."""
    if not os.environ.get("BERNINIR_REPORT_VRAM") or not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    g = 1024 ** 3
    print(f"[BerniniR][VRAM] {tag}: alloc={torch.cuda.memory_allocated()/g:.2f}GB "
          f"pico_alloc={torch.cuda.max_memory_allocated()/g:.2f}GB "
          f"pico_reserved={torch.cuda.max_memory_reserved()/g:.2f}GB", flush=True)


# Tipos personalizados (se pasan tal cual entre nodos):
#   BR_MODEL = BerniniRenderer ; BR_VAE = AutoencoderKLWan
#   BR_COND  = {"pos","neg","task_type"} ; BR_SRC = {"video_latents","image_latents"}
#   BR_LATENT= {"samples": [1,16,T,H,W]}


def _img_to_video(images: torch.Tensor) -> torch.Tensor:
    """ComfyUI IMAGE [T,H,W,C] en 0..1  ->  [1,C,T,H,W] en [-1,1]."""
    x = images.permute(3, 0, 1, 2).unsqueeze(0)      # [1,C,T,H,W]
    return x * 2.0 - 1.0


# Fuentes de pesos: combo del widget -> repo HF (o "local").
_HF_REPOS = {
    "neuregex/Bernini-R-fp8 (auto)": "neuregex/Bernini-R-fp8",
    "ByteDance/Bernini-R-Diffusers (full bf16)": "ByteDance/Bernini-R-Diffusers",
}


def _has_weights(d):
    import glob
    return bool(glob.glob(os.path.join(d, "transformer", "*.safetensors")))


def _resolve_dir(p):
    """Ruta absoluta; relativa se ancla al base_path de ComfyUI (o cwd)."""
    p = os.path.expanduser(p)
    if os.path.isabs(p):
        return p
    try:
        import folder_paths
        return os.path.join(folder_paths.base_path, p)
    except Exception:
        return os.path.abspath(p)


def _comfy_tqdm():
    """tqdm que además refleja el progreso en la barra de ComfyUI (best-effort)."""
    try:
        from comfy.utils import ProgressBar
        from tqdm import tqdm as _tqdm
    except Exception:
        return None

    class _PBTqdm(_tqdm):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._pb = ProgressBar(self.total) if getattr(self, "total", None) else None

        def update(self, n=1):
            super().update(n)
            try:
                if self._pb and self.total:
                    self._pb.update_absolute(self.n, self.total)
            except Exception:
                pass
    return _PBTqdm


def _ensure_weights(repo_id, dst, auto_download):
    """Garantiza pesos en `dst`; si faltan y auto_download, los baja de HF con
    check de espacio + aviso + progreso (tqdm/ProgressBar). Si no, raise claro."""
    if _has_weights(dst):
        return dst
    if not auto_download:
        raise FileNotFoundError(
            f"[BerniniR] Faltan pesos de '{repo_id}' en {dst}. Activa auto_download, o "
            f"descárgalos a mano:\n  huggingface-cli download {repo_id} --local-dir \"{dst}\"")
    import shutil as _sh
    from huggingface_hub import snapshot_download
    parent = os.path.dirname(dst) or "."
    os.makedirs(parent, exist_ok=True)
    free_gb = _sh.disk_usage(parent).free / (1024 ** 3)
    need_gb = 40 if "fp8" in repo_id.lower() else 130
    print(f"[BerniniR] descargando '{repo_id}' -> {dst}  (~{need_gb}GB el PRIMER run; "
          f"{free_gb:.0f}GB libres en {parent})")
    if free_gb < need_gb * 1.1:
        raise RuntimeError(
            f"[BerniniR] espacio insuficiente para '{repo_id}': ~{int(need_gb * 1.1)}GB "
            f"necesarios, solo {free_gb:.0f}GB libres en {parent}.")
    os.makedirs(dst, exist_ok=True)
    kw = {}
    tq = _comfy_tqdm()
    if tq is not None:
        kw["tqdm_class"] = tq
    snapshot_download(repo_id=repo_id, local_dir=dst, **kw)
    if not _has_weights(dst):
        raise RuntimeError(f"[BerniniR] descarga incompleta: sin transformer/*.safetensors en {dst}")
    print(f"[BerniniR] descarga completa: {dst}")
    return dst


class BerniniRModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "source": (list(_HF_REPOS.keys()) + ["local"],
                       {"default": "neuregex/Bernini-R-fp8 (auto)",
                        "tooltip": "fp8 (~40GB, cabe en 24GB) / bf16 full (~126GB) / local (usa model_dir)."}),
            "auto_download": ("BOOLEAN", {"default": True, "tooltip": "Descarga el repo si falta (snapshot_download de HF)."}),
            "download_dir": ("STRING", {"default": "models/bernini", "tooltip": "Carpeta de descarga (relativa a ComfyUI o absoluta)."}),
            "dtype": (["bf16", "fp16"], {"default": "bf16"}),
            "fp8": ("BOOLEAN", {"default": True, "tooltip": "Cuantiza on-the-fly a fp8 si el repo es bf16 (el bundle fp8 ya viene cuantizado)."}),
            "offload_experts": ("BOOLEAN", {"default": True, "tooltip": "Mantén solo el experto activo en GPU (high/low se intercambian)."}),
        }, "optional": {
            "model_dir": ("STRING", {"default": "Bernini-R-Diffusers", "tooltip": "Ruta local de los pesos (solo si source='local')."}),
        }}

    # 2º output BR_PATH = ruta resuelta/descargada: VAE y TextEncode la reciben por
    # conexión, así TODO el grafo usa UNA sola fuente de pesos (incl. auto-download).
    RETURN_TYPES = ("BR_MODEL", "BR_PATH")
    RETURN_NAMES = ("model", "model_path")
    FUNCTION = "load"
    CATEGORY = "BerniniR"

    def load(self, source, auto_download, download_dir, dtype, fp8, offload_experts,
             model_dir="Bernini-R-Diffusers"):
        if source == "local":
            resolved = _resolve_dir(model_dir)
            if not _has_weights(resolved):
                raise FileNotFoundError(f"[BerniniR] No hay pesos en {resolved} (transformer/*.safetensors).")
        else:
            repo_id = _HF_REPOS[source]
            resolved = os.path.join(_resolve_dir(download_dir), repo_id.split("/")[-1])
            _ensure_weights(repo_id, resolved, auto_download)
        hi, lo = load_experts(resolved, dtype=dtype, fp8=fp8, device="cpu")
        renderer = BerniniRenderer(
            high=BerniniExpert(hi, use_src_id_rotary_emb=True),
            low=BerniniExpert(lo, use_src_id_rotary_emb=True) if lo is not None else None,
        )
        if not offload_experts:
            renderer.high.t.to(_device())
            if renderer.low is not None:
                renderer.low.t.to(_device())
        renderer._offload = offload_experts
        renderer._model_dir = resolved
        return ({"renderer": renderer, "offload": offload_experts, "model_dir": resolved}, resolved)


class BerniniRVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_path": ("BR_PATH", {"tooltip": "Conecta el output 'model_path' del Load Model."}),
            "dtype": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
        }}

    RETURN_TYPES = ("BR_VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load"
    CATEGORY = "BerniniR"

    def load(self, model_path, dtype):
        vae = load_vae(os.path.expanduser(model_path), dtype=dtype, device=_offload_device())
        return (vae,)


class BerniniRTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_path": ("BR_PATH", {"tooltip": "Conecta el output 'model_path' del Load Model."}),
            "prompt": ("STRING", {"multiline": True, "default": ""}),
            "task_type": (C.TASK_TYPES, {"default": "t2v"}),
            "add_system_prefix": ("BOOLEAN", {"default": True, "tooltip": "Antepone el system prompt de la tarea (como Bernini)."}),
            "negative_prompt": ("STRING", {"multiline": True, "default": C.DEFAULT_NEG_PROMPT}),
        }}

    RETURN_TYPES = ("BR_COND",)
    RETURN_NAMES = ("cond",)
    FUNCTION = "encode"
    CATEGORY = "BerniniR"

    def encode(self, model_path, prompt, task_type, add_system_prefix, negative_prompt):
        te = TextEncoder(os.path.expanduser(model_path), dtype="bf16", device=_device())
        prefix = C.get_system_prompt_for_task(task_type) if add_system_prefix else ""
        pos = te.encode(prompt, system_prefix=prefix)
        neg = te.encode(negative_prompt, system_prefix="")   # el negativo NO lleva prefijo
        del te
        torch.cuda.empty_cache()
        _report_vram("tras text-encode UMT5 (liberado antes de cargar expertos)")
        return ({"pos": pos.cpu(), "neg": neg.cpu(), "task_type": task_type},)


class BerniniRSourceMedia:
    """VAE-encode del vídeo fuente y/o imágenes de referencia."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae": ("BR_VAE",)},
                "optional": {
                    "source_video": ("IMAGE", {"tooltip": "Frames del vídeo a editar (v2v/rv2v)."}),
                    "reference_images": ("IMAGE", {"tooltip": "Imagen(es) de referencia (i2i/r2v/rv2v)."}),
                }}

    RETURN_TYPES = ("BR_SRC",)
    RETURN_NAMES = ("src",)
    FUNCTION = "encode"
    CATEGORY = "BerniniR"

    def encode(self, vae, source_video=None, reference_images=None):
        dev = _device()
        vae.to(dev)
        video_latents, image_latents = [], []
        if source_video is not None and source_video.shape[0] > 0:
            x = _img_to_video(source_video).to(dev, vae.dtype)
            video_latents.append(L.vae_encode(vae, x).cpu())
        if reference_images is not None and reference_images.shape[0] > 0:
            for i in range(reference_images.shape[0]):
                img = reference_images[i:i + 1]                       # [1,H,W,C]
                x = _img_to_video(img).to(dev, vae.dtype)             # [1,C,1,H,W]
                image_latents.append(L.vae_encode(vae, x).cpu())
        vae.to(_offload_device())
        torch.cuda.empty_cache()
        return ({"video_latents": video_latents, "image_latents": image_latents},)


class BerniniRSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("BR_MODEL",),
            "cond": ("BR_COND",),
            "guidance_mode": (["auto"] + C.GUIDANCE_MODES, {"default": "auto"}),
            "width": ("INT", {"default": 848, "min": 128, "max": 2048, "step": 16}),
            "height": ("INT", {"default": 480, "min": 128, "max": 2048, "step": 16}),
            "num_frames": ("INT", {"default": 81, "min": 1, "max": 257}),
            "steps": ("INT", {"default": 40, "min": 1, "max": 100}),
            "omega_V": ("FLOAT", {"default": 1.25, "min": 0.0, "max": 20.0, "step": 0.05}),
            "omega_I": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 20.0, "step": 0.05}),
            "omega_TI": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.05}),
            "omega_scale": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05}),
            "eta": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            "norm_threshold": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 500.0, "step": 1.0}),
            "momentum": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
        }, "optional": {"src": ("BR_SRC",)}}

    RETURN_TYPES = ("BR_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "BerniniR"

    def sample(self, model, cond, guidance_mode, width, height, num_frames, steps,
               omega_V, omega_I, omega_TI, omega_scale, eta, norm_threshold, momentum, seed, src=None):
        renderer = model["renderer"]
        task_type = cond.get("task_type", "t2v")
        mode = C.TASK_TO_GUIDANCE.get(task_type, "t2v") if guidance_mode == "auto" else guidance_mode

        video_latents = (src or {}).get("video_latents", []) or []
        image_latents = (src or {}).get("image_latents", []) or []
        dev = _device()
        video_latents = [v.to(dev) for v in video_latents]
        image_latents = [v.to(dev) for v in image_latents]

        if task_type in ("t2i", "i2i"):
            num_frames = 1
        target_shape = L.latent_shape(num_frames, height, width)

        params = dict(num_inference_steps=steps, omega_V=omega_V, omega_I=omega_I,
                      omega_TI=omega_TI, omega_scale=omega_scale, eta=eta,
                      norm_threshold=(norm_threshold, norm_threshold, norm_threshold),
                      momentum=momentum, num_frames=num_frames, height=height, width=width)

        sampler = BerniniSampler(renderer, mode, params, offload_experts=model.get("offload", True))
        base_sched = os.path.join(model.get("model_dir", ""), "scheduler")
        base_sched = base_sched if os.path.isdir(base_sched) else None

        latent = sampler.sample(
            video_latents=video_latents, image_latents=image_latents,
            text_pos=cond["pos"].to(dev), text_neg=cond["neg"].to(dev),
            target_shape=target_shape, device=dev, seed=seed, base_scheduler_dir=base_sched)
        _report_vram(f"tras sample ({mode}, offload={model.get('offload')}, {target_shape[2]}x{target_shape[3]}x{target_shape[4]})")
        return ({"samples": latent.cpu()},)


class BerniniRDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vae": ("BR_VAE",), "latent": ("BR_LATENT",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "decode"
    CATEGORY = "BerniniR"

    def decode(self, vae, latent):
        dev = _device()
        vae.to(dev)
        x = latent["samples"].to(dev, vae.dtype)
        frames = L.vae_decode(vae, x)                          # [1,C,T,H,W] en [-1,1]
        vae.to(_offload_device())
        frames = (frames.clamp(-1, 1) + 1.0) / 2.0
        frames = frames[0].permute(1, 2, 3, 0).float().cpu()   # [T,H,W,C]
        _report_vram("tras decode (pico TOTAL del pipeline)")
        return (frames,)


class BerniniRLoadVideo:
    """Carga frames de un vídeo animado (webp/gif) desde ComfyUI/input como un batch
    IMAGE [T,H,W,C] en 0..1. Autocontenido (PIL) — NO depende de VideoHelperSuite.
    Para mp4/avi usa VHS_LoadVideo u otro loader y conéctalo igual a source_video."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video": ("STRING", {"default": "source.webp", "tooltip": "Archivo en ComfyUI/input (webp o gif animado)"}),
            "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 1024, "tooltip": "Máximo de frames (0 = todos)"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "load"
    CATEGORY = "BerniniR"

    def load(self, video, frame_load_cap=0):
        import numpy as np
        from PIL import Image, ImageSequence
        try:
            import folder_paths
            base = folder_paths.get_input_directory()
        except Exception:
            base = "input"
        path = video if os.path.isabs(video) else os.path.join(base, video)
        im = Image.open(path)
        frames = []
        for i, fr in enumerate(ImageSequence.Iterator(im)):
            if frame_load_cap and i >= frame_load_cap:
                break
            frames.append(np.asarray(fr.convert("RGB"), dtype=np.float32) / 255.0)
        if not frames:
            raise ValueError(f"BerniniRLoadVideo: sin frames en {path}")
        return (torch.from_numpy(np.stack(frames, axis=0)),)   # [T,H,W,C]


NODE_CLASS_MAPPINGS = {
    "BerniniRModelLoader": BerniniRModelLoader,
    "BerniniRVAELoader": BerniniRVAELoader,
    "BerniniRTextEncode": BerniniRTextEncode,
    "BerniniRSourceMedia": BerniniRSourceMedia,
    "BerniniRLoadVideo": BerniniRLoadVideo,
    "BerniniRSampler": BerniniRSampler,
    "BerniniRDecode": BerniniRDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BerniniRModelLoader": "BerniniR · Load Model (dual-expert)",
    "BerniniRVAELoader": "BerniniR · Load VAE (Wan)",
    "BerniniRTextEncode": "BerniniR · Text Encode (UMT5 + task prefix)",
    "BerniniRSourceMedia": "BerniniR · Encode Source/Reference",
    "BerniniRLoadVideo": "BerniniR · Load Video (webp/gif, no VHS)",
    "BerniniRSampler": "BerniniR · Sampler (src-id RoPE + APG)",
    "BerniniRDecode": "BerniniR · VAE Decode",
}
