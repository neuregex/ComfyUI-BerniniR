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


# pon True para imprimir picos de VRAM por etapa (debug; silencioso por defecto)
_REPORT_VRAM = False


def _report_vram(tag):
    """Reporte de VRAM (flag _REPORT_VRAM, silencioso por defecto).
    max_memory_allocated()/reserved() son picos desde el inicio del
    proceso -> el último reporte da el pico global del pipeline (objetivo ≤24GB)."""
    if not _REPORT_VRAM or not torch.cuda.is_available():
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


def _dir_size(path):
    """Suma de bytes de TODOS los archivos bajo `path`. Incluye los `.incomplete` que
    huggingface_hub escribe en `<dst>/.cache/huggingface/download/` mientras baja, así
    que es la señal REAL de progreso, independiente de la versión/transporte de HF."""
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _download_with_progress(dst, total, do_download):
    """Corre `do_download()` mientras un hilo VIGILA el tamaño en disco de `dst` y emite
    progreso REAL por bytes (%, GB, MB/s, ETA) cada 2s -> consola + barra de ComfyUI.

    NO depende del tqdm interno de HF: su barra 'Fetching N files' cuenta ARCHIVOS y se
    queda clavada en 5% (=1/19) mientras baja un shard de 14GB -> parecía colgado. Aquí
    vigilamos los BYTES que aterrizan en disco, que suben de verdad. Silenciamos las
    barras de HF para no duplicar la salida."""
    import threading
    import time
    g = 1024 ** 3
    try:
        from comfy.utils import ProgressBar
        pb = ProgressBar(total) if total else None
    except Exception:
        pb = None
    stop = threading.Event()

    def _watch():
        # Solo imprime cuando hay PROGRESO real en disco (xet escribe a ráfagas: si
        # imprimiéramos cada 2s veríamos 100 líneas con el mismo % y un MB/s "medio"
        # que decae solo). MB/s = ritmo del último tramo (real, no decae); ETA = sobre
        # el promedio global (estable). Latido cada ~20s si lleva rato plano (xet
        # llenando su caché fuera de `dst`) para que se vea que sigue vivo.
        t0 = t_prev = last_beat = time.time()
        done_prev = 0
        while not stop.is_set():
            stop.wait(2.0)
            if not total:
                continue
            done = _dir_size(dst)
            now = time.time()
            if done > done_prev:
                inst = (done - done_prev) / max(now - t_prev, 1e-6) / (1024 ** 2)  # MB/s recientes
                avg = done / max(now - t0, 1e-6)                                   # B/s medio
                eta = (total - done) / max(avg, 1.0) / 60
                print(f"[BerniniR]  {100 * min(done, total) / total:4.1f}%  "
                      f"{done / g:5.2f}/{total / g:.2f}GB  {inst:5.1f}MB/s  "
                      f"ETA {max(eta, 0.0):4.1f}min", flush=True)
                if pb:
                    try:
                        pb.update_absolute(min(done, total), total)
                    except Exception:
                        pass
                t_prev = last_beat = now
                done_prev = done
            elif now - last_beat >= 20:
                print(f"[BerniniR]  {100 * min(done, total) / total:4.1f}%  "
                      f"{done / g:.2f}/{total / g:.2f}GB  (sigue bajando; xet escribe a rafagas)",
                      flush=True)
                last_beat = now

    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except Exception:
        pass
    watcher = threading.Thread(target=_watch, daemon=True)
    watcher.start()
    try:
        do_download()
    finally:
        stop.set()
        watcher.join(timeout=3)
        try:
            from huggingface_hub.utils import enable_progress_bars
            enable_progress_bars()
        except Exception:
            pass
        if total:
            print(f"[BerniniR]  100.0%  {total / g:.2f}/{total / g:.2f}GB  (completo)", flush=True)


def _ensure_weights(repo_id, dst, auto_download):
    """Garantiza pesos en `dst`; si faltan y auto_download, los baja de HF con progreso
    REAL por bytes (%, MB/s, ETA) + check de espacio exacto. Usa el motor de HF: xet
    (Rust, multi-conexión, rápido) si 'hf_xet' está instalado, si no HTTPS clásico.
    Resumible: un Ctrl-C deja los .safetensors a medias y el re-run retoma. Si faltan y
    no hay auto_download, raise con la orden manual."""
    if _has_weights(dst):
        return dst
    if not auto_download:
        raise FileNotFoundError(
            f"[BerniniR] Faltan pesos de '{repo_id}' en {dst}. Activa auto_download, o "
            f"descárgalos a mano:\n  huggingface-cli download {repo_id} --local-dir \"{dst}\"")

    # Motor: huggingface_hub usa xet (Rust, multi-conexión, RÁPIDO) si 'hf_xet' está
    # instalado; si no, HTTPS clásico (1 stream/archivo). NO forzamos nada -> dejamos el
    # acelerador nativo activo (antes lo desactivábamos por una hipótesis de cuelgue que
    # no se sostuvo). Un HF_HUB_DISABLE_XET=1 en el entorno lo apaga, por si hiciera falta.
    try:
        import hf_xet  # noqa: F401
        _xet_on = True
    except Exception:
        _xet_on = False
    if not _xet_on:
        print("[BerniniR] tip: 'pip install hf_xet' habilita descargas xet "
              "(multi-conexión; suele ir más rápido si tu conexión da para más).", flush=True)

    import shutil as _sh
    from huggingface_hub import snapshot_download
    parent = os.path.dirname(dst) or "."
    os.makedirs(dst, exist_ok=True)
    g = 1024 ** 3

    # Total REAL en bytes desde la metadata del repo -> % fiable + check exacto.
    total = 0
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo_id, files_metadata=True)
        total = sum((s.size or 0) for s in info.siblings if not s.rfilename.endswith("/"))
    except Exception as e:
        print(f"[BerniniR] aviso: no pude leer tamaños del repo ({e}); sigo sin % global.")

    need = total if total else (40 * g if "fp8" in repo_id.lower() else 130 * g)
    free = _sh.disk_usage(parent).free
    print(f"[BerniniR] descargando '{repo_id}' -> {dst}", flush=True)
    print(f"[BerniniR] total ~{need / g:.1f}GB | libres {free / g:.0f}GB en {parent} | "
          f"motor={'xet (rapido)' if _xet_on else 'https'} | resumible (Ctrl-C y re-run retoma)",
          flush=True)
    if free < need * 1.05:
        raise RuntimeError(
            f"[BerniniR] espacio insuficiente para '{repo_id}': ~{need * 1.05 / g:.0f}GB "
            f"necesarios, solo {free / g:.0f}GB libres en {parent}.")

    _download_with_progress(
        dst, total,
        lambda: snapshot_download(repo_id=repo_id, local_dir=dst),
    )
    if not _has_weights(dst):
        raise RuntimeError(f"[BerniniR] descarga incompleta: sin transformer/*.safetensors en {dst}")
    print(f"[BerniniR] descarga completa: {dst}", flush=True)
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
            # APPEND-ONLY: los widgets NUEVOS van SIEMPRE al final del bloque optional.
            # ComfyUI mapea los `widgets_values` guardados a los slots por ORDEN; insertar
            # un widget en medio desplaza los valores de los workflows ya guardados (p.ej.
            # el model_dir de un .json viejo caería en este INT -> "invalid literal for
            # int()"). Por eso blocks_to_swap va detrás de model_dir: los workflows 0.2.0
            # (que terminaban en model_dir) siguen mapeando bien y blocks_to_swap cae a su
            # default. NUNCA insertar widgets en medio de un bloque ya publicado.
            "blocks_to_swap": ("INT", {"default": 0, "min": 0, "max": 40,
                "tooltip": "N de los 40 bloques del transformer viven en CPU y se streamean a GPU por bloque (baja VRAM, más lento). 0 = off. Sube hasta caber en 16/12GB."}),
        }}

    # 2º output BR_PATH = ruta resuelta/descargada: VAE y TextEncode la reciben por
    # conexión, así TODO el grafo usa UNA sola fuente de pesos (incl. auto-download).
    RETURN_TYPES = ("BR_MODEL", "BR_PATH")
    RETURN_NAMES = ("model", "model_path")
    FUNCTION = "load"
    CATEGORY = "BerniniR"

    def load(self, source, auto_download, download_dir, dtype, fp8, offload_experts,
             blocks_to_swap=0, model_dir="Bernini-R-Diffusers"):
        if source == "local":
            resolved = _resolve_dir(model_dir)
            if not _has_weights(resolved):
                raise FileNotFoundError(f"[BerniniR] No hay pesos en {resolved} (transformer/*.safetensors).")
        else:
            repo_id = _HF_REPOS[source]
            resolved = os.path.join(_resolve_dir(download_dir), repo_id.split("/")[-1])
            _ensure_weights(repo_id, resolved, auto_download)
        bs = int(blocks_to_swap)
        hi, lo = load_experts(resolved, dtype=dtype, fp8=fp8, device="cpu")
        renderer = BerniniRenderer(
            high=BerniniExpert(hi, use_src_id_rotary_emb=True, block_swap=bs),
            low=BerniniExpert(lo, use_src_id_rotary_emb=True, block_swap=bs) if lo is not None else None,
        )
        # con block-swap (o offload) el sampler coloca; sin ninguno, pre-colocamos en GPU.
        if not offload_experts and not bs:
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


_VIDEO_EXTS = (".webp", ".gif", ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")


def _decode_video_frames(path, cap):
    """Decodifica un vídeo (mp4/mov/avi/mkv/webm) a lista de arrays float32 [H,W,3]
    en 0..1. Prueba PyAV, luego OpenCV, luego imageio — el primero instalado.
    cap=0 -> todos los frames."""
    import numpy as np
    fails = []
    try:
        import av
        out = []
        with av.open(path) as container:
            for i, frame in enumerate(container.decode(video=0)):
                if cap and i >= cap:
                    break
                out.append(frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0)
        if out:
            return out
    except Exception as e:
        fails.append(f"av: {e}")
    try:
        import cv2
        out, v = [], cv2.VideoCapture(path)
        while not cap or len(out) < cap:
            ok, bgr = v.read()
            if not ok:
                break
            out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
        v.release()
        if out:
            return out
    except Exception as e:
        fails.append(f"cv2: {e}")
    try:
        import imageio.v3 as iio
        out = []
        for i, fr in enumerate(iio.imiter(path)):
            if cap and i >= cap:
                break
            out.append(np.asarray(fr, dtype=np.float32)[..., :3] / 255.0)
        if out:
            return out
    except Exception as e:
        fails.append(f"imageio: {e}")
    raise RuntimeError(
        "BerniniRLoadVideo: no pude decodificar ese vídeo. Instala un backend: "
        "pip install av (recomendado), opencv-python o imageio[ffmpeg]. "
        "Detalle por backend -> " + " | ".join(fails))


class BerniniRLoadVideo:
    """Carga frames de un vídeo desde ComfyUI/input como batch IMAGE [T,H,W,C] en 0..1.
    webp/gif via PIL (autocontenido); mp4/mov/avi/mkv/webm via PyAV / OpenCV / imageio
    (el que esté instalado; `pip install av` recomendado)."""
    @classmethod
    def INPUT_TYPES(cls):
        # Desplegable con los vídeos de ComfyUI/input + botón de carga (webp/gif).
        # Los mp4/mov/etc. colócalos en ComfyUI/input y aparecerán en la lista.
        files = ["source.webp"]
        try:
            import folder_paths
            d = folder_paths.get_input_directory()
            found = sorted(f for f in os.listdir(d)
                           if os.path.isfile(os.path.join(d, f)) and f.lower().endswith(_VIDEO_EXTS))
            if found:
                files = found
        except Exception:
            pass
        return {"required": {
            "video": (files, {"image_upload": True,
                      "tooltip": "Vídeo en ComfyUI/input (webp/gif/mp4/mov/avi/mkv/webm). "
                                 "El botón carga webp/gif; para mp4, colócalo en ComfyUI/input."}),
            "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 1024, "tooltip": "Máximo de frames (0 = todos)"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "load"
    CATEGORY = "BerniniR"

    def load(self, video, frame_load_cap=0):
        import numpy as np
        try:
            import folder_paths
            base = folder_paths.get_input_directory()
        except Exception:
            base = "input"
        path = video if os.path.isabs(video) else os.path.join(base, video)
        cap = int(frame_load_cap or 0)
        if path.lower().endswith((".webp", ".gif")):
            from PIL import Image, ImageSequence
            im = Image.open(path)
            frames = []
            for i, fr in enumerate(ImageSequence.Iterator(im)):
                if cap and i >= cap:
                    break
                frames.append(np.asarray(fr.convert("RGB"), dtype=np.float32) / 255.0)
        else:
            frames = _decode_video_frames(path, cap)
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
    "BerniniRLoadVideo": "BerniniR · Load Video (webp/gif/mp4/mov/...)",
    "BerniniRSampler": "BerniniR · Sampler (src-id RoPE + APG)",
    "BerniniRDecode": "BerniniR · VAE Decode",
}
