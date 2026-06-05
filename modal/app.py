# Copyright (c) 2026 — ComfyUI-BerniniR. Apache-2.0.
"""Harness de validación en Modal: levanta ComfyUI headless con el custom node
ComfyUI-BerniniR en una GPU grande y ejecuta un workflow (formato API).

Filosofía (skill modal-comfyui-deploy): *local para iterar, Modal para producir*.
Aquí Modal se usa para VALIDAR el modelo de 28B (no entra en consumer), porque
los pesos pesan ~160GB y piden GPU clase H100/A100-80GB.

Uso
----
    pip install modal && modal token new            # una vez
    modal run modal/app.py::download_weights        # baja Bernini-R-Diffusers al Volume (~126GB)
    modal run modal/app.py::smoke                    # validación numérica barata (1 forward)
    modal run modal/app.py::run --workflow workflows/bernini_t2v.json
    modal run modal/app.py::run --workflow workflows/bernini_rv2v.json

Salidas en el Volume 'berninir-out' (descárgalas con `modal volume get`).
"""
import json
import os
import pathlib
import subprocess
import time
import urllib.request

import modal

app = modal.App("comfyui-berninir")

MODELS = modal.Volume.from_name("berninir-models", create_if_missing=True)
OUT = modal.Volume.from_name("berninir-out", create_if_missing=True)
MODELS_DIR = "/models"
OUT_DIR = "/out"
HERE = pathlib.Path(__file__).parent.parent  # carpeta del custom node

# GPU parametrizable por env var (leída al DEFINIR la app, en el cliente local):
#   $env:BERNINIR_GPU="A10G"; modal run modal/app.py::run ...   -> prueba en 24GB
# Default A100-80GB (caben 2 expertos bf16). Para el objetivo ≤24GB usar L4/A10G.
GPU = os.environ.get("BERNINIR_GPU", "A100-80GB")
# RAM de CPU: cargar 2 expertos pasa por bf16 (~28GB c/u) antes de cuantizar a fp8,
# con pico ~42GB al cargar el 2º. Reservamos holgura para no OOMear en CPU.
CPU_MEM = int(os.environ.get("BERNINIR_CPU_MEM", "49152"))

# Imagen: CUDA 12.4 + torch 2.5.1 (alineado con la pila de Bernini-R) + ComfyUI
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    # torchaudio==2.5.1 EXPLÍCITO (trío cu124 consistente): ComfyUI importa torchaudio
    # incondicionalmente (comfy.sd -> audio_vae). Si no lo fijamos aquí, el requirements
    # de ComfyUI lo baja de PyPI con build CUDA 13 -> OSError libcudart.so.13 al arrancar.
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1",
                 index_url="https://download.pytorch.org/whl/cu124")
    .run_commands("git clone --depth 1 https://github.com/comfyanonymous/ComfyUI /root/ComfyUI")
    .run_commands("pip install -r /root/ComfyUI/requirements.txt")
    # transformers==4.49.0: las >=4.56 referencian torch.float8_e8m0fnu (dtype MXFP8
    # de torch>=2.7) en integrations/finegrained_fp8.py, y la imagen fija torch 2.5.1
    # (sin ese dtype) -> import de diffusers.WanTransformer3DModel reventaba. 4.49.0
    # es compatible con torch 2.5.1 y con diffusers 0.35.2 (que solo pide >=4.41.2),
    # y trae UMT5EncoderModel + tokenizer para el text-encode del run.
    .pip_install("diffusers==0.35.2", "transformers==4.49.0", "accelerate>=0.34.2",
                 "einops>=0.7.0", "ftfy>=6.1.0", "safetensors>=0.4.0",
                 "sentencepiece>=0.2.0", "huggingface_hub", "requests")
    # monta el custom node (esta misma carpeta) en ComfyUI/custom_nodes.
    # ignore: NO subir el venv local, caches, tests ni pesos (patrones dockerignore).
    .add_local_dir(str(HERE), "/root/ComfyUI/custom_nodes/ComfyUI-BerniniR", copy=True,
                   ignore=["modal", ".git", ".venv", ".pytest_cache", "tests",
                           "__pycache__", "**/__pycache__",
                           "*.safetensors", "**/*.safetensors", "Bernini-R-Diffusers"])
)


@app.function(image=image, volumes={MODELS_DIR: MODELS}, timeout=60 * 60 * 4,
              container_idle_timeout=60)
def download_weights(repo: str = "ByteDance/Bernini-R-Diffusers"):
    """Descarga el repo diffusers self-contained (vae+t5+tokenizer+2 transformers)."""
    from huggingface_hub import snapshot_download
    dst = f"{MODELS_DIR}/Bernini-R-Diffusers"
    print(f"[*] descargando {repo} -> {dst}")
    snapshot_download(repo_id=repo, local_dir=dst, max_workers=8)
    MODELS.commit()
    print("[ok] pesos en el Volume")


@app.function(image=image, gpu=GPU, volumes={MODELS_DIR: MODELS}, timeout=60 * 30,
              container_idle_timeout=60, memory=CPU_MEM)
def smoke(fp8: bool = False):
    """Validación numérica: carga los DOS expertos reales y corre 1 forward
    multi-stream t2v de baja resolución (2 steps que cruzan la frontera 875, así
    que se ejercitan AMBOS expertos), comprobando shapes y ausencia de NaN.

    Por defecto bf16 SIN fp8 (fidelidad sin la variable de cuantización M8); pasa
    --fp8 para validar también el path cuantizado. En A100-80GB caben los dos
    expertos bf16 (~28GB c/u) en VRAM a la vez."""
    import time
    import torch
    import sys
    sys.path.insert(0, "/root/ComfyUI/custom_nodes/ComfyUI-BerniniR")
    from bernini import load_experts, BerniniRenderer, BerniniExpert, BerniniSampler
    from bernini import latents as L

    md = f"{MODELS_DIR}/Bernini-R-Diffusers"
    t0 = time.time()
    hi, lo = load_experts(md, dtype="bf16", fp8=fp8, device="cuda")
    assert lo is not None, "no se cargó el 2º experto (transformer_2): ¿falta en el repo?"
    r = BerniniRenderer(high=BerniniExpert(hi), low=BerniniExpert(lo))
    print(f"[t] carga de los 2 expertos reales (bf16, fp8={fp8}): {time.time() - t0:.1f}s")

    # text embeds ficticios [1,512,4096] (validación de forma/estabilidad, no de calidad)
    pos = torch.zeros(1, 512, 4096, device="cuda", dtype=torch.bfloat16)
    neg = torch.zeros(1, 512, 4096, device="cuda", dtype=torch.bfloat16)
    shape = L.latent_shape(num_frames=5, height=128, width=128)
    s = BerniniSampler(r, "t2v", dict(num_inference_steps=2), offload_experts=False)
    t1 = time.time()
    out = s.sample([], [], pos, neg, shape, device="cuda", seed=0,
                   base_scheduler_dir=f"{md}/scheduler")
    print(f"[t] forward multi-stream (2 steps, ambos expertos): {time.time() - t1:.1f}s")
    has_nan = bool(torch.isnan(out).any())
    print("salida:", tuple(out.shape), out.dtype, "NaN?", has_nan)
    assert out.shape[1] == 16 and not has_nan, "fallo de validación (shape o NaN)"
    print(f"[ok] smoke (bf16, fp8={fp8}) pasó — los DOS expertos cargan y el "
          f"forward multi-stream corre sin NaN")


def _make_test_landscape(path, w=848, h=848):
    """Paisaje sintético determinista (cielo azul + suelo + sol + nubes + árbol +
    casa) para validar i2i cualitativamente: tras 'make the sky a dramatic sunset'
    el CIELO debe cambiar y el resto de la composición (horizonte, suelo, árbol,
    casa, sol) conservarse. numpy/PIL solo se usan dentro (en el contenedor)."""
    import numpy as np
    from PIL import Image, ImageDraw
    horizon = int(h * 0.58)
    arr = np.zeros((h, w, 3), dtype=np.float32)
    ys = np.arange(h)
    t = (ys[:horizon] / max(horizon - 1, 1))[:, None]
    arr[:horizon] = np.stack([70 + 120 * t, 130 + 95 * t, 210 + 35 * t], axis=-1)[:, 0, :][:, None, :]
    tg = ((ys[horizon:] - horizon) / max(h - horizon - 1, 1))[:, None]
    arr[horizon:] = np.stack([70 - 25 * tg, 150 - 55 * tg, 60 - 25 * tg], axis=-1)[:, 0, :][:, None, :]
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(img)
    sx, sy, sr = int(w * 0.72), int(h * 0.18), int(w * 0.06)
    d.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=(255, 244, 200))           # sol
    for cx, cy, cw in [(0.18, 0.14, 0.14), (0.44, 0.09, 0.11), (0.33, 0.24, 0.10)]:  # nubes
        ex, ey, ew = int(w * cx), int(h * cy), int(w * cw)
        d.ellipse([ex, ey, ex + ew, ey + int(ew * 0.5)], fill=(245, 245, 250))
    tx = int(w * 0.27)                                                               # árbol
    d.rectangle([tx - 11, horizon - 8, tx + 11, horizon + 78], fill=(82, 53, 33))
    d.ellipse([tx - 70, horizon - 100, tx + 70, horizon + 28], fill=(34, 98, 46))
    hx, hy = int(w * 0.6), horizon                                                   # casa
    d.rectangle([hx, hy - 64, hx + 128, hy + 30], fill=(202, 182, 162))
    d.polygon([(hx - 12, hy - 64), (hx + 140, hy - 64), (hx + 64, hy - 118)], fill=(150, 70, 60))
    d.rectangle([hx + 48, hy - 18, hx + 80, hy + 30], fill=(92, 60, 40))
    img.save(path)


class _Comfy:
    """Cliente mínimo del servidor ComfyUI headless."""
    def __init__(self, port=8188):
        self.port = port
        self.proc = subprocess.Popen(
            ["python", "main.py", "--listen", "127.0.0.1", "--port", str(port),
             "--output-directory", OUT_DIR],
            cwd="/root/ComfyUI")
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=2)
                print("[ok] ComfyUI arriba"); return
            except Exception:
                time.sleep(2)
        raise RuntimeError("ComfyUI no arrancó")

    def run(self, graph: dict):
        data = json.dumps({"prompt": graph}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/prompt", data=data,
                                     headers={"Content-Type": "application/json"})
        pid = json.loads(urllib.request.urlopen(req).read())["prompt_id"]
        print("[*] prompt_id:", pid)
        while True:
            h = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/history/{pid}").read())
            if pid in h:
                return h[pid]
            time.sleep(2)


@app.function(image=image, gpu=GPU, volumes={MODELS_DIR: MODELS, OUT_DIR: OUT},
              timeout=60 * 60, container_idle_timeout=60, memory=CPU_MEM)
def run(workflow: str = "workflows/bernini_t2v.json", fp8: bool = False,
        num_frames: int = 0, width: int = 0, height: int = 0, steps: int = 0,
        gen_input: str = "", src_video: str = "", prompt: str = "",
        model_subdir: str = "Bernini-R-Diffusers"):
    """Ejecuta un workflow (formato API) en ComfyUI headless y guarda en el Volume.

    `workflow` es una ruta RELATIVA dentro del custom node (p.ej.
    workflows/bernini_t2v.json), que se copió a la imagen.

    Overrides opcionales (0 = mantener el valor del JSON) para abaratar la primera
    corrida: --fp8 (cuantización del loader; por defecto bf16 puro),
    --num-frames, --width, --height, --steps.

    --gen-input <nombre.png>: genera un paisaje sintético de prueba en
    ComfyUI/input/<nombre> (a la resolución efectiva del sampler) para los workflows
    de edición que usan LoadImage; deja una copia en el Volume out como
    i2i_input_used.png para comparar con el resultado.

    --src-video <archivo>: copia ese archivo DESDE el Volume out a ComfyUI/input
    y apunta el nodo BerniniRLoadVideo a él (para validar v2v/rv2v reusando un
    vídeo ya generado, p.ej. BerniniR_00001_.webp).
    --prompt "<texto>": sobreescribe el prompt del BerniniRTextEncode."""
    import shutil
    import time
    base = pathlib.Path("/root/ComfyUI/custom_nodes/ComfyUI-BerniniR")
    wf_path = pathlib.Path(workflow)
    if not wf_path.is_absolute():
        wf_path = base / workflow
    graph = json.loads(wf_path.read_text())
    # apunta el model_dir al Volume y aplica overrides por class_type
    for node in graph.values():
        ins = node.get("inputs", {})
        ct = node.get("class_type", "")
        if "model_dir" in ins:
            ins["model_dir"] = f"{MODELS_DIR}/{model_subdir}"
        if ct == "BerniniRModelLoader":
            # en Modal usamos los pesos del Volume, no descargamos de HF
            ins["source"] = "local"
            ins["auto_download"] = False
            ins["model_dir"] = f"{MODELS_DIR}/{model_subdir}"
            ins["fp8"] = bool(fp8)
        if ct == "BerniniRSampler":
            if num_frames:
                ins["num_frames"] = int(num_frames)
            if width:
                ins["width"] = int(width)
            if height:
                ins["height"] = int(height)
            if steps:
                ins["steps"] = int(steps)
        if ct == "BerniniRLoadVideo" and src_video:
            ins["video"] = pathlib.Path(src_video).name
        if ct == "BerniniRTextEncode" and prompt:
            ins["prompt"] = prompt

    # imagen de prueba para workflows de edición (LoadImage). Se genera a la
    # resolución efectiva del sampler para que el latente de referencia y el target
    # compartan rejilla espacial (el src-id distingue la MISMA posición por stream).
    if gen_input:
        samp = next((n["inputs"] for n in graph.values()
                     if n.get("class_type") == "BerniniRSampler"), {})
        eff_w, eff_h = int(samp.get("width", 848)), int(samp.get("height", 848))
        inp_dir = pathlib.Path("/root/ComfyUI/input")
        inp_dir.mkdir(parents=True, exist_ok=True)
        dst = inp_dir / gen_input
        _make_test_landscape(str(dst), eff_w, eff_h)
        shutil.copy(str(dst), f"{OUT_DIR}/i2i_input_used.png")
        print(f"[*] imagen de prueba {gen_input} {eff_w}x{eff_h} (copia en Volume out: i2i_input_used.png)")

    # vídeo fuente para v2v/rv2v: copia un archivo del Volume out -> ComfyUI/input
    # (el nodo BerniniRLoadVideo lo lee por nombre, ya parcheado arriba).
    if src_video:
        inp_dir = pathlib.Path("/root/ComfyUI/input")
        inp_dir.mkdir(parents=True, exist_ok=True)
        dst_v = inp_dir / pathlib.Path(src_video).name
        shutil.copy(f"{OUT_DIR}/{src_video}", str(dst_v))
        print(f"[*] source video {src_video} -> {dst_v}")

    # activa el reporte de pico de VRAM en los nodos (el subproceso de ComfyUI
    # hereda este env); nodes.py imprime torch.cuda.max_memory_allocated().
    os.environ["BERNINIR_REPORT_VRAM"] = "1"
    # device REAL (no el string GPU del módulo: en el contenedor se re-importa
    # app.py sin la env var BERNINIR_GPU, así que GPU caería al default y mentiría).
    try:
        import torch
        gpu_name = torch.cuda.get_device_name(0)
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"[*] GPU real: {gpu_name} ({gpu_total:.1f}GB)  (objetivo ≤24GB)")
    except Exception as e:
        print(f"[*] GPU real: ? ({e})")
    comfy = _Comfy()
    t0 = time.time()
    result = comfy.run(graph)
    OUT.commit()
    print(f"[t] workflow {workflow} (fp8={fp8}, frames={num_frames or 'json'}, "
          f"{(width or '?')}x{(height or '?')}, steps={steps or 'json'}): {time.time() - t0:.1f}s")
    print("[ok] terminado. Outputs:", json.dumps(result.get("outputs", {}))[:500])
    print("Descarga con:  modal volume get berninir-out / ./outputs")


@app.local_entrypoint()
def main(workflow: str = "workflows/bernini_t2v.json"):
    """Atajo: corre un workflow. (Lanza antes download_weights una vez.)"""
    run.remote(workflow=workflow)
