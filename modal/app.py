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

# Imagen: CUDA 12.4 + torch 2.5.1 (alineado con la pila de Bernini-R) + ComfyUI
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", index_url="https://download.pytorch.org/whl/cu124")
    .run_commands("git clone --depth 1 https://github.com/comfyanonymous/ComfyUI /root/ComfyUI")
    .run_commands("pip install -r /root/ComfyUI/requirements.txt")
    .pip_install("diffusers>=0.35.2", "transformers>=4.57.0", "accelerate>=0.34.2",
                 "einops>=0.7.0", "ftfy>=6.1.0", "safetensors>=0.4.0",
                 "sentencepiece>=0.2.0", "huggingface_hub", "requests")
    # monta el custom node (esta misma carpeta) en ComfyUI/custom_nodes
    .add_local_dir(str(HERE), "/root/ComfyUI/custom_nodes/ComfyUI-BerniniR", copy=True,
                   ignore=["modal", ".git", "__pycache__"])
)


@app.function(image=image, volumes={MODELS_DIR: MODELS}, timeout=60 * 60 * 4)
def download_weights(repo: str = "ByteDance/Bernini-R-Diffusers"):
    """Descarga el repo diffusers self-contained (vae+t5+tokenizer+2 transformers)."""
    from huggingface_hub import snapshot_download
    dst = f"{MODELS_DIR}/Bernini-R-Diffusers"
    print(f"[*] descargando {repo} -> {dst}")
    snapshot_download(repo_id=repo, local_dir=dst, max_workers=8)
    MODELS.commit()
    print("[ok] pesos en el Volume")


@app.function(image=image, gpu="H100", volumes={MODELS_DIR: MODELS}, timeout=60 * 30)
def smoke():
    """Validación numérica barata: carga UN experto (fp8), 1 forward t2v de
    baja resolución, y comprueba shapes y ausencia de NaN. NO requiere los dos
    expertos ni descargar 160GB completos si solo bajaste 'transformer'."""
    import torch
    import sys
    sys.path.insert(0, "/root/ComfyUI/custom_nodes/ComfyUI-BerniniR")
    from bernini import load_experts, BerniniRenderer, BerniniExpert, BerniniSampler
    from bernini import latents as L

    md = f"{MODELS_DIR}/Bernini-R-Diffusers"
    hi, lo = load_experts(md, dtype="bf16", fp8=True)
    r = BerniniRenderer(high=BerniniExpert(hi), low=BerniniExpert(lo) if lo else None)
    r.high.t.to("cuda")

    # text embeds ficticios [1,512,4096] (validación de forma, no de calidad)
    pos = torch.zeros(1, 512, 4096, device="cuda", dtype=torch.bfloat16)
    neg = torch.zeros(1, 512, 4096, device="cuda", dtype=torch.bfloat16)
    shape = L.latent_shape(num_frames=5, height=128, width=128)
    s = BerniniSampler(r, "t2v", dict(num_inference_steps=2), offload_experts=False)
    out = s.sample([], [], pos, neg, shape, device="cuda", seed=0,
                   base_scheduler_dir=f"{md}/scheduler")
    print("salida:", tuple(out.shape), out.dtype, "NaN?", bool(torch.isnan(out).any()))
    assert out.shape[1] == 16 and not torch.isnan(out).any(), "fallo de validación"
    print("[ok] smoke test pasó — el forward multi-stream + sampler corre end-to-end")


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


@app.function(image=image, gpu="H100", volumes={MODELS_DIR: MODELS, OUT_DIR: OUT}, timeout=60 * 60)
def run(workflow: str = "workflows/bernini_t2v.json"):
    """Ejecuta un workflow (formato API) en ComfyUI headless y guarda en el Volume.

    `workflow` es una ruta RELATIVA dentro del custom node (p.ej.
    workflows/bernini_t2v.json), que se copió a la imagen."""
    base = pathlib.Path("/root/ComfyUI/custom_nodes/ComfyUI-BerniniR")
    wf_path = pathlib.Path(workflow)
    if not wf_path.is_absolute():
        wf_path = base / workflow
    graph = json.loads(wf_path.read_text())
    # apunta el model_dir de los nodos al Volume
    for node in graph.values():
        ins = node.get("inputs", {})
        if "model_dir" in ins:
            ins["model_dir"] = f"{MODELS_DIR}/Bernini-R-Diffusers"
    comfy = _Comfy()
    result = comfy.run(graph)
    OUT.commit()
    print("[ok] terminado. Outputs:", json.dumps(result.get("outputs", {}))[:500])
    print("Descarga con:  modal volume get berninir-out / ./outputs")


@app.local_entrypoint()
def main(workflow: str = "workflows/bernini_t2v.json"):
    """Atajo: corre un workflow. (Lanza antes download_weights una vez.)"""
    run.remote(workflow=workflow)
