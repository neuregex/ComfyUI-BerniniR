# ComfyUI-BerniniR

[![Comfy Registry](https://img.shields.io/badge/Comfy_Registry-comfyui--berninir-1971c2)](https://registry.comfy.org/nodes/comfyui-berninir)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Soporte para **[ByteDance/Bernini-R](https://huggingface.co/ByteDance/Bernini-R)** en ComfyUI — texto-a-vídeo / texto-a-imagen, **edición** de imagen y vídeo, y **referencia-a-vídeo**, con la lógica propia de Bernini (source-id RoPE + guía APG multi-condición) reimplementada fielmente.

> **El hallazgo clave:** Bernini-R **es** Wan2.2-T2V-A14B por dentro. Su `transformer/config.json` declara `"_class_name": "WanTransformer3DModel"` con la config idéntica de A14B (40 capas, 40 heads, ffn 13824, 16 canales), VAE `AutoencoderKLWan` y text encoder UMT5. Las **claves de pesos son 100% Wan estándar** — no hay tensores extra. Todo lo que distingue a Bernini vive en el **código de inferencia**, no en parámetros nuevos. Este paquete reimplementa ese código sobre los módulos de `diffusers` ya probados.

---

## ¿Qué es exactamente lo "Bernini" (y dónde lo reproduce este repo)?

| Mecanismo de Bernini | Qué hace | Dónde, en este repo |
|---|---|---|
| **source-id RoPE** | Cada "stream" visual (target, vídeos, referencias) lleva un `source_id` entero; su rejilla RoPE se multiplica por una fase compleja constante `visual_id_freqs[source_id]`. Distingue la misma posición espacial entre streams sin offsets ni canales extra. | `bernini/rope.py` |
| **Concatenación de streams** (no de canales) | El modelo sigue siendo de 16 canales. Cada condición se *patch-embedea* en su propio bloque de tokens y se concatena **antes** del target en el eje de secuencia; una máscara conserva solo la salida del target. | `bernini/model.py` |
| **Switch dual-expert** | high-noise (`transformer`) si `t ≥ 875`, low-noise (`transformer_2`) si `t < 875` — por **valor de timestep**. Al cambiar, los omegas se escalan ×0.8 una vez. | `bernini/model.py`, `bernini/sampler.py` |
| **7 modos de guía** | `rv2v` (4 fwd), `v2v` (2), `v2v_chain` (3), `t2v` (2), `r2v_apg` (3), `v2v_apg` (2), `t2v_apg` (2). Los `*_apg` hacen **Adaptive Projected Guidance** en x-space. | `bernini/sampler.py`, `bernini/guidance.py` |
| **APG** | Proyección orto/paralela del diff de guía, reducida sobre `{C,H,W}` **por frame** (no sobre T), en float64, con momentum persistente entre pasos. | `bernini/guidance.py` |
| **Scheduler** | UniPCMultistepScheduler con `flow_shift = 3.0` (¡el `flow_shift=5.0` del CLI es código muerto en el path UniPC por defecto!). | `bernini/sampler.py`, `bernini/constants.py` |
| **Texto** | UMT5, prefijo de system-prompt por tarea concatenado al prompt positivo, padding a 512. | `bernini/loader.py`, `bernini/constants.py` |

**Ancla de validación:** con un único stream y `source_id=0`, `visual_id_freqs[0]=1` (fase identidad) ⇒ el forward coincide **exactamente** con Wan2.2 estándar. Por eso `t2v`/`t2i` es el camino más seguro para verificar primero.

---

## Instalación

```bash
cd ComfyUI/custom_nodes
git clone <tu-fork>/ComfyUI-BerniniR
pip install -r ComfyUI-BerniniR/requirements.txt
# (torch lo provee ComfyUI; no lo reinstales)
```

## Pesos

El repo **diffusers self-contained** trae VAE + UMT5 + tokenizer + los dos transformers — nada más que descargar:

```bash
pip install -U huggingface_hub
hf download ByteDance/Bernini-R-Diffusers --local-dir Bernini-R-Diffusers
```

Apunta el campo `model_dir` de los nodos a esa carpeta (puede ser ruta absoluta).

---

## Uso (nodos)

Pipeline mínima **t2v**:

```
BerniniR · Load Model ─┐
BerniniR · Load VAE ───┤
BerniniR · Text Encode ┴─► BerniniR · Sampler ─► BerniniR · VAE Decode ─► SaveAnimatedWEBP
```

Para edición/referencia añade **BerniniR · Encode Source/Reference** (recibe `source_video` y/o `reference_images`) y conéctalo a la entrada `src` del sampler. El `task_type` del Text Encode autoselecciona el `guidance_mode` (puedes forzarlo).

Workflows de ejemplo (formato **API**, listos para `/prompt` o Modal) en [`workflows/`](workflows/):
`bernini_t2v`, `bernini_t2i`, `bernini_i2i`, `bernini_v2v`, `bernini_rv2v`, `bernini_r2v`.

> Los workflows de vídeo (`v2v`, `rv2v`) usan `VHS_LoadVideo` de **ComfyUI-VideoHelperSuite** para cargar frames. Instálalo o sustituye por tu loader de frames preferido.

---

## VRAM y cuantización (≤24GB y menos)

Son **2×14B** (~56GB en bf16 solo transformers). Estrategias, de más a menos VRAM:

| Config | VRAM aprox. (vídeo 480p) | Cómo |
|---|---|---|
| bf16 + offload secuencial | ~30 GB pico | `dtype=bf16`, `offload_experts=True` (default). Un experto en GPU a la vez. |
| **fp8 + offload** | **~16–22 GB** | `fp8=True` (default). Cada experto ~14 GB; el inactivo va a CPU. **Objetivo 24GB.** |
| fp8 + offload, **t2i/i2i** (1 frame) | ~10–14 GB | Imágenes: la secuencia es mucho más corta. Cómodo en 16GB e incluso 12GB. |
| GGUF Q4/Q5 (avanzado) | ~8–12 GB | Solo path `t2v`/`t2i` vía ComfyUI nativo, ver abajo. |

Notas:
- `fp8` aquí almacena los pesos lineales en `float8_e4m3fn` y hace *upcast* a bf16 en cada forward (más lento, mitad de VRAM). Es la palanca principal para 24GB.
- `offload_experts=True` replica el comportamiento de Bernini (mueve el experto high-noise a CPU al cambiar al low-noise).
- Baja `num_frames` y resolución para recortar memoria de activaciones (la atención crece con la longitud de secuencia, y aquí la secuencia incluye los tokens de condición).

### Camino GGUF / ComfyUI-nativo (sub-16GB, solo t2v/t2i por ahora)

Como `t2v`/`t2i` con `source_id=0` es **idéntico a Wan2.2 estándar**, puedes correrlo con los nodos nativos de ComfyUI (que ya soportan fp8 y GGUF) tras convertir los pesos a claves nativas:

```bash
python tools/convert_bernini_to_comfy.py --repo Bernini-R-Diffusers --out-dir comfy_out --dtype fp8_e4m3fn
# -> comfy_out/bernini_r_high_noise_14B_fp8_e4m3fn.safetensors  (+ low_noise)
# copia a ComfyUI/models/diffusion_models/ y carga con dos "Load Diffusion Model"
```

Para GGUF, pasa los `.safetensors` nativos por [`city96/ComfyUI-GGUF`](https://github.com/city96/ComfyUI-GGUF) `tools/convert.py` (solo acepta formato nativo: por eso convertimos antes) y cárgalos con `UnetLoaderGGUF`. El mapeo de claves del conversor está verificado contra el `convert_wan_to_diffusers.py` oficial de 🤗 diffusers (20/20 casos en el test unitario). *Las tareas de edición/referencia requieren el backend de este paquete (fp8+offload), no el path GGUF.*

---

## Validar en Modal (GPU H100 serverless)

Los pesos pesan ~160GB y el modelo de 28B pide GPU clase H100/A100-80GB, así que el end-to-end se valida en la nube. Harness incluido en [`modal/app.py`](modal/app.py):

```bash
pip install modal && modal token new          # una vez

# 1) baja los pesos al Volume de Modal (~126GB, una vez)
modal run modal/app.py::download_weights

# 2) validación numérica barata: 1 forward multi-stream end-to-end (shapes + NaN)
modal run modal/app.py::smoke

# 3) ejecuta un workflow completo headless y guarda el vídeo en el Volume
modal run modal/app.py::run --workflow workflows/bernini_t2v.json
modal run modal/app.py::run --workflow workflows/bernini_rv2v.json

# 4) descarga resultados
modal volume get berninir-out / ./outputs
```

El harness construye una imagen CUDA 12.4 + torch 2.5.1 (la pila de Bernini), clona ComfyUI, monta este custom node, levanta el server headless y postea el workflow al endpoint `/prompt`. Para producción (endpoint unificado, GPU snapshots, R2/CDN) ver la skill `modal-comfyui-deploy`.

---

## Estado de validación (honesto)

Este puerto está construido contra el **código fuente verbatim** de Bernini (github.com/bytedance/Bernini) y el **código real** de `diffusers==0.35.2` y del ComfyUI actual, verificado estáticamente y con un test unitario del conversor (20/20). **No** se ha podido ejecutar el modelo de 160GB end-to-end aquí — eso es justo para lo que sirve el harness de Modal. Detalle de diseño ya resuelto que conviene conocer: diffusers 0.35.2 aplica RoPE en formato **cos/sin real**, pero Bernini la usa **compleja** (necesario para la fase src-id); por eso reemplazamos el *processor* de self-attn de cada bloque por uno complejo (`bernini/model.py::_BerniniSelfAttnProcessor`), dejando cross-attn/FFN/modulación stock.

En tu primer run en Modal, verifica estos puntos (de mayor a menor riesgo):

1. **Ancla de identidad `t2v`** — con `source_id=0`, `visual_id_freqs[0]=1`, así que `t2v` debe coincidir con Wan2.2 estándar usando los mismos pesos. Es el primer test: si `t2v` sale bien pero la edición no, el sospechoso es la fase `visual_id_freqs` o el ensamblado de streams, no la base.
2. **Layout del rope complejo en atención** — el processor transpone a `[B,heads,S,hd]` y multiplica por `freqs [1,1,S,hd/2]`. Confirma (con un print de shapes en el primer bloque) que no hay desalineación heads/seq.
3. **Normalización del VAE** — convención Bernini `(x-mean)/std`. Verifica que encode→decode de un latente sin ruido reconstruye la entrada (si ComfyUI ya normaliza internamente, evita doble normalización).
4. **Unidades del timestep para el switch** — el boundary compara `t` (unidades 0–1000 de UniPC) contra 875; confirma que `scheduler.timesteps` viene en esas unidades.
5. **APG numérico** — `*_apg` corre en float64 reduciendo sobre {C,H,W} por frame; compara una salida `t2v` vs `t2v_apg` para sanity.

Verificado aquí: sintaxis de los 11 módulos, test del conversor (20/20 incl. swap norm2↔norm3 y atención), y la firma de bloque/`condition_embedder` contra `diffusers==0.35.2`.

---

## Créditos y licencia

- Modelo y algoritmo: **Bernini: Latent Semantic Planning for Video Diffusion**, ByteDance ([arXiv:2605.22344](https://arxiv.org/abs/2605.22344), [código](https://github.com/bytedance/Bernini)). Apache-2.0.
- Base: [Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B).
- Mapeo de claves derivado del `convert_wan_to_diffusers.py` de 🤗 diffusers.

Este paquete: **Apache-2.0**. No incluye pesos.
