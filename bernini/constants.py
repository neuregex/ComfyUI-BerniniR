# Copyright (c) 2026 — ComfyUI-BerniniR. Apache-2.0.
"""Constantes canónicas de Bernini-R (tomadas de bernini/cli.py y config.json).

Fuente de verdad: github.com/bytedance/Bernini  (las DEFAULTS del CLI ganan
sobre las defaults internas del pipeline)."""

# --- Defaults del modelo (config.json del repo Bernini-R) ---------------------
SWITCH_DIT_BOUNDARY = 0.875       # frontera high/low-noise (× num_train_timesteps)
MAX_SEQUENCE_LENGTH = 512         # tokens de texto (UMT5), zero-padded a 512
SHIFT = 3.0                       # flow shift EFECTIVO con UniPC (NO 5.0; ver nota)
USE_UNIPC = True
USE_SRC_ID_ROTARY_EMB = True
NUM_TRAIN_TIMESTEPS = 1000

# --- Defaults de generación (CLI; canónicos de cara al usuario) ---------------
DEFAULTS = dict(
    num_frames=81,
    num_inference_steps=40,
    max_image_size=848,
    height=480,
    width=848,
    omega_V=1.25,
    omega_I=4.5,
    omega_TI=4.0,
    omega_scale=0.8,     # escala los 3 omegas UNA vez, al cambiar a low-noise
    flow_shift=5.0,      # SOLO se aplica en el scheduler FlowMatch (no-UniPC). Ver nota.
    eta=0.5,             # peso de la componente paralela en APG
    norm_threshold=(50.0, 50.0, 50.0),
    momentum=0.0,
    seed=42,
    fps=16,
)

# NOTA IMPORTANTE sobre el shift:
#   Con use_unipc=True (default del modelo), UniPCMultistepScheduler fija su
#   flow_shift = config.shift = 3.0 en construcción, y el flow_shift=5.0 del CLI
#   queda MUERTO. Reproducimos esto: el shift efectivo por defecto es 3.0.

# --- Negative prompt estándar de Wan2.2 (cli.DEFAULT_NEG_PROMPT) --------------
DEFAULT_NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)

# --- System prompts por tarea (prompt_enhancer.SYSTEM_PROMPTS) -----------------
# OJO: en Bernini-R estos strings se CONCATENAN literalmente (sin espacio)
# delante del prompt positivo antes de codificar con UMT5; NO van en el negativo.
SYSTEM_PROMPTS = {
    "default": "You are a helpful assistant.",
    "t2i":  "You are a helpful assistant specialized in text-to-image generation.",
    "t2v":  "You are a helpful assistant specialized in text-to-video generation.",
    "i2i":  "You are a helpful assistant specialized in image editing.",
    "r2i":  "You are a helpful assistant specialized in subject-to-image generation.",
    "i2v":  "You are a helpful assistant specialized in image-to-video generation.",
    "v2v":  "You are a helpful assistant specialized in video editing.",
    "r2v":  "You are a helpful assistant specialized in subject-to-video generation.",
    "vi2v": "You are a helpful assistant specialized in video editing on content propagation.",
    "rv2v": "You are a helpful assistant specialized in video editing with reference.",
    "ads2v": "You are a helpful assistant specialized in ads insertion.",
    "vrc2v": "You are a helpful assistant for editing. You may need to adjust the subject's action or position.",
    "mv2v": ("You are a helpful assistant for editing. You might need to adjust the video's "
             "style, lighting, colors, textures, and the subject's pose or action."),
}


def get_system_prompt_for_task(task_type: str) -> str:
    return SYSTEM_PROMPTS.get(task_type, SYSTEM_PROMPTS["default"])


# --- Tipos de tarea y modos de guía -------------------------------------------
TASK_TYPES = ["t2i", "i2i", "t2v", "v2v", "mv2v", "rv2v", "r2v"]
GUIDANCE_MODES = ["rv2v", "v2v", "v2v_chain", "t2v", "r2v_apg", "v2v_apg", "t2v_apg"]

# guidance_mode recomendado por task_type (cuando el usuario no lo fija a mano).
TASK_TO_GUIDANCE = {
    "t2i": "t2v",        # t2i = t2v con num_frames=1
    "t2v": "t2v",
    "i2i": "v2v",        # imagen única -> stream VI, CFG sobre texto
    "v2v": "v2v",
    "mv2v": "v2v_apg",   # cambia movimiento del sujeto -> APG ayuda
    "rv2v": "rv2v",      # video + referencia(s) -> CFG encadenado de 4 forwards
    "r2v": "r2v_apg",    # solo referencias -> APG encadenado en x-space
}

# Para cada guidance_mode: qué "combos" de stream y texto se usan en cada forward,
# y cómo se combinan. Los combos posibles son:
#   none = solo target ;  V = (1er video)+target ;  I = imágenes+target ;
#   VI = videos+imágenes+target
# y el texto es "uncond" (neg) o "cond" (pos).  El plan exacto vive en sampler.py;
# esta tabla solo declara cuántos forwards y si usa APG (x-space).
GUIDANCE_PLAN = {
    "rv2v":     dict(n_fwd=4, apg=False),
    "v2v":      dict(n_fwd=2, apg=False),
    "v2v_chain": dict(n_fwd=3, apg=False),
    "t2v":      dict(n_fwd=2, apg=False),
    "r2v_apg":  dict(n_fwd=3, apg=True),
    "v2v_apg":  dict(n_fwd=2, apg=True),
    "t2v_apg":  dict(n_fwd=2, apg=True),
}
