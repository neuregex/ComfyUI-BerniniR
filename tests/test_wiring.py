# Copyright (c) 2026 — ComfyUI-BerniniR. Apache-2.0.
"""Test de *wiring* end-to-end en CPU con pesos aleatorios.

NO valida calidad ni fidelidad numérica contra Bernini-R (eso es el harness de
Modal con los pesos reales). Aquí solo comprobamos que el ENSAMBLADO está bien
conectado: dos `WanTransformer3DModel` diminutos -> `BerniniExpert` ->
`BerniniRenderer` -> `BerniniSampler` corren los modos de guía sin excepción,
emiten un latente espacial `[1, 16, T, H, W]` y no producen NaN.

Diseño:
  * Modelos diminutos (num_layers=2, heads=4, head_dim=32, ffn=64) con pesos
    aleatorios — caben en CPU y corren en segundos.
  * Entradas pequeñas: 5 frames, 64x64 -> latente (1,16,2,8,8); 1 vídeo fuente y
    2 imágenes de referencia; 2 pasos de denoising.
  * Tres modos representativos: `t2v` (ancla: solo target, CFG sobre texto),
    `rv2v` (4 forwards, CFG encadenado lineal sobre vídeo+referencias) y
    `r2v_apg` (APG encadenada en x-space con las referencias).

Se puede correr con pytest o directamente:  python tests/test_wiring.py
"""
import os
import sys

import torch

# raíz del repo en el path para `import bernini`
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from diffusers import WanTransformer3DModel          # noqa: E402

from bernini.model import BerniniExpert, BerniniRenderer  # noqa: E402
from bernini.sampler import BerniniSampler                # noqa: E402
from bernini import latents as L                          # noqa: E402


# ---- config de un experto DIMINUTO (dim = heads*head_dim = 4*32 = 128) -------
TINY_CFG = dict(
    patch_size=(1, 2, 2),
    num_attention_heads=4,
    attention_head_dim=32,
    in_channels=16,
    out_channels=16,
    text_dim=4096,
    freq_dim=256,
    ffn_dim=64,
    num_layers=2,
    cross_attn_norm=True,
    qk_norm="rms_norm_across_heads",
    eps=1e-6,
    rope_max_seq_len=1024,
)

# 5 frames, 64x64 -> latente Wan (1, 16, T_lat=2, H/8=8, W/8=8)
NUM_FRAMES, HEIGHT, WIDTH = 5, 64, 64
DEVICE = "cpu"
MODES = ["t2v", "rv2v", "r2v_apg"]


def _tiny_transformer(seed: int) -> WanTransformer3DModel:
    torch.manual_seed(seed)
    m = WanTransformer3DModel(**TINY_CFG)
    return m.eval().requires_grad_(False)


def build_renderer() -> BerniniRenderer:
    """Dos expertos (high/low-noise) con pesos aleatorios distintos."""
    hi = BerniniExpert(_tiny_transformer(0))
    lo = BerniniExpert(_tiny_transformer(1))
    return BerniniRenderer(high=hi, low=lo)


def _make_inputs():
    target_shape = L.latent_shape(num_frames=NUM_FRAMES, height=HEIGHT, width=WIDTH)
    g = torch.Generator(device=DEVICE).manual_seed(1234)
    # vídeo fuente: mismo layout temporal que el target (T_lat=2)
    video = torch.randn(1, 16, target_shape[2], target_shape[3], target_shape[4],
                        generator=g, dtype=torch.float32)
    # imágenes de referencia: un solo fotograma (T_lat=1)
    img0 = torch.randn(1, 16, 1, target_shape[3], target_shape[4], generator=g, dtype=torch.float32)
    img1 = torch.randn(1, 16, 1, target_shape[3], target_shape[4], generator=g, dtype=torch.float32)
    # embeds de texto ficticios [1, S_txt, text_dim] (solo importa la forma)
    pos = torch.randn(1, 16, TINY_CFG["text_dim"], generator=g, dtype=torch.float32)
    neg = torch.randn(1, 16, TINY_CFG["text_dim"], generator=g, dtype=torch.float32)
    return target_shape, [video], [img0, img1], pos, neg


def _run_mode(renderer: BerniniRenderer, mode: str) -> torch.Tensor:
    target_shape, vids, imgs, pos, neg = _make_inputs()
    sampler = BerniniSampler(renderer, mode, dict(num_inference_steps=2), offload_experts=False)
    out = sampler.sample(vids, imgs, pos, neg, target_shape, device=DEVICE, seed=0)

    assert isinstance(out, torch.Tensor), f"[{mode}] salida no es Tensor"
    assert tuple(out.shape) == tuple(target_shape), \
        f"[{mode}] shape {tuple(out.shape)} != esperado {tuple(target_shape)}"
    assert out.ndim == 5 and out.shape[0] == 1 and out.shape[1] == 16, \
        f"[{mode}] no es [1, 16, T, H, W]: {tuple(out.shape)}"
    assert not torch.isnan(out).any(), f"[{mode}] la salida contiene NaN"
    assert torch.isfinite(out).all(), f"[{mode}] la salida contiene Inf"
    return out


def test_wiring():
    """Corre los 3 modos sobre el mismo renderer y valida shape + finitud."""
    renderer = build_renderer()
    for mode in MODES:
        out = _run_mode(renderer, mode)
        print(f"[ok] {mode:>8}: salida {tuple(out.shape)} {out.dtype} "
              f"min={out.min().item():+.3f} max={out.max().item():+.3f}")


if __name__ == "__main__":
    test_wiring()
    print("\n[OK] wiring t2v / rv2v / r2v_apg pasó (shape [1,16,2,8,8], sin NaN).")
