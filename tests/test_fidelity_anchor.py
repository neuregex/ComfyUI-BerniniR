# Copyright (c) 2026 — ComfyUI-BerniniR. Apache-2.0.
"""ANCLA DE FIDELIDAD (local, CPU): forward_streams(source_id=0) ≡ Wan2.2 stock.

La pieza más delicada del port es la rope: diffusers 0.35.2 aplica RoPE en
formato REAL cos/sin (pares interleaved), y Bernini la aplica COMPLEJA
(`view_as_complex`). El `_BerniniSelfAttnProcessor` reemplaza SOLO el processor
de self-attn (attn1) por la versión compleja.

Ancla (CLAUDE.md): con UN único stream target y `source_id=0`,
`visual_id_freqs[0] = exp(i·0) = 1` (fase identidad), de modo que la rope
compleja de Bernini DEBE ser numéricamente equivalente a la cos/sin de diffusers.
Por tanto:

  (A) WanTransformer3DModel.forward STOCK sobre un latente [1,16,2,8,8]
  (B) BerniniExpert.forward_streams con [Stream(latent, source_id=0, is_target=True)]
      des-empaquetado a espacial con L.to_spatial

deben coincidir con torch.allclose(atol=1e-3).

Si NO coinciden es el bug de FORMATO DE ROPE (riesgo C1): el test reporta el max
abs diff de la salida y, por bloque, dónde empieza a divergir (hooks forward).
Una divergencia ya en el bloque 0 apunta directo a attn1 / _apply_complex_rope.

Comparten el MISMO transformer (mismos pesos exactos): se calcula (A) ANTES de
envolver en BerniniExpert (que muta los processors de attn1 in-place) y (B) después.
"""
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from diffusers import WanTransformer3DModel          # noqa: E402

from bernini.model import BerniniExpert, Stream       # noqa: E402
from bernini import latents as L                       # noqa: E402


# experto DIMINUTO (dim = heads*head_dim = 4*32 = 128), idéntico al de wiring
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

SEED = 0
SHAPE = (1, 16, 2, 8, 8)   # 5 frames, 64x64 -> latente Wan
TIMESTEP = 500.0
ATOL = 1e-3


def _fixed_inputs():
    """Latente + texto + timestep deterministas (mismos para A y B)."""
    g = torch.Generator().manual_seed(1234)
    latent = torch.randn(*SHAPE, generator=g, dtype=torch.float32)
    text = torch.randn(1, 8, TINY_CFG["text_dim"], generator=g, dtype=torch.float32)
    timestep = torch.tensor([TIMESTEP], dtype=torch.float32)
    return latent, text, timestep


def _capture_blocks(blocks):
    """Hooks forward que guardan la salida (hidden) de cada bloque en float64."""
    store = []

    def hook(_module, _inp, out):
        store.append(out.detach().to(torch.float64))

    handles = [b.register_forward_hook(hook) for b in blocks]
    return store, handles


def test_fidelity_anchor_source_id_0():
    torch.manual_seed(SEED)
    t = WanTransformer3DModel(**TINY_CFG).eval().requires_grad_(False)
    latent, text, timestep = _fixed_inputs()

    # --- (A) forward STOCK de diffusers (procesadores stock, rope cos/sin real) ---
    store_a, h_a = _capture_blocks(t.blocks)
    with torch.no_grad():
        out_stock = t(hidden_states=latent, timestep=timestep,
                      encoder_hidden_states=text, return_dict=False)[0]
    for h in h_a:
        h.remove()
    blocks_a = list(store_a)

    # --- (B) BerniniExpert: attn1 -> rope COMPLEJA; 1 stream target src_id=0 -------
    # (envolver MUTA t.blocks[*].attn1 in-place; por eso (A) se calculó antes)
    expert = BerniniExpert(t)
    store_b, h_b = _capture_blocks(expert.t.blocks)
    out_packed = expert.forward_streams(
        [Stream(latent, source_id=0, is_target=True)], text, timestep)
    for h in h_b:
        h.remove()
    blocks_b = list(store_b)
    out_bernini = L.to_spatial(out_packed, SHAPE)

    # --- diagnóstico por bloque + salida final -----------------------------------
    per_block = []
    first_div = None
    for i, (a, b) in enumerate(zip(blocks_a, blocks_b)):
        d = (a - b).abs().max().item()
        per_block.append(d)
        if first_div is None and d > ATOL:
            first_div = i
    max_abs = (out_stock.double() - out_bernini.double()).abs().max().item()

    report = (
        "\n  visual_id_freqs[0] = exp(i·0) = 1 (fase identidad) => rope compleja ≡ cos/sin diffusers"
        f"\n  max abs diff salida final : {max_abs:.3e}  (atol={ATOL})"
        "\n  max abs diff por bloque   : "
        + ", ".join(f"b{i}={d:.2e}" for i, d in enumerate(per_block))
    )
    if first_div is not None:
        report += (
            f"\n  *** DIVERGE primero en el BLOQUE {first_div} => bug de FORMATO DE ROPE (C1): "
            "revisar _BerniniSelfAttnProcessor/_apply_complex_rope (layout [B,heads,S,hd] / "
            "freqs [1,1,S,hd/2] / emparejamiento interleaved) ***"
        )
    print(report)

    assert tuple(out_stock.shape) == tuple(out_bernini.shape), \
        f"shapes distintas: {tuple(out_stock.shape)} vs {tuple(out_bernini.shape)}"
    assert torch.allclose(out_stock, out_bernini, atol=ATOL), \
        f"ANCLA DE FIDELIDAD FALLÓ (riesgo C1: formato de rope).{report}"

    print(f"\n[OK] ancla de fidelidad: forward_streams(src_id=0) ≡ diffusers stock "
          f"(max abs diff {max_abs:.3e} < {ATOL}).")


if __name__ == "__main__":
    test_fidelity_anchor_source_id_0()
