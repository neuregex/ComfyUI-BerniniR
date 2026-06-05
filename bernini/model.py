# Copyright (c) 2026 — ComfyUI-BerniniR. Apache-2.0.
"""Backend fiel de Bernini-R sobre módulos *diffusers* ya cargados.

Estrategia de MÍNIMO RIESGO: en vez de reimplementar el transformer Wan,
reutilizamos `diffusers.WanTransformer3DModel` (patch_embedding, los 40 bloques,
condition_embedder y la cabeza son código probado). Solo añadimos por encima la
lógica específica de Bernini, que NO toca pesos:

  1. patch-embed de CADA stream por separado y su rope con `source_id`,
  2. concatenación de tokens de condición ANTES del target (eje de secuencia),
  3. self-attention densa sobre la secuencia concatenada (equiv. a la varlen de
     Bernini para batch=1) — la hace el propio bloque diffusers al pasarle la
     rope concatenada,
  4. máscara para quedarnos solo con los tokens del target a la salida.

Anclaje de validación: con UN stream y source_id=0, `visual_id_freqs[0]`=1
(fase identidad) ⇒ el forward coincide EXACTAMENTE con Wan2.2 estándar.

Pin: probado contra la pila de Bernini-R (`diffusers==0.35.2`).
"""
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F

from .rope import WanRotaryPosEmbed
from . import constants as C


# --------------------------------------------------------------------------
# Processor de SELF-ATTENTION con rope COMPLEJA de Bernini.
#
# diffusers 0.35.2 aplica RoPE en formato cos/sin real (tupla freqs_cos/sin).
# Bernini usa rope COMPLEJA (view_as_complex) — necesaria para que la fase
# src-id (multiplicación compleja) sea exacta. Reemplazamos SOLO el processor
# de attn1 (self-attn) de cada bloque; cross-attn y FFN siguen stock.
# Réplica de diffusers WanAttnProcessor con la rotación compleja de Bernini.
# --------------------------------------------------------------------------
def _apply_complex_rope(x, freqs):
    """x: [B, heads, S, head_dim] ; freqs (complejo): [1,1,S,head_dim/2]."""
    x_c = torch.view_as_complex(x.to(torch.float64).unflatten(3, (-1, 2)))   # [B,heads,S,hd/2]
    x_o = torch.view_as_real(x_c * freqs).flatten(3, 4)                        # [B,heads,S,hd]
    return x_o.type_as(x)


class _BerniniSelfAttnProcessor:
    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, rotary_emb=None, **kwargs):
        q = attn.norm_q(attn.to_q(hidden_states))
        k = attn.norm_k(attn.to_k(hidden_states))
        v = attn.to_v(hidden_states)
        q = q.unflatten(2, (attn.heads, -1)).transpose(1, 2)   # [B,heads,S,hd]
        k = k.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        v = v.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        if rotary_emb is not None:
            q = _apply_complex_rope(q, rotary_emb)
            k = _apply_complex_rope(k, rotary_emb)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask,
                                             dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).flatten(2, 3).type_as(hidden_states)         # [B,S,heads*hd]
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


@dataclass
class Stream:
    """Un stream visual a concatenar en la secuencia del transformer."""
    latent: torch.Tensor   # [1, C=16, T, H, W] (latente VAE normalizado)
    source_id: int         # 0 = target (ruido); >0 = vídeo/imagen fuente
    is_target: bool = False


class BerniniExpert(torch.nn.Module):
    """Envoltorio de UN experto (high- o low-noise) = WanTransformer3DModel."""

    def __init__(self, transformer, use_src_id_rotary_emb: bool = True, block_swap: int = 0):
        super().__init__()
        self.t = transformer                     # diffusers WanTransformer3DModel
        cfg = transformer.config
        self.patch_size = tuple(cfg.patch_size)  # (1,2,2)
        self.head_dim = cfg.attention_head_dim   # 128
        self.rope = WanRotaryPosEmbed(
            attention_head_dim=self.head_dim,
            patch_size=self.patch_size,
            max_seq_len=cfg.rope_max_seq_len,    # 1024
            theta=10000.0,
            use_src_id_rotary_emb=use_src_id_rotary_emb,
        )
        # Sustituye el processor de self-attn de cada bloque por el de Bernini
        # (rope compleja). Cross-attn (attn2) se queda con el de diffusers.
        proc = _BerniniSelfAttnProcessor()
        for blk in self.t.blocks:
            blk.attn1.set_processor(proc)
        # block-swap: los ÚLTIMOS `block_swap` de los N bloques viven en CPU y se
        # streamean a GPU durante su propio forward (el resto quedan residentes en
        # GPU). 0 = sin swap (comportamiento actual). Baja el pico de VRAM a costa
        # de transferencias por bloque (más lento). NO cambia la matemática.
        n = len(self.t.blocks)
        self.block_swap = max(0, min(int(block_swap), n))
        self._swap_idx = frozenset(range(n - self.block_swap, n))

    # -- colocación con block-swap / offload -----------------------------------
    def to_active(self, device):
        """Deja el experto listo para inferir en `device`. Con block_swap>0 SOLO los
        bloques residentes + las partes no-bloque (patch_embedding, condition_embedder,
        norm_out, proj_out, scale_shift_table, rope) van a `device`; los bloques swap
        quedan en CPU (se streamean en el forward) — sin pico de experto-completo."""
        if not self.block_swap:
            self.t.to(device)
            return
        for i, blk in enumerate(self.t.blocks):
            blk.to("cpu" if i in self._swap_idx else device)
        for name, mod in self.t.named_children():
            if name != "blocks":
                mod.to(device)
        if hasattr(self.t, "scale_shift_table"):
            self.t.scale_shift_table.data = self.t.scale_shift_table.data.to(device)

    def to_idle(self):
        """Manda TODO el experto (residentes + swap + no-bloque) a CPU (inactivo)."""
        self.t.to("cpu")

    @property
    def dtype(self):
        # dtype de CÓMPUTO/IO = el de la conv de entrada (patch_embedding), NO
        # next(parameters()): al cargar bf16, diffusers mantiene varios módulos en
        # fp32 (_keep_in_fp32_modules: norms, time_embedder, scale_shift_table) y el
        # primer parámetro iterado puede ser uno de esos fp32, lo que provocaría
        # castear el latente a fp32 y chocar con la patch_embedding bf16.
        return self.t.patch_embedding.weight.dtype

    @property
    def device(self):
        return self.t.patch_embedding.weight.device

    # -- patch-embed de un stream -> (tokens [1,S,dim], rope [1,1,S,head/2]) ----
    def patch_stream(self, latent: torch.Tensor, source_id: int):
        latent = latent.to(self.dtype)
        rotary = self.rope(latent, source_id)                 # [1,1,S,head/2]
        x = self.t.patch_embedding(latent)                    # Conv3d -> [1,dim,T,H',W']
        x = x.flatten(2).transpose(1, 2)                      # [1, S, dim]
        return x, rotary

    # -- forward sobre una LISTA de streams (combo) ----------------------------
    @torch.no_grad()
    def forward_streams(self, streams: List[Stream], text_emb: torch.Tensor,
                        timestep: torch.Tensor) -> torch.Tensor:
        """Devuelve el ruido predicho SOLO para el target, en formato PACKED
        [1, S_target, 64] — el mismo layout que el latente del scheduler
        (proj_out emite p_t·p_h·p_w·c = 1·2·2·16 = 64 por token), igual que el
        `_fwd` de Bernini. A espacial se pasa en sampler.py (APG / decode)."""
        toks, ropes, masks = [], [], []
        for s in streams:
            x, r = self.patch_stream(s.latent, s.source_id)
            toks.append(x)
            ropes.append(r)
            m = torch.ones(x.shape[1], dtype=torch.bool, device=x.device) if s.is_target \
                else torch.zeros(x.shape[1], dtype=torch.bool, device=x.device)
            masks.append(m)

        hidden = torch.cat(toks, dim=1)                        # [1, S_total, dim]
        rotary = torch.cat(ropes, dim=2)                       # [1,1,S_total,head/2]
        mask = torch.cat(masks, dim=0)                         # [S_total]

        out = self._run_blocks(hidden, rotary, text_emb, timestep)   # [1,S_total,64]
        return out[:, mask, :]                                 # solo target, packed

    # -- réplica del cuerpo de WanTransformer3DModel.forward (post patch-embed) -
    def _run_blocks(self, hidden, rotary, text_emb, timestep):
        t = self.t
        dev = self.device                                   # device residente (patch_embedding)
        # condition_embedder: (temb, timestep_proj, enc_text, enc_img)
        ce = t.condition_embedder(timestep.to(dev), text_emb.to(self.dtype), None)
        temb, timestep_proj = ce[0], ce[1]
        enc_text = ce[2] if len(ce) > 2 else text_emb
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        # block-swap: el bloque DEBE estar en `dev` antes de su forward (el parche fp8
        # hace weight.to(x.dtype) sobre el peso del bloque, así que device debe coincidir).
        # Activaciones (hidden/rotary/enc_text/timestep_proj) viven en `dev`.
        swap = self._swap_idx if self.block_swap else frozenset()
        for i, block in enumerate(t.blocks):
            if i in swap:
                block.to(dev, non_blocking=True)
                hidden = block(hidden, enc_text, timestep_proj, rotary)
                block.to("cpu", non_blocking=True)
            else:
                hidden = block(hidden, enc_text, timestep_proj, rotary)

        shift, scale = (t.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden = (t.norm_out(hidden.float()) * (1 + scale) + shift).type_as(hidden)
        hidden = t.proj_out(hidden)
        return hidden

    def _unpatchify(self, tokens, shape):
        p_t, p_h, p_w = self.patch_size
        _, c, t_lat, h_lat, w_lat = shape
        ppt, pph, ppw = t_lat // p_t, h_lat // p_h, w_lat // p_w
        x = tokens.reshape(1, ppt, pph, ppw, p_t, p_h, p_w, c)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        x = x.flatten(6, 7).flatten(4, 5).flatten(2, 3)        # [1, C, T, H, W]
        return x


# --------------------------------------------------------------------------
# Renderer = dos expertos + frontera de cambio
# --------------------------------------------------------------------------
@dataclass
class BerniniRenderer:
    high: BerniniExpert                 # transformer  (high-noise)
    low: Optional[BerniniExpert] = None  # transformer_2 (low-noise)
    switch_dit_boundary: float = C.SWITCH_DIT_BOUNDARY
    num_train_timesteps: int = C.NUM_TRAIN_TIMESTEPS

    def boundary_timestep(self) -> float:
        return self.switch_dit_boundary * self.num_train_timesteps

    def expert_for_timestep(self, t: float) -> BerniniExpert:
        """high-noise para t >= 875; low-noise para t < 875 (por VALOR de t)."""
        if t >= self.boundary_timestep() or self.low is None:
            return self.high
        return self.low
