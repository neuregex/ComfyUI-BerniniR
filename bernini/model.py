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

    def __init__(self, transformer, use_src_id_rotary_emb: bool = True):
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

    @property
    def dtype(self):
        return next(self.t.parameters()).dtype

    @property
    def device(self):
        return next(self.t.parameters()).device

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
        # condition_embedder: (temb, timestep_proj, enc_text, enc_img)
        ce = t.condition_embedder(timestep.to(self.device), text_emb.to(self.dtype), None)
        temb, timestep_proj = ce[0], ce[1]
        enc_text = ce[2] if len(ce) > 2 else text_emb
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        for block in t.blocks:
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
