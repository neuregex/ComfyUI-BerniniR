# Copyright (c) 2026 — ComfyUI-BerniniR. Apache-2.0.
"""Rotary position embeddings con **source-id** (el corazón de Bernini-R).

Port fiel de bernini/models/transformer_wan.py::WanRotaryPosEmbed.

Idea: la rejilla espacio-temporal 3D es IDÉNTICA a Wan estándar. Lo nuevo es
que cada "stream" visual (target, vídeos fuente, imágenes de referencia) lleva
un entero `source_id`, y su rejilla compleja se MULTIPLICA por un fasor
constante `visual_id_freqs[source_id]`. Eso rota por igual todos los tokens del
stream, haciendo distinguibles la MISMA posición espacial en streams distintos
sin offsets de posición ni canal extra.
"""
import torch

try:
    # diffusers está presente en cualquier entorno ComfyUI moderno y es dep de
    # Bernini; usamos su implementación canónica de la tabla 1-D.
    from diffusers.models.embeddings import get_1d_rotary_pos_embed
except Exception:  # pragma: no cover - fallback mínimo equivalente
    def get_1d_rotary_pos_embed(dim, pos, theta=10000.0, use_real=False,
                                repeat_interleave_real=False, freqs_dtype=torch.float64):
        if isinstance(pos, int):
            pos = torch.arange(pos)
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype)[: dim // 2] / dim))
        freqs = torch.outer(pos.to(freqs_dtype), freqs)          # [P, dim/2]
        return torch.polar(torch.ones_like(freqs), freqs)        # complejo [P, dim/2]


def _apply_rotary_emb(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Aplica RoPE compleja (igual que Wan/Bernini)."""
    x_rotated = torch.view_as_complex(x.to(torch.float64).unflatten(3, (-1, 2)))
    x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
    return x_out.type_as(x)


class WanRotaryPosEmbed(torch.nn.Module):
    def __init__(self, attention_head_dim: int, patch_size, max_seq_len: int,
                 theta: float = 10000.0, use_src_id_rotary_emb: bool = False):
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        freqs = []
        for dim in [t_dim, h_dim, w_dim]:
            freq = get_1d_rotary_pos_embed(
                dim, max_seq_len, theta, use_real=False,
                repeat_interleave_real=False, freqs_dtype=torch.float64,
            )
            freqs.append(freq)
        self.freqs = torch.cat(freqs, dim=1)

        self.use_src_id_rotary_emb = use_src_id_rotary_emb
        if use_src_id_rotary_emb:
            self.visual_id_freqs = get_1d_rotary_pos_embed(
                attention_head_dim, max_seq_len, theta, use_real=False,
                repeat_interleave_real=False, freqs_dtype=torch.float64,
            )

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, source_id=None) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        self.freqs = self.freqs.to(hidden_states.device)
        freqs = self.freqs.split_with_sizes(
            [
                self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
                self.attention_head_dim // 6,
                self.attention_head_dim // 6,
            ],
            dim=1,
        )

        freqs_f = freqs[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = freqs[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w = freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)

        if self.use_src_id_rotary_emb:
            assert source_id is not None, "source_id es obligatorio con use_src_id_rotary_emb=True"
            self.visual_id_freqs = self.visual_id_freqs.to(hidden_states.device)
            freqs_visual_id = self.visual_id_freqs[source_id: source_id + 1]
            freqs_visual_id = freqs_visual_id.view(1, 1, 1, -1).expand(ppf, pph, ppw, -1)
            freqs_visual_id = freqs_visual_id.reshape(1, 1, ppf * pph * ppw, -1)
            freqs = freqs * freqs_visual_id

        return freqs
