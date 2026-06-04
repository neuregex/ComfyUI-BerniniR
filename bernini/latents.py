# Copyright (c) 2026 — ComfyUI-BerniniR. Apache-2.0.
"""Empaquetado de latentes y normalización VAE (convención Bernini-R/Wan)."""
import torch
from einops import rearrange

# Latente Wan: [B, C=16, T_lat, H_lat, W_lat], con T_lat=(F-1)//4+1, H/8, W/8.
# Tokens "packed" del target:  b (t h w) (ph pw c)  con ph=pw=2.
_PACK = "b (t h w) (pt ph pw c) -> b c (t pt) (h ph) (w pw)"
_UNPACK = "b c (t pt) (h ph) (w pw) -> b (t h w) (pt ph pw c)"


def to_spatial(x, shape):
    return rearrange(x, _PACK, t=shape[2], h=shape[3] // 2, w=shape[4] // 2, pt=1, ph=2, pw=2)


def to_packed(x, shape):
    return rearrange(x, _UNPACK, t=shape[2], h=shape[3] // 2, w=shape[4] // 2, pt=1, ph=2, pw=2)


def snap_num_frames(num_frames: int) -> int:
    """Bernini re-ajusta: num_frames = num_frames//4*4 + 1."""
    return max(1, num_frames // 4 * 4 + 1)


def latent_shape(num_frames: int, height: int, width: int, channels: int = 16):
    t_lat = (snap_num_frames(num_frames) - 1) // 4 + 1
    return (1, channels, t_lat, height // 8, width // 8)


def make_noise(shape, device, dtype=torch.float32, generator=None):
    noise = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    # packed: b c t (h ph) (w pw) -> b (t h w) (ph pw c)
    return rearrange(noise, "b c t (h ph) (w pw) -> b (t h w) (ph pw c)", ph=2, pw=2)


def _mean_std(vae, device, dtype):
    cfg = vae.config if hasattr(vae, "config") else vae
    z = getattr(cfg, "z_dim", 16)
    mean = torch.tensor(cfg.latents_mean, device=device, dtype=dtype).view(1, z, 1, 1, 1)
    std = torch.tensor(cfg.latents_std, device=device, dtype=dtype).view(1, z, 1, 1, 1)
    return mean, std


def vae_encode(vae, x):
    """x: [1,C,T,H,W] en [-1,1] -> latente normalizado estilo Bernini."""
    latents = vae.encode(x).latent_dist.mode()
    mean, std = _mean_std(vae, latents.device, latents.dtype)
    return (latents - mean) / std


def vae_decode(vae, latents):
    mean, std = _mean_std(vae, latents.device, latents.dtype)
    latents = latents * std + mean
    return vae.decode(latents).sample
