# Copyright (c) 2026 — ComfyUI-BerniniR. Apache-2.0.
"""Adaptive Projected Guidance (APG) + CFG encadenado de Bernini-R.

Port VERBATIM de bernini/models/wan_diffusion.py (ByteDance, Apache-2.0).

CLAVE DE FIDELIDAD — no tocar sin validar:
  * Todo opera sobre latentes 5-D en formato espacial [B=1, C=16, T, H, W].
  * Las reducciones son dim=[-1,-2,-4]  ==>  sobre {W, H, C} POR FRAME,
    NUNCA sobre T (-3) ni B (-5). Es lo que hace que la guía sea por-frame.
  * La matemática va en float64 (.double()) y se castea de vuelta al final.
  * Los MomentumBuffer PERSISTEN entre pasos de denoising.
"""
import torch


class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        self.running_average = update_value + self.momentum * self.running_average


def _normalize_diff(diff, base_pred, momentum_buffer, eta, norm_threshold):
    """Proyecta `diff` sobre/contra `base_pred` y recombina con peso `eta`."""
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=[-1, -2, -4], keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor
    v0, v1 = diff.double(), base_pred.double()
    # Normaliza cada volumen (C,H,W) por frame a norma unidad (equivalente a
    # F.normalize(v1, dim=[-1,-2,-4]); explícito para no depender de que
    # F.normalize acepte lista de dims).
    v1 = v1 / v1.norm(p=2, dim=[-1, -2, -4], keepdim=True).clamp_min(1e-12)
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -4], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    diff_parallel, diff_orthogonal = v0_parallel.to(diff.dtype), v0_orthogonal.to(diff.dtype)
    return diff_orthogonal + eta * diff_parallel


def normalized_guidance(pred_cond, pred_uncond, guidance_scale,
                        momentum_buffer=None, eta=1.0, norm_threshold=0.0):
    """APG de UNA condición:  uncond + scale * APG(cond - uncond)."""
    nd = _normalize_diff(pred_cond - pred_uncond, pred_cond,
                         momentum_buffer, eta, norm_threshold)
    return pred_uncond + guidance_scale * nd


def normalized_guidance_chain(pred_uncond, preds, scales,
                              momentum_buffers, eta, norm_thresholds):
    """APG ENCADENADA: el diff de cada condición se toma contra la anterior."""
    bases = [pred_uncond] + list(preds)
    result = pred_uncond
    for i, cond in enumerate(preds):
        nd = _normalize_diff(cond - bases[i], cond,
                             momentum_buffers[i], eta, norm_thresholds[i])
        result = result + scales[i] * nd
    return result
