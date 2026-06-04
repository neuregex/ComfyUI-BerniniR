# Copyright (c) 2026 — ComfyUI-BerniniR. Apache-2.0.
"""Loop de muestreo de Bernini-R (port de GEN_Wanx22.sample).

Reúne: scheduler UniPC, ensamblado de combos de stream con source_id, switch
dual-expert por VALOR de timestep, los 7 modos de guía y el offload secuencial.
"""
from typing import List, Optional

import torch

from . import constants as C
from . import latents as L
from .model import BerniniRenderer, Stream
from .guidance import MomentumBuffer, normalized_guidance, normalized_guidance_chain


def build_scheduler(num_inference_steps, shift=C.SHIFT, use_unipc=True,
                    base_scheduler_dir=None, device="cpu"):
    """UniPCMultistepScheduler con flow_shift = shift (=3.0 efectivo)."""
    from diffusers import UniPCMultistepScheduler
    if base_scheduler_dir:
        sch = UniPCMultistepScheduler.from_pretrained(base_scheduler_dir, flow_shift=shift)
    else:
        sch = UniPCMultistepScheduler(prediction_type="flow_prediction",
                                      use_flow_sigmas=True, flow_shift=shift)
    sch.set_timesteps(num_inference_steps, device=device)
    return sch


class BerniniSampler:
    def __init__(self, renderer: BerniniRenderer, guidance_mode: str,
                 params: dict, offload_experts: bool = True):
        self.r = renderer
        self.mode = guidance_mode
        self.p = {**C.DEFAULTS, **params}
        self.offload = offload_experts

    # -- offload: mantener solo el experto activo en GPU ----------------------
    def _activate(self, expert, device):
        if not self.offload or self.r.low is None:
            return
        other = self.r.low if expert is self.r.high else self.r.high
        if other is not None and next(other.t.parameters()).device.type != "cpu":
            other.t.to("cpu")
            torch.cuda.empty_cache()
        if next(expert.t.parameters()).device != torch.device(device):
            expert.t.to(device)

    def _apg_sigma(self, t_idx):
        # sigmas[t_idx] = nivel de ruido al INICIO del paso actual. Equivale a
        # la lógica step_index de Bernini (tras t_idx steps, step_index==t_idx),
        # pero sin la ambigüedad de leer step_index antes de scheduler.step().
        return self.scheduler.sigmas[t_idx].to(torch.float64)

    @torch.no_grad()
    def sample(self, video_latents: List[torch.Tensor], image_latents: List[torch.Tensor],
               text_pos: torch.Tensor, text_neg: torch.Tensor,
               target_shape, device, seed: int = 0,
               base_scheduler_dir: Optional[str] = None) -> torch.Tensor:
        p = self.p
        self.scheduler = build_scheduler(p["num_inference_steps"], shift=C.SHIFT,
                                         base_scheduler_dir=base_scheduler_dir, device=device)
        timesteps = self.scheduler.timesteps.to(device)

        gen = torch.Generator(device=device).manual_seed(int(seed))
        noisy = L.make_noise(target_shape, device=device, dtype=torch.float32, generator=gen)  # packed

        # --- streams de condición (constantes en el tiempo) ------------------
        sid = 1
        v_cond, vi_cond = [], []
        for idx, vid in enumerate(video_latents or []):
            st = Stream(vid, sid); sid += 1
            if idx == 0:
                v_cond.append(st)
            vi_cond.append(st)
        i_cond = []
        sid_img = 1
        for img in (image_latents or []):
            vi_cond.append(Stream(img, sid)); sid += 1
            i_cond.append(Stream(img, sid_img)); sid_img += 1

        # --- omegas (mutables: se escalan al cambiar a low-noise) ------------
        oV, oI, oTI = float(p["omega_V"]), float(p["omega_I"]), float(p["omega_TI"])
        eta = float(p["eta"]); momentum = float(p["momentum"])
        nt = p["norm_threshold"]
        nt = list(nt) if isinstance(nt, (list, tuple)) else [nt, nt, nt]
        while len(nt) < 3:
            nt.append(nt[-1])
        mb1 = MomentumBuffer(momentum); mb2 = MomentumBuffer(momentum); mb = MomentumBuffer(momentum)

        switched = False
        boundary = self.r.boundary_timestep()

        for t_idx, t in enumerate(timesteps):
            tv = float(t)
            expert = self.r.expert_for_timestep(tv)
            if (not switched) and tv < boundary and self.r.low is not None:
                expert = self.r.low
                oV *= float(p["omega_scale"]); oI *= float(p["omega_scale"]); oTI *= float(p["omega_scale"])
                switched = True
            self._activate(expert, device)
            t_tensor = t.reshape(1).to(device)

            noisy_spatial = L.to_spatial(noisy, target_shape).to(expert.dtype)
            target = Stream(noisy_spatial, 0, is_target=True)
            none_c = [target]
            V_c = v_cond + [target]
            I_c = i_cond + [target]
            VI_c = vi_cond + [target]

            def fwd(streams, text):
                return expert.forward_streams(streams, text, t_tensor).float()

            mode = self.mode
            if mode == "rv2v":
                eu = fwd(none_c, text_neg); eV = fwd(V_c, text_neg)
                eVI = fwd(VI_c, text_neg); eVTI = fwd(VI_c, text_pos)
                noise_pred = eu + oV * (eV - eu) + oI * (eVI - eV) + oTI * (eVTI - eVI)
            elif mode == "v2v":
                eu = fwd(VI_c, text_neg); eVTI = fwd(VI_c, text_pos)
                noise_pred = eu + oTI * (eVTI - eu)
            elif mode == "v2v_chain":
                eu = fwd(none_c, text_neg); eV = fwd(V_c, text_neg); eVTI = fwd(VI_c, text_pos)
                noise_pred = eu + oV * (eV - eu) + oTI * (eVTI - eV)
            elif mode == "t2v":
                eu = fwd(none_c, text_neg); eT = fwd(none_c, text_pos)
                noise_pred = eu + oTI * (eT - eu)
            elif mode in ("r2v_apg", "v2v_apg", "t2v_apg"):
                sigma = self._apg_sigma(t_idx)
                noisy_s = L.to_spatial(noisy, target_shape).double()

                def to_x(eps_packed):
                    return noisy_s - sigma * L.to_spatial(eps_packed, target_shape).double()

                if mode == "r2v_apg":
                    eu = fwd(none_c, text_neg); eI = fwd(I_c, text_neg); eTI = fwd(I_c, text_pos)
                    x_guided = normalized_guidance_chain(
                        to_x(eu), [to_x(eI), to_x(eTI)], [oI, oTI], [mb1, mb2], eta, [nt[0], nt[1]])
                elif mode == "v2v_apg":
                    eu = fwd(VI_c, text_neg); eVTI = fwd(VI_c, text_pos)
                    x_guided = normalized_guidance(to_x(eVTI), to_x(eu), oTI, mb, eta, nt[0])
                else:  # t2v_apg
                    eu = fwd(none_c, text_neg); eT = fwd(none_c, text_pos)
                    x_guided = normalized_guidance(to_x(eT), to_x(eu), oTI, mb, eta, nt[0])
                noise_pred = L.to_packed(((noisy_s - x_guided) / sigma).to(noisy.dtype), target_shape)
            else:
                raise ValueError(f"guidance_mode desconocido: {mode}")

            noisy = self.scheduler.step(noise_pred.to(noisy.dtype), t, noisy, return_dict=False)[0]

        return L.to_spatial(noisy, target_shape)   # latente final espacial [1,16,T,H,W]
