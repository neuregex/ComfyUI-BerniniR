# Copyright (c) 2026 — ComfyUI-BerniniR. Apache-2.0.
"""M3b — Guider de Bernini-R sobre el sampling NATIVO de ComfyUI.

`BerniniGuider` subclasea `comfy.samplers.CFGGuider` y reemplaza `predict_noise`
para implementar los 7 modos de guía de Bernini (CFG encadenada + APG), haciendo
hasta 4 forwards por paso con DISTINTOS subconjuntos de streams y texto pos/neg.

POR QUÉ ESTO ES FIEL Y SIMPLE EN NATIVO:
  * `model.apply_model` (vía `calc_cond_batch`) devuelve la predicción **x0
    (denoised)**, que es justo el espacio donde Bernini aplica APG. No hay que
    convertir eps<->x: APG se aplica directo a los x0.
  * Los modos NO-APG (rv2v/v2v/v2v_chain/t2v) son combinaciones **afines**
    (Σ coeficientes = 1). Para una combinación afín, `Σ aᵢ·epsᵢ` y `Σ aᵢ·x0ᵢ`
    se corresponden por el mismo mapa eps<->x0, así que aplicar la MISMA fórmula
    sobre los x0 y devolver x0 es exacto. El sampler de ComfyUI hace x0->derivada.

Los streams de condición se leen de `model.model_options["transformer_options"]
["bernini_streams"]` (los inyecta `BerniniRSourceStream`). Se separan en VIDEO
(latente con dim temporal > 1) e IMAGEN (T == 1). Por cada forward se fija el
SUBCONJUNTO correcto en `transformer_options` antes de llamar al modelo:
  none = solo target ;  V = primer vídeo + target ;  I = imágenes + target ;
  VI = vídeos+imágenes + target   (el target es el id 0, lo añade el forward).

Switch dual-expert: si se pasa `model_low`, al cruzar el boundary (t < 875) se
carga el experto low-noise (lazy, vía load_models_gpu — con GGUF cabe junto al
high y el load/offload no crashea) y se escala los omegas ×omega_scale una vez.
Sin `model_low` -> single-expert, sin switch.
"""
import torch

import comfy.samplers

from . import constants as C
from .guidance import MomentumBuffer, normalized_guidance, normalized_guidance_chain


def _is_video_latent(lat: torch.Tensor) -> bool:
    """[B, C, T, H, W] -> vídeo si T > 1; imagen si T == 1."""
    return lat.dim() >= 5 and lat.shape[-3] > 1


class BerniniGuider(comfy.samplers.CFGGuider):
    def __init__(self, model_patcher, mode: str, params: dict, model_low=None):
        super().__init__(model_patcher)
        self.bmode = mode
        self.bp = {**C.DEFAULTS, **(params or {})}
        self.model_low = model_low
        self.boundary_t = float(self.bp.get("boundary", C.SWITCH_DIT_BOUNDARY)) * C.NUM_TRAIN_TIMESTEPS

        # streams inyectados por BerniniRSourceStream (en el model_options del patcher)
        to = (getattr(model_patcher, "model_options", {}) or {}).get("transformer_options", {})
        streams = list(to.get("bernini_streams", []))
        self.streams_video, self.streams_image = [], []
        for st in streams:
            lat = st.get("latent")
            if lat is None:
                continue
            (self.streams_video if _is_video_latent(lat) else self.streams_image).append(st)

        self._reset_runtime()

    # -- estado que persiste DENTRO de un sampling y se resetea entre runs --------
    def _reset_runtime(self):
        m = float(self.bp["momentum"])
        self.mb1, self.mb2, self.mb = MomentumBuffer(m), MomentumBuffer(m), MomentumBuffer(m)
        self.oV = float(self.bp["omega_V"])
        self.oI = float(self.bp["omega_I"])
        self.oTI = float(self.bp["omega_TI"])
        nt = self.bp["norm_threshold"]
        nt = list(nt) if isinstance(nt, (list, tuple)) else [nt, nt, nt]
        while len(nt) < 3:
            nt.append(nt[-1])
        self.nt = [float(v) for v in nt]
        self.eta = float(self.bp["eta"])
        self.switched = False
        self._inner_low = None   # inner model del experto low (lazy, al cruzar el boundary)

    def sample(self, *args, **kwargs):
        self._reset_runtime()
        try:
            return super().sample(*args, **kwargs)
        finally:
            # Auto-cleanup: el experto low lo cargamos a mano (load_models_gpu) y ComfyUI no lo
            # libera por su cuenta -> se acumulan GGUF entre runs/workflows y la VRAM hace
            # thrashing al encadenar (p.ej. v2v -> i2i). Liberamos su clon aquí. El high lo
            # gestiona el cleanup normal del CFGGuider.
            if self.model_low is not None and self._inner_low is not None:
                try:
                    import comfy.model_management as mm
                    mm.unload_model_clones(self.model_low)
                    mm.soft_empty_cache()
                except Exception:
                    pass
            self._inner_low = None

    # -- resolución de modo -------------------------------------------------------
    def _resolve_mode(self) -> str:
        if self.bmode and self.bmode != "auto":
            return self.bmode
        has_v, has_i = bool(self.streams_video), bool(self.streams_image)
        if not has_v and not has_i:
            return "t2v"          # sin streams -> CFG de texto puro
        if has_v and has_i:
            return "rv2v"         # vídeo + referencia -> CFG encadenada 4-fwd
        return "v2v"              # solo imagen (i2i) o solo vídeo -> CFG de texto con stream VI

    # -- experto activo según el nivel de ruido (high-noise -> low-noise en el boundary) --
    def _active_inner(self, x, timestep):
        """high para t >= boundary; low para t < boundary (Wan2.2). El low se carga
        perezosamente la primera vez que se cruza (con GGUF cabe junto al high y el
        load/offload NO crashea, a diferencia de fp8 — ver F09)."""
        if self.model_low is None:
            return self.inner_model
        try:
            t_val = float(self.inner_model.model_sampling.timestep(timestep).reshape(-1)[0])
        except Exception:
            t_val = float("inf")
        if t_val >= self.boundary_t:
            return self.inner_model                       # régimen high-noise
        if not self.switched:                             # primer paso en low-noise
            sc = float(self.bp["omega_scale"])
            self.oV *= sc; self.oI *= sc; self.oTI *= sc
            self.switched = True
        if self._inner_low is None:
            try:
                import comfy.model_management as mm
                mm.load_models_gpu([self.model_low])
                # calc_cond_batch hace `model.current_patcher.prepare_state(...)`; ese enlace
                # BaseModel->ModelPatcher lo fija pre_run() (load_models_gpu NO lo pone). Sin
                # esto: AttributeError('NoneType' ... prepare_state).
                if hasattr(self.model_low, "pre_run"):
                    self.model_low.pre_run()
                self._inner_low = self.model_low.model
                if getattr(self._inner_low, "current_patcher", None) is None:
                    self._inner_low.current_patcher = self.model_low
                print(f"[BerniniR] switch -> experto LOW-noise (t<{self.boundary_t:.0f}); "
                      f"omegas ×{self.bp['omega_scale']}", flush=True)
            except Exception as e:
                print(f"[BerniniR] aviso: no pude cargar el experto low ({e}); sigo con high", flush=True)
                self._inner_low = self.inner_model
        return self._inner_low

    # -- un forward del experto DADO con un subconjunto de streams + un texto ---
    def _x0(self, model, streams, cond, x, timestep, model_options):
        mo = dict(model_options)
        to = dict(mo.get("transformer_options", {}))
        to["bernini_streams"] = streams          # [] -> el forward usa el camino nativo (solo target)
        mo["transformer_options"] = to
        out = comfy.samplers.calc_cond_batch(model, [cond], x, timestep, mo)
        return out[0].float()

    # -- núcleo: predicción denoised (x0) combinada según el modo de Bernini ------
    def predict_noise(self, x, timestep, model_options={}, seed=None):
        pos = self.conds.get("positive", None)
        neg = self.conds.get("negative", None)

        # experto activo según el ruido (escala los omegas ×omega_scale al cruzar a low-noise).
        active = self._active_inner(x, timestep)

        oV, oI, oTI = self.oV, self.oI, self.oTI
        none_c = []
        V_c = self.streams_video[:1]
        I_c = list(self.streams_image)
        VI_c = list(self.streams_video) + list(self.streams_image)

        def f(streams, cond):
            return self._x0(active, streams, cond, x, timestep, model_options)

        mode = self._resolve_mode()

        if mode == "rv2v":
            eu = f(none_c, neg); eV = f(V_c, neg); eVI = f(VI_c, neg); eVTI = f(VI_c, pos)
            out = eu + oV * (eV - eu) + oI * (eVI - eV) + oTI * (eVTI - eVI)
        elif mode == "v2v":
            eu = f(VI_c, neg); eVTI = f(VI_c, pos)
            out = eu + oTI * (eVTI - eu)
        elif mode == "v2v_chain":
            eu = f(none_c, neg); eV = f(V_c, neg); eVTI = f(VI_c, pos)
            out = eu + oV * (eV - eu) + oTI * (eVTI - eV)
        elif mode == "t2v":
            eu = f(none_c, neg); eT = f(none_c, pos)
            out = eu + oTI * (eT - eu)
        elif mode == "r2v_apg":
            eu = f(none_c, neg); eI = f(I_c, neg); eTI = f(I_c, pos)
            out = normalized_guidance_chain(eu, [eI, eTI], [oI, oTI],
                                            [self.mb1, self.mb2], self.eta, [self.nt[0], self.nt[1]])
        elif mode == "v2v_apg":
            eu = f(VI_c, neg); eVTI = f(VI_c, pos)
            out = normalized_guidance(eVTI, eu, oTI, self.mb, self.eta, self.nt[0])
        elif mode == "t2v_apg":
            eu = f(none_c, neg); eT = f(none_c, pos)
            out = normalized_guidance(eT, eu, oTI, self.mb, self.eta, self.nt[0])
        else:
            raise ValueError(f"[BerniniR] guidance_mode desconocido: {mode}")

        return out.to(x.dtype)


class BerniniRGuiderNode:
    """Construye el GUIDER de Bernini (APG + CFG encadenada) para SamplerCustomAdvanced.
    Lee los streams de condición del MODEL (inyectados con BerniniRSourceStream) y los
    separa en vídeo/imagen por la dim temporal del latente."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL", {"tooltip": "MODEL de BerniniR (Load Model native), opcionalmente con streams."}),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "mode": (["auto"] + C.GUIDANCE_MODES,
                     {"default": "auto", "tooltip": "auto: t2v sin streams, v2v con imagen/vídeo, rv2v con ambos."}),
            "omega_V": ("FLOAT", {"default": C.DEFAULTS["omega_V"], "min": 0.0, "max": 30.0, "step": 0.05}),
            "omega_I": ("FLOAT", {"default": C.DEFAULTS["omega_I"], "min": 0.0, "max": 30.0, "step": 0.05}),
            "omega_TI": ("FLOAT", {"default": C.DEFAULTS["omega_TI"], "min": 0.0, "max": 30.0, "step": 0.05}),
            "eta": ("FLOAT", {"default": C.DEFAULTS["eta"], "min": 0.0, "max": 1.0, "step": 0.05,
                              "tooltip": "APG: peso de la componente paralela (1.0 = CFG normal)."}),
            "momentum": ("FLOAT", {"default": C.DEFAULTS["momentum"], "min": -1.0, "max": 1.0, "step": 0.05}),
            "norm_threshold": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 1000.0, "step": 1.0,
                                         "tooltip": "APG: recorte de norma del diff (0 = sin recorte)."}),
        }, "optional": {
            "model_low": ("MODEL", {"tooltip": "Experto low-noise (Wan2.2). Si está, switch por boundary (necesita GGUF/pagefile)."}),
            "omega_scale": ("FLOAT", {"default": C.DEFAULTS["omega_scale"], "min": 0.0, "max": 2.0, "step": 0.05,
                                      "tooltip": "Escala los omegas al cambiar a low-noise."}),
            "boundary": ("FLOAT", {"default": C.SWITCH_DIT_BOUNDARY, "min": 0.0, "max": 1.0, "step": 0.005,
                                   "tooltip": "Frontera high/low (× 1000 timesteps). Wan2.2 A14B = 0.875."}),
        }}

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "get_guider"
    CATEGORY = "BerniniR"

    def get_guider(self, model, positive, negative, mode, omega_V, omega_I, omega_TI,
                   eta, momentum, norm_threshold, model_low=None, omega_scale=None, boundary=None):
        params = dict(
            omega_V=omega_V, omega_I=omega_I, omega_TI=omega_TI, eta=eta, momentum=momentum,
            norm_threshold=float(norm_threshold),
        )
        if omega_scale is not None:
            params["omega_scale"] = omega_scale
        if boundary is not None:
            params["boundary"] = boundary
        g = BerniniGuider(model, mode, params, model_low=model_low)
        g.set_conds(positive, negative)
        g.set_cfg(1.0)   # Bernini usa omegas, no el cfg escalar de ComfyUI
        print(f"[BerniniR] guider listo (modo={g._resolve_mode()}, "
              f"streams: {len(g.streams_video)} vídeo / {len(g.streams_image)} imagen)", flush=True)
        return (g,)


NODE_CLASS_MAPPINGS = {"BerniniRGuider": BerniniRGuiderNode}
NODE_DISPLAY_NAME_MAPPINGS = {"BerniniRGuider": "BerniniR · Guider (APG + dual-expert)"}
