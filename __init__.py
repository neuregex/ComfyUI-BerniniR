# Copyright (c) 2026 — ComfyUI-BerniniR. Apache-2.0.
"""ComfyUI-BerniniR — soporte para ByteDance/Bernini-R (Wan2.2-T2V-A14B
   con source-id RoPE + guía APG multi-condición) en ComfyUI."""

# ComfyUI SIEMPRE importa este paquete con contexto de paquete (`__package__`
# definido), así que el import relativo funciona en producción. Cuando se importa
# fuera de ese contexto —p.ej. pytest recolectando el árbol del repo, que tiene
# este __init__.py en la raíz— el import relativo lanzaría "attempted relative
# import with no known parent package". Lo evitamos sin enmascarar errores reales
# (que sí ocurren cuando `__package__` está definido y el import se ejecuta).
if __package__:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
else:  # pragma: no cover - solo en import standalone (tests/herramientas)
    NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = {}, {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
