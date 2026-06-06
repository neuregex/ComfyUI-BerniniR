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
    # Backend nativo (v0.4): añade los loaders que esquivan el crash de load_torch_file
    # con fp8 e4m3 en torch 2.8/Windows. Tolerante a fallos: si algo va mal, el paquete
    # sigue funcionando con los nodos diffusers.
    try:
        from .bernini.native_nodes import (
            NODE_CLASS_MAPPINGS as _NCM_NATIVE,
            NODE_DISPLAY_NAME_MAPPINGS as _NDN_NATIVE,
        )
        NODE_CLASS_MAPPINGS.update(_NCM_NATIVE)
        NODE_DISPLAY_NAME_MAPPINGS.update(_NDN_NATIVE)
    except Exception as _e:  # pragma: no cover
        print(f"[BerniniR] aviso: nodos nativos (v0.4) no cargados: {_e}")
else:  # pragma: no cover - solo en import standalone (tests/herramientas)
    NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = {}, {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
