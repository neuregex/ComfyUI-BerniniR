# Copyright (c) 2026 — ComfyUI-BerniniR. Apache-2.0.
"""M2 — lógica de Bernini sobre el WanModel NATIVO de ComfyUI, vía add_object_patch.

Dos cosas, sin forkear `comfy/`:

1. **source-id RoPE.** Cada stream visual (target, vídeo fuente, imagen ref) lleva una fase
   constante. En diffusers Bernini multiplica las freqs complejas por `visual_id_freqs[source_id]`
   = e^{i·source_id·inv_freq_d}. El RoPE nativo de ComfyUI es REAL (rotaciones 2×2 estilo flux),
   y como `e^{i(φ+θ)} == R(φ+θ)` (validado), la fase se inyecta **componiendo R(θ_s)** en las
   freqs nativas por matmul. `θ_s,d = source_id · theta^(−2d/head_dim)` (head_dim COMPLETO, igual
   que el visual_id_freqs de Bernini, cuyos ejes coinciden con el axes_dim del EmbedND nativo).

2. **stream-concat + máscara.** Cada stream se patch-embedea y se concatena en el eje de secuencia
   ANTES del target (mismo patrón que el `ref_conv` nativo); tras los bloques se recortan los
   tokens del target. Los streams los alimenta el guider de M3 vía
   `transformer_options["bernini_streams"] = [{"latent": Tensor[B,C,F,H,W], "source_id": int}, ...]`.
   Sin streams -> el forward es IDÉNTICO al nativo (no-op para t2v).
"""
import torch


def compose_src_id_freqs(freqs, source_id, head_dim, theta=10000.0):
    """Compone la fase source-id (rotación real R(θ_s)) en las freqs nativas de flux/Wan.

    `freqs`: tensor con las matrices de rotación 2×2 en las dos últimas dims y `head_dim//2`
    en la antepenúltima -> shape [..., head_dim//2, 2, 2]. Devuelve R(θ_s) @ freqs = R(φ+θ_s).
    `source_id == 0` -> identidad (no-op, p.ej. el target)."""
    if not source_id:
        return freqs
    half = head_dim // 2
    d = torch.arange(half, device=freqs.device, dtype=torch.float32)
    omega = theta ** (-(2.0 * d) / head_dim)                  # inv_freq del head_dim completo
    ang = float(source_id) * omega                            # θ_s,d  [half]
    c, s = torch.cos(ang), torch.sin(ang)
    rs = torch.stack([torch.stack([c, -s], dim=-1),
                      torch.stack([s,  c], dim=-1)], dim=-2)   # [half, 2, 2] = R(θ_s)
    rs = rs.to(freqs.dtype)
    return torch.matmul(rs, freqs)                            # broadcast sobre las dims previas


def _head_dim(diff):
    hd = getattr(diff, "head_dim", None)
    if hd:
        return hd
    dim = getattr(diff, "dim", None)
    nh = getattr(diff, "num_heads", None)
    return (dim // nh) if (dim and nh) else 128


def apply_bernini_patches(model_patcher, theta: float = 10000.0):
    """Instala los patches de Bernini sobre un ModelPatcher (WanModel nativo) vía
    add_object_patch. Reversible (ComfyUI los quita al hacer unpatch). Si no hay streams
    en transformer_options, el forward llama al original -> 0 cambios para t2v."""
    diff = model_patcher.model.diffusion_model
    head_dim = _head_dim(diff)
    orig_forward_orig = diff.forward_orig          # bound method original (capturado antes del patch)

    # patch_embedding tolerante a dtype: el forward_orig NATIVO de Wan hace
    # `self.patch_embedding(x.float())`. Con el 1.3B bf16 (ops regulares) el conv recibe
    # float32 y el bias es bf16 -> "Input type (float) and bias type (BFloat16)". Casteamos
    # el input al dtype del peso. Cubre la rama sin-streams (t2v + término incondicional del
    # guider) y la multi-stream. Guarda: solo para pesos float -> no-op en fp8/GGUF (manual_cast).
    _pe = getattr(diff, "patch_embedding", None)
    if _pe is not None and not getattr(_pe, "_bernini_dtype_safe", False):
        _pe_orig_forward = _pe.forward
        def _pe_dtype_safe(inp, *a, _f=_pe_orig_forward, _pe=_pe, **kw):
            w = getattr(_pe, "weight", None)
            if (w is not None and w.dtype in (torch.float16, torch.bfloat16, torch.float32)
                    and inp.dtype != w.dtype):
                inp = inp.to(w.dtype)
            return _f(inp, *a, **kw)
        _pe.forward = _pe_dtype_safe
        _pe._bernini_dtype_safe = True

    def forward_orig_bernini(x, t, context, clip_fea=None, freqs=None, transformer_options={}, **kwargs):
        streams = transformer_options.get("bernini_streams", None)
        if not streams:
            # t2v / sin condición extra -> forward nativo intacto.
            return orig_forward_orig(x, t, context, clip_fea=clip_fea, freqs=freqs,
                                     transformer_options=transformer_options, **kwargs)

        # --- camino multi-stream (edición) ---
        from comfy.ldm.wan.model import sinusoidal_embedding_1d
        m = diff

        # target: parchea y saca SU grid REAL de parches (Tg, Hg, Wg).
        # dtype de la conv: con pesos float NORMALES (ops normales, p.ej. el 1.3B bf16) el input
        # DEBE igualar ese dtype; con pesos fp8/GGUF (manual-cast) la op castea el peso al dtype
        # del input, así que basta el de cómputo (x.dtype). NO forzar float32 (rompía el bf16 puro).
        _w = getattr(m.patch_embedding, "weight", None)
        emb_dtype = _w.dtype if (_w is not None and _w.dtype in (torch.float16, torch.bfloat16, torch.float32)) else x.dtype
        xt_conv = m.patch_embedding(x.to(emb_dtype)).to(x.dtype)
        grid_sizes = xt_conv.shape[2:]
        transformer_options["grid_sizes"] = grid_sizes
        xt = xt_conv.flatten(2).transpose(1, 2)               # [B, Lt, dim]
        Lt = xt.shape[1]
        # Freqs del target desde SU grid de parches, NO desde las dims del latente: con vídeo
        # (alguna dim impar) rope_encode(dims) y patch_embedding discrepan ((h+1)//2 vs
        # (h-2)//2+1). rope_encode(Tg, 2*Hg, 2*Wg) -> grid (Tg,Hg,Wg) == el de patch_embedding,
        # y para dims pares es IDÉNTICO a las freqs nativas (i2i/t2v intactos).
        freqs_t = m.rope_encode(grid_sizes[0], 2 * grid_sizes[1], 2 * grid_sizes[2],
                                device=x.device, dtype=x.dtype, transformer_options=transformer_options)

        # streams de condición (cada uno con su source_id -> su fase RoPE)
        bsz = xt.shape[0]                              # batch del target (2 si hay CFG: cond+uncond)
        toks, frqs = [], []
        for st in streams:
            lat = st["latent"].to(x.device)            # el stream puede venir en CPU
            sid = int(st.get("source_id", 0))
            xe_conv = m.patch_embedding(lat.to(emb_dtype)).to(x.dtype)   # [B, dim, Ts, Hs, Ws]
            gs = xe_conv.shape[2:]
            xe = xe_conv.flatten(2).transpose(1, 2)                # [B, Ls, dim]
            if xe.shape[0] != bsz:                     # alinear batch con el target (CFG)
                xe = xe.expand(bsz, -1, -1)
            fe = m.rope_encode(gs[0], 2 * gs[1], 2 * gs[2],        # freqs desde el grid REAL de xe
                               device=x.device, dtype=x.dtype, transformer_options=transformer_options)
            fe = compose_src_id_freqs(fe, sid, head_dim, theta)
            toks.append(xe)
            frqs.append(fe)

        if not getattr(forward_orig_bernini, "_dbg", False):
            forward_orig_bernini._dbg = True
            print(f"[BerniniR] dbg: x={tuple(x.shape)} grid_t={tuple(grid_sizes)} Lt={Lt} "
                  f"Ls={[t.shape[1] for t in toks]} freqs_t_seq={freqs_t.shape[1]} "
                  f"fe_seq={[f.shape[1] for f in frqs]}", flush=True)

        # concatenar streams ANTES del target (como ref_conv). Las freqs nativas tienen shape
        # [1, seq, 1, head_dim//2, 2, 2] tras rope_embedder(...).movedim(1,2): el eje de TOKENS
        # es el 1 (el 2 es el broadcast de cabezas, size 1). Hay que concatenar por el 1; si se
        # hace por el 2 ese singleton pasa a 2 y apply_rope1 peta (2 vs num_heads).
        x_all = torch.cat(toks + [xt], dim=1)
        freqs_all = torch.cat(frqs + [freqs_t], dim=1)

        # time + context embeddings (réplica fiel del forward_orig nativo)
        e = m.time_embedding(sinusoidal_embedding_1d(m.freq_dim, t.flatten()).to(dtype=x_all[0].dtype))
        e = e.reshape(t.shape[0], -1, e.shape[-1])
        e0 = m.time_projection(e).unflatten(2, (6, m.dim))

        context = m.text_embedding(context)
        context_img_len = None
        if clip_fea is not None and getattr(m, "img_emb", None) is not None:
            context = torch.cat([m.img_emb(clip_fea), context], dim=1)
            context_img_len = clip_fea.shape[-2]

        transformer_options["total_blocks"] = len(m.blocks)
        for i, block in enumerate(m.blocks):
            transformer_options["block_index"] = i
            x_all = block(x_all, e=e0, freqs=freqs_all, context=context,
                          context_img_len=context_img_len, transformer_options=transformer_options)

        # quedarnos solo con los tokens del target (van al final) y rematar
        x_out = x_all[:, -Lt:]
        x_out = m.head(x_out, e)
        x_out = m.unpatchify(x_out, grid_sizes)
        return x_out

    model_patcher.add_object_patch("diffusion_model.forward_orig", forward_orig_bernini)
    return model_patcher
