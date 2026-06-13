// ComfyUI-BerniniR — frontend: carga + preview de vídeo (BerniniRLoadVideo) y
// preview del vídeo generado (BerniniRSaveVideo). Defensivo (con fallbacks).
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const ANIM = ["webp", "gif"];

function viewURL(name, type = "input", subfolder = "") {
    return api.apiURL(`/view?filename=${encodeURIComponent(name)}&type=${type}&subfolder=${encodeURIComponent(subfolder)}&t=${Date.now()}`);
}

// ---------- BerniniRLoadVideo: preview + botón de carga (acepta vídeo) ----------
app.registerExtension({
    name: "BerniniR.LoadVideoPreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "BerniniRLoadVideo") return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = onCreated?.apply(this, arguments);
            const node = this;
            const wrap = document.createElement("div");
            wrap.style.cssText = "width:100%;display:flex;justify-content:center;align-items:center;";
            const video = document.createElement("video");
            video.controls = true; video.loop = true; video.muted = true; video.playsInline = true;
            video.style.cssText = "max-width:100%;max-height:240px;border-radius:8px;display:none;";
            const img = document.createElement("img");
            img.style.cssText = "max-width:100%;max-height:240px;border-radius:8px;display:none;";
            wrap.appendChild(video); wrap.appendChild(img);
            node.addDOMWidget("br_preview", "preview", wrap, { serialize: false });

            const show = (name) => {
                const empty = !name || name === "source.webp";
                const ext = empty ? "" : (name.split(".").pop() || "").toLowerCase();
                if (empty) { video.style.display = "none"; img.style.display = "none"; video.removeAttribute("src"); }
                else if (ANIM.includes(ext)) { img.src = viewURL(name); img.style.display = "block"; video.style.display = "none"; video.removeAttribute("src"); }
                else { video.src = viewURL(name); video.style.display = "block"; img.style.display = "none"; video.load(); }
                node.setDirtyCanvas?.(true, true);
            };
            const combo = node.widgets?.find((w) => w.name === "video");
            if (combo) {
                const prev = combo.callback;
                combo.callback = function () { const r = prev?.apply(this, arguments); show(combo.value); return r; };
                setTimeout(() => show(combo.value), 100);
            }

            const picker = document.createElement("input");
            picker.type = "file"; picker.accept = "video/*,image/webp,image/gif"; picker.style.display = "none";
            document.body.appendChild(picker);
            picker.onchange = async () => {
                const file = picker.files?.[0]; picker.value = ""; if (!file) return;
                const body = new FormData(); body.append("image", file); body.append("type", "input"); body.append("subfolder", "");
                try {
                    const resp = await api.fetchApi("/upload/image", { method: "POST", body });
                    if (resp.status !== 200) throw new Error(await resp.text());
                    const data = await resp.json();
                    const name = data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
                    if (combo) {
                        const vals = combo.options?.values;
                        if (Array.isArray(vals) && !vals.includes(name)) vals.push(name);
                        combo.value = name; combo.callback?.(name);
                    }
                } catch (e) { alert("BerniniR: fallo al cargar el vídeo -> " + e); }
            };
            node.addWidget("button", "📹 Cargar vídeo (mp4 / webp / gif)", null, () => picker.click());
            return ret;
        };
    },
});

// ---------- BerniniRSaveVideo: preview del vídeo generado (mp4/webm) ----------
app.registerExtension({
    name: "BerniniR.SaveVideoPreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "BerniniRSaveVideo") return;
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const node = this;
            const items = message?.berninir_video;
            if (!items || !items.length) return;
            const it = items[0];
            const url = viewURL(it.filename, it.type || "output", it.subfolder || "");
            if (!node._brOutWrap) {
                const wrap = document.createElement("div");
                wrap.style.cssText = "width:100%;display:flex;justify-content:center;align-items:center;";
                node._brOutWrap = wrap;
                node.addDOMWidget("br_out_preview", "preview", wrap, { serialize: false });
            }
            const wrap = node._brOutWrap;
            wrap.innerHTML = "";
            const v = document.createElement("video");
            v.src = url; v.controls = true; v.loop = true; v.muted = true; v.autoplay = true; v.playsInline = true;
            v.style.cssText = "max-width:100%;max-height:280px;border-radius:8px;";
            wrap.appendChild(v);
            node.setDirtyCanvas?.(true, true);
        };
    },
});
