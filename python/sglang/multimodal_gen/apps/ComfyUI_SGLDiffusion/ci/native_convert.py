"""Rewrite an SGLD integrated-mode workflow into native ComfyUI nodes.

Integrated-mode workflows load the DiT through SGLDUNETLoader, so sampling
runs in the SGLang engine. The converter swaps those nodes for stock ones so
the same graph can be run by ComfyUI itself as the baseline.

Server-mode workflows (SGLDiffusionServerModel / Generate* nodes) call a
separate `sglang serve` process and have no native twin; `classify` tells them
apart and `to_native` refuses them.
"""

from __future__ import annotations

import copy

SERVER_CLASSES = frozenset(
    {
        "SGLDiffusionServerModel",
        "SGLDiffusionGenerateImage",
        "SGLDiffusionGenerateVideo",
        "SGLDiffusionGenerateH3",
        "SGLDiffusionServerSetLora",
        "SGLDiffusionServerUnsetLora",
        # Stale name still present in shipped sgld_text2img.json.
        "SGLDiffusionSetLora",
    }
)
INTEGRATED_CLASSES = frozenset({"SGLDUNETLoader", "SGLDOptions", "SGLDLoraLoader"})


class NotConvertible(ValueError):
    pass


def classify(wf) -> str:
    """Return 'server', 'integrated' or 'native' for a workflow."""
    classes = {n.get("class_type") for n in wf.values()}
    if classes & SERVER_CLASSES:
        return "server"
    if classes & INTEGRATED_CLASSES:
        return "integrated"
    return "native"


def _is_link(v):
    return isinstance(v, list) and len(v) == 2 and isinstance(v[1], int)


def to_native(wf, lora_map=None):
    """Return (native_workflow, notes). Input is not modified.

    lora_map maps an SGLD lora_name to the native lora_name to load; names not
    in the map are kept as they are.
    """
    if classify(wf) == "server":
        raise NotConvertible("server-mode workflow has no native equivalent")
    lora_map = lora_map or {}
    out = copy.deepcopy(wf)
    notes = []

    for nid, node in list(out.items()):
        cls = node["class_type"]
        inp = node["inputs"]
        if cls == "SGLDUNETLoader":
            dropped = inp.pop("sgld_options", None)
            if dropped is not None:
                notes.append(f"{nid}: dropped sgld_options link")
            node["class_type"] = "UNETLoader"
        elif cls == "SGLDLoraLoader":
            node["class_type"] = "LoraLoaderModelOnly"
            name = inp.get("lora_name")
            if name in lora_map:
                inp["lora_name"] = lora_map[name]
            for k in ("nickname", "target"):
                if inp.pop(k, None) is not None:
                    notes.append(f"{nid}: dropped {k}")
        elif cls == "SGLDOptions":
            del out[nid]
            notes.append(f"{nid}: removed SGLDOptions")

    for nid, node in out.items():
        for k, v in node["inputs"].items():
            if _is_link(v) and str(v[0]) not in out:
                raise NotConvertible(
                    f"node {nid} input '{k}' links to removed node {v[0]}"
                )
    return out, notes
