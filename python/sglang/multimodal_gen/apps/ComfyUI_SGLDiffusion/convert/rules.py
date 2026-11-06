# SPDX-License-Identifier: Apache-2.0
"""Which nodes the SGLD workflow converter rewrites or reports, as data."""

import re

# Diffusion models SGLD integrated mode serves, by file name. LTX, Wan and other
# models stay on ComfyUI's stock loader.
_SGLD_NAME = re.compile(
    r"(?i)minimax|fasth3|(^|[^a-z0-9])h3|qwen|z[-_]?image|flux1|flux\.1|flux-1"
)
_NOT_SGLD = re.compile(
    r"(?i)ltx|wan2|hunyuan|rt_detr|sdpose|detr|yolo|sam3|flux2|flux\.2|krea"
)

SGLD_LOADER = "SGLDUNETLoader"
SGLD_LOADERS = {"SGLDUNETLoader", "SGLDNativeModelLoader"}
# Stock diffusion-model loaders whose file SGLDUNETLoader can load as-is.
STOCK_LOADERS = {"UNETLoader", "UnetLoaderGGUF"}

# Core LoRA nodes rewritten to SGLDLoraLoader (widgets: lora_name, strength_model,
# nickname, target). Only their diffusion-model half carries over.
LORA_NODES = {"LoraLoader", "LoraLoaderModelOnly"}

# Multi-LoRA stack nodes: reported, to be replaced by hand with chained
# SGLDLoraLoader nodes (their CLIP / trigger-word outputs may still be needed).
STACK_NODES = {
    "Power Lora Loader (rgthree)",
    "AUSBOSS_NODES_LoraLoader",
    "Lora Loader (LoraManager)",
}

# MODEL -> MODEL nodes that only change the sampler's sigma schedule or the
# preview, which stay in ComfyUI; they work on an SGLD model, so they are not
# reported. (The bundled SGLD workflows use ModelSampling*.)
SAMPLING_ONLY_PREFIX = "ModelSampling"
SAMPLING_ONLY = {"MiniMaxH3SigmaShift", "ModelPreviewOverrideKJ"}

# KJNodes' named wiring: a GetNode outputs whatever the SetNode of the same name
# takes. Both are followed when tracing a MODEL and never reported.
SET_NODE, GET_NODE = "SetNode", "GetNode"


def file_name(name) -> str:
    return name.replace("\\", "/").split("/")[-1] if isinstance(name, str) else ""


def is_sgld_model(name) -> bool:
    base = file_name(name)
    return bool(_SGLD_NAME.search(base)) and not _NOT_SGLD.search(base)


def is_sgld_node(kind) -> bool:
    return isinstance(kind, str) and kind.startswith("SGLD")
