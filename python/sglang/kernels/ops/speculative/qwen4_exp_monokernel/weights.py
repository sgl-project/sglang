# SPDX-License-Identifier: Apache-2.0
"""Checkpoint access for the Qwen3.8-Flash-Next (Qwen4-Exp) decode monokernel."""

from __future__ import annotations

import json
import os
from functools import lru_cache

from safetensors import safe_open

MODEL_DIR = ""  # checkpoint directory, set by the sglang runner from --model-path
PFX = "model.language_model."

# Qwen3.8-Flash-Next text_config
HIDDEN = 2560
HC = 4
HC_DIM = HIDDEN * HC  # 10240, hyper-connection residual stream width
HC_LOWRANK = 320
EPS = 1e-6
# Gated DeltaNet
GDN_HK = 16
GDN_HV = 48
GDN_DK = 128
GDN_DV = 128
GDN_KEY_DIM = GDN_HK * GDN_DK  # 2048
GDN_VAL_DIM = GDN_HV * GDN_DV  # 6144
GDN_CONV_DIM = 2 * GDN_KEY_DIM + GDN_VAL_DIM  # 10240
GDN_QKVZ = 2 * GDN_KEY_DIM + 2 * GDN_VAL_DIM  # 16384
GDN_CONV_K = 4
# MoE
N_EXPERTS = 512
TOPK = 10
MOE_INTER = 640
# full attention (QSA)
ATT_HQ = 24
ATT_HKV = 2
ATT_D = 256
ROTARY_DIM = 64


@lru_cache(maxsize=1)
def _index():
    with open(os.path.join(MODEL_DIR, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def load(names, device="cuda"):
    """Load checkpoint tensors by full name -> dict."""
    idx = _index()
    by_file = {}
    for n in names:
        by_file.setdefault(idx[n], []).append(n)
    out = {}
    for fname, ns in by_file.items():
        with safe_open(os.path.join(MODEL_DIR, fname), "pt", device="cpu") as f:
            for n in ns:
                out[n] = f.get_tensor(n).to(device)
    return out


# projections stored as FP8 E4M3 with F32 per-output-channel scales (activations: FP8 per-token dynamic)
FP8_QSA_PROJ = ("q_proj", "k_proj", "v_proj", "indexer.index_qk_proj", "o_proj")


def layer_names(layer: int, kind: str):
    """Checkpoint names of the tensors the kernel packs itself; the experts and the FP8 projections are sglang's
    own params, read in place."""
    p = f"{PFX}layers.{layer}."
    names = []
    for hc in ("attn_hyper_connection", "mlp_hyper_connection"):
        for w in (
            "hc_norm.weight",
            "input_mix_weight_down.weight",
            "input_mix_weight_up.weight",
            "block_inject_weight.weight",
        ):
            names.append(f"{p}{hc}.{w}")
    if kind == "gdn":
        for w in ("A_log", "conv1d.weight", "dt_bias", "norm.weight"):
            names.append(f"{p}linear_attn.{w}")
    else:
        for w in (
            "q_norm.weight",
            "k_norm.weight",
            "indexer.q_layernorm.weight",
            "indexer.k_layernorm.weight",
        ):
            names.append(f"{p}self_attn.{w}")
    names += [f"{p}mlp.gate.weight", f"{p}mlp.shared_expert_gate.weight"]
    return names


def load_layer(layer: int, kind: str, device="cuda", ple=False):
    """Return a dict of short names -> tensors for one decoder layer."""
    names = layer_names(layer, kind)
    if ple:
        names += [
            f"{PFX}layers.{layer}.ple.{n}"
            for n in (
                "key_proj.weight",
                "value_proj.weight",
                "norm_key.weight",
                "norm_query.weight",
                "norm_conv.weight",
                "conv1d.weight",
            )
        ]
    raw = load(names, device)
    p = f"{PFX}layers.{layer}."
    return {k[len(p) :]: v for k, v in raw.items()}


def is_expert_tensor(name: str) -> bool:
    return ".mlp.experts." in name or ".mlp.shared_expert." in name


def is_fp8_proj_tensor(name: str) -> bool:
    return any(f".self_attn.{p}.weight" in name for p in FP8_QSA_PROJ)


def load_mtp(device="cuda"):
    """MTP (NEXTN) draft head without its experts / FP8 projections (sglang's params): norms, router, mixers and
    input fusion -> dict of names without the "mtp." prefix (layer weights under "layers.0.")."""
    idx = _index()
    names = [
        k
        for k in idx
        if k.startswith("mtp.")
        and not is_expert_tensor(k)
        and not is_fp8_proj_tensor(k)
    ]
    raw = load(names, device)
    return {k[len("mtp.") :]: v for k, v in raw.items()}
