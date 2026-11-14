"""Only the double-anchor hidden-state residual at T1 is supported."""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


def normalize_residual_config(config):
    """Translate only the explicit dual-anchor residual export schema."""
    cfg = config.dflash_config
    if cfg.get("mtp_prefix_injection") != "add_to_t1_input_embedding":
        return config
    required = dict(
        mtp_prefix=True,
        mtp_t1_residual=True,
        mtp_hidden_only=True,
        mtp_drop_anchor=False,
        mtp_t1_fusion=False,
        known_prefix_tokens=2,
        block_size=8,
        mtp_t1_input_slot=1,
        input_layout=["A", "T1"] + ["MASK"] * 6,
        mtp_hidden_index_relative_to_T1=-2,
        target_context_end_offset=0,
        context_kv_boundary="strictly_before_anchor",
        block_attention="bidirectional",
        swa_window_size=4096,
        inference_requires_t1_residual_adapter=True,
    )
    for name, value in required.items():
        if cfg.get(name) != value:
            raise ValueError(
                f"Conflicting dual-anchor residual export {name}: {cfg.get(name)!r}"
            )
    if (
        getattr(config, "is_causal", None) is not False
        or cfg.get("draft_layer_context_windows") != [4096] * config.num_hidden_layers
    ):
        raise ValueError(
            "Residual export requires explicit bidirectional uniform anchor windows"
        )
    derived = dict(
        mtp_injection_mode="t1_residual",
        mtp_prefix_concat_order=[],
        draft_prediction_offsets=list(range(2, 8)),
        context_fc_input_size=5 * config.hidden_size,
        context_window_boundary="anchor_relative_not_query_relative",
    )
    out = deepcopy(config)
    for name, value in derived.items():
        if name in cfg and cfg[name] != value:
            raise ValueError(f"Conflicting residual export field {name}: {cfg[name]!r}")
        out.dflash_config[name] = value
    out.dflash_config["mtp_prefix_injection"] = "t1_residual"
    return out


def validate_config(config):
    cfg = config.dflash_config
    expected = dict(
        mtp_prefix=True,
        mtp_injection_mode="t1_residual",
        mtp_t1_residual=True,
        mtp_hidden_only=True,
        known_prefix_tokens=2,
        block_size=8,
        mtp_prefix_injection="t1_residual",
        mtp_proposal_token_count=1,
        dflash_proposal_token_count=6,
        draft_prediction_offsets=list(range(2, 8)),
        context_fc_input_size=5 * config.hidden_size,
        context_kv_boundary="strictly_before_anchor",
        context_window_boundary="anchor_relative_not_query_relative",
        block_attention="bidirectional",
        swa_window_size=4096,
        target_kv_layer_ids=[],
        legacy_double_mlp_norm=False,
    )
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(
                f"Only double-anchor T1 residual is supported: {key}={cfg.get(key)!r}"
            )
    for name in (
        "mtp_drop_anchor",
        "mtp_t1_fusion",
        "mtp_token_only",
        "mtp_context_only",
        "mtp_context_fusion",
        "mtp_context_residual",
        "mtp_native_input",
        "mtp_kv_consumer",
    ):
        if cfg.get(name, False) is not False:
            raise ValueError(f"T1 residual cannot enable {name}")
    layers = cfg.get("target_layer_ids", [])
    if (
        config.block_size != 8
        or config.num_hidden_layers != 5
        or config.head_dim != 256
        or len(layers) != 5
        or layers != sorted(set(layers))
        or layers[0] < 0
        or layers[-1] >= config.num_target_layers
        or cfg.get("draft_layer_context_windows") != [4096] * 5
        or cfg.get("mtp_prefix_rank", 0) < 1
        or cfg.get("mtp_prefix_eps") != config.rms_norm_eps
        or cfg.get("mtp_prefix_concat_order") != []
    ):
        raise ValueError(
            "Unsupported T1 residual dimensions, windows or hidden-only contract"
        )
    theta = getattr(config, "rope_theta", None)
    if theta is None:
        theta = (getattr(config, "rope_parameters", None) or {}).get("rope_theta")
    if theta != 10000000.0 or config.partial_rotary_factor != 0.25:
        raise ValueError("Unsupported T1 residual RoPE")
    if config.layer_types not in (["full_attention"] * 5, ["sliding_attention"] * 5):
        raise ValueError("Unsupported T1 residual layer types")


class MTPPrefixConditioner(nn.Module):
    """Hidden-only low-rank residual at slot 1; never changes A or MASK slots."""

    draft_slots = 8

    def __init__(self, hidden_size, rank, eps):
        super().__init__()
        self.eps = eps
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)

    def forward(self, blocks, mtp_hidden):
        if blocks.ndim != 3 or blocks.shape[1:] != (8, self.up.out_features):
            raise ValueError("Expected [B,8,H]=[A,T1,MASK*6]")
        if mtp_hidden.shape != blocks[:, 1].shape:
            raise ValueError("Expected the actual MTP hidden predicting T1")
        x = mtp_hidden.float()
        x = (x * x.square().mean(-1, keepdim=True).add(self.eps).rsqrt()).to(
            mtp_hidden.dtype
        )
        residual = self.up(F.silu(self.down(x)))
        return torch.cat(
            (blocks[:, :1], blocks[:, 1:2] + residual[:, None], blocks[:, 2:]), 1
        )
