"""Translate DModel DeepSeek-V4-Flash HF configs to the V4 runtime's flat schema.

These checkpoints keep the public V4-Flash geometry but differ from the release in
config-visible ways: softmax noaux_tc routing with an optional FP32 router, dense
FFNs in the first layers, identity hyper-connections, and RoPE settings nested
under ``main``/``compress``. Only the variant the runtime implements is accepted;
anything else is rejected here rather than mis-executed later.
"""

from transformers import DeepseekV3Config

DMODEL_DEEPSEEK_V4_MODEL_TYPE = "dmodel_deepseek_v4_flash"

# DModel-only keys with no runtime meaning. ``layer_types`` uses custom names
# ("compressed_sparse_attention", ...) that generic sliding-window detection must
# never see; ``compress_ratios`` already carries the same layout.
_DROPPED_KEYS = ("layer_types", "compress_rates", "auto_map")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(
            f"Unsupported {DMODEL_DEEPSEEK_V4_MODEL_TYPE} checkpoint: {message}"
        )


def _flatten_rope(values: dict) -> None:
    """Rewrite ``{"main": ..., "compress": ...}`` RoPE into the public V4 form.

    V4 already selects RoPE per layer: SWA-only layers use the unscaled
    ``rope_theta`` and compressed layers use ``compress_rope_theta`` with the
    YaRN ``rope_scaling``. mscale is dropped because V4 attention never applies it.
    """
    rope = values.get("rope_scaling")
    if not isinstance(rope, dict) or not ({"main", "compress"} & rope.keys()):
        return
    main = rope.get("main") or {}
    compress = rope.get("compress") or {}
    main_type = main.get("rope_type", main.get("type", "default"))
    _require(main_type == "default", f"main RoPE must be unscaled, got {main_type!r}")
    compress_type = compress.get("rope_type", compress.get("type"))
    _require(
        compress_type == "yarn", f"compressed RoPE must be YaRN, got {compress_type!r}"
    )
    for branch in (main, compress):
        factor = branch.get("partial_rotary_factor")
        if factor is not None:
            _require(
                factor * values["head_dim"] == values["qk_rope_head_dim"],
                f"partial_rotary_factor={factor} does not give "
                f"qk_rope_head_dim={values['qk_rope_head_dim']}",
            )
    if "rope_theta" in main:
        values["rope_theta"] = main["rope_theta"]
    if "rope_theta" in compress:
        values["compress_rope_theta"] = compress["rope_theta"]
    values["rope_scaling"] = {
        "type": "yarn",
        "factor": compress["factor"],
        "original_max_position_embeddings": compress[
            "original_max_position_embeddings"
        ],
        "beta_fast": compress.get("beta_fast", 32),
        "beta_slow": compress.get("beta_slow", 1),
    }


def _normalize_dense_layers(values: dict) -> None:
    num_dense = int(
        values.get("n_dense_layers", values.get("first_k_dense_replace", 0)) or 0
    )
    _require(
        int(values.get("first_k_dense_replace", num_dense) or 0) == num_dense,
        "first_k_dense_replace and n_dense_layers disagree",
    )
    if num_dense:
        _require(
            values.get("intermediate_size") is not None,
            "dense layers require intermediate_size",
        )
        for key in ("dense_intermediate_size", "dense_inter_dim"):
            _require(
                values.get(key, values["intermediate_size"])
                == values["intermediate_size"],
                f"{key} differs from intermediate_size",
            )
    # The runtime keys dense layers off n_dense_layers: DeepseekV3Config defaults
    # first_k_dense_replace to 3, so the public V4 config must not reach that path.
    values["n_dense_layers"] = num_dense
    values["first_k_dense_replace"] = num_dense


def _check_supported_layout(values: dict) -> None:
    _require(
        values.get("csa_stage_mode", "sparse") == "sparse",
        f"csa_stage_mode={values.get('csa_stage_mode')!r}; only 'sparse' runs",
    )
    _require(
        bool(values.get("dsa_indexer_enabled", True))
        and bool(values.get("dsa_indexer_use_for_attention", True)),
        "sparse attention requires an indexer that selects attention entries",
    )
    _require(
        not values.get("enable_attnres", False), "AttnRes residuals are unsupported"
    )
    _require(
        bool(values.get("enable_hyper_connection", True))
        and int(values.get("hc_mult", 1)) > 1,
        "the V4 runtime requires multi-stream hyper-connections",
    )
    _require(
        values.get("head_dim") == 512 and values.get("qk_rope_head_dim") == 64,
        "only 512-wide attention heads with 64 RoPE dims are supported",
    )
    num_layers = values["num_hidden_layers"]
    ratios = values["compress_ratios"]
    _require(
        len(ratios) >= num_layers and set(ratios[:num_layers]) <= {0, 4, 128},
        f"compress_ratios must cover {num_layers} layers with ratios 0/4/128",
    )
    _require(
        int(values.get("num_nextn_predict_layers") or 0) == 0,
        "next-token prediction layers are unsupported",
    )
    _require(
        int(values.get("num_hash_layers") or 0) == 0, "hash-routed MoE is unsupported"
    )
    scoring_func = values.get("scoring_func")
    _require(
        scoring_func in ("softmax", "sqrtsoftplus"),
        f"scoring_func={scoring_func!r}",
    )


def normalize_dmodel_deepseek_v4_config(values: dict) -> dict:
    values = dict(values)
    # Transformers builds default instances (e.g. in to_diff_dict); only a
    # checkpoint config is validated and rewritten.
    if "compress_ratios" not in values:
        return values

    _check_supported_layout(values)
    hc_residual_mode = values.setdefault("hc_residual_mode", "sinkhorn")
    _require(
        hc_residual_mode in ("identity", "sinkhorn"),
        f"hc_residual_mode={hc_residual_mode!r}",
    )
    _normalize_dense_layers(values)
    _flatten_rope(values)

    router_dtype = values.get("dmodel_router_dtype")
    _require(
        router_dtype in (None, "float32", "bfloat16"),
        f"dmodel_router_dtype={router_dtype!r}",
    )
    values["router_fp32"] = router_dtype == "float32"

    # generation_config.json carries V4's ids; config.json leaves them null.
    if values.get("bos_token_id") is None:
        values["bos_token_id"] = 0
    if values.get("eos_token_id") is None:
        values["eos_token_id"] = 1

    for key in _DROPPED_KEYS:
        values.pop(key, None)
    return values


class DModelDeepseekV4FlashConfig(DeepseekV3Config):
    model_type = DMODEL_DEEPSEEK_V4_MODEL_TYPE
    # The stock DeepSeek-V4 runtime fields (see _DeepseekV4ConfigAlias).
    hc_pre_from_prev_sublayer = False
    q_head_norm = True
    kv_source_layer_ids = ()
    index_source_layer_ids = ()
    candidate_source_layer_id = -1
    candidate_topk_blocks = 0
    candidate_block_size = 0
    engram_layer_ids = ()
    engram_num_embeddings = ()
    engram_max_ngram_size = 1
    engram_vocab_size = 0
    engram_n_heads = 0
    engram_head_dim = 0
    engram_pad_token_id = 2
    engram_compressed_vocab_size = 0
    # The DModel fields, at their stock DeepSeek-V4 values.
    hc_residual_mode = "sinkhorn"
    n_dense_layers = 0
    router_fp32 = False

    def __init__(self, **kwargs):
        kwargs = normalize_dmodel_deepseek_v4_config(kwargs)
        kwargs["model_type"] = DMODEL_DEEPSEEK_V4_MODEL_TYPE
        super().__init__(**kwargs)

    def to_dict(self):
        values = super().to_dict()
        values["model_type"] = DMODEL_DEEPSEEK_V4_MODEL_TYPE
        return values
