import typing

from sglang_simulator.simulation.types import SchedulerConfig
from sglang_simulator.spec import DataType, ModelInfo

if typing.TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs


def _resolve_model_config(server_args: "ServerArgs", model_config=None):
    if model_config is not None:
        return model_config

    get_model_config = getattr(server_args, "get_model_config", None)
    if callable(get_model_config):
        return get_model_config()

    return server_args.model_config


def _resolved_server_args(server_args: "ServerArgs") -> dict:
    """Return effective ServerArgs values when the runtime exposes them."""
    resolved_dict = getattr(server_args, "resolved_dict", None)
    if not callable(resolved_dict):
        return {}

    try:
        values = resolved_dict()
    except (AttributeError, RuntimeError, TypeError):
        return {}

    return values if isinstance(values, dict) else {}


def resolve_scheduler_config(
    server_args: "ServerArgs",
    model_config: typing.Optional["ModelConfig"] = None,
) -> SchedulerConfig:
    from sglang.version import __version__

    resolved = _resolved_server_args(server_args)

    def get_arg(name: str, default=None):
        value = resolved.get(name)
        if value is None:
            value = getattr(server_args, name, None)
        return default if value is None else value

    dtype = get_arg("dtype", "auto")
    if dtype == "auto":
        model_config = _resolve_model_config(server_args, model_config)
        dtype = str(model_config.dtype).strip("torch.")
    data_type = DataType.from_torch_dtype(dtype)
    return SchedulerConfig(
        data_type=data_type,
        kv_cache_data_type=DataType.from_torch_dtype(get_arg("kv_cache_dtype"))
        or data_type,
        mem_fraction_static=get_arg("mem_fraction_static"),
        max_total_tokens=get_arg("max_total_tokens"),
        tp_size=get_arg("tp_size"),
        ep_size=get_arg("ep_size"),
        dp_size=get_arg("dp_size"),
        pp_size=get_arg("pp_size"),
        cp_size=get_arg("attn_cp_size", 1),
        cp_style=get_arg("cp_style", "none"),
        page_size=get_arg("page_size"),
        swa_full_tokens_ratio=get_arg("swa_full_tokens_ratio"),
        kv_bytes_per_token_per_gpu=get_arg("kv_bytes_per_token_per_gpu"),
        hicache_ratio=get_arg("hicache_ratio"),
        enable_hierarchical_cache=get_arg("enable_hierarchical_cache"),
        backend_name="sglang",
        backend_version=__version__,
    )


def resolve_model_info(model_config: "ModelConfig") -> ModelInfo:
    from sglang.srt.configs.model_config import AttentionArch

    torch_dtype = str(model_config.dtype).strip("torch.")
    if model_config.attention_arch == AttentionArch.MHA:
        return ModelInfo(
            hf_config=model_config.hf_text_config,
            model_path=model_config.model_path,
            attention_arch="MHA",
            context_len=model_config.context_len,
            hidden_size=model_config.hidden_size,
            head_dim=model_config.head_dim,
            num_attention_heads=model_config.num_attention_heads,
            num_hidden_layers=model_config.num_hidden_layers,
            num_key_value_heads=model_config.num_key_value_heads,
            v_head_dim=model_config.v_head_dim,
            vocab_size=model_config.vocab_size,
            # DSv4-style models (e.g. DSv4-Pro) report attention_arch=MHA because
            # sglang routes them through a custom `attention_backend='dsv4'`, not
            # MLA. But they still carry compress_ratios + indexer + SWA fields on
            # ModelConfig, and is_dsv4() needs them to take the right calculator
            # branch. getattr makes this a no-op for true MHA models.
            compression_ratios=getattr(model_config, "compress_ratios", None),
            indexer_head_dim=getattr(model_config, "index_head_dim", None),
            window_size=getattr(model_config, "window_size", None),
            qk_nope_head_dim=getattr(model_config, "qk_nope_head_dim", None),
            qk_rope_head_dim=getattr(model_config, "qk_rope_head_dim", None),
            torch_dtype=torch_dtype,
        )
    elif model_config.attention_arch == AttentionArch.MLA:
        return ModelInfo(
            hf_config=model_config.hf_text_config,
            model_path=model_config.model_path,
            attention_arch="MLA",
            context_len=model_config.context_len,
            hidden_size=model_config.hidden_size,
            head_dim=model_config.head_dim,
            num_attention_heads=model_config.num_attention_heads,
            num_hidden_layers=model_config.num_hidden_layers,
            num_key_value_heads=model_config.num_key_value_heads,
            v_head_dim=model_config.v_head_dim,
            vocab_size=model_config.vocab_size,
            qk_rope_head_dim=model_config.qk_rope_head_dim,
            qk_nope_head_dim=model_config.qk_nope_head_dim,
            kv_lora_rank=model_config.kv_lora_rank,
            compression_ratios=getattr(model_config, "compress_ratios", None),
            indexer_head_dim=getattr(model_config, "index_head_dim", None),
            window_size=getattr(model_config, "window_size", None),
            torch_dtype=torch_dtype,
        )
    else:
        raise ValueError(
            f"The attention type of `{model_config.attention_arch}` is not supported now."
        )
