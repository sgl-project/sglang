import typing

from sglang_simulator.simulation.types import SchedulerConfig
from sglang_simulator.spec import DataType, ModelInfo

if typing.TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs


def resolve_scheduler_config(
    server_args: "ServerArgs",
) -> SchedulerConfig:
    from sglang.version import __version__

    dtype = server_args.dtype
    if dtype == "auto":
        dtype = str(server_args.model_config.dtype).strip("torch.")
    data_type = DataType.from_torch_dtype(dtype)
    return SchedulerConfig(
        data_type=data_type,
        kv_cache_data_type=DataType.from_torch_dtype(server_args.kv_cache_dtype)
        or data_type,
        mem_fraction_static=server_args.mem_fraction_static,
        max_total_tokens=server_args.max_total_tokens,
        tp_size=server_args.tp_size,
        ep_size=server_args.ep_size,
        dp_size=server_args.dp_size,
        pp_size=server_args.pp_size,
        page_size=getattr(server_args, "page_size", None),
        swa_full_tokens_ratio=getattr(server_args, "swa_full_tokens_ratio", None),
        kv_bytes_per_token_per_gpu=getattr(
            server_args, "kv_bytes_per_token_per_gpu", None
        ),
        hicache_ratio=getattr(server_args, "hicache_ratio", None),
        enable_hierarchical_cache=getattr(
            server_args, "enable_hierarchical_cache", None
        ),
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
