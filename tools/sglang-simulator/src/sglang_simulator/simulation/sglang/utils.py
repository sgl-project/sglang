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
            torch_dtype=torch_dtype,
        )
    else:
        raise ValueError(
            f"The attention type of `{model_config.attention_arch}` is not supported now."
        )
