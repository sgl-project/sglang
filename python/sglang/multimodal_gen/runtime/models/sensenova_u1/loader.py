# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch
from transformers import AutoTokenizer

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.fsdp_load import maybe_load_fsdp_model
from sglang.multimodal_gen.runtime.loader.utils import _list_safetensors_files
from sglang.multimodal_gen.runtime.models import (  # noqa: F401
    sensenova_u1 as _sensenova_u1,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.configuration_neo_chat import (
    NEOChatConfig,
    NEOLLMConfig,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_neo_chat import (
    NEOChatModel,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.precision_types import PRECISION_TO_TYPE

_SUPPORTED_LLM_CONFIG = {
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_hidden_layers": 42,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
}


def _validate_supported_checkpoint(config: NEOChatConfig) -> None:
    llm_config = config.llm_config
    if not isinstance(llm_config, NEOLLMConfig):
        raise TypeError(
            "SenseNova tensor parallelism currently supports only the dense "
            "sensenova/SenseNova-U1.5-8B-MoT checkpoint; MoE backbones are unsupported."
        )
    mismatches = {
        name: (getattr(llm_config, name, None), expected)
        for name, expected in _SUPPORTED_LLM_CONFIG.items()
        if getattr(llm_config, name, None) != expected
    }
    if mismatches:
        details = ", ".join(
            f"{name}={actual!r} (expected {expected!r})"
            for name, (actual, expected) in mismatches.items()
        )
        raise ValueError(
            "SenseNova tensor parallelism currently supports only "
            f"sensenova/SenseNova-U1.5-8B-MoT; incompatible config: {details}"
        )
    if bool(getattr(llm_config, "tie_word_embeddings", False)):
        raise ValueError(
            "SenseNova-U1.5-8B-MoT expects independent embedding and lm_head weights"
        )


def load_model_and_tokenizer(
    model_path: str,
    server_args: ServerArgs,
) -> dict[str, Any]:
    dtype = PRECISION_TO_TYPE.get(
        server_args.pipeline_config.model_precision, torch.bfloat16
    )
    tokenizer_kwargs: dict[str, Any] = {}
    if server_args.trust_remote_code:
        tokenizer_kwargs["trust_remote_code"] = True
    if server_args.revision is not None:
        tokenizer_kwargs["revision"] = server_args.revision

    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    config = NEOChatConfig.from_pretrained(model_path)
    _validate_supported_checkpoint(config)

    # This flag is intentionally injected by the native loader rather than
    # serialized into the upstream config. Direct Transformers users retain the
    # original nn.Linear model, while SGLang constructs rank-local TP layers.
    config.llm_config.use_sglang_tp = True

    weight_files = _list_safetensors_files(model_path)
    if not weight_files:
        raise ValueError(f"No safetensors checkpoint found at {model_path!r}")

    device = get_local_torch_device()
    current_platform.set_device(device)
    model = maybe_load_fsdp_model(
        model_cls=NEOChatModel,
        init_params={"config": config},
        weight_dir_list=weight_files,
        device=device,
        hsdp_replicate_dim=1,
        hsdp_shard_dim=1,
        param_dtype=dtype,
        reduce_dtype=torch.float32,
        component_starts_on_cpu=False,
        fsdp_inference=False,
        pin_cpu_memory=getattr(server_args, "pin_cpu_memory", True),
        strict=True,
    )
    return {"model": model, "tokenizer": tokenizer}
