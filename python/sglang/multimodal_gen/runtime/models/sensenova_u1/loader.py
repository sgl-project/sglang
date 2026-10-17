# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.models import (  # noqa: F401
    sensenova_u1 as _sensenova_u1,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify.modeling_qwen3 import (
    Qwen3MLP,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.precision_types import PRECISION_TO_TYPE


def load_model_and_tokenizer(
    model_path: str,
    server_args: ServerArgs,
) -> dict[str, Any]:
    dtype = PRECISION_TO_TYPE.get(
        server_args.pipeline_config.model_precision, torch.bfloat16
    )
    model_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if server_args.trust_remote_code:
        model_kwargs["trust_remote_code"] = True
    if server_args.revision is not None:
        model_kwargs["revision"] = server_args.revision

    tokenizer_kwargs: dict[str, Any] = {}
    if server_args.trust_remote_code:
        tokenizer_kwargs["trust_remote_code"] = True
    if server_args.revision is not None:
        tokenizer_kwargs["revision"] = server_args.revision

    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    model = AutoModel.from_pretrained(model_path, **model_kwargs).eval()
    device = get_local_torch_device()
    current_platform.set_device(device)
    model = model.to(device)
    if device.type != "cpu":
        _pack_mlp_gate_up_weights(model, device)
    return {"model": model, "tokenizer": tokenizer}


def _pack_mlp_gate_up_weights(model: torch.nn.Module, device: torch.device) -> None:
    for module in model.modules():
        if isinstance(module, Qwen3MLP):
            module.pack_gate_up_weight()
    # Each pack frees a gate/up pair whose blocks cannot hold the next packed
    # buffer; return them now rather than keep them reserved while serving.
    torch.get_device_module(device).empty_cache()
