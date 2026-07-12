# SPDX-License-Identifier: Apache-2.0
"""DreamZero-DROID DiT checkpoint loader.

DreamZero checkpoints are packaged as a full VLA under ``action_head.*``.  The
causal Wan DiT lives below ``action_head.model.*`` and already uses parameter
names that match :mod:`dreamzero_causal` after that prefix is stripped.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.dreamzero_checkpoint_utils import (
    DreamZeroCheckpointLoadReport,
    assign_tensor,
    iter_indexed_safetensors,
    iter_prefixed_safetensors,
    iter_safetensor_file,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.dreamzero_config import (
    dreamzero_dit_init_kwargs_from_checkpoint_config,
    load_dreamzero_checkpoint_config,
)
from sglang.multimodal_gen.runtime.models.dits.dreamzero_causal import (
    DreamZeroCausalWanModel,
    rope_params,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_DROID_DIT_PREFIX = "action_head.model."
_DROID_INDEX_NAME = "model.safetensors.index.json"
_WAN_DIT_INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
_WAN_DIT_SINGLE_NAME = "diffusion_pytorch_model.safetensors"


def remap_dreamzero_dit_key(checkpoint_key: str) -> str | None:
    if not checkpoint_key.startswith(_DROID_DIT_PREFIX):
        return None
    return checkpoint_key[len(_DROID_DIT_PREFIX) :]


def _has_wan_component_dit(model_path: str | os.PathLike[str]) -> bool:
    model_dir = Path(model_path)
    return (model_dir / _WAN_DIT_INDEX_NAME).is_file() or (
        model_dir / _WAN_DIT_SINGLE_NAME
    ).is_file()


def _load_config(model_path: str | os.PathLike[str]) -> dict[str, Any]:
    return load_dreamzero_checkpoint_config(model_path)


def _iter_wan_component_safetensors(
    model_path: str | os.PathLike[str],
) -> Iterator[tuple[str, torch.Tensor]]:
    model_dir = Path(model_path)
    index_path = model_dir / _WAN_DIT_INDEX_NAME
    single_path = model_dir / _WAN_DIT_SINGLE_NAME
    if index_path.is_file():
        yield from iter_indexed_safetensors(model_dir, index_name=_WAN_DIT_INDEX_NAME)
    elif single_path.is_file():
        yield from iter_safetensor_file(single_path)
    else:
        raise FileNotFoundError(
            f"Cannot find {_WAN_DIT_INDEX_NAME} or {_WAN_DIT_SINGLE_NAME} in {model_dir}"
        )


def _iter_dit_tensors(
    model_path: str | os.PathLike[str],
) -> Iterator[tuple[str, torch.Tensor]]:
    if _has_wan_component_dit(model_path):
        yield from _iter_wan_component_safetensors(model_path)
    else:
        yield from iter_prefixed_safetensors(model_path, _DROID_DIT_PREFIX)


def _remap_dit_key(
    model_path: str | os.PathLike[str], checkpoint_key: str
) -> str | None:
    if _has_wan_component_dit(model_path):
        return checkpoint_key
    return remap_dreamzero_dit_key(checkpoint_key)


def _copy_weight_loader_attrs(src: torch.Tensor, dst: torch.Tensor) -> None:
    for attr in (
        "input_dim",
        "output_dim",
        "is_sharded_weight",
        "needs_scalar_to_array",
        "is_metadata",
    ):
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))


def _materialize_rope_tensors(
    model: DreamZeroCausalWanModel, device: torch.device
) -> None:
    head_dim = model.dim // model.num_heads
    model.freqs_action = rope_params(1024 * 10, head_dim).to(device)
    model.freqs_state = rope_params(1024, head_dim).to(device)
    model.freqs = [
        rope_params(1024, head_dim - 4 * (head_dim // 6)).to(device),
        rope_params(1024, 2 * (head_dim // 6)).to(device),
        rope_params(1024, 2 * (head_dim // 6)).to(device),
    ]


def load_dreamzero_dit_checkpoint(
    model: DreamZeroCausalWanModel,
    model_path: str | os.PathLike[str],
    *,
    device: torch.device,
    strict: bool = False,
) -> DreamZeroCheckpointLoadReport:
    meta_sd = model.state_dict()
    param_dict = dict(model.named_parameters())
    loaded_keys: list[str] = []
    unexpected_keys: list[str] = []
    shape_mismatches: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}

    with torch.no_grad():
        for checkpoint_key, full_tensor in _iter_dit_tensors(model_path):
            target_name = _remap_dit_key(model_path, checkpoint_key)
            if target_name is None:
                continue
            meta_tensor = meta_sd.get(target_name)
            actual_param = param_dict.get(target_name)
            if meta_tensor is None:
                unexpected_keys.append(target_name)
                continue

            target_dtype = meta_tensor.dtype
            full_tensor = full_tensor.to(device=device, dtype=target_dtype)
            weight_loader = (
                getattr(actual_param, "weight_loader", None)
                if actual_param is not None
                else None
            )

            if tuple(full_tensor.shape) == tuple(meta_tensor.shape):
                target_tensor = full_tensor
            elif weight_loader is not None and actual_param is not None:
                target_tensor = torch.empty(
                    tuple(meta_tensor.shape), device=device, dtype=target_dtype
                )
                temp_param = nn.Parameter(target_tensor, requires_grad=False)
                _copy_weight_loader_attrs(actual_param, temp_param)
                weight_loader(temp_param, full_tensor)
                target_tensor = temp_param.data
            else:
                shape_mismatches[target_name] = (
                    tuple(meta_tensor.shape),
                    tuple(full_tensor.shape),
                )
                continue

            assign_tensor(model, target_name, target_tensor)
            loaded_keys.append(target_name)

    missing_keys = sorted(set(meta_sd) - set(loaded_keys))
    report = DreamZeroCheckpointLoadReport(
        loaded_keys=loaded_keys,
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
        shape_mismatches=shape_mismatches,
    )
    allow_missing = _has_wan_component_dit(model_path)
    if strict and (
        (report.missing_keys and not allow_missing)
        or (report.unexpected_keys and not allow_missing)
        or report.shape_mismatches
    ):
        raise RuntimeError(f"DreamZero DiT checkpoint load failed: {report.as_dict()}")
    return report


def build_dreamzero_dit_from_checkpoint(
    model_path: str | os.PathLike[str],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    use_tensor_parallel: bool = False,
) -> DreamZeroCausalWanModel:
    kwargs = dreamzero_dit_init_kwargs_from_checkpoint_config(
        model_path,
        use_tensor_parallel=use_tensor_parallel,
    )
    if _has_wan_component_dit(model_path):
        model = DreamZeroCausalWanModel(**kwargs).to(device=device, dtype=dtype)
    else:
        with torch.device("meta"):
            model = DreamZeroCausalWanModel(**kwargs).to(dtype=dtype)
    _materialize_rope_tensors(model, device)
    model.eval()
    return model


class DreamZeroDiTLoader(ComponentLoader):
    """Loader entry for DreamZero DiT weights embedded in a full VLA checkpoint."""

    component_names = ["dreamzero_dit"]
    expected_library = "diffusers"

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ) -> DreamZeroCausalWanModel:
        use_tensor_parallel = bool(
            getattr(server_args.pipeline_config, "dreamzero_use_tensor_parallel", False)
        )
        device = get_local_torch_device()
        model = build_dreamzero_dit_from_checkpoint(
            component_model_path,
            device=device,
            dtype=torch.bfloat16,
            use_tensor_parallel=use_tensor_parallel,
        )
        report = load_dreamzero_dit_checkpoint(
            model,
            component_model_path,
            device=device,
            strict=True,
        )
        logger.info("Loaded DreamZero DiT checkpoint: %s", report.as_dict())
        return model
