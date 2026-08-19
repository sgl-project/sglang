# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
    MiniMaxH3DiTArchConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MiniMaxH3AdalnCache,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)

_ARCH = MiniMaxH3DiTArchConfig(
    num_layers=2,
    hidden_size=4,
    time_embed_dim=3,
)
_BLOCK_WIDTH = 6 * MINIMAX_H3_ADALN_MODALITY_NUM * _ARCH.hidden_size
_FINAL_WIDTH = 2 * _ARCH.hidden_size


def _ensure_single_process_parallel_runtime() -> None:
    if model_parallel_is_initialized():
        return
    ensure_distributed_env_defaults()
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


def _write_online_weights(
    path: Path,
    *,
    omit: str | None = None,
) -> None:
    # These cache-state tests need checkpoint-compatible shapes, not values.
    tensors: dict[str, torch.Tensor] = {}
    for layer in range(_ARCH.num_layers):
        prefix = f"blocks.{layer}.adaln_proj.linear"
        tensors[f"{prefix}.weight"] = torch.zeros(_BLOCK_WIDTH, _ARCH.time_embed_dim)
        tensors[f"{prefix}.bias"] = torch.zeros(_BLOCK_WIDTH)
    prefix = "final_layer.adaln_proj.linear"
    tensors[f"{prefix}.weight"] = torch.zeros(_FINAL_WIDTH, _ARCH.time_embed_dim)
    tensors[f"{prefix}.bias"] = torch.zeros(_FINAL_WIDTH)
    if omit is not None:
        tensors.pop(omit)
    save_file(tensors, path)


def _online_cache(
    tmp_path: Path,
    *,
    max_plans: int = 2,
    max_plan_width: int = 2,
    omit: str | None = None,
) -> MiniMaxH3AdalnCache:
    _ensure_single_process_parallel_runtime()
    weight_path = tmp_path / "model.safetensors"
    _write_online_weights(weight_path, omit=omit)
    cache = MiniMaxH3AdalnCache(
        _ARCH,
        weight_files=[str(weight_path)],
        max_plans=max_plans,
        max_plan_width=max_plan_width,
    )
    cache.load(torch.device("cpu"))
    return cache


def _embed(timesteps: torch.Tensor) -> torch.Tensor:
    return timesteps[:, None].expand(-1, _ARCH.time_embed_dim)


def test_minimax_h3_adaln_cache_matches_bf16_embedding(tmp_path):
    cache_path = tmp_path / "adaln.safetensors"
    plan_timesteps = torch.tensor([[0.0, 0.0], [1.0, 2.0]])
    plan_lengths = torch.tensor([1, 2], dtype=torch.int64)
    block_params = (
        torch.arange(2 * 2 * 2 * _BLOCK_WIDTH, dtype=torch.float32)
        .reshape(2, 2, 2, _BLOCK_WIDTH)
        .bfloat16()
    )
    final_params = (
        torch.arange(2 * 2 * _FINAL_WIDTH, dtype=torch.float32)
        .reshape(2, 2, _FINAL_WIDTH)
        .bfloat16()
    )
    save_file(
        {
            "plan_timesteps": plan_timesteps,
            "plan_lengths": plan_lengths,
            "block_params": block_params,
            "final_params": final_params,
        },
        cache_path,
        metadata={"format_version": "2", "model_variant": "fl2va"},
    )

    cache = MiniMaxH3AdalnCache(
        _ARCH,
        path=str(cache_path),
        model_variant="fl2va",
    )
    cache.load(torch.device("cpu"))

    cache_plan_index = cache.lookup(plan_timesteps[1])
    block = cache.block(1, cache_plan_index, 2)
    final = cache.final(cache_plan_index, 2)

    # block() hands the forward pass six [num_timesteps * modality, hidden]
    # chunks, while the checkpoint stores a plan as one flat
    # [num_timesteps, 6 * modality * hidden] row -- same elements, and the
    # modality axis folds into the leading one rather than staying separate.
    assert torch.equal(
        torch.cat(block, dim=-1).reshape(block_params[1, :, 1].shape),
        block_params[1, :, 1],
    )
    assert torch.equal(torch.cat(final, dim=-1), final_params[1])


def test_online_cache_reset_rebuilds_previously_resident_request_plans(tmp_path):
    """A capacity reset must not drop plans reused by the current request."""
    cache = _online_cache(tmp_path, max_plan_width=1)
    plan_a = torch.tensor([1.0])
    plan_b = torch.tensor([2.0])
    plan_c = torch.tensor([3.0])

    cache.build([plan_a, plan_b], embed=_embed)
    cache.build([plan_a, plan_c], embed=_embed)

    cache.lookup(plan_a)
    cache.lookup(plan_c)


def test_online_cache_failed_rebuild_can_be_retried(tmp_path):
    """A failed rebuild must not publish a cache hit that blocks its retry."""
    missing_name = "final_layer.adaln_proj.linear.bias"
    cache = _online_cache(tmp_path, omit=missing_name)
    plan_a = torch.tensor([1.0])

    with pytest.raises(KeyError, match=missing_name):
        cache.build([plan_a], embed=_embed)

    _write_online_weights(tmp_path / "model.safetensors")
    cache.build([plan_a], embed=_embed)
    cache.lookup(plan_a)


def test_online_cache_width_rejection_preserves_resident_plans(tmp_path):
    """Rejecting an over-width plan must not evict usable resident plans."""
    cache = _online_cache(tmp_path, max_plan_width=1)
    plan_a = torch.tensor([1.0])
    plan_b = torch.tensor([2.0])
    wide_plan = torch.tensor([3.0, 4.0])

    cache.build([plan_a, plan_b], embed=_embed)
    with pytest.raises(ValueError, match="--minimax-h3-adaln-plan-width"):
        cache.build([wide_plan], embed=_embed)

    cache.lookup(plan_a)
    cache.lookup(plan_b)
