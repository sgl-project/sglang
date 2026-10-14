# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Wan ModelSlim offline Q/K rotation descriptors."""

import pytest
import torch

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackendEnum,
)
from sglang.multimodal_gen.runtime.layers.linear import UnquantizedLinearMethod
from sglang.multimodal_gen.runtime.models.dits.wanvideo import WanTransformerBlock


class _RotatedQuantConfig:
    """A quantized checkpoint that ships offline Q/K rotation matrices for one block."""

    def __init__(
        self, prefix: str, names: tuple[str, str] = ("q_rot", "k_rot")
    ) -> None:
        q_name, k_name = names
        self.quant_description = {
            f"{prefix}.attn1.{q_name}": "FLOAT",
            f"{prefix}.attn1.{k_name}": "FLOAT",
        }

    def get_quant_method(self, layer, prefix: str) -> UnquantizedLinearMethod:
        return UnquantizedLinearMethod()


@pytest.fixture(autouse=True)
def _single_process_parallel_runtime(monkeypatch):
    # The DiT's parallel linear layers need the TP group even on a meta device.
    if model_parallel_is_initialized():
        return
    for name, value in {
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29505",
        "RANK": "0",
        "LOCAL_RANK": "0",
        "WORLD_SIZE": "1",
    }.items():
        monkeypatch.setenv(name, value)
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


def test_block_registers_offline_qk_rotation_from_to_prefixed_descriptors():
    """msmodelslim names the exported rotations `to_q_rot`/`to_k_rot` after the
    to_q/to_k projections; the block used to look up only `q_rot`/`k_rot`, so the
    rotation buffers never registered and the tensors were skipped at load."""
    with torch.device("meta"):
        block = WanTransformerBlock(
            64,
            128,
            2,
            "rms_norm",
            True,
            1e-6,
            None,
            {AttentionBackendEnum.TORCH_SDPA},
            prefix="blocks.0",
            quant_config=_RotatedQuantConfig(
                "blocks.0", names=("to_q_rot", "to_k_rot")
            ),
        )
    assert block.use_offline_qk_rotation
    buffers = dict(block.named_buffers())
    assert buffers["q_rot"].shape == (32, 32)
    assert buffers["k_rot"].shape == (32, 32)


def test_block_accepts_legacy_bare_rotation_descriptor_names():
    with torch.device("meta"):
        block = WanTransformerBlock(
            64,
            128,
            2,
            "rms_norm",
            True,
            1e-6,
            None,
            {AttentionBackendEnum.TORCH_SDPA},
            prefix="blocks.0",
            quant_config=_RotatedQuantConfig("blocks.0"),
        )
    assert block.use_offline_qk_rotation


def test_block_rejects_a_q_rotation_exported_without_k():
    # Rotating Q but not K changes every attention score; half an export must
    # fail loudly instead of loading.
    config = _RotatedQuantConfig("blocks.0", names=("to_q_rot", "to_k_rot"))
    config.quant_description.pop("blocks.0.attn1.to_k_rot")
    with torch.device("meta"), pytest.raises(ValueError, match="both Q/K rotation"):
        WanTransformerBlock(
            64,
            128,
            2,
            "rms_norm",
            True,
            1e-6,
            None,
            {AttentionBackendEnum.TORCH_SDPA},
            prefix="blocks.0",
            quant_config=config,
        )
