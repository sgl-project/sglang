"""Unit tests for running DFLASH speculative decoding on CPU.

Covers the pieces that used to be CUDA/NPU-only: the server-arg device gate and
draft attention backend resolution, the linear-chain `tree_topk` that the
intel_amx backend branches on, and the AMX-prepacked weight guards that keep
DFlash from reading packed weights as if they were dense matrices.
"""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import (
    _disable_overlap_schedule_for_cpu,
    _handle_dflash,
    _resolve_dflash_draft_attention_backend,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_utils import (
    can_dflash_slice_qkv_weight,
    can_dflash_use_fused_qkv_proj,
    dflash_head_logits,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _server_args(**overrides) -> ServerArgs:
    # ServerArgs(model_path="dummy") early-returns the resolution pipeline, so
    # the raw fields below are still writable.
    server_args = ServerArgs(model_path="dummy")
    server_args.device = "cpu"
    server_args.speculative_algorithm = "DFLASH"
    server_args.speculative_draft_model_path = "/tmp/draft"
    server_args.speculative_draft_attention_backend = None
    for key, value in overrides.items():
        setattr(server_args, key, value)
    return server_args


def test_dflash_device_gate_rejects_an_unsupported_device():
    with pytest.raises(ValueError, match="CUDA, NPU and CPU"):
        _handle_dflash(_server_args(device="xpu"))


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:0", "npu"])
def test_dflash_device_gate_admits_the_supported_devices(device):
    # dp attention is rejected immediately after the device gate, so reaching
    # that error is what proves the device itself was accepted.
    args = _server_args(device=device, enable_dp_attention=True)
    with pytest.raises(ValueError, match="dp attention"):
        _handle_dflash(args)


def test_dflash_cpu_disables_overlap_schedule():
    args = _server_args()
    _disable_overlap_schedule_for_cpu(args)
    assert resolution_result(args, "disable_overlap_schedule") is True


def test_dflash_leaves_overlap_schedule_alone_off_cpu():
    args = _server_args(device="cuda")
    _disable_overlap_schedule_for_cpu(args)
    assert resolution_result(args, "disable_overlap_schedule") is not True


def test_dflash_cpu_draft_backend_defaults_to_intel_amx():
    # Without this the draft inherits a GPU default and fails at its first
    # forward.
    args = _server_args()
    _resolve_dflash_draft_attention_backend(args)
    assert resolution_result(args, "speculative_draft_attention_backend") == "intel_amx"


def test_dflash_cpu_draft_backend_is_kept_when_explicit():
    args = _server_args(speculative_draft_attention_backend="intel_amx")
    _resolve_dflash_draft_attention_backend(args)
    assert resolution_result(args, "speculative_draft_attention_backend") == "intel_amx"


def test_dflash_verify_input_is_a_linear_chain():
    # intel_amx skips the explicit mask (and takes the faster kernel path) only
    # when tree_topk == 1.
    verify_input = DFlashVerifyInput(
        draft_token=torch.zeros(4, dtype=torch.long),
        positions=torch.zeros(4, dtype=torch.long),
        draft_token_num=4,
        custom_mask=None,
        capture_hidden_mode=CaptureHiddenMode.FULL,
    )
    assert verify_input.tree_topk == 1


def _fake_qkv_proj(*, packed: bool):
    proj = SimpleNamespace(
        quant_method=UnquantizedLinearMethod(),
        weight=torch.zeros(6, 4),
        bias=None,
    )
    if packed:
        proj.use_intel_amx_backend = True
    return proj


def test_amx_prepacked_qkv_cannot_be_sliced():
    eligible, reason = can_dflash_slice_qkv_weight(_fake_qkv_proj(packed=True))
    assert not eligible
    assert "AMX" in reason
    assert not can_dflash_use_fused_qkv_proj(_fake_qkv_proj(packed=True))[0]

    assert can_dflash_slice_qkv_weight(_fake_qkv_proj(packed=False))[0]


def test_dflash_head_logits_dense_path_matches_matmul():
    hidden = torch.randn(3, 8)
    lm_head = SimpleNamespace(weight=torch.randn(10, 8))

    torch.testing.assert_close(
        dflash_head_logits(lm_head, hidden), hidden @ lm_head.weight.T
    )
    torch.testing.assert_close(
        dflash_head_logits(lm_head, hidden, 0, 6), hidden @ lm_head.weight[:6].T
    )
    torch.testing.assert_close(
        dflash_head_logits(lm_head, hidden, 6, 10), hidden @ lm_head.weight[6:10].T
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
