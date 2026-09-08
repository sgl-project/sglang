import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.lora.backend.base_backend import (
    BaseLoRABackend,
    _compute_moe_lora_info,
)
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cuda_ci,
    register_xpu_ci,
)

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=5, stage="stage-b", runner_config="1-gpu-small-amd")
register_xpu_ci(est_time=20, suite="stage-a-test-1-gpu-xpu")

DEVICE = get_device()


def _expected_adapter_enabled(
    lora_ranks: torch.Tensor,
    weight_indices: torch.Tensor,
) -> torch.Tensor:
    expected = torch.zeros_like(lora_ranks)
    expected.scatter_(
        0,
        weight_indices.long(),
        (lora_ranks[weight_indices.long()] > 0).to(torch.int32),
    )
    return expected


@pytest.mark.parametrize("use_preallocated_buffers", [False, True])
def test_compute_moe_lora_info_expands_segments(use_preallocated_buffers: bool):
    device = DEVICE
    seg_lens = torch.tensor([5, 1, 7, 3, 9, 2], dtype=torch.int32, device=device)
    seg_indptr = torch.zeros((seg_lens.numel() + 1,), dtype=torch.int32, device=device)
    seg_indptr[1:] = torch.cumsum(seg_lens, dim=0)

    weight_indices = torch.tensor([2, 0, 5, 2, 3, 7], dtype=torch.int32, device=device)
    lora_ranks = torch.tensor(
        [0, 12, 16, 32, 24, 8, 0, 4], dtype=torch.int32, device=device
    )
    num_tokens = int(seg_indptr[-1].item())

    if use_preallocated_buffers:
        adapter_enabled = torch.full_like(lora_ranks, 123)
        token_lora_mapping = torch.full(
            (num_tokens + 11,), 456, dtype=torch.int32, device=device
        )
    else:
        adapter_enabled = None
        token_lora_mapping = None

    actual_enabled, actual_mapping = _compute_moe_lora_info(
        num_tokens,
        seg_indptr,
        lora_ranks,
        weight_indices,
        adapter_enabled,
        token_lora_mapping,
        max_len=int(seg_lens.max().item()),
    )
    torch.get_device_module(device).synchronize()

    expected_mapping = torch.repeat_interleave(weight_indices, seg_lens)
    expected_enabled = _expected_adapter_enabled(lora_ranks, weight_indices)

    torch.testing.assert_close(actual_mapping, expected_mapping)
    torch.testing.assert_close(actual_enabled, expected_enabled)

    if use_preallocated_buffers:
        assert actual_mapping.data_ptr() == token_lora_mapping.data_ptr()


def test_moe_graph_metadata_uses_matching_static_buffers():
    """Capture fixes the align kernel's request count; include empty tail slots."""
    num_slots, max_loras = 8, 4
    backend = BaseLoRABackend.__new__(BaseLoRABackend)
    backend._is_moe_lora = True
    backend.prefill_cuda_graph_batch_info = None
    with torch.device(DEVICE):
        backend.moe_cg_buffers = {
            "adapter_enabled": torch.zeros(max_loras, dtype=torch.int32),
            "token_lora_mapping": torch.full((8,), 7, dtype=torch.int32),
        }
        backend.prefill_moe_cg_buffers = {
            "adapter_enabled": torch.zeros(max_loras, dtype=torch.int32),
            "token_lora_mapping": torch.full((64,), 7, dtype=torch.int32),
        }

    for prefill, buffers in (
        (False, backend.moe_cg_buffers),
        (True, backend.prefill_moe_cg_buffers),
    ):
        with torch.device(DEVICE):
            info = LoRABatchInfo(
                bs=num_slots,
                use_cuda_graph=True,
                num_segments=2,
                seg_lens=torch.tensor(
                    [5, 3] + [0] * (num_slots - 2), dtype=torch.int32
                ),
                seg_indptr=torch.zeros(num_slots + 1, dtype=torch.int32),
                max_len=5,
                weight_indices=torch.tensor(
                    [2, 1] + [0] * (num_slots - 2), dtype=torch.int32
                ),
                lora_ranks=torch.tensor([0, 16, 8, 0], dtype=torch.int32),
                scalings=torch.zeros(max_loras, dtype=torch.float),
                permutation=None,
            )
        if prefill:
            backend.prefill_cuda_graph_batch_info = info
        torch.cumsum(info.seg_lens, dim=0, out=info.seg_indptr[1:])
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            batch_size=2,
            extend_num_tokens=8,
            extend_seq_lens_cpu=[5, 3],
        )
        moe = backend._add_moe_lora_info(forward_batch, info).moe_lora_info
        torch.get_device_module(DEVICE).synchronize()

        assert moe.adapter_enabled.data_ptr() == buffers["adapter_enabled"].data_ptr()
        assert (
            moe.token_lora_mapping.data_ptr()
            == buffers["token_lora_mapping"].data_ptr()
        )
        expected = torch.tensor([2] * 5 + [1] * 3, dtype=torch.int32, device=DEVICE)
        torch.testing.assert_close(moe.token_lora_mapping, expected)
        assert moe.adapter_enabled.tolist() == [0, 1, 1, 0]
        if prefill:
            assert moe.seg_indptr.shape[0] == num_slots + 1
            assert moe.req_to_lora.shape[0] == num_slots
            assert torch.all(moe.seg_indptr[2:] == 8)
            assert torch.all(buffers["token_lora_mapping"][8:] == -1)


def test_moe_lora_prefill_graph_admission_is_breakable_only(monkeypatch):
    """Full capture omits the MoE LoRA hooks; only breakable may be admitted."""
    from sglang.srt.lora.lora_manager import LoRAManager
    from sglang.srt.model_executor import cuda_graph_config

    manager = LoRAManager.__new__(LoRAManager)
    manager.enable_dp_attention = False
    manager.lora_backend = BaseLoRABackend.__new__(BaseLoRABackend)
    manager.lora_backend.supports_prefill_cuda_graph = True
    manager.lora_backend._is_moe_lora = True

    for backend, admitted in (
        ("breakable", True),
        ("full", False),
    ):
        monkeypatch.setattr(
            cuda_graph_config,
            "check_cuda_graph_backend",
            lambda phase, b, _cur=backend: b == _cur,
        )
        assert manager.supports_prefill_cuda_graph is admitted, backend

    # Dense LoRA is admitted under any capturing backend.
    manager.lora_backend._is_moe_lora = False
    monkeypatch.setattr(
        cuda_graph_config, "check_cuda_graph_backend", lambda phase, b: False
    )
    assert manager.supports_prefill_cuda_graph is True
    manager.enable_dp_attention = True
    assert manager.supports_prefill_cuda_graph is False


def test_compute_moe_lora_info_rejects_undercovered_launch():
    device = DEVICE
    seg_indptr = torch.tensor([0, 300], dtype=torch.int32, device=device)
    weight_indices = torch.tensor([0], dtype=torch.int32, device=device)
    lora_ranks = torch.tensor([16], dtype=torch.int32, device=device)

    with pytest.raises(AssertionError, match="under-covers tokens"):
        _compute_moe_lora_info(
            300,
            seg_indptr,
            lora_ranks,
            weight_indices,
            None,
            None,
            max_len=1,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
