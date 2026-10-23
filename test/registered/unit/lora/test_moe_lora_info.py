"""MoE LoRA batch metadata: token-to-adapter mapping and CUDA-graph buffers."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.lora.backend.base_backend import (
    BaseLoRABackend,
    _compute_moe_lora_info,
    _compute_token_lora_mapping,
)
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cuda_ci,
    register_xpu_ci,
)

register_cuda_ci(est_time=7, stage="base-b", runner_config="1-gpu-small")
# backend-specific: ROCm's Triton build of _compute_moe_lora_info_kernel,
# graph-padding clear included.
register_amd_ci(est_time=5, stage="stage-b", runner_config="1-gpu-small-amd")
# backend-specific: XPU takes the native torch mapping path instead of the
# Triton launch.
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

    raw_mapping = torch.repeat_interleave(weight_indices, seg_lens)
    expected_mapping = torch.where(
        lora_ranks[raw_mapping.long()] > 0,
        raw_mapping,
        -1,
    )
    expected_enabled = _expected_adapter_enabled(lora_ranks, weight_indices)

    torch.testing.assert_close(actual_mapping, expected_mapping)
    torch.testing.assert_close(actual_enabled, expected_enabled)

    if use_preallocated_buffers:
        assert actual_mapping.data_ptr() == token_lora_mapping.data_ptr()
        assert token_lora_mapping[num_tokens:].tolist() == [-1] * 11


def test_compute_moe_lora_info_cpu_fallback_normalizes_inactive_slots():
    # One CPU pin of the torch fallback, with mixed active and inactive slots.
    seg_lens = torch.tensor([2, 3, 1], dtype=torch.int32)
    seg_indptr = torch.tensor([0, 2, 5, 6], dtype=torch.int32)
    weight_indices = torch.tensor([0, 1, 2], dtype=torch.int32)
    lora_ranks = torch.tensor([0, 8, 0], dtype=torch.int32)
    adapter_enabled = torch.full((3,), 123, dtype=torch.int32)
    mapping_buffer = torch.full((9,), 456, dtype=torch.int32)

    actual_enabled, actual_mapping = _compute_moe_lora_info(
        6,
        seg_indptr,
        lora_ranks,
        weight_indices,
        adapter_enabled,
        mapping_buffer,
        max_len=int(seg_lens.max().item()),
    )

    assert actual_mapping.data_ptr() == mapping_buffer.data_ptr()
    assert actual_mapping.tolist() == [-1, -1, 1, 1, 1, -1]
    assert actual_enabled.tolist() == [0, 1, 0]
    assert mapping_buffer[6:].tolist() == [-1, -1, -1]


@pytest.mark.parametrize("mode", ["legacy", "mapping_only"])
def test_token_mapping_reuse_clears_padding_and_optional_mask(mode):
    storage = torch.full((516,), 777, dtype=torch.int32, device=DEVICE)
    mask, mapping_storage = storage[:4], storage[4:]
    mask.fill_(13)
    cases = (
        ([257, 0, 7], [1, 2, 3], [0, 16, 8, 4]),
        ([1, 0, 3], [2, 1, 0], [0, 0, 8, 0]),
        ([2, 1], [1, 3], [0, 0, 0, 0]),
        ([0, 0], [1, 2], [0, 16, 8, 4]),
        ([], [], [0, 16, 8, 4]),
    )
    for lengths, slots, ranks in cases:
        pointers = [0]
        expected = []
        for length, slot in zip(lengths, slots):
            pointers.append(pointers[-1] + length)
            expected.extend([slot if ranks[slot] > 0 else -1] * length)
        indptr = torch.tensor(pointers, dtype=torch.int32, device=DEVICE)
        assignments = torch.tensor(slots, dtype=torch.int32, device=DEVICE)
        rank_tensor = torch.tensor(ranks, dtype=torch.int32, device=DEVICE)
        common = dict(max_len=max(lengths, default=0), bucket_len=512)
        if mode == "mapping_only":
            actual = _compute_token_lora_mapping(
                len(expected),
                indptr,
                rank_tensor,
                assignments,
                mapping_storage,
                **common,
            )
        else:
            _, actual = _compute_moe_lora_info(
                len(expected),
                indptr,
                rank_tensor,
                assignments,
                mask,
                mapping_storage,
                **common,
            )
        assert actual.tolist() == expected
        assert mapping_storage[len(expected) :].tolist() == [-1] * (512 - len(expected))
        if mode == "legacy":
            assert mask.tolist() == [
                int(i in slots and rank > 0) for i, rank in enumerate(ranks)
            ]
        else:
            assert mask.tolist() == [13] * 4


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
        if prefill:
            assert moe.seg_indptr.shape[0] == num_slots + 1
            assert moe.req_to_lora.shape[0] == num_slots
            assert torch.all(moe.seg_indptr[2:] == 8)
            assert torch.all(buffers["token_lora_mapping"][8:] == -1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
