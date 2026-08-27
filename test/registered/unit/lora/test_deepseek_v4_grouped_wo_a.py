import pytest
import torch
from torch import nn

from sglang.srt.layers.cp.cp_decode_attn_tp import CpDecodeAttnTpContext
from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.layers import ColumnParallelLinearWithLoRA
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _batch_info(*, permutation=None):
    return LoRABatchInfo(
        use_cuda_graph=False,
        bs=2,
        num_segments=2,
        seg_indptr=torch.tensor([0, 1, 3], dtype=torch.int32),
        weight_indices=torch.tensor([0, 1], dtype=torch.int32),
        lora_ranks=torch.tensor([3, 3], dtype=torch.int32),
        scalings=torch.tensor([1.0, 1.0]),
        max_len=2,
        seg_lens=torch.tensor([1, 2], dtype=torch.int32),
        permutation=permutation,
        expected_tokens=3,
    )


class _TorchSegmentedBackend(BaseLoRABackend):
    name = "test"
    supports_repeated_sgemm_batch_info = True

    def __init__(self, batch_info, *, supports_repeated=True):
        self.batch_info = batch_info
        self.supports_repeated_sgemm_batch_info = supports_repeated
        self.repeated_batch_info = None

    def get_repeated_sgemm_batch_info(self, repeats):
        self.repeated_batch_info = super().get_repeated_sgemm_batch_info(repeats)
        return self.repeated_batch_info

    def _adapter_per_row(self, batch_info, num_rows):
        adapters = torch.empty(num_rows, dtype=torch.long)
        permutation = batch_info.permutation
        for segment in range(batch_info.num_segments):
            start = int(batch_info.seg_indptr[segment])
            end = int(batch_info.seg_indptr[segment + 1])
            physical_rows = (
                torch.arange(start, end)
                if permutation is None
                else permutation[start:end].long()
            )
            adapters[physical_rows] = batch_info.weight_indices[segment].long()
        return adapters

    def run_lora_a_sgemm(self, x, weights, pruned_batch_info):
        assert pruned_batch_info is self.repeated_batch_info
        adapters = self._adapter_per_row(pruned_batch_info, x.shape[0])
        return torch.bmm(x.unsqueeze(1), weights[adapters].transpose(1, 2)).squeeze(1)

    def run_lora_b_sgemm(self, x, weights, pruned_batch_info, **_kwargs):
        assert pruned_batch_info is self.repeated_batch_info
        adapters = self._adapter_per_row(pruned_batch_info, x.shape[0])
        output = torch.bmm(x.unsqueeze(1), weights[adapters].transpose(1, 2)).squeeze(1)
        return output * pruned_batch_info.scalings[adapters].unsqueeze(1)


def _layer(backend, lora_a, lora_b):
    layer = object.__new__(ColumnParallelLinearWithLoRA)
    nn.Module.__init__(layer)
    layer.base_layer = nn.Linear(lora_a.shape[-1], lora_b.shape[-2], bias=False)
    layer.set_lora = True
    layer.A_buffer = lora_a
    layer.B_buffer = lora_b
    layer.lora_backend = backend
    layer.output_offset = torch.tensor([0, lora_b.shape[-2]], dtype=torch.int32)
    layer.output_offset_cpu = layer.output_offset
    return layer


def test_grouped_wo_a_lora_selects_matching_group_diagonal():
    torch.manual_seed(0)
    tokens, groups, input_dim, output_dim, rank = 3, 4, 5, 2, 3
    x = torch.randn(tokens, groups, input_dim)
    base_output = torch.randn(tokens, groups, output_dim)
    lora_a = torch.randn(2, rank, input_dim)
    lora_b = torch.randn(2, groups * output_dim, rank)
    backend = _TorchSegmentedBackend(_batch_info())
    layer = _layer(backend, lora_a, lora_b)

    output = layer.apply_grouped_lora(base_output, x)

    adapter_per_token = torch.tensor([0, 1, 1])
    delta_weight = torch.bmm(lora_b, lora_a).view(2, groups, output_dim, input_dim)
    expected = base_output + torch.einsum(
        "tgd,tgrd->tgr", x, delta_weight[adapter_per_token]
    )
    torch.testing.assert_close(output, expected)


def test_grouped_batch_metadata_repeats_segments_and_permutation():
    info = _batch_info(permutation=torch.tensor([2, 0, 1], dtype=torch.int32))
    backend = _TorchSegmentedBackend(info)
    groups = 2
    layer = _layer(
        backend,
        torch.randn(2, 3, 4),
        torch.randn(2, groups * 5, 3),
    )

    layer.apply_grouped_lora(
        torch.randn(3, groups, 5),
        torch.randn(3, groups, 4),
    )

    repeated = backend.repeated_batch_info
    torch.testing.assert_close(
        repeated.seg_indptr, torch.tensor([0, 2, 6], dtype=torch.int32)
    )
    torch.testing.assert_close(
        repeated.seg_lens, torch.tensor([2, 4], dtype=torch.int32)
    )
    torch.testing.assert_close(
        repeated.permutation,
        torch.tensor([4, 5, 0, 1, 2, 3], dtype=torch.int32),
    )
    assert repeated.expected_tokens == 6


def test_grouped_wo_a_rejects_backend_without_repeated_metadata():
    groups = 2
    backend = _TorchSegmentedBackend(_batch_info(), supports_repeated=False)
    layer = _layer(
        backend,
        torch.randn(1, 3, 4),
        torch.randn(1, groups * 5, 3),
    )

    with pytest.raises(RuntimeError, match="does not support repeated"):
        layer.apply_grouped_lora(
            torch.randn(3, groups, 5),
            torch.randn(3, groups, 4),
        )


def test_cp_decode_tp_rejects_active_lora_and_unwraps_inactive_wrapper():
    backend = _TorchSegmentedBackend(_batch_info())
    layer = _layer(
        backend,
        torch.randn(2, 3, 4),
        torch.randn(2, 10, 3),
    )
    context = object.__new__(CpDecodeAttnTpContext)

    with pytest.raises(RuntimeError, match="does not support active LoRA"):
        context._unwrap_inactive_lora(layer)

    backend.batch_info = None
    assert context._unwrap_inactive_lora(layer) is layer.base_layer
