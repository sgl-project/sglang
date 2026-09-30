import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.lora.backend.base_backend import (
    BaseLoRABackend,
    _compute_moe_lora_info,
)
from sglang.srt.lora.backend.triton_backend import TritonLoRABackend
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cuda_ci,
    register_xpu_ci,
)
from sglang.test.test_utils import CustomTestCase

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
        if prefill:
            assert moe.seg_indptr.shape[0] == num_slots + 1
            assert moe.req_to_lora.shape[0] == num_slots
            assert torch.all(moe.seg_indptr[2:] == 8)
            assert torch.all(buffers["token_lora_mapping"][8:] == -1)


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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires CUDA graph capture",
)
class TestDenseLoRAPrefillGraph(CustomTestCase):
    def test_replay_preserves_ragged_adapters(self):
        """Ragged replays retain adapters without a token bucket per request."""
        device, dtype = torch.device("cuda"), torch.float16
        capacity, num_requests, rank, width = 1024, 64, 32, 64
        ranks, scalings = [0, 16, 32], [0.0, 0.5, 1.0]
        backend = TritonLoRABackend(max_loras_per_batch=3, device=device)
        backend.init_prefill_cuda_graph_batch_info(
            capacity, max_num_requests=num_requests
        )
        generator = torch.Generator().manual_seed(0)
        cpu_a, cpu_b, cpu_embedding = [
            torch.randint(-4, 5, shape, generator=generator).float() / 16
            for shape in ((3, rank, width), (3, width, rank), (3, rank, width))
        ]
        a_weights, b_weights, embedding_weights = [
            weight.to(device=device, dtype=dtype)
            for weight in (cpu_a, cpu_b, cpu_embedding)
        ]
        x = torch.empty((capacity, width), device=device, dtype=dtype)
        input_ids = torch.empty(capacity, device=device, dtype=torch.int64)
        output = torch.full_like(x, 0.25)
        ragged = [1, 3, 7, 15, 16, 17, 23, 31] * 8
        ragged[-1] += capacity - sum(ragged)
        cases = (
            ([capacity], [0]),
            (ragged, [(i + 1) % 3 for i in range(num_requests)]),
            ([1, 17], [2, 0]),
        )
        for phase, (lengths, adapters) in enumerate(cases):
            cpu_x = torch.randint(-4, 5, x.shape, generator=generator).float() / 16
            cpu_ids = (torch.arange(capacity) + phase) % width
            x.copy_(cpu_x)
            input_ids.copy_(cpu_ids)
            backend.prepare_lora_batch(
                SimpleNamespace(
                    forward_mode=ForwardMode.EXTEND,
                    batch_size=len(lengths),
                    extend_num_tokens=sum(lengths),
                    extend_seq_lens_cpu=lengths,
                    extend_seq_lens=torch.tensor(
                        lengths, device=device, dtype=torch.int32
                    ),
                    return_logprob=False,
                ),
                weight_indices=adapters,
                lora_ranks=ranks,
                scalings=scalings,
                use_cuda_graph=False,
                use_prefill_cuda_graph=True,
            )
            if phase == 0:
                info = backend._sgemm_info()
                # Allow one partial 16-token tile per request, not a bucket per slot.
                assert info.bs * info.max_len <= capacity + 16 * num_requests
                graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                for capture in (False, True):
                    with (
                        torch.cuda.graph(graph, stream=stream)
                        if capture
                        else torch.cuda.stream(stream)
                    ):
                        a_output = backend.run_lora_a_sgemm(x, a_weights)
                        backend.run_lora_b_sgemm(
                            a_output, b_weights, base_output=output
                        )
                        embedding_output = backend.run_lora_a_embedding(
                            input_ids, embedding_weights, vocab_size=width
                        )
                    torch.cuda.synchronize()
                continue

            output.fill_(0.25)
            graph.replay()
            torch.cuda.synchronize()
            expected = torch.full((capacity, width), 0.25, dtype=dtype)
            expected_embedding = torch.zeros((capacity, rank), dtype=dtype)
            start = 0
            for length, adapter in zip(lengths, adapters):
                rows, r = slice(start, start + length), ranks[adapter]
                if r:
                    expected_a = (cpu_x[rows] @ cpu_a[adapter, :r].T).to(dtype)
                    delta = (
                        expected_a.float() @ cpu_b[adapter, :, :r].T * scalings[adapter]
                    ).to(dtype)
                    expected[rows] += delta
                    expected_embedding[rows, :r] = cpu_embedding[adapter, :r][
                        :, cpu_ids[rows]
                    ].T.to(dtype)
                start += length
            torch.testing.assert_close(output.cpu(), expected, atol=1e-3, rtol=1e-3)
            torch.testing.assert_close(
                embedding_output.cpu(), expected_embedding, atol=0, rtol=0
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
