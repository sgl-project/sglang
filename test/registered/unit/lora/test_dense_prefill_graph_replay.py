"""Dense legacy-Triton LoRA prefill CUDA-graph replay with ragged adapters."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.lora.backend.triton_backend import TritonLoRABackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


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
