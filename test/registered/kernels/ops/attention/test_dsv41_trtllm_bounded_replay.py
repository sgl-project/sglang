"""TRT-LLM bounded-tail attention, including changing batches across BCG replay."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.deepseek_v4_trtllm_backend import (
    DeepseekV4TrtllmAttnBackend,
    _install_persistent_trtllm_semaphores,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    eager_on_graph,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="TRT-LLM sparse MLA requires Blackwell",
)


class _Pool:
    def __init__(self):
        self.swa_kv_pool = SimpleNamespace(page_size=64)
        self.swa = torch.randn(16, 64 * 512, device="cuda").to(torch.float8_e4m3fn)
        self.extra = torch.randn_like(self.swa.float()).to(torch.float8_e4m3fn)

    def get_swa_key_buffer_radix(self, layer_id):
        return self.swa

    def get_extra_key_buffer(self, layer_id):
        return self.extra

    def get_extra_key_page_size(self, layer_id):
        return 64


def _metadata(tail_lens):
    return SimpleNamespace(
        late_layer_tail=SimpleNamespace(
            extend_seq_lens_cpu=tail_lens,
            extend_seq_lens=torch.tensor(tail_lens, device="cuda", dtype=torch.int32),
        ),
        core_attn_metadata=SimpleNamespace(
            trtllm_prefill_qmeta=None,
            trtllm_prefill_swa_lens=None,
            trtllm_prefill_c4_indices=None,
            trtllm_prefill_c128=None,
        ),
    )


@pytest.mark.parametrize("ratio", [0, 1, 2, 4, 128])
@pytest.mark.parametrize("bounded", [False, True])
def test_tail_varlen_and_breakable_replay(ratio, bounded):
    torch.manual_seed(413)
    _install_persistent_trtllm_semaphores(512)
    backend = DeepseekV4TrtllmAttnBackend.__new__(DeepseekV4TrtllmAttnBackend)
    backend.device = torch.device("cuda")
    backend.cuda_int32_kwargs = dict(device="cuda", dtype=torch.int32)
    backend.token_to_kv_pool = _Pool()
    backend.softmax_scale = 512**-0.5
    backend.trtllm_workspace_buffer = torch.zeros(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda"
    )
    layer = SimpleNamespace(layer_id=0, k_scale_float=None)
    batch = SimpleNamespace(extend_seq_lens_cpu=None, seq_lens=None)
    q = torch.randn(256, 16, 512, device="cuda").to(torch.float8_e4m3fn)
    sinks = torch.randn(16, device="cuda")
    swa_indices = torch.full((256, 128), -1, **backend.cuda_int32_kwargs)
    extra_indices = torch.zeros((256, 4), **backend.cuda_int32_kwargs)
    extra_lens = torch.full((256,), 4, **backend.cuda_int32_kwargs)

    def prepare(full_lens, tail_lens, seq_lens):
        batch.extend_seq_lens_cpu = full_lens if bounded else tail_lens
        batch.seq_lens = torch.tensor(seq_lens, **backend.cuda_int32_kwargs)
        backend.forward_metadata = _metadata(tail_lens)
        if not bounded:
            backend.forward_metadata.late_layer_tail = None
        swa_indices.fill_(-1)
        offset = 0
        for req, length in enumerate(tail_lens):
            # Only the tail has SWA writes: old context remains masked even
            # when the absolute sequence length is much larger than the tail.
            for row in range(length):
                visible = (
                    row + 1 if bounded else min(128, seq_lens[req] - length + row + 1)
                )
                swa_indices[offset + row, :visible] = torch.arange(
                    req * 256, req * 256 + visible, device="cuda"
                )
            offset += length
        extra_indices.random_(0, 1024)

    def attention(query):
        return backend._forward_trtllm_prefill(
            q=query,
            layer=layer,
            compress_ratio=ratio,
            forward_batch=batch,
            attn_sink=sinks,
            swa_page_indices=swa_indices,
            extra_indices=extra_indices if ratio else None,
            extra_topk_lengths=extra_lens if ratio else None,
        )

    prepare([512, 17], [128, 17], [768, 1024])
    attention(q)
    torch.cuda.synchronize()
    graph = BreakableCUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with BreakableCUDAGraphCapture(graph, stream=stream):
        query = q.clone()
        output = eager_on_graph(True)(attention)(query)
        result = output.clone()
    torch.cuda.current_stream().wait_stream(stream)

    # Same captured capacity, different real tail counts and cached prefixes.
    for full_lens, tail_lens, seq_lens in (
        ([512, 17], [128, 17], [768, 1024]),
        ([5, 1024], [5, 128], [5, 1536]),
        ([4096], [128], [8192]),
        ([7, 3], [7, 3], [263, 515]),
    ):
        prepare(full_lens, tail_lens, seq_lens)
        graph.replay()
        torch.cuda.synchronize()
        real_rows = sum(tail_lens)
        cum_q, max_q, sum_q, swa_lens = (
            backend.forward_metadata.core_attn_metadata.trtllm_prefill_qmeta
        )
        assert cum_q.tolist() == [0] + list(torch.tensor(tail_lens).cumsum(0).tolist())
        assert (max_q, sum_q) == (max(tail_lens), real_rows)
        assert swa_lens.tolist() == (tail_lens if bounded else seq_lens)
        assert batch.seq_lens.tolist() == seq_lens
        assert torch.count_nonzero(result[real_rows:]).item() == 0
        assert torch.isfinite(result).all().item()
        for row in sorted({0, tail_lens[0] - 1, real_rows - 1}):
            valid = swa_indices[row][swa_indices[row] >= 0].long()
            keys = backend.token_to_kv_pool.swa.float().view(-1, 512)[valid]
            if ratio:
                keys = torch.cat(
                    [
                        keys,
                        backend.token_to_kv_pool.extra.float().view(-1, 512)[
                            extra_indices[row].long()
                        ],
                    ]
                )
            logits = q[row].float() @ keys.T * backend.softmax_scale
            probs = torch.cat([logits, sinks[:, None]], dim=-1).softmax(-1)
            expected = probs[:, :-1] @ keys
            # TRT-LLM also rounds attention probabilities for the FP8 PV GEMM.
            torch.testing.assert_close(
                result[row].float(), expected, atol=0.05, rtol=0.05
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
