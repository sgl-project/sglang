"""Local Mooncake integration checks. Requires a CUDA-enabled Mooncake build.

Run with pytest on a CUDA host; no model download or distributed group needed.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.mem_cache.engram.storage import ByteRows, init_deepseek, lookup_deepseek

pytest.importorskip("mooncake._engram_cuda")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture
def config(tmp_path):
    with patch(
        "sglang.srt.runtime_context.get_memory",
        return_value=SimpleNamespace(engram_local_dir=str(tmp_path)),
    ):
        yield


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_rows_prefetch_graph_and_scale(config, dtype):
    weights = torch.randn(37, 128).to(dtype)
    rows = ByteRows()
    rows.initialize(weights.view(torch.uint8))
    ids = torch.tensor([[0, 36], [4, 7]], device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        rows.lookup(ids)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = rows.lookup(ids).view(dtype).to(torch.bfloat16) * 1.5
    for indices in ([[3, 9], [31, 17]], [[0, 36], [1, 1]]):
        ids.copy_(torch.tensor(indices, device="cuda"))
        graph.replay()
        expected = weights.float()[torch.tensor(indices)].to(torch.bfloat16) * 1.5
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
    del graph
    rows.close()


def test_deepseek_block_scales_and_shard_mask(config):
    from sglang.kernels.ops.embeddings.engram_gather import engram_gather

    module = SimpleNamespace(rows=17, dim=256, row_start=100)
    init_deepseek(module, 32)
    module.weight.data.copy_(torch.randn(17, 256).to(torch.float8_e4m3fn))
    module.scale.data.copy_(
        (2.0 ** torch.randint(-3, 4, (17, 8))).to(torch.float8_e8m0fnu)
    )
    ids = torch.tensor([[99, 100, 116, 117], [104, 107, 102, 103]], device="cuda")
    result = lookup_deepseek(module, ids, 32)
    local = (ids.cpu() - 100).clamp(0, 16)
    expected = (
        (
            module.weight.float()[local].view(2, 4, 8, 32)
            * module.scale.float()[local].unsqueeze(-1)
        )
        .flatten(-2)
        .bfloat16()
    )
    expected[0, [0, 3]] = 0
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)

    device_weight = module.weight.cuda()
    device_scale = module.scale.cuda()
    native = torch.empty_like(result)

    def native_lookup():
        engram_gather(
            device_weight.data_ptr(),
            device_scale.data_ptr(),
            ids.flatten(),
            native.view(-1, module.dim),
            module.dim,
            32,
            row_lo=module.row_start,
            row_hi=module.row_start + module.rows,
        )

    native_lookup()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = lookup_deepseek(module, ids, 32)
        native_lookup()
    for indices in (
        [[99, 100, 116, 117], [104, 107, 102, 103]],
        [[101, 105, 108, 110], [99, 117, 116, 100]],
    ):
        ids.copy_(torch.tensor(indices, device="cuda"))
        graph.replay()
        torch.testing.assert_close(actual, native, rtol=0, atol=0)
    del graph
    module._mooncake.close()


def test_qwen_prefetch_matches_native_kernel(config):
    from sglang.srt.mem_cache.engram.embedding import MooncakeVocabParallelEmbedding
    from sglang.srt.models.qwen4_exp import _gather_ple_embedding_from_pinned_kernel

    adapter = MooncakeVocabParallelEmbedding.__new__(MooncakeVocabParallelEmbedding)
    torch.nn.Module.__init__(adapter)
    adapter.weight = torch.nn.Parameter(
        torch.randn(37, 160).to(torch.float8_e4m3fn), requires_grad=False
    )
    adapter.tp_size = 2
    adapter.output_dtype = torch.bfloat16
    start = 100_000_000
    adapter.shard_indices = SimpleNamespace(
        org_vocab_start_index=start,
        org_vocab_end_index=start + 37,
        num_org_vocab_padding=0,
        added_vocab_start_index=start + 37,
        added_vocab_end_index=start + 37,
    )
    adapter._rows = ByteRows()
    weight = adapter.weight.cuda()
    ids = torch.tensor(
        [[start - 1, start, start + 36], [start + 37, start + 4, start + 7]],
        device="cuda",
    )
    native = torch.empty(2, 3, 160, dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.Stream()
    with (
        patch(
            "sglang.srt.mem_cache.engram.embedding.get_parallel",
            return_value=SimpleNamespace(tp_group=None),
        ),
        patch(
            "sglang.srt.mem_cache.engram.embedding.is_allocation_symmetric",
            return_value=False,
        ),
    ):
        actual = adapter.allocate_output(native.shape, ids.device)

        def run():
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                adapter.gather(ids, out=actual)
            torch.cuda.current_stream().wait_stream(stream)
            _gather_ple_embedding_from_pinned_kernel[(ids.numel(),)](
                weight.data_ptr(),
                ids,
                native,
                160,
                start,
                start + 37,
                is_fp8=True,
                BLOCK_D=256,
            )

        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for offsets in ([[-1, 0, 36], [37, 4, 7]], [[17, 3, 2], [0, 36, 37]]):
            ids.copy_(torch.tensor(offsets, device="cuda") + start)
            graph.replay()
            torch.testing.assert_close(actual, native, rtol=0, atol=0)
        del graph
    adapter._rows.close()
