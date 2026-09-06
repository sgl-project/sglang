import pytest
import torch

from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
from sglang.srt.layers.attention.dsa.utils import fp8_mqa_logits_make_fused_kv
from sglang.srt.model_executor.runner_utils.capture_owner import (
    collect_full_cuda_graph_owners,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_outer_torch_compile_preserves_capture_owner():
    torch.manual_seed(48)
    indexer = object.__new__(Indexer)
    values = torch.randn(16, 32, device="cuda")

    def outer(value):
        produced = value.float().square()
        produced = indexer._retain_full_graph_capture_owner(produced)
        return produced.to(torch.int32)

    compiled_outer = torch.compile(
        outer, mode="max-autotune-no-cudagraphs", dynamic=False
    )
    compiled_outer(values)
    torch.cuda.synchronize()

    with collect_full_cuda_graph_owners() as owners:
        output = compiled_outer(values)
    torch.cuda.synchronize()

    assert output.dtype == torch.int32
    assert len(owners) == 1
    assert owners[0].dtype == torch.float32
    expected_replay = values.float().square()
    torch.testing.assert_close(owners[0], expected_replay)

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        compiled_outer(values)
        compiled_outer(values)
    torch.cuda.current_stream().wait_stream(capture_stream)

    graph = torch.cuda.CUDAGraph()
    with collect_full_cuda_graph_owners() as capture_owners:
        with torch.cuda.graph(graph, stream=capture_stream):
            captured_output = compiled_outer(values)

    del captured_output
    graph.replay()
    torch.cuda.synchronize()

    assert len(capture_owners) == 1
    torch.testing.assert_close(capture_owners[0], expected_replay)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_deepgemm_paged_logits_owner_survives_shared_pool_capture():
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("DeepGEMM paged MQA requires SM90 or newer")
    deep_gemm = pytest.importorskip("deep_gemm")

    torch.manual_seed(52)
    torch._dynamo.reset()
    batch_size = 24
    max_context_len = 2240
    block_size = 64
    num_heads = 32
    head_dim = 128
    num_blocks = max_context_len // block_size

    q_fp8 = torch.randn(batch_size, 1, num_heads, head_dim, device="cuda").to(
        torch.float8_e4m3fn
    )
    kv_fp8 = torch.randn(num_blocks, block_size, head_dim, device="cuda").to(
        torch.float8_e4m3fn
    )
    kv_scales = torch.ones(num_blocks, block_size, device="cuda")
    kv_fused = fp8_mqa_logits_make_fused_kv(kv_fp8, kv_scales, block_size, head_dim)
    weights = torch.randn(batch_size, num_heads, device="cuda")
    context_lens = torch.full(
        (batch_size, 1), max_context_len, dtype=torch.int32, device="cuda"
    )
    block_table = torch.arange(num_blocks, dtype=torch.int32, device="cuda").repeat(
        batch_size, 1
    )
    schedule = deep_gemm.get_paged_mqa_logits_metadata(
        context_lens, block_size, deep_gemm.get_num_sms()
    )
    indexer = object.__new__(Indexer)

    def outer(query):
        logits = deep_gemm.fp8_paged_mqa_logits(
            query,
            kv_fused,
            weights,
            context_lens,
            block_table,
            schedule,
            max_context_len,
            clean_logits=False,
        )
        logits = indexer._retain_full_graph_capture_owner(logits)
        return logits[:, :1]

    compiled_outer = torch.compile(
        outer, mode="max-autotune-no-cudagraphs", dynamic=False
    )
    expected = compiled_outer(q_fp8).detach().clone()
    torch.cuda.synchronize()

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        compiled_outer(q_fp8)
        compiled_outer(q_fp8)
    torch.cuda.current_stream().wait_stream(capture_stream)

    pool = torch.cuda.graph_pool_handle()
    graph = torch.cuda.CUDAGraph()
    with collect_full_cuda_graph_owners() as capture_owners:
        with torch.cuda.graph(graph, pool=pool, stream=capture_stream):
            captured_output = compiled_outer(q_fp8)

    assert len(capture_owners) == 1
    owner = capture_owners[0]
    assert owner.shape == (batch_size, max_context_len)
    assert owner.dtype == torch.float32
    assert (
        owner.untyped_storage().data_ptr()
        == captured_output.untyped_storage().data_ptr()
    )

    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(owner[:, :1], expected)
    del captured_output
    expected_owner = owner.detach().clone()

    # A second capture sharing the pool must not receive the retained
    # owner's storage; without retention the freed block is reusable and
    # replaying the first graph would overwrite the second graph's tensor.
    other_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(other_graph, pool=pool, stream=capture_stream):
        other = torch.empty_like(owner)
        other.fill_(17.0)
    owner_start = owner.untyped_storage().data_ptr()
    owner_end = owner_start + owner.untyped_storage().nbytes()
    other_start = other.untyped_storage().data_ptr()
    other_end = other_start + other.untyped_storage().nbytes()
    assert owner_end <= other_start or other_end <= owner_start
    del other

    other_graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(owner, expected_owner)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_backend_stores_helper_retained_owner_with_the_shape():
    from unittest import mock

    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
        FullCudaGraphBackend,
    )

    indexer = object.__new__(Indexer)
    runner = mock.Mock()
    runner.device_module = torch.cuda
    runner.model_runner.tp_group.barrier = lambda: None
    runner.enable_profile_cuda_graph = False
    backend = FullCudaGraphBackend(runner)

    values = torch.randn(8, device="cuda")
    produced = {}

    def forward_fn():
        tensor = values.float().square()
        produced["tensor"] = indexer._retain_full_graph_capture_owner(tensor)
        return object()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with backend.capture_session(stream):
        backend.capture_one("shape", forward_fn)
    torch.cuda.current_stream().wait_stream(stream)

    assert produced["tensor"] in backend._capture_owners["shape"]
    backend.cleanup()
    assert backend._capture_owners == {}


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
