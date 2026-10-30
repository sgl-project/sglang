from types import SimpleNamespace

import pytest

from sglang.srt.arg_groups.cuda_graph_hook import (
    generate_decode_cuda_graph_batch_sizes,
)
from sglang.srt.model_executor.runner import base_cuda_graph_runner as graph_runner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "pool_size,configured_max,width,alignment,tbo,padding,expected_max",
    [
        # An aligned pool endpoint is kept when --cuda-graph-max-bs exceeds the pool.
        (15, 256, 1, 4, False, True, 16),
        # An unaligned --cuda-graph-max-bs below the pool still gets a covering bucket.
        (15, 15, 1, 4, False, True, 16),
        # Four rows per request already satisfy alignment 4 at any request count.
        (15, 256, 4, 4, False, True, 15),
        (15, 256, 3, 4, False, True, 16),
        (17, 256, 2, 8, False, True, 20),
        # TBO aligns per request, not per token row.
        (15, 256, 4, 8, True, True, 16),
        (128, 12, 1, 4, False, True, 12),
        # Without padding, only exact configured sizes are captured.
        (15, 256, 1, 4, False, False, 12),
        (15, 256, 4, 4, False, False, 15),
    ],
)
def test_capture_buckets_cover_every_serviceable_batch(
    monkeypatch,
    pool_size,
    configured_max,
    width,
    alignment,
    tbo,
    padding,
    expected_max,
):
    args = SimpleNamespace(
        disable_cuda_graph_padding=not padding,
        speculative_algorithm="EAGLE",
        dllm_algorithm=None,
    )
    buckets = generate_decode_cuda_graph_batch_sizes(args, configured_max)
    context = SimpleNamespace(
        graph=SimpleNamespace(
            cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(bs=buckets)),
            disable_cuda_graph_padding=not padding,
            torch_compile_max_bs=8,
        ),
        overlap=SimpleNamespace(enable_two_batch_overlap=tbo),
    )
    monkeypatch.setattr(graph_runner, "get_exec", lambda: context)
    monkeypatch.setattr(
        graph_runner,
        "get_flags",
        lambda: SimpleNamespace(capture=SimpleNamespace(enable_torch_compile=True)),
    )
    monkeypatch.setattr(
        graph_runner,
        "get_cuda_graph_batch_size_alignment",
        lambda gathered_buffer_required=None: alignment,
    )
    runner = SimpleNamespace(
        server_args=args, req_to_token_pool=SimpleNamespace(size=pool_size)
    )

    capture, compile_bs = graph_runner.get_batch_sizes_to_capture(runner, width)

    assert max(capture) == expected_max
    assert all(bs * (1 if tbo else width) % alignment == 0 for bs in capture)
    assert compile_bs == [bs for bs in capture if bs <= 8]
    assert max(buckets) == configured_max  # The phase config list is not mutated.
    if padding:
        assert all(
            any(bucket >= bs for bucket in capture)
            for bs in range(1, min(pool_size, configured_max) + 1)
        )
    else:
        assert set(capture) <= set(buckets)
