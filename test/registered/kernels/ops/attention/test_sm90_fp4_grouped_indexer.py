"""H32/H64 grouped logits, current paged reference and independent graph streams."""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="requires SM90, CUDA JIT and CUTLASS",
)


def _inputs(heads=32, group=6, width=131, ratio=2, page_size=64, partial=False):
    from sglang.kernels.ops.attention.dsv4.fp4_indexer import store_fp4_index_k_cache
    from sglang.kernels.ops.attention.dsv4.torch_quant import fake_quant_fp4

    torch.manual_seed(31)
    rows = 3 * group - int(partial)
    q = torch.randn(rows, heads, 4, 32, device="cuda", dtype=torch.bfloat16)
    # Exercise distinct per-32 scales as well as signed head weights.
    q *= torch.tensor([0.125, 0.5, 2, 8], device="cuda").view(1, 1, 4, 1)
    q = fake_quant_fp4(q.flatten(-2))
    weights = torch.randn(rows, heads, device="cuda", dtype=torch.bfloat16) / heads
    req = torch.arange(3, device="cuda", dtype=torch.int64).repeat_interleave(group)[
        :rows
    ]
    lens = torch.full((rows,), width, device="cuda", dtype=torch.int32)
    lens[:group] = torch.arange(group, device="cuda") % (width + 1)
    lens[group : 2 * group] = 0  # Entirely empty request, including poisoned map.
    slots_count = ((max(width, 1) + page_size - 1) // page_size + 2) * page_size
    cache = torch.empty(
        (slots_count // page_size, page_size * 68), device="cuda", dtype=torch.uint8
    )
    keys = torch.randn(slots_count, 128, device="cuda", dtype=torch.bfloat16)
    store_fp4_index_k_cache(
        keys,
        cache,
        torch.arange(slots_count, device="cuda"),
        page_size=page_size,
        rne=True,
    )
    # A non-contiguous row stride and nontrivial physical mapping must survive.
    req_to_token = torch.full(
        (3, width * ratio + 11), -999999, device="cuda", dtype=torch.int32
    )[:, : width * ratio]
    valid_map = torch.stack(
        [torch.randperm(slots_count, device="cuda")[:width] * ratio for _ in range(3)]
    ).to(torch.int32)
    req_to_token[:, ::ratio] = valid_map
    req_to_token[1].fill_(-999999)
    return (q, weights, req_to_token, req, lens, cache, page_size, ratio, width, group)


def _reference(args):
    from sglang.kernels.ops.attention.dsv4.fp4_indexer import fp4_index_logits_paged

    q, weights, mapping, req, lens, cache, page_size, ratio, width, _ = args
    return fp4_index_logits_paged(
        q, weights, req, mapping, lens, cache, page_size, width, ratio
    )


def _assert_scores(actual, expected, lens):
    visible = (
        torch.arange(actual.shape[1], device=actual.device)[None, :] < lens[:, None]
    )
    assert torch.isneginf(actual[~visible]).all()
    a, e = actual[visible].float(), expected[visible].float()
    assert torch.isfinite(a).all()
    if not a.numel():
        return
    # WGMMA changes accumulation order. Check normalized error in addition to
    # the exact-coordinate/top-k test below; full-model quality is still needed.
    delta = a - e
    assert (delta.norm() / e.norm().clamp_min(1e-6)).item() <= 0.005
    assert (delta.abs().max() / e.abs().max().clamp_min(1e-6)).item() <= 0.01


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("group", [2, 3, 4, 6])
@pytest.mark.parametrize(
    "ratio,page_size,width",
    [(1, 16, 65), (2, 64, 4097), (2, 128, 8193), (1, 256, 32769)],
)
def test_grouped_matches_current_paged_scorer(heads, group, ratio, page_size, width):
    from sglang.kernels.ops.attention.dsv4.sm90_fp4_grouped_indexer import (
        fp4_index_logits_grouped_sm90,
    )

    args = _inputs(heads, group, width, ratio, page_size, partial=True)
    actual = fp4_index_logits_grouped_sm90(*args)
    _assert_scores(actual, _reference(args), args[4])
    assert actual.stride(0) % 8 == 0
    if heads == 32:
        padded = (
            torch.nn.functional.pad(args[0], (0, 0, 0, 32)),
            torch.nn.functional.pad(args[1], (0, 32)),
            *args[2:],
        )
        torch.testing.assert_close(
            actual, fp4_index_logits_grouped_sm90(*padded), rtol=0, atol=0
        )


@pytest.mark.parametrize("rows,width", [(0, 65), (6, 0)])
def test_empty_inputs_do_not_launch(rows, width):
    from unittest.mock import patch

    from sglang.kernels.ops.attention.dsv4 import sm90_fp4_grouped_indexer as grouped

    args = list(_inputs(width=width))
    for index in (0, 1, 3, 4):
        args[index] = args[index][:rows]
    with patch.object(grouped, "_sm90_fp4_grouped_indexer_op") as launch:
        actual = grouped.fp4_index_logits_grouped_sm90(*args)
    assert actual.shape == (rows, width)
    launch.assert_not_called()


def test_h32_topk_and_candidate_source_mapping():
    from sglang.kernels.ops.attention.dsv4.candidate_blocks import amax_topk_blocks
    from sglang.kernels.ops.attention.dsv4.fp4_indexer import store_fp4_index_k_cache
    from sglang.kernels.ops.attention.dsv4.sm90_fp4_grouped_indexer import (
        fp4_index_logits_grouped_sm90,
    )

    args = list(_inputs(width=131))
    q, weights, mapping, _, lens, cache, page_size, ratio, width, _ = args
    q.zero_()
    q[:, 0, 0] = 1
    weights.zero_()
    weights[:, 0] = 1
    lens.fill_(width)
    slots = torch.arange(width, device="cuda", dtype=torch.int32)
    keys = torch.zeros(width, 128, device="cuda", dtype=torch.bfloat16)
    keys[:, 0] = (slots + 1).to(torch.bfloat16)
    store_fp4_index_k_cache(keys, cache, slots, page_size=page_size, rne=True)
    mapping[:, ::ratio] = slots * ratio
    expected = _reference(args)
    actual = fp4_index_logits_grouped_sm90(*args)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual.topk(16).indices, expected.topk(16).indices)
    # This downstream consumer needs an eight-float-aligned score row stride.
    torch.testing.assert_close(
        amax_topk_blocks(actual, lens, (lens + 7) // 8, 8),
        amax_topk_blocks(expected, lens, (lens + 7) // 8, 8),
    )


def test_two_stream_graph_replay_changes_queries_mapping_and_visibility():
    from sglang.kernels.ops.attention.dsv4.sm90_fp4_grouped_indexer import (
        fp4_index_logits_grouped_sm90,
    )

    args = [_inputs(width=4097), _inputs(width=4097)]
    # Model cache may be shared, while queries, maps, output and packing are owned
    # by each lane. The operator must not synchronize the host with the device.
    args[1] = (*args[1][:5], args[0][5], *args[1][6:])
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    graphs, outputs = [], []
    for stream, inputs in zip(streams, args):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fp4_index_logits_grouped_sm90(*inputs)
    torch.cuda.synchronize()
    for stream, inputs in zip(streams, args):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = fp4_index_logits_grouped_sm90(*inputs)
        graphs.append(graph)
        outputs.append(output)

    for visible in (4097, 0, 63, 4097):
        for i, (stream, inputs, graph) in enumerate(zip(streams, args, graphs)):
            q, weights, mapping, req, lens, _, _, ratio, _, _ = inputs
            with torch.cuda.stream(stream):
                lens.fill_(visible)
                mapping[:, ::ratio] = (
                    torch.arange(4097, device="cuda", dtype=torch.int32) + i
                ) * ratio
                q.neg_()
                weights.mul_(-1)
                req.copy_((req + 1) % 3)
                outputs[i].fill_(float("nan"))
                graph.replay()
        for stream in streams:
            torch.cuda.current_stream().wait_stream(stream)
        for inputs, actual in zip(args, outputs):
            _assert_scores(actual, _reference(inputs), inputs[4])
            torch.testing.assert_close(
                actual, fp4_index_logits_grouped_sm90(*inputs), rtol=0, atol=0
            )
        torch.cuda.synchronize()
