import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

from sglang.kernels.ops.speculative.eagle import (
    prepare_draft_extend_inputs,
    prepare_draft_extend_lengths,
    prepare_verify_commit_outputs,
)


@pytest.mark.parametrize("bs", [0, 1, 3, 128, 256])
@pytest.mark.parametrize("width", [1, 4, 7])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_prepare_integer_outputs(bs, width, dtype):
    depth = min(width, 4)
    accepted = torch.ones(bs * 2, device="cuda", dtype=dtype)[::2]
    seq = torch.arange(bs * 2, device="cuda", dtype=dtype)[::2] + 8192
    predict = torch.arange(bs * width * 2, device="cuda", dtype=dtype)[::2]
    storage = torch.zeros((bs, depth * 2), device="cuda", dtype=torch.int64)
    indices = storage[:, ::2]
    if bs:
        indices.copy_(
            torch.randperm(bs * width, device="cuda").reshape(bs, width)[:, :depth]
        )
        indices[0, -1] = -1
    prepare_verify_commit_outputs(predict, indices, accepted, seq)
    prepare_draft_extend_inputs(accepted, predict, width)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = prepare_verify_commit_outputs(predict, indices, accepted, seq)
        draft = prepare_draft_extend_inputs(accepted, predict, width)
    for replay in range(depth + 1):
        accepted.fill_(1 + replay % depth)
        seq.add_(1)
        predict.add_(17)
        graph.replay()
        expected_bonus = predict[indices][
            torch.arange(bs, device="cuda"), accepted.long() - 1
        ].int()
        references = (
            seq + accepted,
            expected_bonus,
            accepted - 1,
            torch.arange(bs, device="cuda") * width + accepted - 1,
            predict.long(),
        )
        for actual, expected in zip(outputs + draft, references):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("bs", [0, 1, 128, 257])
@pytest.mark.parametrize("front", [0, 3])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_draft_extend_lengths(bs, front, dtype):
    seq = torch.arange(bs * 2, device="cuda", dtype=dtype)[::2]
    outputs = prepare_draft_extend_lengths(seq, 4, front)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = prepare_draft_extend_lengths(seq, 4, front)
    for delta in [0, 1, 8192]:
        seq.add_(delta)
        graph.replay()
        expected = (
            (seq - front).clamp(min=0).int(),
            torch.full((bs,), 4 + front, dtype=torch.int32, device="cuda"),
            seq + 4,
        )
        for a, b in zip(outputs, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("mixed", [False, True])
def test_spec_mrope_positions_graph(dtype, mixed):
    from types import SimpleNamespace

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    positions = torch.arange(24, device="cuda", dtype=dtype)[::2]
    mm_inputs = [None, None, None]
    if mixed:
        mm_inputs[1] = SimpleNamespace(mrope_position_delta=torch.tensor([[7]]))
    batch = SimpleNamespace(
        multimodal_inputs=mm_inputs, spec_info=SimpleNamespace(positions=positions)
    )
    runner = SimpleNamespace(device="cuda")
    output = SimpleNamespace(seq_lens=torch.ones(3, device="cuda"))
    ForwardBatch.compute_spec_mrope_positions(output, runner, batch)
    graph = torch.cuda.CUDAGraph()
    if not mixed:
        with torch.cuda.graph(graph):
            ForwardBatch.compute_spec_mrope_positions(output, runner, batch)
    for delta in [0, 17]:
        positions.add_(delta)
        if mixed:
            ForwardBatch.compute_spec_mrope_positions(output, runner, batch)
        else:
            graph.replay()
        offsets = torch.tensor([0, 7 if mixed else 0, 0], device="cuda").view(3, 1)
        expected = (positions.view(3, 4) + offsets).flatten().unsqueeze(0).repeat(3, 1)
        torch.testing.assert_close(output.mrope_positions, expected, rtol=0, atol=0)


@pytest.mark.parametrize("bs", [0, 1, 3, 129])
@pytest.mark.parametrize("width", [1, 4, 7])
@pytest.mark.parametrize("mrope", [False, True])
def test_draft_extend_layout_graph(bs, width, mrope):
    from sglang.kernels.ops.attention.position import compute_position_triton
    from sglang.kernels.ops.speculative.eagle import prepare_draft_extend_layout

    seq = torch.arange(bs * 2, device="cuda", dtype=torch.int64)[::2]
    prepare_draft_extend_layout(seq, width, mrope)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prefix, lengths, post, info = prepare_draft_extend_layout(seq, width, mrope)
    for value in [0, 1, 131000]:
        seq.fill_(value)
        graph.replay()
        torch.testing.assert_close(prefix, seq.clamp(min=0).int(), rtol=0, atol=0)
        torch.testing.assert_close(
            lengths, torch.full_like(lengths, width), rtol=0, atol=0
        )
        torch.testing.assert_close(post, seq + width, rtol=0, atol=0)
        if bs:
            positions, starts = compute_position_triton(
                seq.clamp(min=0).int(), torch.full_like(lengths, width), bs * width
            )
            torch.testing.assert_close(info[0], positions, rtol=0, atol=0)
            torch.testing.assert_close(info[1], starts, rtol=0, atol=0)
        else:
            assert info[0].numel() == info[1].numel() == 0
        if mrope:
            torch.testing.assert_close(
                info[2], info[0].unsqueeze(0).repeat(3, 1), rtol=0, atol=0
            )
        else:
            assert info[2] is None


@pytest.mark.parametrize("bs", [1, 3, 128, 256])
@pytest.mark.parametrize("depth", [1, 3, 7])
@pytest.mark.parametrize("preallocated", [False, True])
def test_chain_tree_graph(bs, depth, preallocated):
    from sglang.kernels.ops.speculative.eagle import build_chain_tree
    from sglang.srt.speculative.eagle_utils import (
        TreeMaskMode,
        build_tree_kernel_efficient,
    )

    bonus = torch.arange(bs * 2, device="cuda", dtype=torch.int32)[::2]
    draft = torch.arange(bs * depth * 4, device="cuda", dtype=torch.int64).reshape(
        bs * 2, depth * 2
    )[::2, ::2]
    seq = torch.arange(bs * 2, device="cuda", dtype=torch.int64)[::2]
    parents = (
        torch.arange(-1, depth - 1, device="cuda").repeat(bs, 1)
        if depth > 1
        else torch.empty((bs, 0), device="cuda", dtype=torch.int64)
    )
    selected = torch.arange(depth, device="cuda").repeat(bs, 1)
    width = depth + 1
    buf = (
        torch.empty((bs * width * width + 19,), dtype=torch.bool, device="cuda")
        if preallocated
        else None
    )
    build_chain_tree(bonus, draft, seq, buf)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = build_chain_tree(bonus, draft, seq, buf)
    for length in [0, 8192, 131000]:
        seq.fill_(length)
        bonus.add_(1)
        draft.add_(3)
        if buf is not None:
            buf.fill_(False)
        graph.replay()
        expected = build_tree_kernel_efficient(
            bonus,
            parents,
            selected,
            draft,
            seq.contiguous(),
            bs * length,
            1,
            depth,
            width,
            TreeMaskMode.QLEN_ONLY,
        )
        for i, (a, e) in enumerate(zip(actual, expected)):
            if i == 0:
                a = a[: bs * width * width]
            torch.testing.assert_close(a, e, rtol=0, atol=0)
        if buf is not None:
            assert not buf[-19:].any()


@pytest.mark.parametrize("bs", [0, 1, 3, 128])
@pytest.mark.parametrize("depth", [1, 3, 7])
@pytest.mark.parametrize("mrope", [False, True])
def test_chain_tree_prepared_outputs_graph(bs, depth, mrope):
    from sglang.kernels.ops.speculative.eagle import build_chain_tree

    table = torch.arange((bs + 3) * 128, dtype=torch.int32, device="cuda").reshape(
        bs + 3, 128
    )[:, ::2]
    requests = torch.arange(bs * 2, device="cuda")[::2] % (bs + 3)
    seq = torch.arange(bs * 2, device="cuda")[::2] % 48
    bonus = torch.arange(bs, device="cuda")
    draft = torch.arange(bs * depth, device="cuda").reshape(bs, depth)

    def run():
        return build_chain_tree(
            bonus,
            draft,
            seq,
            req_pool_indices=requests,
            req_to_token=table,
            with_mrope=mrope,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    for i in range(3):
        requests.add_(1).remainder_(bs + 3)
        seq.add_(1).remainder_(48)
        table.add_(13)
        graph.replay()
        positions = seq[:, None] + torch.arange(depth + 1, device="cuda")
        expected = table[requests[:, None], positions].flatten().long()
        torch.testing.assert_close(out[6], expected, rtol=0, atol=0)
        torch.testing.assert_close(out[1], positions.flatten(), rtol=0, atol=0)
        if mrope:
            torch.testing.assert_close(
                out[7], positions.flatten()[None, :].repeat(3, 1), rtol=0, atol=0
            )
        else:
            assert out[7] is None


@pytest.mark.parametrize("bs", [1, 3, 128])
@pytest.mark.parametrize("steps", [1, 3, 7])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("mrope", [False, True])
def test_draft_cache_positions_graph(bs, steps, dtype, mrope):
    from sglang.kernels.ops.speculative.cache_locs import (
        assign_draft_cache_locs_contiguous,
    )

    table = torch.arange((bs + 3) * 64, dtype=torch.int32, device="cuda").reshape(
        bs + 3, 64
    )
    requests = torch.arange(bs, device="cuda")
    seq = torch.arange(bs, device="cuda", dtype=dtype) % 48
    cache = torch.empty((bs * steps,), device="cuda", dtype=torch.int64)
    positions = torch.empty_like(seq)
    rope = torch.empty((3, bs), device="cuda", dtype=torch.int64) if mrope else None

    def run():
        assign_draft_cache_locs_contiguous[(bs,)](
            requests,
            table,
            seq,
            cache,
            64,
            1,
            steps,
            positions=positions,
            mrope=rope,
            BS=bs,
            WRITE_POSITIONS=True,
            WRITE_MROPE=mrope,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for i in range(3):
        requests.add_(1).remainder_(bs + 3)
        seq.add_(1).remainder_(48)
        table.add_(17)
        graph.replay()
        expected = (
            table[
                requests[:, None],
                seq[:, None].long() + torch.arange(steps, device="cuda"),
            ]
            .flatten()
            .long()
        )
        torch.testing.assert_close(cache, expected, rtol=0, atol=0)
        torch.testing.assert_close(positions, seq, rtol=0, atol=0)
        if mrope:
            torch.testing.assert_close(
                rope, seq.long()[None, :].repeat(3, 1), rtol=0, atol=0
            )


@pytest.mark.parametrize("bs", [0, 1, 7, 128, 257])
@pytest.mark.parametrize("width", [1, 4, 7])
@pytest.mark.parametrize("interval", [0, 16, 512])
def test_combined_verify_commit_metadata(bs, width, interval):
    seq = torch.arange(bs * 2, device="cuda", dtype=torch.int64)[::2] + 509
    accepted = torch.ones(bs * 2, device="cuda", dtype=torch.int32)[::2]
    predict = torch.arange(bs * width * 2, device="cuda", dtype=torch.int32)[::2]
    accept = torch.arange(bs * width, device="cuda", dtype=torch.int64).reshape(
        bs, width
    )
    outputs = prepare_verify_commit_outputs(
        predict, accept, accepted, seq, width, interval
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = prepare_verify_commit_outputs(
            predict, accept, accepted, seq, width, interval
        )
    for n in range(1, width + 1):
        accepted.fill_(n)
        seq.add_(1)
        predict.add_(13)
        graph.replay()
        new_seq, bonus, (last, tracking), draft = outputs
        rows = torch.arange(bs, device="cuda")
        torch.testing.assert_close(new_seq, seq + accepted, rtol=0, atol=0)
        torch.testing.assert_close(
            bonus, predict[accept[rows, n - 1]].int(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            last, accept[rows, n - 1] - rows * width, rtol=0, atol=0
        )
        references = (accepted - 1, rows * width + n - 1, predict.long())
        for actual, expected in zip(draft, references):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if interval:
            crossed = seq // interval != (seq + accepted) // interval
            offsets = (((seq + accepted) // interval) * interval - seq - 1).clamp(min=0)
            ref = torch.where(crossed, accept[rows, offsets] - rows * width, -1)
            torch.testing.assert_close(tracking, ref, rtol=0, atol=0)
        else:
            assert tracking is None
