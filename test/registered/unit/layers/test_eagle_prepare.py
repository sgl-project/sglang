import pytest
import torch

from sglang.kernels.ops.speculative.eagle import (
    prepare_draft_extend_inputs,
    prepare_draft_extend_lengths,
    prepare_verify_commit_outputs,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


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


@pytest.mark.parametrize("bs", [0, 1, 129])
@pytest.mark.parametrize("width", [1, 17, 2560])
@pytest.mark.parametrize("strided", [False, True])
def test_relay_scatter_graph(bs, width, strided):
    from sglang.kernels.ops.speculative.gather_spec_extras import scatter_spec_extras

    slots = bs + 3
    indices = torch.randperm(slots, device="cuda")[:bs].repeat_interleave(2)[::2]
    if bs:
        indices[0] -= slots
    pairs = []
    for n, src_type, dst_type in [
        (1, torch.int32, torch.int64),
        (3, torch.float32, torch.float32),
        (width, torch.float32, torch.bfloat16),
        (7, torch.int64, torch.int64),
    ]:
        stride = 2 if strided else 1
        src = (
            torch.arange(bs * n * stride, device="cuda")
            .reshape(bs, n * stride)[:, ::stride]
            .to(src_type)
        )
        dst = torch.full((slots, n), -7, dtype=dst_type, device="cuda")
        pairs.append((dst, src))
    scatter_spec_extras(indices, pairs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        scatter_spec_extras(indices, pairs)
    for replay in range(3):
        expected = []
        for dst, src in pairs:
            dst.fill_(-7)
            src.add_(replay)
            ref = dst.clone()
            ref[indices] = src.to(dst.dtype)
            expected.append(ref)
        graph.replay()
        for (actual, _), ref in zip(pairs, expected):
            torch.testing.assert_close(actual, ref, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [0, 1, 4, 128])
@pytest.mark.parametrize("cols", [32771, 151936, 248320])
def test_speculative_argmax_graph(rows, cols):
    from sglang.kernels.ops.speculative.row_argmax import speculative_argmax

    logits = torch.randn((rows * 2, cols), device="cuda")[::2]
    speculative_argmax(logits)
    speculative_argmax(logits, with_probs=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ids = speculative_argmax(logits)
        probabilities, ids2 = speculative_argmax(logits, with_probs=True)
    for case in range(4):
        if case == 1:
            logits.fill_(-float("inf"))
        elif case == 2:
            logits[:, 2049] = float("inf")
            logits[:, -1] = float("inf")
        elif case == 3:
            logits[:, 15] = float("nan")
            logits[:, -17] = float("nan")
        graph.replay()
        expected = logits.argmax(dim=-1)
        torch.testing.assert_close(ids, expected, rtol=0, atol=0)
        torch.testing.assert_close(ids2[:, 0], expected, rtol=0, atol=0)
        torch.testing.assert_close(
            probabilities, torch.ones_like(probabilities), rtol=0, atol=0
        )


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


@pytest.mark.parametrize("bs", [0, 1, 3, 129])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_mamba_track_gather_graph(bs, dtype):
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        gather_mamba_track_indices,
    )

    mapping = torch.arange(300 * 12, device="cuda", dtype=dtype).reshape(300, 12)[
        ::2, ::3
    ]
    requests = torch.arange(bs * 2, device="cuda", dtype=dtype)[::2] % 150
    requests -= 150
    positions = torch.arange(bs * 2, device="cuda", dtype=dtype)[::2] % 4
    gather_mamba_track_indices(mapping, requests, positions)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = gather_mamba_track_indices(mapping, requests, positions)
    for i in range(3):
        positions.add_(1).remainder_(4)
        mapping.add_(i)
        graph.replay()
        expected = (
            mapping[requests.long()]
            .gather(1, positions.long().unsqueeze(1))
            .squeeze(1)
            .long()
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("bs", [1, 3, 128])
@pytest.mark.parametrize("width", [1, 2, 4, 7])
@pytest.mark.parametrize("vocab", [131073, 248320])
def test_greedy_verify_chain_graph(bs, width, vocab):
    from sglang.kernels.ops.speculative.row_argmax import greedy_verify_chain
    from sglang.srt.speculative.eagle_utils import verify_tree_greedy_func

    logits = torch.randn((bs * width, vocab), device="cuda")
    candidates = torch.empty((bs * 2, width), device="cuda", dtype=torch.int64)[::2]
    indices = torch.arange(bs * width, device="cuda").reshape(bs, width)
    next_tokens = torch.arange(1, width + 1, device="cuda").repeat(bs, 1)
    next_tokens[:, -1] = -1
    siblings = torch.full_like(indices, -1)
    greedy_verify_chain(logits, candidates)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = greedy_verify_chain(logits, candidates)
    for case in range(3):
        if case == 1:
            logits.fill_(-float("inf"))
            logits[:, -2:] = float("inf")
        elif case == 2:
            logits[:, 3] = float("nan")
            logits[:, -1] = float("nan")
        target = logits.argmax(-1).reshape(bs, width)
        for accepted in range(width):
            candidates[:, 0] = 17
            candidates[:, 1:] = target[:, :-1]
            if accepted < width - 1:
                candidates[:, accepted + 1].add_(1).remainder_(vocab)
            graph.replay()
            expected = verify_tree_greedy_func(
                predicts=torch.zeros(bs * width, device="cuda", dtype=torch.int32),
                accept_index=torch.full(
                    (bs, width), -1, device="cuda", dtype=torch.int32
                ),
                accept_token_num=torch.empty(bs, device="cuda", dtype=torch.int32),
                candidates=candidates.contiguous(),
                retrieve_index=indices,
                retrieve_next_token=next_tokens,
                retrieve_next_sibling=siblings,
                target_predict=target,
                topk=1,
            )
            for a, e in zip(actual[:3], expected):
                torch.testing.assert_close(a, e, rtol=0, atol=0)
            torch.testing.assert_close(actual[3], target, rtol=0, atol=0)


@pytest.mark.parametrize("bs", [0, 1, 3, 129])
@pytest.mark.parametrize("position", [0, 1, 3])
def test_mamba_uniform_track_gather_graph(bs, position):
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        gather_mamba_track_indices,
    )

    mapping = torch.arange(300 * 12, device="cuda").reshape(300, 12)[::2, ::3]
    requests = torch.arange(bs * 2, device="cuda")[::2] % 150
    gather_mamba_track_indices(mapping, requests, uniform_position=position)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = gather_mamba_track_indices(
            mapping, requests, uniform_position=position
        )
    for i in range(3):
        requests.add_(1).remainder_(150)
        mapping.add_(i)
        graph.replay()
        torch.testing.assert_close(actual, mapping[requests, position], rtol=0, atol=0)


@pytest.mark.parametrize("positions", [[0], [1], [0, 0, 0], [0, 1, 0]])
@pytest.mark.parametrize("override", [False, True])
def test_mamba_track_indices_from_reqs(positions, override):
    from types import SimpleNamespace

    from sglang.srt.managers.schedule_batch import set_mamba_track_indices_from_reqs

    mapping = torch.arange(20, device="cuda", dtype=torch.int64).reshape(10, 2)
    requests = torch.arange(len(positions), device="cuda", dtype=torch.int64) + 2
    batch = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            req_index_to_mamba_ping_pong_track_buffer_mapping=mapping
        ),
        req_pool_indices=requests,
        reqs=[
            SimpleNamespace(kv=SimpleNamespace(mamba_next_track_idx=(p if p else None)))
            for p in positions
        ],
    )
    set_mamba_track_indices_from_reqs(
        batch, track_positions=positions if override else None
    )
    expected = mapping[requests, torch.tensor(positions, device="cuda")]
    torch.testing.assert_close(batch.mamba_track_indices, expected, rtol=0, atol=0)
    assert batch.mamba_track_buffer_indices == positions


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


@pytest.mark.parametrize("bs", [0, 1, 3, 128])
@pytest.mark.parametrize("overlap", ["none", "same", "cross"])
@pytest.mark.parametrize("prepared", [False, True])
def test_combined_state_commit_tracking(bs, overlap, prepared):
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        fused_mamba_state_scatter_multi,
        prepare_mamba_state_scatter_multi,
    )

    device = "cuda"
    pool = 2 * bs + 2
    indices = torch.arange(bs * 2, device=device, dtype=torch.int32)[::2] // 2
    steps = torch.full((bs * 2,), 2, device=device, dtype=torch.int64)[::2]
    track = indices + bs
    if overlap == "same":
        track = indices.clone()
    elif overlap == "cross":
        track = indices.roll(1)
    tracking = torch.full((bs * 2,), 1, device=device, dtype=torch.int32)[::2]
    storage = torch.randn((2, bs, 65, 6), device=device, dtype=torch.bfloat16)
    pairs = [
        (
            torch.empty((2, pool, 65, 3), device=device, dtype=torch.bfloat16),
            storage.unfold(-1, 3, 1).permute(0, 1, 3, 2, 4),
        ),
        (
            torch.empty((1, pool, 2), device=device, dtype=torch.int64),
            torch.randint(0, 2**40, (1, bs, 4, 2), device=device),
        ),
    ]
    metadata = prepare_mamba_state_scatter_multi(pairs) if prepared else None
    fused_mamba_state_scatter_multi(
        pairs, indices, steps, track, tracking, _metadata=metadata
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused_mamba_state_scatter_multi(
            pairs, indices, steps, track, tracking, _metadata=metadata
        )
    for step in [-1, 0, 3, 4]:
        tracking.fill_(step)
        if bs > 1:
            tracking[1] = -1
        for dst, src in pairs:
            dst.fill_(-7)
        graph.replay()
        for dst, src in pairs:
            expected = torch.full_like(dst, -7)
            rows = torch.arange(bs, device=device)
            expected[:, indices.long()] = src[:, rows, steps]
            valid = (tracking >= 0) & (tracking < 4)
            expected[:, track[valid].long()] = src[
                :, rows[valid], tracking[valid].long()
            ]
            assert torch.equal(dst, expected)


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


def test_state_scatter_metadata_cache_invalidation():
    from types import SimpleNamespace

    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
    )

    def tensor(shape, dtype=torch.bfloat16):
        return torch.zeros(shape, device="cuda", dtype=dtype)

    states = SimpleNamespace(
        temporal=tensor((2, 4, 7)),
        intermediate_ssm=tensor((2, 1, 4, 7)),
        conv=[tensor((2, 4, 3, 2))],
        intermediate_conv_window=[tensor((2, 1, 4, 3, 2))],
    )
    pool = SimpleNamespace(
        short_conv_pool=SimpleNamespace(
            conv_state=tensor((1, 4, 5)), intermediate_conv_state=tensor((1, 1, 4, 5))
        ),
        ngram_pool=SimpleNamespace(
            context=tensor((4, 2), torch.int64),
            intermediate_context=tensor((1, 4, 2), torch.int64),
        ),
    )
    backend = object.__new__(HybridLinearAttnBackend)
    backend.linear_attn_backend = SimpleNamespace(req_to_token_pool=pool)
    first = backend._prepare_verify_state_scatter(states)
    assert backend._prepare_verify_state_scatter(states) is first
    pool.ngram_pool.context = tensor((4, 2), torch.int64)
    second = backend._prepare_verify_state_scatter(states)
    assert second is not first
    assert second[0][-1][0].data_ptr() == pool.ngram_pool.context.data_ptr()
    old_storage = pool.ngram_pool.context
    old_storage.data = tensor((4, 2), torch.int64)
    third = backend._prepare_verify_state_scatter(states)
    assert third is not second
    assert third[0][-1][0].data_ptr() == old_storage.data_ptr()
