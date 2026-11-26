import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-small")

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
    output = SimpleNamespace(
        seq_lens=torch.ones(3, device="cuda"), ragged_verify_layout=None
    )
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
    from sglang.kernels.ops.speculative.spec_tree import TreeMaskMode
    from sglang.srt.speculative.eagle_utils import build_tree_kernel_efficient

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


@pytest.mark.parametrize("bs", [1, 128])
@pytest.mark.parametrize("steps", [1, 7])
@pytest.mark.parametrize("mrope", [False, True])
def test_draft_cache_positions_graph(bs, steps, mrope):
    from sglang.kernels.ops.speculative.cache_locs import (
        assign_draft_cache_locs_contiguous_func,
    )

    table = torch.arange((bs + 3) * 64, dtype=torch.int32, device="cuda").reshape(
        bs + 3, 64
    )
    requests = torch.arange(bs, device="cuda")
    seq = torch.arange(bs, device="cuda", dtype=torch.int64) % 48
    cache = torch.empty((bs * steps,), device="cuda", dtype=torch.int64)
    positions = torch.empty_like(seq)
    rope = torch.empty((3, bs), device="cuda", dtype=torch.int64) if mrope else None

    def run():
        assign_draft_cache_locs_contiguous_func(
            requests, table, seq, cache, 1, steps, positions=positions, mrope=rope
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


def _legacy_chain_accept_indices(num_accept_tokens, block_size):
    steps = torch.arange(block_size, device=num_accept_tokens.device)
    rows = torch.arange(num_accept_tokens.numel(), device=num_accept_tokens.device)
    flat = rows[:, None] * block_size + steps
    return torch.where(steps < num_accept_tokens[:, None], flat, -1)


@pytest.mark.parametrize("block_size", [2, 3, 5, 8, 16])
@pytest.mark.parametrize("bs", [1, 7, 64])
@pytest.mark.parametrize("accept", ["greedy", "sampled", "reject_all", "accept_all"])
def test_dflash_commit_block_graph(block_size, bs, accept):
    """The graph-replayed committed block and chain indices equal the eager ops
    the DFlash worker ran before, bit for bit and dtype for dtype."""
    from sglang.kernels.ops.speculative.dflash import (
        compute_dflash_correct_drafts_and_bonus,
        dflash_chain_accept_indices,
        dflash_commit_block,
    )
    from sglang.srt.speculative.dflash_worker_v2 import _commit_accept

    gen = torch.Generator(device="cuda").manual_seed(bs * 31 + block_size)

    def draw(*shape):
        return torch.randint(0, 1 << 40, shape, device="cuda", generator=gen)

    candidates = draw(bs, block_size)
    prefix_lens = draw(bs)
    # A strided int32 view, as a sampler's accept column can be.
    num_correct_drafts = torch.zeros((bs, 2), device="cuda", dtype=torch.int32)[:, 0]
    bonus_tokens = draw(bs)
    out_tokens = torch.full((bs, block_size), -7, device="cuda", dtype=torch.int64)
    num_accept_tokens = torch.full((bs,), -7, device="cuda", dtype=torch.int32)
    new_seq_lens = torch.full((bs,), -7, device="cuda", dtype=torch.int64)

    def launch():
        dflash_commit_block(
            candidates=candidates,
            num_correct_drafts=num_correct_drafts,
            bonus_tokens=bonus_tokens,
            prefix_lens=prefix_lens,
            out_tokens=out_tokens,
            num_accept_tokens_out=num_accept_tokens,
            new_seq_lens_out=new_seq_lens,
        )
        return dflash_chain_accept_indices(
            num_accept_tokens=num_accept_tokens, block_size=block_size
        )

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        accept_indices = launch()
    for _ in range(3):
        candidates.copy_(draw(bs, block_size))
        prefix_lens.copy_(draw(bs))
        if accept == "greedy":
            # Each target token matches the next draft with probability 3/4.
            hit = torch.rand((bs, block_size), device="cuda", generator=gen) < 0.75
            target = torch.where(hit, candidates.roll(-1, dims=1), draw(bs, block_size))
            correct, bonus = compute_dflash_correct_drafts_and_bonus(
                candidates=candidates, target_predict=target
            )
        else:
            correct = {
                "sampled": torch.randint(
                    0, block_size, (bs,), device="cuda", generator=gen
                ),
                "reject_all": torch.zeros((bs,), device="cuda"),
                "accept_all": torch.full((bs,), block_size - 1, device="cuda"),
            }[accept]
            bonus = draw(bs)
        num_correct_drafts.copy_(correct)
        bonus_tokens.copy_(bonus)
        graph.replay()

        ref_tokens, ref_num_accept = _commit_accept(
            candidates, num_correct_drafts, bonus_tokens
        )
        expected = (
            ref_tokens,
            ref_num_accept,
            prefix_lens + ref_num_accept.to(prefix_lens.dtype),
            _legacy_chain_accept_indices(ref_num_accept, block_size),
        )
        actual = (out_tokens, num_accept_tokens, new_seq_lens, accept_indices)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, rtol=0, atol=0)


@pytest.mark.parametrize(
    "operation",
    ["cache", "commit", "commit_basic", "inputs", "lengths", "layout", "tree"],
)
def test_dynamic_batch_reuses_kernel(operation, monkeypatch):
    """Changing request counts must not create new eager preparation variants."""
    from collections import defaultdict

    from sglang.kernels.ops.speculative import cache_locs, eagle

    kernels = {
        "cache": cache_locs.assign_draft_cache_locs_contiguous,
        "commit": eagle._verify_commit_outputs_kernel,
        "commit_basic": eagle._verify_commit_outputs_kernel,
        "inputs": eagle._draft_extend_inputs_kernel,
        "lengths": eagle._prepare_draft_extend_lengths_kernel,
        "layout": eagle._prepare_draft_extend_lengths_kernel,
        "tree": eagle._build_chain_tree_kernel,
    }
    kernel = kernels[operation]
    monkeypatch.setattr(kernel, "device_caches", defaultdict(kernel.create_binder))
    cache = kernel.device_caches[torch.cuda.current_device()][0]
    width = 4
    max_bs = 257
    requests = torch.arange(max_bs, device="cuda")
    seq = torch.full((max_bs,), 17, device="cuda", dtype=torch.int64)
    accepted = torch.full((max_bs,), 3, device="cuda", dtype=torch.int32)
    predict = torch.arange(max_bs * width + 3, device="cuda", dtype=torch.int32)
    indices = torch.arange(max_bs * width, device="cuda").reshape(max_bs, width)
    draft = torch.arange(max_bs * (width - 1), device="cuda").reshape(max_bs, width - 1)
    table = torch.arange(max_bs * 64, device="cuda", dtype=torch.int32).reshape(
        max_bs, 64
    )
    sizes = [1, 2, 15, 16, 17, 31, 32, 33, 127, 128, 129, 257, 257]
    for i, bs in enumerate(sizes):
        token_count = bs * width + (3 if i == len(sizes) - 1 else 0)
        if operation == "cache":
            locations = torch.empty((bs * 3,), device="cuda", dtype=torch.int64)
            positions = torch.empty_like(seq[:bs])
            mrope = torch.empty((3, bs), device="cuda", dtype=torch.int64)
            cache_locs.assign_draft_cache_locs_contiguous_func(
                requests[:bs],
                table,
                seq[:bs],
                locations,
                1,
                3,
                positions=positions,
                mrope=mrope,
            )
            torch.testing.assert_close(positions, seq[:bs], rtol=0, atol=0)
            torch.testing.assert_close(
                mrope, seq[:bs][None, :].expand(3, -1), rtol=0, atol=0
            )
        elif operation in ("commit", "commit_basic"):
            out = eagle.prepare_verify_commit_outputs(
                predict[:token_count],
                indices[:bs],
                accepted[:bs],
                seq[:bs],
                num_draft_tokens=width if operation == "commit" else 0,
                mamba_track_interval=16 if operation == "commit" else 0,
            )
            torch.testing.assert_close(out[0], seq[:bs] + accepted[:bs], rtol=0, atol=0)
        elif operation == "inputs":
            out = eagle.prepare_draft_extend_inputs(
                accepted[:bs], predict[:token_count], width
            )
            torch.testing.assert_close(
                out[2], predict[:token_count].long(), rtol=0, atol=0
            )
        elif operation == "lengths":
            eagle.prepare_draft_extend_lengths(seq[:bs], width, 3)
        elif operation == "layout":
            eagle.prepare_draft_extend_layout(seq[:bs], width, True)
        else:
            eagle.build_chain_tree(
                predict[:bs],
                draft[:bs],
                seq[:bs],
                req_pool_indices=requests[:bs],
                req_to_token=table,
                with_mrope=True,
            )
        torch.cuda.synchronize()
        assert len(cache) == 1, (operation, bs, token_count, len(cache))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
