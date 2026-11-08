"""Frozen FP64 interval and full-row selection contract for candidate scorers.

Run explicitly on a Hopper GPU.
The frozen acceptance plan lives beside the benchmark evidence, not in a runtime
skip. No dense scorer or candidate_bf16_mqa implementation is used as an oracle.
"""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.fp4_candidate import SEED, check, implementations, make_inputs

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="requires SM90",
)


@pytest.mark.parametrize("ratio", [1, 2])
def test_public_entry_matches_cuda(ratio):
    from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
        fp4_index_logits_candidates,
    )
    from sglang.kernels.ops.attention.dsv4.fp4_indexer_cuda import (
        fp4_index_logits_candidates_cuda,
        is_fp4_index_logits_candidates_cuda_supported,
    )

    x = make_inputs(candidates=136, ratio=ratio, padded=True, strided=True)
    assert is_fp4_index_logits_candidates_cuda_supported(*x.args())
    actual = fp4_index_logits_candidates(*x.args())
    expected = fp4_index_logits_candidates_cuda(*x.args())
    assert actual.stride() == (136, 1)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("block_count", [0, 1, 7, 8, 9, 17])
@pytest.mark.parametrize("q_kind", ["arbitrary", "fakequant"])
def test_candidates(ratio, block_count, q_kind):
    x = make_inputs(candidates=block_count * 8, ratio=ratio, q_kind=q_kind, padded=True)
    x.lens.copy_(torch.tensor([0, 63, 64, 65], dtype=x.lens.dtype, device="cuda"))
    if block_count:
        # Each verify query has a separate list; includes negative padding,
        # out-of-capacity block, visible 63/64/65 and physical-page boundaries.
        ids = torch.tensor(
            [0, 7, 8, 16, 17, -1, 2, 1, 9, 3, 5, 4, 10, 6, 12, 11, 13],
            device="cuda",
            dtype=x.blocks.dtype,
        )
        for row in range(4):
            x.blocks[row].copy_(ids.roll(row)[:block_count])
    check(x)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("ratio", [1, 2])
def test_strides_metadata_and_guards(dtype, ratio):
    x = make_inputs(candidates=72, ratio=ratio, dtype=dtype, padded=True, strided=True)
    # Metadata types can differ independently at the public boundary.
    x.req = x.req.to(torch.int64 if dtype == torch.int32 else torch.int32)
    before = [a.clone() for a in x.args() if isinstance(a, torch.Tensor)]
    check(x)
    after = [a for a in x.args() if isinstance(a, torch.Tensor)]
    assert all(torch.equal(a, b) for a, b in zip(before, after)), (
        "scorer mutated an input"
    )
    # Cache/table backing padding is visible through the parent allocation.
    cache_storage = x.table.as_strided(
        (x.table.shape[0], x.table.stride(0)), x.table.stride()
    )
    mapping_storage = x.req_table.as_strided(
        (x.req_table.shape[0], x.req_table.stride(0)), x.req_table.stride()
    )
    assert (cache_storage[:, x.table.shape[1] :] == 0xA5).all()
    assert (mapping_storage[:, x.req_table.shape[1] :] == -31337).all()


@pytest.mark.parametrize("page_size", [8, 16, 128])
def test_generic_pages_with_unaligned_storage(page_size):
    x = make_inputs(candidates=136, ratio=2, padded=True)
    payload = x.table[:, : 64 * 64].reshape(-1, 64)
    scales = x.table[:, 64 * 64 :].reshape(-1, 4)
    pages = payload.shape[0] // page_size
    packed = torch.cat(
        (
            payload.reshape(pages, page_size * 64),
            scales.reshape(pages, page_size * 4),
        ),
        dim=1,
    )
    backing = torch.full(
        (pages, page_size * 68 + 3), 0xA5, device="cuda", dtype=torch.uint8
    )
    x.table = backing[:, 1 : page_size * 68 + 1]
    x.table.copy_(packed)
    x.page_size = page_size
    q_backing = torch.empty(x.q.numel() + 1, device="cuda", dtype=x.q.dtype)
    q = q_backing[1:].view_as(x.q)
    q.copy_(x.q)
    x.q = q
    assert x.q.data_ptr() % 16 == 2 and x.table.data_ptr() % 16 == 1
    check(x)
    assert (backing[:, 0] == 0xA5).all() and (backing[:, -2:] == 0xA5).all()


@pytest.mark.parametrize("rows,candidates", [(0, 0), (0, 72), (2, 0)])
def test_empty(rows, candidates):
    x = make_inputs(requests=rows, queries=1, candidates=candidates)
    check(x)


def test_zero_capacity():
    x = make_inputs(requests=2, candidates=72, max_len=0, padded=True)
    check(x, analytic=True)


@pytest.mark.parametrize("ratio", [1, 2])
def test_exact_analytic_ties_and_zero(ratio):
    x = make_inputs(candidates=72, ratio=ratio, analytic=True, padded=True)
    check(x, analytic=True)
    x.q.zero_()
    check(x, analytic=True)
    x.lens.zero_()
    check(x, analytic=True)


def test_wide_scale_range_and_scale_byte_zero():
    x = make_inputs(requests=1, queries=1, candidates=8, max_len=8, analytic=True)
    x.table[:, : 64 * 64].zero_()
    x.table[:, 64 * 64 :].reshape(-1, 64, 4)[:, :, :] = torch.tensor(
        [0, 27, 127, 227], device="cuda", dtype=torch.uint8
    )
    # Four isolated coordinates exercise subnormal scale and scales impossible
    # to represent in FP8. Separate runs prevent dynamic-range cancellation.
    payload = x.table[:, : 64 * 64].reshape(-1, 64, 64)
    for group in range(4):
        payload.zero_()
        payload[:, :, group * 16] = 2  # LUT[2] == 1
        x.q.zero_()
        x.q[:, 0, group * 32] = 1
        check(x, analytic=True)
    # Keep the distinguishing mathematical decode visible: byte0 scale is
    # mathematically2^-127 but compiled Triton flushes scale to0 before multiply.
    from sglang.test.kernels.fp4_candidate import decode_slots

    payload.zero_()
    payload[:, :, 0] = 4  # E2M1 magnitude2
    x.table[:, 64 * 64 :].fill_(0)
    x.q.zero_()
    x.q[:, 0, 0] = 2.0**100
    physical = torch.arange(8)
    assert (
        decode_slots(x.table, physical, 64, reference_ftz=False)[:, 0] == 2.0**-126
    ).all()
    assert (decode_slots(x.table, physical, 64) == 0).all()
    check(x, analytic=True)


@pytest.mark.parametrize("q_kind", ["arbitrary", "fakequant"])
def test_full_medium_oracle(q_kind):
    x = make_inputs(
        requests=2,
        queries=6,
        candidates=512,
        max_len=1025,
        q_kind=q_kind,
        seed=SEED + 1,
    )
    check(x)


def test_changed_input_graph_replay():
    funcs = implementations()
    x = make_inputs(candidates=136, ratio=2, padded=True)
    graphs, outputs = {}, {}
    for name, fn in funcs.items():
        for _ in range(3):
            fn(*x.args())
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs[name] = fn(*x.args())
        graphs[name] = graph
    for step in range(4):
        if step == 1:
            x.lens.copy_(
                torch.tensor([137, 65, 64, 63], dtype=x.lens.dtype, device="cuda")
            )
            x.blocks.copy_(x.blocks.flip(1))
            x.req_table.copy_(x.req_table.flip(0))
            x.q.neg_()
            x.weights.neg_()
            x.req.copy_(x.req.roll(1))
        elif step == 2:
            x.blocks.fill_(-1)
        elif step == 3:
            x.blocks.copy_(
                torch.arange(17, device="cuda", dtype=x.blocks.dtype).expand(4, -1)
            )
            x.lens.copy_(
                torch.tensor([0, 1, 64, 137], dtype=x.lens.dtype, device="cuda")
            )
            x.table[:, : 64 * 64].zero_()
        for out in outputs.values():
            out.fill_(12345)
        for graph in graphs.values():
            graph.replay()
        # check evaluates graph outputs against independent oracle and fresh
        # Triton output; graph result closures perform no new scoring.
        for name, fn in funcs.items():
            fresh = fn(*x.args())
            assert torch.equal(outputs[name].view(torch.int32), fresh.view(torch.int32))
        check(
            x,
            {name: (lambda *_, value=value: value) for name, value in outputs.items()},
        )


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("selection", ["torch", "paged_v2", "ragged_v2"])
def test_existing_topk_finalizer_mapping_and_underfill(ratio, selection):
    from sglang.kernels.ops.attention.dsv4.fp4_indexer import finish_paged_indexer_topk
    from sglang.kernels.ops.attention.dsv4.topk import (
        plan_topk_v2,
        topk_transform_paged_v2,
        topk_transform_ragged_v2,
    )
    from sglang.test.kernels.fp4_candidate import coordinates

    x = make_inputs(candidates=72, max_len=137, ratio=ratio, analytic=True, padded=True)
    # Exact strictly increasing scores encode physical slots in four base-4
    # coordinates. Above255 use even scores256..510 so every output is BF16
    # representable and all physical slots retain a unique score.
    slots = x.table.shape[0] * x.page_size
    values = torch.arange(slots, device="cuda")
    encoded = torch.where(values < 256, values, values - 128)
    exponent_bump = (values >= 256).byte()
    payload = x.table[:, : x.page_size * 64].reshape(-1, 64, 64)
    payload.zero_()
    codes = torch.tensor([0, 2, 4, 5], dtype=torch.uint8, device="cuda")
    for group in range(4):
        payload[:, :, group * 16] = codes[(encoded // (4**group)) % 4].reshape(-1, 64)
    x.table[:, x.page_size * 64 :].reshape(-1, 64, 4)[:] = (
        torch.tensor([127, 129, 131, 133], device="cuda", dtype=torch.uint8)[None, :]
        + exponent_bump[:, None]
    ).reshape(-1, 64, 4)
    x.q.zero_()
    x.weights.zero_()
    x.q[:, 0, ::32] = 1
    x.weights[:, 0] = 1
    x.lens.copy_(torch.tensor([0, 5, 65, 137], device="cuda", dtype=x.lens.dtype))
    for row in range(4):
        x.blocks[row] = torch.arange(9, device="cuda", dtype=x.blocks.dtype).roll(row)
    check(x, analytic=True)
    logical, physical, valid = coordinates(x)
    expected_raw = torch.full((5, 19), -1, dtype=torch.int32)
    expected_pages = expected_raw.clone()
    for row in range(4):
        columns = valid[row].nonzero().flatten()
        selected = columns[
            physical[row, columns].topk(min(16, columns.numel())).indices
        ]
        positions = logical[row, selected].sort().values
        expected_raw[row, : positions.numel()] = positions.int()
        expected_pages[row, : positions.numel()] = (
            x.req_table.cpu()[int(x.req[row]), positions * ratio] // ratio
        ).int()
    for fn in implementations().values():
        scores = fn(*x.args())
        if selection == "torch":
            selected = scores.topk(16, sorted=False).indices.int()
        else:
            selected = torch.empty((4, 16), device="cuda", dtype=torch.int32)
            scan_lens = torch.full((4,), 72, device="cuda", dtype=torch.int32)
            if selection == "paged_v2":
                topk_transform_paged_v2(
                    scores, scan_lens, None, selected, 1, plan_topk_v2(scan_lens)
                )
            else:
                topk_transform_ragged_v2(
                    scores,
                    scan_lens,
                    out_offsets=torch.zeros_like(scan_lens),
                    out_indices=selected,
                )
        raw_storage = torch.full((5, 38), 12345, dtype=torch.int32, device="cuda")
        page_storage = raw_storage.clone()
        raw, pages = raw_storage[:, :19], page_storage[:, :19]
        finish_paged_indexer_topk(
            selected,
            scores,
            x.lens,
            x.req,
            x.req_table,
            pages,
            raw,
            ratio,
            False,
            candidate_blocks=x.blocks,
        )
        assert torch.equal(raw.cpu(), expected_raw)
        assert torch.equal(pages.cpu(), expected_pages)
        assert (raw_storage[:, 19:] == 12345).all()
        assert (page_storage[:, 19:] == 12345).all()


def test_frozen_case2_rotation_payload_regression():
    # Frozen benchmark case2 layer1: payload seed40861, metadata seed40860.
    # Natural D128 can change rounded scores within the frozen FP64 interval.
    # Check both complete 8192-candidate rows and full-row 512/1024 selection.
    # The historical bitwise diagnostic remains in the frozen baseline.
    kwargs = dict(
        requests=1,
        queries=2,
        candidates=8192,
        max_len=16384,
        ratio=2,
        mapping="random",
        q_kind="fakequant",
    )
    base = make_inputs(**kwargs, seed=40860)
    layer = make_inputs(**kwargs, seed=40861)
    for name in ("req", "req_table", "lens", "blocks"):
        setattr(layer, name, getattr(base, name))
    check(layer)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
