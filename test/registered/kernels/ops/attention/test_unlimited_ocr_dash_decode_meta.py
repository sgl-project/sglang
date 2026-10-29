"""Fused one-token decode metadata (page ids, Page2D coordinates, candidate mask,
local-window union, cos/sin table) and the Page2D RoPE kernels must match the
PyTorch path the backend uses for every other step."""

import copy
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.unlimited_ocr_dash.decode_meta import (
    local_union_fused,
)
from sglang.kernels.ops.attention.unlimited_ocr_dash.page2d_rope import (
    apply_page2d_rope,
    apply_page2d_rope_pair,
)
from sglang.srt.layers.attention.unlimited_dash_ops.decode_meta import (
    cos_sin_2d_fused,
    decode_allow_fused,
    decode_meta_fused,
)
from sglang.srt.layers.attention.unlimited_dash_ops.page2d_rope import (
    PageCoordState,
    PageRope2D,
)
from sglang.srt.layers.attention.unlimited_dash_ops.page_state import (
    BatchLayout,
    PageIsolationState,
)
from sglang.srt.layers.attention.unlimited_ocr_v2_backend import (
    UnlimitedOCRV2AttnBackend,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.unlimited_ocr_dash import (
    cos_sin_2d_reference,
    local_union_reference,
    page2d_rope_reference,
    torch_decode_page_meta,
)

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

BS, IMG, PAGE, MM_PAD = 128, 99, 7, 1_000_000
NUM_BLOCKS, WIDTH = 64, 12


def _prompt(n_pages: int, images_per_page: int, with_marker: bool) -> list[int]:
    ids = [1, 2, 3]
    for p in range(n_pages):
        ids += ([PAGE] if with_marker else []) + [MM_PAD + p] * images_per_page
        ids += [4, 5]
    return ids


def _prefilled_state(prompts, with_marker: bool):
    """Page / coordinate state after a PyTorch prefill of ``prompts``."""
    cfg = {"image_token_id": IMG, "per_page_image_tokens": 50}
    if with_marker:
        cfg["page_token_id"] = PAGE
    page_state = PageIsolationState(cfg, BS)
    coord_state = PageCoordState(cfg, max_reqs=8, max_pages=16)
    # Block 0 is reserved: never live, so stale table columns point at no image page.
    bt = torch.zeros(len(prompts), WIDTH, dtype=torch.int32)
    for r in range(len(prompts)):
        bt[r] = 1 + r * WIDTH + torch.arange(WIDTH)
    bt = bt.cuda()
    lens = torch.tensor([len(p) for p in prompts], device="cuda")
    q_start = torch.cat([torch.zeros(1, device="cuda"), lens.cumsum(0)]).long()
    ids = torch.tensor(sum(prompts, []), device="cuda")
    req = torch.arange(len(prompts), device="cuda").repeat_interleave(lens)
    pos = torch.arange(ids.numel(), device="cuda") - q_start[req]
    slots = bt.long()[req].gather(1, (pos // BS)[:, None]).squeeze(1) * BS + pos % BS
    layout = BatchLayout(
        num_tokens=ids.numel(),
        num_reqs=len(prompts),
        q_start=q_start,
        start_pos=torch.zeros(len(prompts), dtype=torch.int64, device="cuda"),
        block_table=bt,
        req_of_token=req,
    )
    page_state.step(ids, layout, slots, NUM_BLOCKS * BS)
    coord_state.step(ids, layout)
    return page_state, coord_state, bt, lens.to(torch.int32)


@pytest.mark.parametrize("with_marker", [True, False])
def test_fused_decode_meta_matches_torch_path(with_marker):
    # Prompt ends land mid-block, exactly on a block edge, and one past it.
    prompts = [_prompt(2, 60, with_marker), _prompt(3, 40, with_marker)]
    prompts.append([1] * (2 * BS - 1))
    prompts.append(_prompt(1, 121, with_marker))
    fused_page, fused_coord, bt, seq = _prefilled_state(prompts, with_marker)
    torch_page, torch_coord = copy.deepcopy(fused_page), copy.deepcopy(fused_coord)
    gen = torch.Generator().manual_seed(int(with_marker))
    num_slots = NUM_BLOCKS * BS
    for _ in range(6):
        seq = seq + 1
        pos = seq.long() - 1
        slots = bt.long().gather(1, (pos // BS)[:, None]).squeeze(1) * BS + pos % BS
        choices = torch.tensor([PAGE, IMG, MM_PAD + 3, 4, 5])
        ids = choices[torch.randint(0, 5, (seq.numel(),), generator=gen)].cuda()
        q_page, coords = decode_meta_fused(
            input_ids=ids,
            block_table=bt,
            seq_lens=seq,
            slot_mapping=slots,
            page_state=fused_page,
            coord_state=fused_coord,
            num_slots=num_slots,
            block_size=BS,
        )
        allow = decode_allow_fused(
            block_table=bt,
            page_state=fused_page,
            q_page=q_page,
            seq_lens=seq,
            max_block=WIDTH + 3,
            num_blocks=NUM_BLOCKS,
            block_size=BS,
        )
        ref_q_page, ref_allow, ref_coords = torch_decode_page_meta(
            page_state=torch_page,
            coord_state=torch_coord,
            input_ids=ids,
            block_table=bt,
            seq_lens=seq,
            slot_mapping=slots,
            num_slots=num_slots,
            max_block=WIDTH + 3,
            num_blocks=NUM_BLOCKS,
        )
        assert torch.equal(q_page, ref_q_page)
        assert torch.equal(coords[0], ref_coords[0])
        assert torch.equal(coords[1], ref_coords[1])
        assert torch.equal(allow.bool(), ref_allow)
        for name in ("_slot_img_page", "_slot_marker_cum", "_slot_img_run"):
            assert torch.equal(getattr(fused_page, name), getattr(torch_page, name))
        # The last coordinate row is scratch for masked-out writes.
        live = slice(0, fused_coord.max_reqs)
        for name in ("st_marker", "st_pp", "cnt"):
            fused, ref = getattr(fused_coord, name), getattr(torch_coord, name)
            assert torch.equal(fused[live], ref[live])


@pytest.mark.parametrize("local_blocks", [1, 3])
@pytest.mark.parametrize("max_block", [5, 16, 300])
def test_local_union(local_blocks, max_block):
    gen = torch.Generator().manual_seed(max_block)
    allow = (torch.rand(4, 16, generator=gen) < 0.2).to(torch.int8).cuda()
    # seq 1 sits in block 0; 300 * BS overruns a narrow table and is clamped.
    seq = torch.tensor([1, BS, 2 * BS + 1, 300 * BS], dtype=torch.int32).cuda()
    out = torch.full((4, max_block + 7), 3, dtype=torch.int8, device="cuda")
    view = local_union_fused(
        allow=allow,
        seq_lens=seq,
        out=out,
        max_block=max_block,
        block_size=BS,
        local_blocks=local_blocks,
    )
    ref = local_union_reference(
        allow=allow,
        seq_lens=seq,
        max_block=max_block,
        block_size=BS,
        local_blocks=local_blocks,
    )
    assert torch.equal(view.cpu(), ref)
    assert bool((out[:, max_block:] == 3).all())


@pytest.mark.parametrize(
    "dim,page_pairs,mode,page_mod,dtype",
    [
        (64, 8, "resample", 0, torch.bfloat16),
        (128, 24, "truncate", 0, torch.float32),
        (96, 16, "resample", 5, torch.bfloat16),
    ],
)
def test_cos_sin_2d(dim, page_pairs, mode, page_mod, dtype):
    rope = PageRope2D(dim, 10000.0, 100.0, page_pairs, mode, page_mod)
    gen = torch.Generator().manual_seed(dim)
    tok = torch.randint(0, 5000, (37,), generator=gen).cuda()
    page = torch.randint(0, 900, (37,), generator=gen).cuda()
    out_cos = torch.empty(40, dim, dtype=dtype, device="cuda")
    out_sin = torch.empty_like(out_cos)
    cos, sin = cos_sin_2d_fused(
        rope=rope,
        tok_pos=tok,
        page_pos=page,
        dtype=dtype,
        out_cos=out_cos,
        out_sin=out_sin,
    )
    assert cos.data_ptr() == out_cos.data_ptr()
    ref_cos, ref_sin = cos_sin_2d_reference(
        inv_tok=rope.inv_tok.cuda(),
        inv_page=rope.inv_page.cuda(),
        tok_pos=tok,
        page_pos=page,
        page_mod=page_mod,
        dtype=dtype,
    )
    tol = dict(atol=1e-5, rtol=0) if dtype == torch.float32 else dict(atol=8e-3, rtol=0)
    torch.testing.assert_close(cos, ref_cos, **tol)
    torch.testing.assert_close(sin, ref_sin, **tol)


@pytest.mark.parametrize(
    "tokens,heads_x,heads_y,head_dim", [(1, 8, 1, 64), (33, 16, 2, 128)]
)
def test_page2d_rope_is_bit_identical_to_torch(tokens, heads_x, heads_y, head_dim):
    """The fused bf16 RoPE rounds each product like PyTorch, so outputs are equal."""
    gen = torch.Generator().manual_seed(tokens)

    def rand(*shape):
        return torch.randn(*shape, generator=gen).to(torch.bfloat16).cuda()

    x, y = rand(tokens, heads_x * head_dim), rand(tokens, heads_y * head_dim)
    cos, sin = rand(tokens, head_dim), rand(tokens, head_dim)
    ref_x = page2d_rope_reference(x, cos, sin, head_dim)
    assert torch.equal(apply_page2d_rope(x, cos, sin, head_dim), ref_x)
    ox, oy = apply_page2d_rope_pair(x, y, cos, sin, head_dim)
    assert torch.equal(ox, ref_x)
    assert torch.equal(oy, page2d_rope_reference(y, cos, sin, head_dim))


def _decode_backend(prompts, page_state, coord_state, bt, *, graph: bool):
    """A backend holding only the decode-metadata state, on a prefilled batch."""
    b = object.__new__(UnlimitedOCRV2AttnBackend)
    b.device = torch.device("cuda")
    b.block_size = BS
    b.local_blocks = 1
    b.buf_width = 4
    b.num_layers = 0
    b.max_spec_slots = 1
    b.image_bucket_constraint = True
    b.max_context_len = WIDTH * BS
    b.max_block_static = WIDTH
    b.model_dtype = torch.float32
    b.page_state, b.coord_state = page_state, coord_state
    b.routers = [SimpleNamespace(image_only=True)]
    b.rope_main = PageRope2D(64, 10000.0, 256.0, 8, "truncate", 0)
    b.rope_index = PageRope2D(32, 10000.0, 256.0, 8, "resample", 0)
    b.kv_pool = SimpleNamespace(
        get_index_k_buffer=lambda _: torch.empty(NUM_BLOCKS * BS, 1, device="cuda")
    )
    pos = torch.arange(WIDTH * BS, device="cuda")
    b.req_to_token = (bt.long()[:, pos // BS] * BS + pos % BS).to(torch.int32)
    # Non-zero page bases so the RoPE page shift is part of the comparison.
    b._page_base_row = torch.arange(len(prompts), device="cuda", dtype=torch.int64)
    b._cos_sin_cache = {}
    b._logged_first = True
    b.forward_metadata = None
    b.cg_max_bs = b.cg_max_tokens = 0
    b._cg_ready = False
    b._init_decode_buffer_slots()
    b._init_prefill_buffer_slots()
    if graph:
        b.init_cuda_graph_state(max_bs=8, max_num_tokens=8)
    return b


def test_eager_decode_metadata_matches_graph_path():
    """Decode without CUDA graphs must build the same step geometry as replay.

    Bug: the eager decode step took the "refreshed inside the graph" branch,
    which binds page ids, Page2D coordinates and the candidate mask to graph
    buffers and defers filling them to the in-graph hook. Eager forwards never
    run that hook, so decode tokens got no 2D RoPE and no page mask and the
    model looped on block coordinates after the first decode token.
    """
    prompts = [_prompt(2, 60, True), _prompt(3, 40, True), _prompt(1, 121, True)]
    page, coord, bt, seq = _prefilled_state(prompts, True)
    eager = _decode_backend(prompts, page, coord, bt, graph=False)
    graph = _decode_backend(
        prompts, copy.deepcopy(page), copy.deepcopy(coord), bt, graph=True
    )
    mode = SimpleNamespace(
        is_idle=lambda: False,
        is_decode_or_idle=lambda: True,
        is_target_verify=lambda: False,
    )
    rows = torch.arange(len(prompts), device="cuda")
    gen = torch.Generator().manual_seed(0)
    choices = torch.tensor([PAGE, 4, 5, 6])
    for _ in range(6):
        seq = seq + 1
        pos = seq.long() - 1
        fb = SimpleNamespace(
            forward_mode=mode,
            input_ids=choices[
                torch.randint(0, 4, (len(prompts),), generator=gen)
            ].cuda(),
            seq_lens=seq.long(),
            seq_lens_cpu=seq.long().cpu(),
            out_cache_loc=bt.long().gather(1, (pos // BS)[:, None]).squeeze(1) * BS
            + pos % BS,
            req_pool_indices=rows,
        )
        eager.init_forward_metadata(fb)
        graph.init_forward_metadata_out_graph(fb)
        graph.init_forward_metadata_in_graph(fb)
        e, g = eager.forward_metadata, graph.forward_metadata
        assert e.coords is not None and e.query_allow is not None
        w = e.max_block
        assert torch.equal(e.block_table, g.block_table[:, :w])
        assert torch.equal(e.q_page, g.q_page)
        assert torch.equal(e.coords[0], g.coords[0])
        assert torch.equal(e.coords[1], g.coords[1])
        assert torch.equal(e.query_allow.bool(), g.query_allow[:, :w].bool())
        assert torch.equal(e.image_allow.bool(), g.image_allow[:, :w].bool())
        for which in ("main", "index"):
            ec, es = eager.cos_sin(which, torch.float32)
            gc, gs = graph.cos_sin(which, torch.float32)
            # Graph buffers are filled by the fused kernel, eager by PyTorch.
            torch.testing.assert_close(ec, gc, atol=1e-5, rtol=0)
            torch.testing.assert_close(es, gs, atol=1e-5, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
