import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.models.dflash import (
    CandidateSelector,
    DFlash2DraftModel,
    _validate_max_num_nodes,
    _beam_walk_torch,
    _grouped_conv,
)
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=38, suite="base-a-test-cpu")
# The triton beam walk needs a device; the rest of this file needs no kernel.
register_cuda_ci(est_time=4, stage="base-b", runner_config="1-gpu-small")


def test_dflash_unary_logit_transform():
    logits = torch.tensor([[-100.0, 0.0, 100.0]], dtype=torch.bfloat16)
    for fields in ({}, {"output_multiplier": 0.2, "final_logit_softcapping": 20.0}):
        config = parse_dflash_draft_config(
            draft_hf_config={
                "num_hidden_layers": 5,
                "dflash_config": {
                    "selector_rank": 256,
                    "selector_top_k": 16,
                    **fields,
                },
            }
        )
        actual = DFlash2DraftModel._transform_unary_logits(
            SimpleNamespace(draft_config=config), logits
        )
        expected = logits.float() * config.output_multiplier
        if config.final_logit_softcapping is not None:
            expected = torch.tanh(expected / config.final_logit_softcapping)
            expected *= config.final_logit_softcapping
        torch.testing.assert_close(actual, expected)


def test_selector_greedy_row_walk_is_deterministic_in_a_mixed_batch():
    """A greedy row walks the argmax, so the q it hands verify has to be the point
    mass there. Greedy reaches the selector as top_k=1 with the temperature reset
    to 1.0, so a softmax q stays a real distribution and verify would
    rejection-sample a deterministic request against it. The row must also not
    depend on who else is in the batch."""
    selector = CandidateSelector(hidden_size=4, vocab_size=16, state_rank=2, top_k=4)
    torch.manual_seed(1)
    candidate_ids = torch.randint(0, 16, (2, 3, 4))
    scores = torch.randn(2, 3, 4, 4)
    uniforms = torch.tensor([[0.2, 0.7, 0.4], [0.8, 0.1, 0.6]])
    temperatures = torch.tensor([1.0, 0.7])
    greedy_mask = torch.tensor([True, False])

    mixed_tokens, mixed_q = selector.sample_path(
        candidate_ids=candidate_ids,
        scores=scores,
        uniforms=uniforms,
        temperatures=temperatures,
        greedy_mask=greedy_mask,
    )
    assert torch.all((mixed_q[0] == 0) | (mixed_q[0] == 1))
    for row in range(2):
        tokens, q_rows = selector.sample_path(
            candidate_ids=candidate_ids[row : row + 1],
            scores=scores[row : row + 1],
            uniforms=uniforms[row : row + 1],
            temperatures=temperatures[row : row + 1],
            greedy_mask=greedy_mask[row : row + 1],
        )
        torch.testing.assert_close(mixed_tokens[row], tokens[0])
        torch.testing.assert_close(mixed_q[row], q_rows[0])


def test_selector_rejects_a_quantized_target_lm_head():
    """Candidate scoring requires a dense target lm_head."""
    model = SimpleNamespace(
        lm_head=SimpleNamespace(weight=torch.empty(8, 4, dtype=torch.int8)),
        candidate_selector=SimpleNamespace(top_k=4),
    )
    with pytest.raises(RuntimeError, match="requires a dense"):
        DFlash2DraftModel.compute_candidates(model, torch.randn(2, 4))


def _flashinfer_contract_topk(scores, k, sorted=False, deterministic=False):
    """Stand-in for flashinfer.top_k pinning its call contract: contiguous
    input (its CHECK_INPUT) and the explicit sorted/deterministic flags
    _radix_topk relies on (the real kernel defaults both to False)."""
    assert scores.is_contiguous()
    assert sorted and deterministic
    return torch.topk(scores, k, dim=-1)


class _FakeQuantMethod:
    """Projects through a captured dense weight, asserting the packed-head
    call contract (packed dtype, no bias). The padded tail comes out as
    dominant garbage so a masking regression surfaces as wrong candidates."""

    def __init__(self, dense_weight, num_padded):
        self.dense_weight = dense_weight
        self.num_padded = num_padded
        self.called = False

    def apply(self, layer, x, bias):
        self.called = True
        assert layer.weight.dtype == torch.int8
        assert bias is None
        logits = torch.matmul(x, self.dense_weight.T)
        pad = logits.new_full((logits.shape[0], self.num_padded), 100.0)
        full = torch.cat([logits, pad], dim=-1)
        # A strided view, like a kernel writing into a wider workspace: the
        # projection must materialize it before flashinfer's radix top-k.
        return torch.stack([full, full], dim=-1)[..., 0]


def test_selector_projects_a_quantized_target_lm_head_through_its_quant_method(
    monkeypatch,
):
    """Packed head weights must be projected through their quantization method,
    with the padded-vocab tail masked out of the top-k on contiguous logits:
    flashinfer's radix top-k rejects non-contiguous input, so a plain crop view
    would fail at capture on any padded vocab."""
    torch.manual_seed(0)
    hidden = torch.randn(2, 4)
    dense_weight = torch.randn(6, 4)

    quant_method = _FakeQuantMethod(dense_weight, num_padded=2)
    lm_head = SimpleNamespace(
        # Mimic a 2:1 packed head and two padded vocabulary rows.
        weight=torch.empty(8, 2, dtype=torch.int8),
        quant_method=quant_method,
        org_vocab_size=6,
    )
    model = SimpleNamespace(
        lm_head=lm_head,
        candidate_selector=SimpleNamespace(top_k=4),
        _transform_unary_logits=lambda logits: logits.float(),
    )
    monkeypatch.setattr(
        "sglang.srt.models.dflash.get_parallel",
        lambda: SimpleNamespace(tp_size=1),
    )
    monkeypatch.setattr(
        "sglang.srt.models.dflash._flashinfer_top_k", _flashinfer_contract_topk
    )

    candidate_ids, unary_logits = DFlash2DraftModel.compute_candidates(model, hidden)

    expected_logits, expected_ids = torch.topk(
        torch.matmul(hidden, dense_weight.T), 4, dim=-1
    )
    assert quant_method.called
    torch.testing.assert_close(candidate_ids, expected_ids)
    torch.testing.assert_close(unary_logits, expected_logits)


def test_selector_gathers_global_candidates_across_vocab_shards(monkeypatch):
    """Pins the TP gather contract on the quantized path: the per-shard
    org-vocab restriction, the global id offset, and the fp32 cast before the
    all-gather -- a
    regression in any of them returns wrong global candidates only under TP,
    which no single-rank test observes."""
    torch.manual_seed(0)
    k = 4
    # bf16 like production: makes the fp32 upcast before the gather observable.
    hidden = torch.randn(2, 4, dtype=torch.bfloat16)
    full_weight = torch.randn(12, 4, dtype=torch.bfloat16)  # org vocab 12, 6+6

    # This process plays rank 1 of tp=2: org rows 6..12 as local rows 0..6,
    # plus two dominant padded columns that must never reach the candidates.
    quant_method = _FakeQuantMethod(full_weight[6:], num_padded=2)
    lm_head = SimpleNamespace(
        weight=torch.empty(8, 2, dtype=torch.int8),
        quant_method=quant_method,
        shard_indices=SimpleNamespace(num_org_elements=6, org_vocab_start_index=6),
    )
    model = SimpleNamespace(
        lm_head=lm_head,
        candidate_selector=SimpleNamespace(top_k=k),
        _transform_unary_logits=lambda logits: logits.float(),
    )

    # Rank 0's gathered contribution, synthesized from the reference weights.
    rank0_vals, rank0_ids = torch.topk(
        torch.matmul(hidden, full_weight[:6].T), k, dim=-1
    )

    def fake_all_gather(x, dim):
        if x.is_floating_point():
            assert x.dtype == torch.float32
            return torch.cat([rank0_vals.float(), x], dim=dim)
        return torch.cat([rank0_ids.long(), x], dim=dim)

    monkeypatch.setattr(
        "sglang.srt.models.dflash.get_parallel",
        lambda: SimpleNamespace(tp_size=2),
    )
    monkeypatch.setattr(
        "sglang.srt.models.dflash.tensor_model_parallel_all_gather", fake_all_gather
    )
    monkeypatch.setattr(
        "sglang.srt.models.dflash._flashinfer_top_k", _flashinfer_contract_topk
    )

    candidate_ids, unary_logits = DFlash2DraftModel.compute_candidates(model, hidden)

    expected_logits, expected_ids = torch.topk(
        torch.matmul(hidden, full_weight.T), k, dim=-1
    )
    torch.testing.assert_close(candidate_ids, expected_ids)
    torch.testing.assert_close(unary_logits, expected_logits.float())


def test_worker_folds_a_gate_admitted_quantized_selector_head(monkeypatch):
    """The pre-capture screen decides whether a quantized head reaches the
    graph-folded selector sampler or silently degrades to the eager per-round
    fallback -- a revert there keeps every compute_candidates test green, so
    the admission (and the rejection of an unsupported packed head) needs its
    own guard."""
    from sglang.srt.speculative import dflash_worker_v2 as worker_mod

    built = {}
    built_sampler = object()

    def build_sampler(**kwargs):
        built.update(kwargs)
        return built_sampler

    monkeypatch.setattr(
        worker_mod,
        "_SelectorDraftSampler",
        build_sampler,
    )
    monkeypatch.setattr(
        worker_mod,
        "get_exec",
        lambda: SimpleNamespace(
            graph=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(bs=[1]))
            )
        ),
    )
    quant_head = SimpleNamespace(
        weight=torch.empty(8, 2, dtype=torch.int8),
        quant_method=_FakeQuantMethod(torch.randn(6, 4), num_padded=2),
    )
    worker = SimpleNamespace(
        block_size=8,
        selector=object(),
        ps=SimpleNamespace(tp_rank=0),
        draft_model=SimpleNamespace(lm_head=None),
        device="cpu",
        _use_tree_verify=False,
        _selector_sampling_enabled=True,
        _target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(model=SimpleNamespace(lm_head=quant_head))
        ),
    )

    sampler = worker_mod.DFlashWorkerV2._maybe_build_draft_sampler(worker)
    assert sampler is built_sampler
    assert built["sampling_enabled"] is True
    assert worker.draft_model.lm_head is quant_head

    # A packed head without an applicable quant method must stay eager.
    worker._target_worker.model_runner.model.lm_head = SimpleNamespace(
        weight=torch.empty(8, 2, dtype=torch.int8)
    )
    worker.draft_model.lm_head = None
    assert worker_mod.DFlashWorkerV2._maybe_build_draft_sampler(worker) is None
    assert worker.draft_model.lm_head is None


def test_worker_warns_once_when_selector_sampling_is_disabled(monkeypatch):
    from sglang.srt.speculative import dflash_worker_v2 as worker_mod

    warnings = []
    monkeypatch.setattr(
        worker_mod.logger, "warning", lambda *args: warnings.append(args)
    )
    worker = SimpleNamespace(
        selector=object(),
        _use_tree_verify=False,
        _selector_sampling_enabled=False,
        _warned_sampling_fallback=False,
        ps=SimpleNamespace(tp_rank=0),
    )
    batch = SimpleNamespace(sampling_info=SimpleNamespace(is_all_greedy=False))

    worker_mod.DFlashWorkerV2._validate_phase1_sampling_support(worker, batch)
    worker_mod.DFlashWorkerV2._validate_phase1_sampling_support(worker, batch)

    assert worker._warned_sampling_fallback
    assert len(warnings) == 1
    assert "sampling distribution will not be preserved" in warnings[0][0]

    worker._selector_sampling_enabled = True
    worker._warned_sampling_fallback = False
    worker_mod.DFlashWorkerV2._validate_phase1_sampling_support(worker, batch)
    assert len(warnings) == 1


def test_disabled_selector_sampling_forces_greedy_draft():
    from sglang.srt.speculative import dflash_worker_v2 as worker_mod

    sampling_info = SimpleNamespace(
        temperatures=torch.tensor([[0.7]]),
        top_ks=torch.tensor([[8]]),
        is_all_greedy=False,
    )

    sampler = worker_mod._SelectorDraftSampler.__new__(worker_mod._SelectorDraftSampler)
    sampler.temperatures = torch.zeros(1)
    sampler.greedy_mask = torch.zeros(1, dtype=torch.bool)
    sampler.sampling_enabled = False
    sampler.stage_sampling_params(bs=1, sampling_info=sampling_info)
    torch.testing.assert_close(sampler.temperatures, torch.ones(1))
    assert sampler.greedy_mask.tolist() == [True]

    sampler.sampling_enabled = True
    sampler.stage_sampling_params(bs=1, sampling_info=sampling_info)
    torch.testing.assert_close(sampler.temperatures, torch.tensor([0.7]))
    assert sampler.greedy_mask.tolist() == [False]
    observed = {}

    def sample_path(**kwargs):
        observed.update(kwargs)
        return torch.zeros((1, 1), dtype=torch.int64), torch.zeros((1, 1, 2))

    selector = SimpleNamespace(
        build_lattice=lambda **kwargs: torch.zeros((1, 1, 2, 2)),
        sample_path=sample_path,
    )
    draft_model = SimpleNamespace(
        lm_head=None,
        candidate_selector=selector,
        compute_candidates=lambda hidden: (
            torch.zeros((1, 2), dtype=torch.int64),
            torch.zeros((1, 2)),
        ),
    )
    worker = SimpleNamespace(
        draft_model=draft_model,
        selector=selector,
        block_size=2,
        _selector_sampling_enabled=False,
        _selector_sample=None,
    )
    draft_logits_output = SimpleNamespace(hidden_states=torch.zeros((2, 4)))

    worker_mod.DFlashWorkerV2._propose_selector_block(
        worker,
        draft_logits_output=draft_logits_output,
        bs=1,
        lm_head=object(),
        anchor_token_ids=torch.zeros(1, dtype=torch.int64),
        sampling_info=sampling_info,
    )

    torch.testing.assert_close(observed["temperatures"], torch.ones(1))
    assert observed["greedy_mask"].tolist() == [True]
    assert worker._selector_sample is None


def test_selector_accept_uses_greedy_fallback_without_staged_sample(monkeypatch):
    from sglang.srt.speculative import dflash_worker_v2 as worker_mod

    monkeypatch.setattr(
        worker_mod, "is_dflash_sampling_verify_available", lambda: False
    )
    monkeypatch.setattr(
        worker_mod,
        "compute_dflash_correct_drafts_and_bonus",
        lambda **kwargs: (torch.tensor([0]), torch.tensor([7])),
    )

    sync_sites = []
    worker = SimpleNamespace(
        _selector_sample=None,
        _selector_sampling_accept=lambda **kwargs: pytest.fail(
            "selector sampling must not run without a staged sample"
        ),
        _tp_sync=SimpleNamespace(sync=lambda site, tensor: sync_sites.append(site)),
        _use_triton_accept_bonus=False,
        block_size=2,
    )

    result = worker_mod.DFlashWorkerV2._accept_block(
        worker,
        candidates=torch.tensor([[9, 1]]),
        next_token_logits=torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]),
        sampling_info=SimpleNamespace(is_all_greedy=False),
        draft_input=object(),
        prefix_lens=torch.tensor([3]),
        bs=1,
    )

    accept_len, commit_lens, bonus, out_tokens, _, target_predict = result
    assert accept_len.tolist() == [0]
    assert commit_lens.tolist() == [1]
    assert bonus.tolist() == [7]
    assert out_tokens.tolist() == [[7, 0]]
    assert target_predict.tolist() == [[1, 0]]
    assert sync_sites == [worker_mod.SpecTpSyncSite.DFLASH_ACCEPT_GREEDY]


def _lattice(*, bs, slots, top_k, seed=0, spread=8.0):
    """Build a lattice with score gaps large enough for exact comparisons."""
    generator = torch.Generator().manual_seed(seed)
    scores = torch.randint(
        -5, 6, (bs, slots, top_k, top_k), generator=generator
    ).float()
    candidate_ids = (
        torch.arange(slots * top_k).view(1, slots, top_k).expand(bs, slots, top_k)
    )
    anchor = torch.arange(bs) + 9001
    return candidate_ids.contiguous(), scores * spread, anchor


@pytest.mark.parametrize("slots,top_k", [(3, 4), (7, 16)])
def test_beam_width_one_reproduces_the_greedy_chain(slots, top_k):
    """Width 1 must reproduce the existing greedy chain."""
    bs = 3
    candidate_ids, scores, anchor = _lattice(bs=bs, slots=slots, top_k=top_k)
    selector = CandidateSelector(
        hidden_size=4, vocab_size=64, state_rank=2, top_k=top_k
    )

    chain, _ = selector.sample_path(
        candidate_ids=candidate_ids,
        scores=scores,
        uniforms=torch.zeros(bs, slots),
        temperatures=torch.ones(bs),
        greedy_mask=torch.ones(bs, dtype=torch.bool),
    )
    tokens, parents = _beam_walk_torch(
        candidate_ids=candidate_ids,
        scores=scores,
        anchor_token_ids=anchor,
        beam_width=1,
    )

    assert torch.equal(tokens[:, 0], anchor)
    assert torch.equal(tokens[:, 1:], chain)
    expected_parents = torch.cat([torch.tensor([-1]), torch.arange(slots)])
    assert torch.equal(parents, expected_parents.expand(bs, slots + 1))


@pytest.mark.parametrize("beam_width", [1, 2, 3, 4])
@pytest.mark.parametrize("slots,top_k", [(3, 4), (7, 16)])
def test_tree_is_fixed_width_and_bfs_ordered(beam_width, slots, top_k):
    """Tree nodes are fixed-width and BFS ordered."""
    bs = 2
    candidate_ids, scores, anchor = _lattice(bs=bs, slots=slots, top_k=top_k)
    tokens, parents = _beam_walk_torch(
        candidate_ids=candidate_ids,
        scores=scores,
        anchor_token_ids=anchor,
        beam_width=beam_width,
    )

    num_nodes = 1 + slots * beam_width
    assert tokens.shape == (bs, num_nodes)
    assert parents.shape == (bs, num_nodes)
    assert torch.equal(parents[:, 0], torch.full((bs,), -1))
    assert (parents[:, 1:] < torch.arange(1, num_nodes)).all()
    for slot in range(slots):
        base = 1 + slot * beam_width
        block = parents[:, base : base + beam_width]
        lower = 1 + (slot - 1) * beam_width if slot else 0
        assert (block >= lower).all()
        assert (block < base).all()


def test_spine_is_independent_of_the_beam_width():
    """The greedy spine must be independent of beam width."""
    slots, top_k = 7, 16
    candidate_ids, scores, anchor = _lattice(bs=2, slots=slots, top_k=top_k, seed=3)

    spines = []
    for beam_width in (1, 2, 4, 8):
        tokens, _ = _beam_walk_torch(
            candidate_ids=candidate_ids,
            scores=scores,
            anchor_token_ids=anchor,
            beam_width=beam_width,
        )
        spines.append(tokens[:, [1 + slot * beam_width for slot in range(slots)]])

    for spine in spines[1:]:
        assert torch.equal(spine, spines[0])


def test_same_parent_siblings_carry_distinct_tokens():
    """Siblings selected under one parent must carry distinct tokens."""
    slots, top_k, beam_width = 7, 16, 4
    candidate_ids, scores, anchor = _lattice(bs=3, slots=slots, top_k=top_k, seed=5)
    tokens, parents = _beam_walk_torch(
        candidate_ids=candidate_ids,
        scores=scores,
        anchor_token_ids=anchor,
        beam_width=beam_width,
    )

    for row in range(tokens.shape[0]):
        for slot in range(slots):
            base = 1 + slot * beam_width
            groups = {}
            for beam in range(beam_width):
                parent = int(parents[row, base + beam])
                groups.setdefault(parent, []).append(int(tokens[row, base + beam]))
            for siblings in groups.values():
                assert len(siblings) == len(set(siblings))


def test_rows_are_normalized_before_the_scores_accumulate():
    """Row offsets must not affect beam competition."""
    beam_width = top_k = 3
    scores = torch.zeros(1, 2, top_k, top_k)
    scores[0, 1, 0] = torch.tensor([0.0, -1.0, -2.0])
    scores[0, 1, 1] = torch.tensor([0.0, -1.0, -2.0])
    scores[0, 1, 2] = torch.tensor([10.0, 9.0, 8.0])
    candidate_ids = torch.arange(2 * top_k).view(1, 2, top_k)

    _, parents = _beam_walk_torch(
        candidate_ids=candidate_ids,
        scores=scores,
        anchor_token_ids=torch.tensor([7]),
        beam_width=beam_width,
    )

    # Depth 1 is nodes 1..3; every one of them must have kept a child.
    assert sorted(int(p) for p in parents[0, 1 + beam_width :]) == [1, 2, 3]


def test_exact_ties_break_beam_major():
    """Exact ties must use beam-major flattening."""
    beam_width = top_k = 2
    scores = torch.zeros(1, 2, top_k, top_k)
    candidate_ids = torch.arange(2 * top_k).view(1, 2, top_k)

    _, parents = _beam_walk_torch(
        candidate_ids=candidate_ids,
        scores=scores,
        anchor_token_ids=torch.tensor([7]),
        beam_width=beam_width,
    )

    # Depth 2 is nodes 3 and 4. Node 3 is the spine, under node 1. Every remaining
    # pool entry ties, so beam-major hands node 4 to beam 0 -- node 1, not node 2.
    assert parents[0, 3] == 1
    assert parents[0, 4] == 1


def test_beam_walk_rejects_a_width_above_the_candidate_count():
    """Reject a beam wider than the candidate axis."""
    selector = CandidateSelector(hidden_size=4, vocab_size=64, state_rank=2, top_k=4)
    candidate_ids, scores, anchor = _lattice(bs=1, slots=3, top_k=4)
    with pytest.raises(ValueError, match="selector_top_k"):
        selector.beam_walk(
            candidate_ids=candidate_ids,
            scores=scores,
            anchor_token_ids=anchor,
            beam_width=8,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="triton needs a GPU")
@pytest.mark.parametrize("beam_width", [1, 2, 3, 4, 8])
def test_triton_beam_walk_matches_the_reference(beam_width):
    """The Triton walk must match the device-agnostic reference."""
    slots, top_k = 7, 16
    candidate_ids, scores, anchor = _lattice(bs=4, slots=slots, top_k=top_k, seed=11)
    selector = CandidateSelector(
        hidden_size=4, vocab_size=slots * top_k, state_rank=2, top_k=top_k
    )

    expected = _beam_walk_torch(
        candidate_ids=candidate_ids,
        scores=scores,
        anchor_token_ids=anchor,
        beam_width=beam_width,
    )
    actual = selector.beam_walk(
        candidate_ids=candidate_ids.cuda(),
        scores=scores.cuda(),
        anchor_token_ids=anchor.cuda(),
        beam_width=beam_width,
    )

    assert torch.equal(actual[0].cpu(), expected[0])
    assert torch.equal(actual[1].cpu(), expected[1])


def _root_path(*, tokens, parents, node):
    """Token sequence from the root down to `node`, root first."""
    path = []
    while node != -1:
        path.append(int(tokens[node]))
        node = int(parents[node])
    return path[::-1]


# scores drawn from {0, -SPLIT}: fp32 log_softmax returns exactly 0.0 and -SPLIT
# there (the other candidate's mass underflows fp32 eps), so every `cum` is an exact
# multiple of -SPLIT and both the ranking and its ties are hand-derivable.
_SPLIT = 40.0


def _two_slot_lattice():
    """slots=2, top_k=2; at beam_width=2 the walk builds 5 nodes with known `cum`.

    Hand-derived: node 1 and node 3 have cum 0, node 2 and node 4 have cum -SPLIT, so
    ranking by (-cum, index) over the non-anchor nodes gives 1, 3, 2, 4.
    """
    scores = torch.tensor(
        [[[0.0, -_SPLIT], [0.0, -_SPLIT]], [[0.0, -_SPLIT], [0.0, -_SPLIT]]]
    )[None]
    return torch.arange(4).view(1, 2, 2), scores, torch.tensor([9001])


@pytest.mark.parametrize(
    "max_num_nodes,tokens,parents",
    [
        (2, [9001, 0], [-1, 0]),
        (3, [9001, 0, 2], [-1, 0, 1]),
        (4, [9001, 0, 1, 2], [-1, 0, 0, 1]),
    ],
)
def test_prune_keeps_the_best_cum_nodes(max_num_nodes, tokens, parents):
    """Keep the expected cumulative-score nodes and remap their parents."""
    candidate_ids, scores, anchor = _two_slot_lattice()
    actual = _beam_walk_torch(
        candidate_ids=candidate_ids,
        scores=scores,
        anchor_token_ids=anchor,
        beam_width=2,
        max_num_nodes=max_num_nodes,
    )
    assert actual[0][0].tolist() == tokens
    assert actual[1][0].tolist() == parents


_CAPS = [1, 2, 8, 15, 30]


def _caps_for(built):
    """The sweep at one built size: fixed points plus both sides of the boundary."""
    return sorted({cap for cap in _CAPS if cap <= built} | {built - 1, built})


@pytest.mark.parametrize("beam_width", [1, 2, 4, 8])
def test_prune_emits_a_bfs_ordered_tree_at_every_cap(beam_width):
    """Pruned trees must remain BFS ordered."""
    slots, top_k = 7, 16
    built = 1 + slots * beam_width
    candidate_ids, scores, anchor = _lattice(bs=3, slots=slots, top_k=top_k, seed=5)
    for max_num_nodes in _caps_for(built):
        tokens, parents = _beam_walk_torch(
            candidate_ids=candidate_ids,
            scores=scores,
            anchor_token_ids=anchor,
            beam_width=beam_width,
            max_num_nodes=max_num_nodes,
        )
        assert tokens.shape == (3, max_num_nodes)
        assert parents.shape == (3, max_num_nodes)
        assert (parents[:, 0] == -1).all()
        ceiling = torch.arange(1, max_num_nodes)
        assert (parents[:, 1:] < ceiling).all()
        assert (parents[:, 1:] >= 0).all()


@pytest.mark.parametrize("beam_width", [2, 4, 8])
def test_prune_preserves_every_surviving_root_path(beam_width):
    """Pruning may remove paths but must not rewrite survivors."""
    slots, top_k = 7, 16
    built = 1 + slots * beam_width
    candidate_ids, scores, anchor = _lattice(bs=2, slots=slots, top_k=top_k, seed=7)
    full_tokens, full_parents = _beam_walk_torch(
        candidate_ids=candidate_ids,
        scores=scores,
        anchor_token_ids=anchor,
        beam_width=beam_width,
    )
    paths = [
        {
            tuple(
                _root_path(
                    tokens=full_tokens[row], parents=full_parents[row], node=node
                )
            )
            for node in range(built)
        }
        for row in range(full_tokens.shape[0])
    ]
    for max_num_nodes in _caps_for(built):
        tokens, parents = _beam_walk_torch(
            candidate_ids=candidate_ids,
            scores=scores,
            anchor_token_ids=anchor,
            beam_width=beam_width,
            max_num_nodes=max_num_nodes,
        )
        for row in range(tokens.shape[0]):
            for node in range(max_num_nodes):
                path = tuple(
                    _root_path(tokens=tokens[row], parents=parents[row], node=node)
                )
                assert path in paths[row]


def test_prune_breaks_cum_ties_towards_the_lower_node_index():
    """Equal cumulative scores must break ties by lower node index."""
    slots, top_k, bs = 7, 16, 2
    candidate_ids, _, anchor = _lattice(bs=bs, slots=slots, top_k=top_k)
    uniform = torch.zeros(bs, slots, top_k, top_k)
    for beam_width in (2, 4, 8):
        built = 1 + slots * beam_width
        full = _beam_walk_torch(
            candidate_ids=candidate_ids,
            scores=uniform,
            anchor_token_ids=anchor,
            beam_width=beam_width,
        )
        for max_num_nodes in (5, 15, 30):
            if max_num_nodes >= built:
                continue
            actual = _beam_walk_torch(
                candidate_ids=candidate_ids,
                scores=uniform,
                anchor_token_ids=anchor,
                beam_width=beam_width,
                max_num_nodes=max_num_nodes,
            )
            assert torch.equal(actual[0], full[0][:, :max_num_nodes])
            assert torch.equal(actual[1], full[1][:, :max_num_nodes])


def test_prune_rejects_a_cap_outside_the_built_node_count():
    """Reject caps that do not describe a subset of the built tree."""
    candidate_ids, scores, anchor = _lattice(bs=1, slots=7, top_k=16)
    for bad in (0, 1 + 7 * 4 + 1):
        with pytest.raises(ValueError, match="max_num_nodes"):
            _beam_walk_torch(
                candidate_ids=candidate_ids,
                scores=scores,
                anchor_token_ids=anchor,
                beam_width=4,
                max_num_nodes=bad,
            )


def test_prune_allows_an_uncapped_tree_above_kernel_limit():
    _validate_max_num_nodes(max_num_nodes=257, num_nodes=257)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="triton needs a GPU")
@pytest.mark.parametrize("beam_width", [1, 2, 4, 8])
def test_triton_prune_matches_the_reference(beam_width):
    """The Triton prune must match the independent torch reference."""
    slots, top_k = 7, 16
    built = 1 + slots * beam_width
    candidate_ids, scores, anchor = _lattice(bs=4, slots=slots, top_k=top_k, seed=11)
    selector = CandidateSelector(
        hidden_size=4, vocab_size=slots * top_k, state_rank=2, top_k=top_k
    )
    for max_num_nodes in _caps_for(built):
        expected = _beam_walk_torch(
            candidate_ids=candidate_ids,
            scores=scores,
            anchor_token_ids=anchor,
            beam_width=beam_width,
            max_num_nodes=max_num_nodes,
        )
        actual = selector.beam_walk(
            candidate_ids=candidate_ids.cuda(),
            scores=scores.cuda(),
            anchor_token_ids=anchor.cuda(),
            beam_width=beam_width,
            max_num_nodes=max_num_nodes,
        )
        assert torch.equal(actual[0].cpu(), expected[0])
        assert torch.equal(actual[1].cpu(), expected[1])


def test_grouped_conv_supports_runtime_block_sizes():
    """The conv indexes a position inside the block, so it must follow whatever
    block size the worker resolved -- including one that is not a power of two."""
    torch.manual_seed(0)
    groups, group_size, taps = 3, 2, 2
    hidden_size = groups * group_size
    batch_size = 2

    for block_size in (5, 8, 16):
        hidden = torch.randn(batch_size * block_size, hidden_size)
        delta = torch.randn(batch_size * block_size, taps, groups)
        base = torch.randn(taps, hidden_size)

        actual = _grouped_conv(
            hidden, delta, base, block_size, groups, group_size, taps
        )

        expected = torch.empty_like(hidden)
        hidden_3d = hidden.view(batch_size, block_size, groups, group_size)
        delta_4d = delta.view(batch_size, block_size, taps, groups)
        base_3d = base.view(taps, groups, group_size)
        for batch in range(batch_size):
            for position in range(block_size):
                value = torch.zeros(groups, group_size)
                for tap in range(min(taps, position + 1)):
                    coefficient = base_3d[tap] + delta_4d[batch, position, tap, :, None]
                    value += coefficient * hidden_3d[batch, position - tap]
                expected[batch * block_size + position] = value.flatten()
        torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="triton needs a GPU")
def test_prune_kernel_bounds_its_writes_on_non_finite_cums():
    """A NaN `cum` compares false against everything, so without the normalization
    every node ranks 0, `keep` is all-true, and the compaction pushes up to `num_nodes`
    elements into a `max_num_nodes`-wide row -- an out-of-bounds write, silent memory
    corruption rather than a wrong answer. Mapping NaN to -inf instead makes
    `better[i, j]` reduce to `j < i`, so exactly `max_num_nodes` nodes survive.

    Driven at the kernel rather than through `beam_walk`, because the walk cannot
    produce a tree at all from a NaN lattice: `_first_max_index` returns its
    out-of-range sentinel there.

    The output buffers are flat with `guard` sentinel elements past the end, because
    the kernel derives each row's offset from `max_num_nodes` itself -- padding the
    rows instead would put row `r + 1`'s legitimate writes inside row `r`'s padding
    and detect nothing."""
    import triton

    from sglang.kernels.ops.speculative.dflash import _dflash_tree_prune_by_cum_kernel

    bs, num_nodes, cap, guard = 3, 57, 15, 8
    sentinel = -12345
    tokens = torch.arange(bs * num_nodes, device="cuda").view(bs, num_nodes)
    parents = torch.zeros(bs, num_nodes, dtype=torch.int64, device="cuda")
    parents[:, 0] = -1
    cums = torch.full((bs, num_nodes), float("nan"), device="cuda")
    out_tokens = torch.full(
        (bs * cap + guard,), sentinel, dtype=torch.int64, device="cuda"
    )
    out_parents = torch.full_like(out_tokens, sentinel)

    _dflash_tree_prune_by_cum_kernel[(bs,)](
        tokens,
        parents,
        cums,
        out_tokens,
        out_parents,
        num_nodes=num_nodes,
        max_num_nodes=cap,
        NODES=triton.next_power_of_2(num_nodes),
        num_warps=4,
    )

    assert (out_tokens[bs * cap :] == sentinel).all()
    assert (out_parents[bs * cap :] == sentinel).all()
    # All-equal cum degenerates to the BFS prefix, exactly `cap` nodes per row.
    assert torch.equal(out_tokens[: bs * cap].view(bs, cap), tokens[:, :cap])
    for row in range(bs):
        assert out_parents[row * cap : (row + 1) * cap].tolist() == [-1] + [0] * (
            cap - 1
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
