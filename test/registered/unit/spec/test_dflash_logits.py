import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.models.dflash import (
    CandidateSelector,
    DFlash2DraftModel,
    _grouped_conv,
)
from sglang.srt.speculative.dflash_utils import (
    parse_dflash_draft_config,
    select_dflash_pred_hidden,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=40, suite="base-a-test-cpu")


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
    """The candidate matmuls read the lm_head weight directly, so a packed or
    absent weight would be read as if it were dense."""
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


def _layout_config(dflash_config):
    return {"num_hidden_layers": 5, "dflash_config": dflash_config}


def test_every_draft_query_prediction_is_used():
    """No query's prediction may be discarded: the draft block is sized so that the
    rows from pred_start on are exactly the drafts the verify block has room for.

    An anchor-first checkpoint predicts the first draft from query 0, so reading from
    position 1 (the 1+N layout) shifts every draft position by one.
    """
    bs, verify_block_size, hidden_size = 2, 4, 3

    for pred_start, num_queries, expected in ((1, 4, [1, 2, 3]), (0, 3, [0, 1, 2])):
        # Each row carries its own block position, so a shifted read is visible.
        block_positions = torch.arange(bs * num_queries) % num_queries
        hidden_states = (
            block_positions.float().unsqueeze(1).expand(-1, hidden_size).contiguous()
        )
        selected = select_dflash_pred_hidden(
            hidden_states,
            bs=bs,
            num_draft_queries=num_queries,
            pred_start=pred_start,
        )
        # Nothing dropped: every query from pred_start on lands in the verify block.
        assert selected.shape[1] == num_queries - pred_start
        assert tuple(selected.shape) == (bs, verify_block_size - 1, hidden_size)
        torch.testing.assert_close(
            selected[:, :, 0],
            torch.tensor([expected] * bs, dtype=torch.float32),
        )


def test_anchor_first_draft_block_is_one_query_narrower():
    """Both layouts fill the same verify block, so the anchor-first drafter runs one
    query fewer rather than computing a prediction it would have to discard."""
    for dflash_config, expected_queries in (
        ({}, 8),
        ({"query_zero_predicts_next": True}, 7),
    ):
        config = parse_dflash_draft_config(
            draft_hf_config=_layout_config(dflash_config)
        )
        assert config.resolve_num_draft_queries(verify_block_size=8) == expected_queries


def test_draft_block_layout_defaults_to_one_plus_n():
    config = parse_dflash_draft_config(draft_hf_config=_layout_config({}))
    assert config.anchor_first is False
    assert config.draft_pred_start == 1


@pytest.mark.parametrize("field", ("query_zero_predicts_next", "sample_from_anchor"))
@pytest.mark.parametrize("nested", (True, False))
def test_both_published_spellings_declare_the_anchor_first_layout(field, nested):
    """Checkpoints declare this layout under two names; neither may be dropped."""
    raw = (
        _layout_config({field: True})
        if nested
        else {"num_hidden_layers": 5, field: True, "dflash_config": {}}
    )
    config = parse_dflash_draft_config(draft_hf_config=raw)
    assert config.anchor_first is True
    assert config.draft_pred_start == 0


def test_contradictory_block_layout_declarations_are_rejected():
    with pytest.raises(ValueError, match="declared inconsistently"):
        parse_dflash_draft_config(
            draft_hf_config=_layout_config(
                {"query_zero_predicts_next": True, "sample_from_anchor": False}
            )
        )


def test_non_bool_block_layout_declaration_is_rejected():
    with pytest.raises(ValueError, match="must be a bool"):
        parse_dflash_draft_config(
            draft_hf_config=_layout_config({"sample_from_anchor": "true"})
        )


def _domino_layout_config(**overrides):
    dflash_config = {
        "projector_type": "domino",
        "shift_label": True,
        "pure_draft_prefix_len": 1,
        "gru_hidden_dim": 4,
        "emb_dim": 5,
    }
    dflash_config.update(overrides)
    return _layout_config(dflash_config)


def test_domino_shift_label_drives_the_block_layout():
    """Domino spells the same layout choice as shift_label; the two must not diverge."""
    for shift_label, expected_start in ((True, 0), (False, 1)):
        config = parse_dflash_draft_config(
            draft_hf_config=_domino_layout_config(shift_label=shift_label)
        )
        assert config.anchor_first is shift_label
        assert config.draft_pred_start == expected_start


def test_domino_rejects_a_layout_that_contradicts_shift_label():
    with pytest.raises(ValueError, match="declared inconsistently"):
        parse_dflash_draft_config(
            draft_hf_config=_domino_layout_config(
                shift_label=True, query_zero_predicts_next=False
            )
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
