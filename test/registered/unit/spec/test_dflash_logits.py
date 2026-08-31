import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.models.dflash import (
    CandidateSelector,
    DFlash2DraftModel,
    _grouped_conv,
)
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=37, suite="base-a-test-cpu")


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
    monkeypatch.setattr(
        worker_mod,
        "_SelectorDraftSampler",
        lambda **kwargs: built.setdefault("sampler", object()),
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
        _target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(model=SimpleNamespace(lm_head=quant_head))
        ),
    )

    sampler = worker_mod.DFlashWorkerV2._maybe_build_draft_sampler(worker)
    assert sampler is built["sampler"]
    assert worker.draft_model.lm_head is quant_head

    # A packed head without an applicable quant method must stay eager.
    worker._target_worker.model_runner.model.lm_head = SimpleNamespace(
        weight=torch.empty(8, 2, dtype=torch.int8)
    )
    worker.draft_model.lm_head = None
    assert worker_mod.DFlashWorkerV2._maybe_build_draft_sampler(worker) is None
    assert worker.draft_model.lm_head is None


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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
