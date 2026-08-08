import logging
from types import SimpleNamespace

import torch

from sglang.srt.speculative.dspark_components.dspark_config import (
    parse_dspark_draft_config,
    resolve_runtime_config,
)
from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
    DsparkDraftSampler,
    _resolve_corrected_logits_dtype,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _config(**overrides):
    values = dict(
        block_size=8,
        mask_token_id=990,
        markov_rank=512,
        markov_head_type="vanilla",
        target_layer_ids=[1, 5, 19, 29, 41, 51],
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_bonus_anchor_block_size_resolves_to_seven_proposals_and_eight_queries():
    hf_config = _config(sample_from_anchor=False, dspark_bonus_anchor=True)

    parsed = parse_dspark_draft_config(draft_hf_config=hf_config)
    runtime = resolve_runtime_config(
        draft_hf_config=hf_config,
        speculative_num_draft_tokens=8,
        target_vocab_size=131072,
    )

    assert parsed.gamma == 7
    assert not parsed.sample_from_anchor
    assert runtime.gamma == 7
    assert runtime.verify_num_draft_tokens == 8
    assert runtime.query_token_num == 8
    assert not runtime.sample_from_anchor


def test_missing_layout_fields_preserve_sample_from_anchor_behavior():
    parsed = parse_dspark_draft_config(draft_hf_config=_config())

    assert parsed.gamma == 8
    assert parsed.sample_from_anchor


def test_gamma_override_warning_names_resolved_config_gamma(caplog):
    hf_config = _config(sample_from_anchor=False, dspark_bonus_anchor=True)

    with caplog.at_level(logging.WARNING):
        runtime = resolve_runtime_config(
            draft_hf_config=hf_config,
            speculative_num_draft_tokens=7,
            target_vocab_size=131072,
        )

    assert runtime.gamma == 6
    assert "resolved draft config gamma=7" in caplog.text
    assert "draft config block_size=7" not in caplog.text


def test_draft_graph_width_tracks_dspark_query_layout():
    from sglang.srt.model_executor.model_runner import ModelRunner

    def width(config):
        runner = SimpleNamespace(
            spec_algorithm=SpeculativeAlgorithm.DSPARK,
            is_draft_worker=True,
            server_args=SimpleNamespace(speculative_num_draft_tokens=8),
            model_config=SimpleNamespace(hf_config=config),
        )
        return ModelRunner.decode_num_tokens_per_req(runner, num_draft_tokens=8)

    assert width(_config(sample_from_anchor=False, dspark_bonus_anchor=True)) == 8
    assert width(_config(sample_from_anchor=True, dspark_bonus_anchor=False)) == 7


def test_folded_sampler_skips_bonus_anchor_hidden_state():
    class FakeModel:
        lm_head = SimpleNamespace(org_vocab_size=4, weight=torch.empty(1))
        confidence_head = None

        def __init__(self):
            self.seen_hidden = None
            self.markov_head = self

        def compute_base_logits(self, hidden):
            self.seen_hidden = hidden.clone()
            return torch.zeros(hidden.shape[0], 4), None

        def sample_block(
            self, base_logits, *, first_prev_tokens, hidden_states, sampler
        ):
            del first_prev_tokens, hidden_states
            tokens = torch.stack(
                [sampler(base_logits[:, i], i) for i in range(base_logits.shape[1])],
                dim=1,
            )
            return tokens, base_logits

    model = FakeModel()
    sampler = DsparkDraftSampler(
        model=model,
        gamma=2,
        sample_from_anchor=False,
        max_bs=1,
        device="cpu",
        folded_sampling=False,
    )
    hidden = torch.tensor([[10.0], [20.0], [30.0]])
    input_ids = torch.tensor([7, 990, 990])

    sampler(hidden, input_ids)

    torch.testing.assert_close(model.seen_hidden, torch.tensor([[20.0], [30.0]]))


def test_folded_sampler_uses_logit_dtype_for_quantized_lm_head():
    model = SimpleNamespace(
        config=SimpleNamespace(torch_dtype=torch.bfloat16),
        lm_head=SimpleNamespace(
            org_vocab_size=4,
            weight=torch.empty((4, 1), dtype=torch.uint8),
        ),
        markov_head=SimpleNamespace(),
        confidence_head=None,
    )

    assert _resolve_corrected_logits_dtype(model) == torch.bfloat16
    sampler = DsparkDraftSampler(
        model=model,
        gamma=2,
        sample_from_anchor=False,
        max_bs=1,
        device="cpu",
        folded_sampling=True,
    )

    assert sampler.corrected_out.dtype == torch.bfloat16
