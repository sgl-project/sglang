import logging
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative.dspark_components.dspark_config import (
    parse_dspark_draft_config,
    resolve_runtime_config,
)
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockProposer
from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
    DsparkDraftSampler,
    _resolve_corrected_logits_dtype,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import resolve_num_tokens_per_req
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
    def width(*, sample_from_anchor, is_draft_worker=True):
        server_args = SimpleNamespace(
            speculative_num_draft_tokens=8,
            speculative_dspark_sample_from_anchor=sample_from_anchor,
        )
        return resolve_num_tokens_per_req(
            phase="target_verify",
            server_args=server_args,
            spec_algorithm=SpeculativeAlgorithm.DSPARK,
            is_draft_worker=is_draft_worker,
        )

    assert width(sample_from_anchor=False) == 8
    assert width(sample_from_anchor=True) == 7
    assert width(sample_from_anchor=False, is_draft_worker=False) == 8
    assert width(sample_from_anchor=True, is_draft_worker=False) == 8


def test_non_dspark_target_verify_width_is_unchanged():
    server_args = SimpleNamespace(speculative_num_draft_tokens=8)

    assert (
        resolve_num_tokens_per_req(
            phase="target_verify",
            server_args=server_args,
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
            is_draft_worker=True,
        )
        == 8
    )


def test_bonus_anchor_eager_forward_uses_draft_embedding_and_counts_all_queries():
    class RecordingEmbedding:
        def __init__(self):
            self.input_ids = None

        def __call__(self, input_ids):
            self.input_ids = input_ids.clone()
            return torch.zeros((*input_ids.shape, 4))

    class RecordingRunner:
        device = "cpu"
        decode_cuda_graph_runner = None

        def __init__(self):
            self.forward_batch = None

        def forward(self, forward_batch):
            self.forward_batch = forward_batch
            hidden_states = torch.zeros(forward_batch.input_ids.numel(), 4)
            return SimpleNamespace(
                logits_output=SimpleNamespace(hidden_states=hidden_states),
                can_run_graph=False,
            )

    embedding = RecordingEmbedding()
    runner = RecordingRunner()
    draft_model = SimpleNamespace(get_input_embeddings=lambda: embedding)
    proposer = DraftBlockProposer(
        draft_model=draft_model,
        draft_model_runner=runner,
        gamma=2,
        sample_from_anchor=False,
        mask_token_id=990,
        draft_block_spec_info=SimpleNamespace(),
    )
    batch = SimpleNamespace(
        seq_lens=torch.tensor([4, 5]),
        seq_lens_cpu=torch.tensor([4, 5]),
        req_pool_indices=torch.tensor([0, 1]),
        can_run_dp_cuda_graph=False,
        global_num_tokens=None,
    )
    draft_input = SimpleNamespace(
        bonus_tokens=torch.tensor([7, 8]),
        reserved_seq_lens_cpu=None,
    )
    verify_window = SimpleNamespace(
        positions_2d=torch.arange(6).view(2, 3),
        verify_cache_loc_2d=torch.arange(6).view(2, 3),
    )

    with patch(
        "sglang.srt.speculative.dspark_components.dspark_draft."
        "enable_num_token_non_padded",
        return_value=True,
    ):
        proposer._run_forward(
            batch=batch,
            draft_input=draft_input,
            verify_window=verify_window,
            bs=2,
            device="cpu",
            embed_module=proposer._embed_module,
        )

    assert embedding.input_ids.shape == (2, 3)
    assert runner.forward_batch.input_ids.numel() == 6
    assert runner.forward_batch.num_token_non_padded.item() == 6
    assert runner.forward_batch.num_token_non_padded_cpu == 6


def test_folded_sampler_skips_bonus_anchor_hidden_state():
    class FakeModel:
        config = SimpleNamespace(torch_dtype=torch.float32)
        embed_tokens = SimpleNamespace(weight=torch.empty(1))
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
        embed_tokens=SimpleNamespace(weight=torch.empty(1, dtype=torch.bfloat16)),
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


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
