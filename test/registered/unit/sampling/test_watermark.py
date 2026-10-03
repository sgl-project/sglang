import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.logprob_processor import OutputLogprobProcessor
from sglang.srt.layers.sampler import Sampler
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.sampling.watermarking.core import (
    build_watermark_batch_config,
    normalize_watermark_request,
    resolve_watermark_request,
)
from sglang.srt.speculative import eagle_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


_DEFAULT_KEY = "0123456789abcdef"
_REQUEST_KEY = "fedcba9876543210"
_REQUEST_FORMS = {
    "omitted": None,
    "disabled": normalize_watermark_request({"enabled": False}),
    "enabled": normalize_watermark_request({"enabled": True}),
    "key": normalize_watermark_request({"key": _REQUEST_KEY}),
}


@pytest.mark.parametrize(
    (
        "server_enabled",
        "default_enabled",
        "enforce_all",
        "request_form",
        "expected_key",
        "expected_enabled",
    ),
    [
        pytest.param(False, False, False, "omitted", None, False, id="off-omitted"),
        pytest.param(False, False, False, "disabled", None, False, id="off-opt-out"),
        pytest.param(False, False, False, "enabled", ValueError, None, id="off-opt-in"),
        pytest.param(False, False, False, "key", ValueError, None, id="off-key"),
        pytest.param(True, False, False, "omitted", None, False, id="opt-in-omitted"),
        pytest.param(True, False, False, "disabled", None, False, id="opt-in-opt-out"),
        pytest.param(
            True, True, False, "enabled", _DEFAULT_KEY, True, id="default-on-enabled"
        ),
        pytest.param(True, True, False, "key", _REQUEST_KEY, True, id="default-on-key"),
        pytest.param(
            True, False, True, "enabled", _DEFAULT_KEY, True, id="enforce-enabled"
        ),
        pytest.param(True, False, True, "key", _REQUEST_KEY, True, id="enforce-key"),
    ],
)
def test_request_enablement_matrix(
    server_enabled,
    default_enabled,
    enforce_all,
    request_form,
    expected_key,
    expected_enabled,
):
    def resolve():
        return resolve_watermark_request(
            _REQUEST_FORMS[request_form],
            server_enabled=server_enabled,
            default_key=_DEFAULT_KEY,
            default_context_window=4,
            default_enabled=default_enabled,
            enforce_all=enforce_all,
        )

    if expected_key is ValueError:
        with pytest.raises(ValueError):
            resolve()
        return

    key, context_window, enabled = resolve()
    assert key == expected_key
    assert context_window == 4
    assert enabled is expected_enabled


def test_per_request_batch_config_and_admission():
    secret = _REQUEST_KEY
    requests = [
        SimpleNamespace(
            sampling_params=SimpleNamespace(
                watermark=normalize_watermark_request(
                    {"key": secret, "context_window": 2}
                ),
                top_k=2,
            )
        ),
        SimpleNamespace(sampling_params=SimpleNamespace(watermark=None, top_k=2)),
        SimpleNamespace(
            sampling_params=SimpleNamespace(
                watermark=normalize_watermark_request({"enabled": False}),
                top_k=2,
            )
        ),
        SimpleNamespace(
            sampling_params=SimpleNamespace(
                watermark=normalize_watermark_request({"enabled": True}),
                top_k=1,
            )
        ),
    ]

    config = build_watermark_batch_config(
        requests,
        default_key=_DEFAULT_KEY,
        default_context_window=4,
        default_enabled=False,
        enforce_all=False,
        device="cpu",
    )

    assert config.keys.tolist() == [
        0xFEDCBA9876543210 - (1 << 64),
        0,
        0,
        0x0123456789ABCDEF,
    ]
    assert config.context_windows.tolist() == [2, 4, 4, 4]
    assert config.enabled.tolist() == [True, False, False, True]
    assert config.candidates_host == [True, False, False, False]
    assert config.has_candidates

    with pytest.raises(ValueError, match="unknown fields"):
        normalize_watermark_request({"key": secret, "provider": "textseal"})
    with pytest.raises(ValueError, match="enabled must be a boolean"):
        normalize_watermark_request({"enabled": 1})
    with pytest.raises(ValueError, match="must set enabled or key"):
        resolve_watermark_request(
            normalize_watermark_request({"context_window": 2}),
            server_enabled=True,
            default_key=_DEFAULT_KEY,
            default_context_window=4,
            default_enabled=True,
            enforce_all=False,
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_watermark_request(
            normalize_watermark_request({"enabled": False, "key": secret}),
            server_enabled=True,
            default_key=_DEFAULT_KEY,
            default_context_window=4,
            default_enabled=False,
            enforce_all=False,
        )


def _sampling_info(batch_size, **overrides):
    return SamplingBatchInfo(
        temperatures=torch.ones(batch_size, 1),
        top_ps=torch.ones(batch_size),
        top_ks=torch.full((batch_size,), TOP_K_ALL, dtype=torch.int32),
        min_ps=torch.zeros(batch_size),
        is_all_greedy=False,
        is_any_greedy=False,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        need_min_p_sampling=False,
        vocab_size=32,
        device="cpu",
        penalizer_orchestrator=Mock(is_required=False),
        **overrides,
    )


def test_watermark_rows_track_filter_and_merge():
    info = _sampling_info(
        2,
        watermark_enabled=torch.tensor([False, True]),
        watermark_candidates_host=[False, True],
        has_watermark_candidates=True,
    )
    info.filter_batch([0], torch.tensor([0]))
    assert info.watermark_enabled.tolist() == [False]
    assert info.watermark_candidates_host == [False]
    assert not info.has_watermark_candidates

    info.merge_batch(
        _sampling_info(
            1,
            watermark_enabled=torch.tensor([True]),
            watermark_candidates_host=[True],
            has_watermark_candidates=True,
        )
    )
    assert info.watermark_enabled.tolist() == [False, True]
    assert info.watermark_candidates_host == [False, True]
    assert info.has_watermark_candidates


def test_idle_batch_max_top_k_is_merge_identity():
    exec_context = SimpleNamespace(
        deterministic=SimpleNamespace(enable_deterministic_inference=False),
        features=SimpleNamespace(
            enable_custom_logit_processor=False,
            enable_watermark=False,
            watermark_default_enabled=False,
            watermark_enforce_all=False,
        ),
    )
    with patch(
        "sglang.srt.sampling.sampling_batch_info.get_exec", return_value=exec_context
    ):
        info = SamplingBatchInfo.from_schedule_batch(Mock(reqs=[], device="cpu"), 32)
    assert len(info) == 0
    assert info.max_top_k == 1


def test_default_watermark_policy_requires_schedule_batch():
    exec_context = SimpleNamespace(
        deterministic=SimpleNamespace(enable_deterministic_inference=False),
        features=SimpleNamespace(
            enable_custom_logit_processor=False,
            enable_watermark=True,
            watermark_default_enabled=True,
            watermark_enforce_all=False,
        ),
    )
    with (
        patch(
            "sglang.srt.sampling.sampling_batch_info.get_exec",
            return_value=exec_context,
        ),
        pytest.raises(RuntimeError, match="requires a ScheduleBatch"),
    ):
        SamplingBatchInfo.from_schedule_batch(Mock(reqs=[], device="cpu"), 32)


def test_watermark_logprobs_use_pre_force_distribution():
    sampling_info = _sampling_info(
        1,
        watermark_candidates_host=[True],
        has_watermark_candidates=True,
    )
    sampling_info.top_ks = torch.tensor([2], dtype=torch.int32)
    sampling_info.max_top_k = 2
    sampling_info.update_regex_vocab_mask = Mock()
    sampling_info.apply_logits_bias = Mock()

    sampler = object.__new__(Sampler)
    torch.nn.Module.__init__(sampler)
    sampler.rl_on_policy_target = None
    sampler.enable_deterministic = False
    sampler.use_log_softmax_logprob = False
    sampler.use_ascend_backend = False
    sampler.sampling_mask_max_tokens = 4096
    sampler.output_logprob_processor = OutputLogprobProcessor()

    runner = object.__new__(ModelRunner)
    runner._sampling_observer = None
    runner.sampler = sampler
    runner.ngram_embedding_manager = Mock()
    runner.watermark_state = Mock()

    def force_selected_token(logits, *_):
        logits.fill_(-torch.inf)
        logits[:, 1] = 0

    runner.watermark_state.force.side_effect = force_selected_token
    logits_output = LogitsProcessorOutput(
        next_token_logits=torch.tensor([[3.0, 1.0, -2.0]])
    )
    forward_batch = SimpleNamespace(
        sampling_info=sampling_info,
        req_pool_indices=torch.tensor([0], dtype=torch.int32),
        watermark_prompt_tail_ids=[[]],
        watermark_context_hash_history=[[]],
        return_logprob=True,
        top_logprobs_nums=[2],
        token_ids_logprobs=[[]],
        positions=torch.tensor([0], dtype=torch.int64),
        seq_lens=torch.tensor([1], dtype=torch.int64),
        forward_mode=SimpleNamespace(is_decode=lambda: True),
    )

    next_token_ids = runner.sample(logits_output, forward_batch)

    expected = torch.log_softmax(torch.tensor([3.0, 1.0, -2.0]), dim=-1)
    assert next_token_ids.tolist() == [1]
    torch.testing.assert_close(logits_output.next_token_logprobs, expected[1:2])
    assert [row.tolist() for row in logits_output.next_token_top_logprobs_idx] == [
        [0, 1]
    ]


def test_disabled_speculative_batch_skips_watermark_state():
    def fake_verify_tree_greedy(**kwargs):
        kwargs["predicts"].fill_(3)
        kwargs["accept_index"].fill_(0)
        kwargs["accept_token_num"].fill_(1)
        return (
            kwargs["predicts"],
            kwargs["accept_index"],
            kwargs["accept_token_num"],
        )

    verify_input = SimpleNamespace(
        draft_token_num=2,
        draft_token=torch.tensor([1, 2], dtype=torch.int32),
        max_tree_depth=2,
        tree_topk=1,
        retrieve_index=torch.zeros((1, 2), dtype=torch.int32),
        retrieve_next_token=torch.zeros((1, 2), dtype=torch.int32),
        retrieve_next_sibling=torch.zeros((1, 2), dtype=torch.int32),
        draft_probs=None,
        custom_mask=torch.zeros((1, 2), dtype=torch.bool),
        positions=torch.zeros(2, dtype=torch.int32),
    )
    sampling_info = SimpleNamespace(
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        logit_bias=None,
        is_all_greedy=True,
        temperatures=torch.ones((1, 1)),
        need_top_k_sampling=False,
        need_top_p_sampling=False,
        sampling_seed=None,
        has_watermark_candidates=False,
    )
    batch = SimpleNamespace(
        device="cpu",
        seq_lens=torch.tensor([4], dtype=torch.int32),
        req_pool_indices=torch.tensor([0], dtype=torch.int32),
        sampling_info=sampling_info,
        return_logprob=False,
        forward_mode=SimpleNamespace(is_idle=lambda: False),
    )
    logits_output = SimpleNamespace(next_token_logits=torch.randn((2, 8)))
    watermark_state = Mock()
    spec_config = SimpleNamespace(
        speculative_use_rejection_sampling=False,
        speculative_accept_threshold_single=1.0,
        speculative_accept_threshold_acc=1.0,
    )
    tp_group = SimpleNamespace(world_size=1)

    with (
        patch.object(eagle_utils, "get_spec", return_value=spec_config),
        patch(
            "sglang.srt.runtime_context.get_parallel",
            return_value=SimpleNamespace(
                tp_group=tp_group,
                attn_tp_group=tp_group,
            ),
        ),
        patch.object(
            eagle_utils,
            "verify_tree_greedy_func",
            side_effect=fake_verify_tree_greedy,
        ),
    ):
        eagle_utils.eagle_sample(
            verify_input,
            batch,
            logits_output,
            watermark_state=watermark_state,
        )

    watermark_state.speculative_contexts.assert_not_called()
    watermark_state.force_speculative.assert_not_called()
    watermark_state.record_speculative.assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
