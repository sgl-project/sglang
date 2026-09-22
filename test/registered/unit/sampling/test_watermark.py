import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.sampling.watermark import (
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
            True, False, False, "enabled", _DEFAULT_KEY, True, id="opt-in-enabled"
        ),
        pytest.param(True, False, False, "key", _REQUEST_KEY, True, id="opt-in-key"),
        pytest.param(
            True, True, False, "omitted", _DEFAULT_KEY, True, id="default-on-omitted"
        ),
        pytest.param(
            True, True, False, "disabled", None, False, id="default-on-opt-out"
        ),
        pytest.param(
            True, True, False, "enabled", _DEFAULT_KEY, True, id="default-on-enabled"
        ),
        pytest.param(True, True, False, "key", _REQUEST_KEY, True, id="default-on-key"),
        pytest.param(
            True, False, True, "omitted", _DEFAULT_KEY, True, id="enforce-omitted"
        ),
        pytest.param(
            True, False, True, "disabled", ValueError, None, id="enforce-opt-out"
        ),
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
                top_k=2,
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
    assert config.candidates_host == [True, False, False, True]
    assert config.has_candidates

    config = build_watermark_batch_config(
        requests[:3],
        default_key=_DEFAULT_KEY,
        default_context_window=4,
        default_enabled=True,
        enforce_all=False,
        device="cpu",
    )
    assert config.keys.tolist() == [
        0xFEDCBA9876543210 - (1 << 64),
        0x0123456789ABCDEF,
        0,
    ]
    assert config.context_windows.tolist() == [2, 4, 4]
    assert config.enabled.tolist() == [True, True, False]
    assert config.candidates_host == [True, True, False]
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
