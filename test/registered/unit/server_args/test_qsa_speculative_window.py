"""Reject oversized QSA verify windows before loading model weights."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.arg_groups import speculative_hook
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _qsa_config(ratio=4, *, nested=False):
    text_config = SimpleNamespace(
        architectures=["Qwen4ExpForConditionalGeneration"],
        indexer_n_heads=8,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=512 * ratio,
        indexer_compress_ratio=ratio,
    )
    if nested:
        return SimpleNamespace(
            architectures=text_config.architectures, text_config=text_config
        )
    return text_config


class TestQsaSpeculativeWindow(CustomTestCase):
    def _resolve(self, hf_config, *, maximum=None, **fields):
        values = {
            "speculative_algorithm": "EAGLE",
            "device": "cuda",
            "max_running_requests": 48,
            "disable_overlap_schedule": True,
            "enable_mixed_chunk": False,
            "speculative_draft_model_path": "draft",
            "model_path": "target",
            "revision": None,
            "speculative_adaptive": False,
            "speculative_num_steps": 3,
            "speculative_eagle_topk": 1,
            "speculative_num_draft_tokens": 4,
            "speculative_use_rejection_sampling": False,
            "speculative_token_map": None,
            "speculative_accept_threshold_single": 1.0,
            "speculative_accept_threshold_acc": 1.0,
            "enable_deterministic_inference": False,
            "enable_multi_layer_eagle": False,
            "attention_backend": "triton",
            "page_size": 1,
        }
        values.update(fields)
        args = SimpleNamespace(**values)
        max_tokens = Mock(
            side_effect=lambda sa: (
                sa.speculative_num_draft_tokens if maximum is None else maximum
            )
        )

        def declare(sa, _origin, **updates):
            for key, value in updates.items():
                setattr(sa, key, value)

        with ExitStack() as stack:
            for name, value in {
                "resolving_view": lambda sa: sa,
                "resolved_view": lambda sa: sa,
                "model_config_of": lambda sa: SimpleNamespace(hf_config=hf_config),
                "declare_resolution": declare,
                "max_speculative_num_draft_tokens": max_tokens,
                "attn_dp_enabled_of": lambda sa: False,
                "attention_backends_of": lambda sa: [sa.attention_backend],
                "get_platform": lambda: SimpleNamespace(is_hip=False),
                "envs": SimpleNamespace(
                    SGLANG_SIMULATE_ACC_LEN=SimpleNamespace(get=lambda: -1)
                ),
                "_check_oot_speculative_algorithm": Mock(),
                "_disable_overlap_schedule_for_cpu": Mock(),
                "_handle_iquest_q1_mtp_draft": Mock(),
                "_fall_back_xpu_draft_attention_for_head_size": Mock(),
                "_auto_choose_speculative_params": Mock(return_value=(3, 1, 4)),
            }.items():
                stack.enter_context(patch.object(speculative_hook, name, value))
            speculative_hook._handle_eagle_family(args)
        return args, max_tokens

    def test_rejects_window_above_configured_ratio(self):
        for nested in (False, True):
            for ratio in (2, 4, 8):
                with (
                    self.subTest(nested=nested, ratio=ratio),
                    self.assertRaisesRegex(
                        ValueError, rf"QSA.*ratio \({ratio}\).*got {ratio + 1}"
                    ),
                ):
                    self._resolve(
                        _qsa_config(ratio, nested=nested),
                        speculative_num_steps=ratio,
                        speculative_num_draft_tokens=ratio + 1,
                    )

    def test_accepts_window_at_configured_ratio(self):
        for ratio in (2, 4, 8):
            with self.subTest(ratio=ratio):
                args, _ = self._resolve(
                    _qsa_config(ratio),
                    speculative_num_steps=ratio - 1,
                    speculative_num_draft_tokens=ratio,
                )
                self.assertEqual(args.speculative_num_draft_tokens, ratio)

    def test_validates_after_chain_count_is_increased(self):
        with self.assertRaisesRegex(ValueError, r"ratio \(4\).*got 5"):
            self._resolve(
                _qsa_config(),
                speculative_num_steps=4,
                speculative_num_draft_tokens=4,
            )

    def test_accepts_chain_count_normalized_down_to_supported_window(self):
        args, _ = self._resolve(_qsa_config(), speculative_num_draft_tokens=99)
        self.assertEqual(args.speculative_num_draft_tokens, 4)

    def test_automatic_parameters_are_validated(self):
        fields = {
            "speculative_num_steps": None,
            "speculative_eagle_topk": None,
            "speculative_num_draft_tokens": None,
        }
        args, _ = self._resolve(_qsa_config(), **fields)
        self.assertEqual(args.speculative_num_draft_tokens, 4)
        with self.assertRaisesRegex(ValueError, r"ratio \(2\).*got 4"):
            self._resolve(_qsa_config(2), **fields)

    def test_rejects_oversized_adaptive_maximum(self):
        with self.assertRaisesRegex(ValueError, r"ratio \(4\).*got 7"):
            self._resolve(
                _qsa_config(),
                maximum=7,
                speculative_adaptive=True,
                speculative_num_steps=2,
                speculative_num_draft_tokens=3,
            )

    def test_accepts_supported_adaptive_maximum(self):
        self._resolve(
            _qsa_config(),
            maximum=4,
            speculative_adaptive=True,
            speculative_num_steps=2,
            speculative_num_draft_tokens=3,
        )

    def test_non_qsa_does_not_read_qsa_maximum(self):
        config = SimpleNamespace(architectures=["LlamaForCausalLM"])
        _, maximum = self._resolve(
            config, speculative_num_steps=8, speculative_num_draft_tokens=9
        )
        maximum.assert_not_called()


if __name__ == "__main__":
    unittest.main()
