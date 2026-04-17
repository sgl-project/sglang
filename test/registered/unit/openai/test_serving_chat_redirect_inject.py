"""Unit tests for OpenAIServingChat._maybe_attach_redirect_processor.

We avoid constructing a real OpenAIServingChat (which needs a tokenizer manager
+ template manager). Instead we exercise the helper directly through a thin
subclass that bypasses __init__ and provides only the attributes the helper
relies on.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")

import json
import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.parser.reasoning_redirect_registry import ReasoningRedirectConfig
from sglang.srt.sampling.custom_logit_processor import (
    ReasoningEosRedirectLogitProcessor,
)
from sglang.test.test_utils import CustomTestCase


def _make_handler(
    *,
    redirect_unclosed_reasoning=True,
    enable_custom_logit_processor=True,
    cfg=None,
    reasoning_parser="qwen3",
    redirect_eos_prob_threshold=0.5,
):
    """Build an OpenAIServingChat without invoking __init__."""

    handler = OpenAIServingChat.__new__(OpenAIServingChat)
    handler.tokenizer_manager = SimpleNamespace(
        server_args=SimpleNamespace(
            redirect_unclosed_reasoning=redirect_unclosed_reasoning,
            enable_custom_logit_processor=enable_custom_logit_processor,
            redirect_eos_prob_threshold=redirect_eos_prob_threshold,
        ),
        tokenizer=SimpleNamespace(),  # placeholder; build_redirect_config is mocked via cfg
    )
    handler.reasoning_parser = reasoning_parser
    # Pre-populate the lazy cache so build_redirect_config is not called.
    handler._redirect_cfg = cfg
    handler._redirect_cfg_built = True
    handler._redirect_processor_str = (
        ReasoningEosRedirectLogitProcessor.to_str() if cfg is not None else None
    )
    return handler


def _basic_request(**overrides):
    base = {
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
    }
    base.update(overrides)
    return ChatCompletionRequest(**base)


_DUMMY_CFG = ReasoningRedirectConfig(
    think_start_token_id=151667,
    think_end_token_id=151668,
    redirect_eos_token_ids=[151645],
    force_reasoning=False,
)


class TestRedirectInjection(CustomTestCase):
    def test_attaches_processor_when_eligible(self):
        handler = _make_handler(cfg=_DUMMY_CFG)
        sampling_params = {"custom_params": None, "temperature": 1.0}
        result = handler._maybe_attach_redirect_processor(
            _basic_request(), sampling_params
        )
        self.assertIsNotNone(result)
        # Round-trip the processor string and make sure it deserializes to
        # ReasoningEosRedirectLogitProcessor.
        from sglang.srt.sampling.custom_logit_processor import (
            CustomLogitProcessor,
        )

        proc = CustomLogitProcessor.from_str(result)
        self.assertIsInstance(proc, ReasoningEosRedirectLogitProcessor)

        # custom_params merged with redirect params
        cp = sampling_params["custom_params"]
        self.assertEqual(cp["think_end_token_id"], 151668)
        self.assertEqual(cp["think_start_token_id"], 151667)
        self.assertEqual(cp["redirect_eos_token_ids"], [151645])
        self.assertEqual(cp["prob_threshold"], 0.5)
        self.assertFalse(cp["force_reasoning"])

    def test_skips_when_cfg_is_none(self):
        handler = _make_handler(cfg=None)
        sampling_params = {"custom_params": None}
        result = handler._maybe_attach_redirect_processor(
            _basic_request(), sampling_params
        )
        self.assertIsNone(result)
        self.assertIsNone(sampling_params["custom_params"])

    def test_skips_when_user_already_has_custom_logit_processor(self):
        handler = _make_handler(cfg=_DUMMY_CFG)
        sampling_params = {"custom_params": None}
        req = _basic_request(custom_logit_processor='{"callable": "custom"}')
        result = handler._maybe_attach_redirect_processor(req, sampling_params)
        self.assertIsNone(result)
        # custom_params remains untouched (we did not inject our params).
        self.assertIsNone(sampling_params["custom_params"])

    def test_user_custom_params_take_precedence(self):
        handler = _make_handler(cfg=_DUMMY_CFG, redirect_eos_prob_threshold=0.5)
        sampling_params = {
            "custom_params": {
                "prob_threshold": 0.9,
                "extra_user_field": "keep_me",
            }
        }
        result = handler._maybe_attach_redirect_processor(
            _basic_request(custom_params={"prob_threshold": 0.9}),
            sampling_params,
        )
        self.assertIsNotNone(result)
        cp = sampling_params["custom_params"]
        # User override wins on the conflicting key.
        self.assertEqual(cp["prob_threshold"], 0.9)
        self.assertEqual(cp["extra_user_field"], "keep_me")
        # Redirect-only fields are still present.
        self.assertEqual(cp["think_end_token_id"], 151668)

    def test_skips_when_custom_params_is_not_a_dict(self):
        handler = _make_handler(cfg=_DUMMY_CFG)
        sampling_params = {"custom_params": "not-a-dict"}
        result = handler._maybe_attach_redirect_processor(
            _basic_request(), sampling_params
        )
        self.assertIsNone(result)


class TestLazyBuildPath(CustomTestCase):
    """Cover the lazy build path: server flag off / enable_custom flag off."""

    def _fresh_handler(self, **kwargs):
        handler = OpenAIServingChat.__new__(OpenAIServingChat)
        handler.tokenizer_manager = SimpleNamespace(
            server_args=SimpleNamespace(
                redirect_unclosed_reasoning=kwargs.get(
                    "redirect_unclosed_reasoning", True
                ),
                enable_custom_logit_processor=kwargs.get(
                    "enable_custom_logit_processor", True
                ),
                redirect_eos_prob_threshold=0.5,
            ),
            tokenizer=None,  # build_redirect_config sees None and bails
        )
        handler.reasoning_parser = kwargs.get("reasoning_parser", "qwen3")
        handler._redirect_cfg = None
        handler._redirect_cfg_built = False
        handler._redirect_processor_str = None
        return handler

    def test_bails_when_redirect_flag_off(self):
        h = self._fresh_handler(redirect_unclosed_reasoning=False)
        self.assertIsNone(h._get_or_build_redirect_cfg())
        self.assertTrue(h._redirect_cfg_built)

    def test_bails_when_enable_custom_logit_off(self):
        h = self._fresh_handler(enable_custom_logit_processor=False)
        self.assertIsNone(h._get_or_build_redirect_cfg())
        self.assertTrue(h._redirect_cfg_built)

    def test_bails_when_tokenizer_missing(self):
        # Even with all flags on, a None tokenizer means the registry
        # cannot resolve token ids → cfg should be None.
        h = self._fresh_handler()
        self.assertIsNone(h._get_or_build_redirect_cfg())

    def test_caches_result_after_first_call(self):
        h = self._fresh_handler(redirect_unclosed_reasoning=False)
        self.assertIsNone(h._get_or_build_redirect_cfg())
        # Flip the flag — cached value should still be returned.
        h.tokenizer_manager.server_args.redirect_unclosed_reasoning = True
        self.assertIsNone(h._get_or_build_redirect_cfg())


if __name__ == "__main__":
    unittest.main()
