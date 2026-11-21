"""
Unit tests for the input_embeds x speculative-decoding gate in
TokenizerManager._tokenize_one_request.

A request carrying input_embeds on a server launched with any
--speculative-algorithm must be rejected with a per-request error at
tokenization time. Without the gate, the embeddings reach the draft
model's prefill: a draft whose hidden size differs from the target's
(e.g. STANDALONE) dies on the shape mismatch and kills the scheduler,
and an EAGLE-family draft silently builds its prompt KV from embeddings
that violate its embedding/hidden-state alignment contract.

Covers:
  - input_embeds + speculative algorithm -> ValueError naming speculative
    decoding, raised before validation/dispatch
  - input_embeds without a speculative algorithm proceeds
  - token-id requests are not rejected when speculation is enabled
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import GenerateReqInput  # noqa: E402
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_EMBEDS = [[0.1] * 8 for _ in range(4)]


def _make_tm() -> TokenizerManager:
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.model_config = MagicMock()
    tm.model_config.hf_config = SimpleNamespace(architectures=["LlamaForCausalLM"])
    tm.served_model_name = "test-model"
    tm.max_req_input_len = 8192
    tm.mm_processor = None
    return tm


def _run_tokenize(
    tm,
    request,
    speculative_algorithm,
    disable_radix_cache: bool = True,
):
    async def drive():
        with (
            patch(
                "sglang.srt.managers.tokenizer_manager.get_memory",
                return_value=SimpleNamespace(disable_radix_cache=disable_radix_cache),
            ),
            patch(
                "sglang.srt.managers.tokenizer_manager.get_spec",
                return_value=SimpleNamespace(
                    speculative_algorithm=speculative_algorithm
                ),
            ),
        ):
            return await tm._tokenize_one_request(request)

    return asyncio.run(drive())


def _stub_pipeline_past_gate(tm: TokenizerManager, sentinel: object) -> None:
    tm._validate_one_request = Mock()
    tm._create_tokenized_object = Mock(return_value=sentinel)


class TestInputEmbedsSpecRejection(CustomTestCase):
    def test_input_embeds_rejected_when_speculation_enabled(self):
        tm = _make_tm()
        tm._validate_one_request = Mock()
        request = GenerateReqInput(input_embeds=_EMBEDS)

        with self.assertRaisesRegex(ValueError, "speculative decoding"):
            _run_tokenize(tm, request, speculative_algorithm="EAGLE3")

        tm._validate_one_request.assert_not_called()

    def test_input_embeds_allowed_without_speculation(self):
        tm = _make_tm()
        sentinel = object()
        _stub_pipeline_past_gate(tm, sentinel)
        request = GenerateReqInput(input_embeds=_EMBEDS)

        self.assertIs(_run_tokenize(tm, request, speculative_algorithm=None), sentinel)

    def test_token_requests_not_rejected_when_speculation_enabled(self):
        tm = _make_tm()
        sentinel = object()
        _stub_pipeline_past_gate(tm, sentinel)
        request = GenerateReqInput(input_ids=[1, 2, 3])

        self.assertIs(
            _run_tokenize(tm, request, speculative_algorithm="EAGLE3"), sentinel
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
