"""Request-admission validation for UNO speculative decoding."""

import unittest
from types import SimpleNamespace

from sglang.srt.speculative.uno_validation import validate_uno_request
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _make_request(**overrides):
    sampling_params = SimpleNamespace(
        min_p=0.0,
        json_schema=None,
        regex=None,
        ebnf=None,
        structural_tag=None,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        min_new_tokens=0,
        logit_bias=None,
    )
    request = SimpleNamespace(
        sampling_params=sampling_params,
        grammar=None,
        return_logprob=False,
        return_hidden_states_mode=SimpleNamespace(need_capture=lambda: False),
        custom_logit_processor=None,
        lora_id=None,
    )
    for name, value in overrides.items():
        target, field = name.split("__", maxsplit=1)
        owner = sampling_params if target == "sampling_params" else request
        setattr(owner, field, value)
    return request


class TestUnoRequestValidation(CustomTestCase):
    def test_supported_request_is_accepted(self):
        self.assertIsNone(validate_uno_request(_make_request()))

    def test_unsupported_request_features_are_rejected(self):
        cases = {
            "min_p": ({"sampling_params__min_p": 0.1}, "min_p"),
            "grammar": ({"sampling_params__regex": "[0-9]+"}, "grammar"),
            "logprobs": ({"request__return_logprob": True}, "logprobs"),
            "hidden states": (
                {
                    "request__return_hidden_states_mode": SimpleNamespace(
                        need_capture=lambda: True
                    )
                },
                "return_hidden_states",
            ),
            "frequency penalty": (
                {"sampling_params__frequency_penalty": 0.1},
                "penalties",
            ),
            "presence penalty": (
                {"sampling_params__presence_penalty": 0.1},
                "penalties",
            ),
            "repetition penalty": (
                {"sampling_params__repetition_penalty": 1.1},
                "penalties",
            ),
            "minimum new tokens": (
                {"sampling_params__min_new_tokens": 1},
                "penalties",
            ),
            "logit bias": (
                {"sampling_params__logit_bias": {1: 0.5}},
                "logit_bias",
            ),
            "custom processor": (
                {"request__custom_logit_processor": "processor"},
                "custom logit processors",
            ),
            "public LoRA": ({"request__lora_id": "adapter"}, "LoRA"),
        }

        for name, (overrides, expected) in cases.items():
            with self.subTest(name=name):
                error = validate_uno_request(_make_request(**overrides))
                self.assertIsNotNone(error)
                self.assertIn(expected, error)


if __name__ == "__main__":
    unittest.main()
