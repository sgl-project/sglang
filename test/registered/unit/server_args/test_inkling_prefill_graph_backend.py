"""Inkling's attention-backend and prefill-graph defaults must agree.

Regression: the backend default is fa4 on SM100 and triton otherwise, while the
full-graph prefill opt-in fired unconditionally. Only fa4 can capture a prefill
graph, so off SM100 a bare launch died in capture_prefill_graph with
`ValueError: Invalid forward mode: forward_mode=<ForwardMode.EXTEND: 1>`.

The two decisions live in different files and run at different times, which is
how they drifted apart. Every Inkling test pins --attention-backend fa4, so
nothing covered the default path.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.overrides import resolve_inkling_attention_backend
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase

_INKLING_ARCH = "InklingForConditionalGeneration"


def _inkling_args(arch: str = _INKLING_ARCH, **overrides) -> ServerArgs:
    args = ServerArgs(model_path="dummy")
    args.model_config = SimpleNamespace(hf_config=SimpleNamespace(architectures=[arch]))
    args.cuda_graph_backend_prefill = None
    args.disable_prefill_cuda_graph = False
    args.attention_backend = None
    args.prefill_attention_backend = None
    args.decode_attention_backend = None
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _resolved_prefill_backend(args: ServerArgs, *, sm100: bool):
    with patch(
        "sglang.srt.arg_groups.overrides.is_sm100_supported", return_value=sm100
    ):
        args._apply_inkling_prefill_cuda_graph_default()
    return args.cuda_graph_backend_prefill


class TestInklingPrefillGraphBackend(CustomTestCase):
    def test_hopper_default_does_not_opt_into_prefill_graph(self):
        args = _inkling_args()
        self.assertEqual(resolve_inkling_attention_backend(args), "triton")
        self.assertIsNone(_resolved_prefill_backend(args, sm100=False))

    def test_sm100_default_still_opts_into_prefill_graph(self):
        args = _inkling_args()
        self.assertIsNotNone(_resolved_prefill_backend(args, sm100=True))

    def test_explicit_fa4_opts_in_on_hopper_too(self):
        args = _inkling_args(attention_backend="fa4")
        self.assertEqual(resolve_inkling_attention_backend(args), "fa4")
        self.assertIsNotNone(_resolved_prefill_backend(args, sm100=False))

    def test_explicit_triton_does_not_opt_in_on_sm100(self):
        args = _inkling_args(attention_backend="triton")
        self.assertIsNone(_resolved_prefill_backend(args, sm100=True))

    def test_non_inkling_arch_untouched(self):
        args = _inkling_args(arch="Qwen3ForCausalLM")
        self.assertIsNone(_resolved_prefill_backend(args, sm100=True))


if __name__ == "__main__":
    unittest.main()
