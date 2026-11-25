"""DeepSeek-V4 explicit attention-backend and page-size validation."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.arg_utils import record_fields
from sglang.srt.arg_groups.cuda_graph_hook import handle_cuda_graph_config
from sglang.srt.arg_groups.model_override_base import (
    attention_backends_of,
    resolved_view,
)
from sglang.srt.arg_groups.model_overrides.deepseek_v4 import (
    DSV4_NPU_PAGE_SIZE,
    DSV4_PAGE_SIZE,
)
from sglang.srt.arg_groups.overrides import (
    collect_model_override_declarations,
    resolution_result,
    validate_declarations,
)
from sglang.srt.hardware_backend.npu.utils import set_default_server_args
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_DSV4_ARCH = "DeepseekV4ForCausalLM"


def _dsv4_hf_config() -> SimpleNamespace:
    return SimpleNamespace(architectures=[_DSV4_ARCH])


def _make_server_args(device: str, **fields) -> ServerArgs:
    server_args = ServerArgs(model_path="dummy", device=device, **fields)
    server_args._model_config = SimpleNamespace(
        hf_config=_dsv4_hf_config(),
        is_fp4_experts=False,
        nvfp4_moe_meta=None,
    )
    server_args._raw_input = {
        field.name: getattr(server_args, field.name)
        for field in record_fields(type(server_args))
    }
    server_args._resolved_overrides = []
    return server_args


def _apply_dsv4_overrides(server_args: ServerArgs) -> dict:
    declarations = collect_model_override_declarations(
        _DSV4_ARCH, server_args, hf_config=_dsv4_hf_config()
    )
    validate_declarations(server_args, declarations)
    server_args._resolved_overrides.extend(declarations)
    return {
        field: value
        for _, declaration in declarations
        for field, value in declaration.items()
    }


def _apply_npu_defaults_then_dsv4_overrides(server_args: ServerArgs) -> dict:
    handle_cuda_graph_config(server_args)
    with patch(
        "sglang.srt.hardware_backend.npu.utils.get_npu_memory_capacity",
        return_value=64 * 1024,
    ):
        set_default_server_args(server_args)
    return _apply_dsv4_overrides(server_args)


class TestDeepSeekV4CudaOverrides(CustomTestCase):
    def test_unset_defaults_resolve_to_dsv4_and_256(self):
        server_args = _make_server_args("cuda")
        declared = _apply_dsv4_overrides(server_args)
        self.assertEqual(declared["attention_backend"], "dsv4")
        self.assertEqual(declared["page_size"], DSV4_PAGE_SIZE)
        self.assertNotIn("prefill_attention_backend", declared)
        self.assertNotIn("decode_attention_backend", declared)
        self.assertEqual(resolution_result(server_args, "attention_backend"), "dsv4")
        self.assertEqual(
            attention_backends_of(resolved_view(server_args)), ("dsv4", "dsv4")
        )
        self.assertEqual(resolution_result(server_args, "page_size"), DSV4_PAGE_SIZE)
        self.assertIsNone(server_args.attention_backend)
        self.assertIsNone(server_args.page_size)

    def test_explicit_supported_values_are_accepted(self):
        cases = (
            {"attention_backend": "dsv4"},
            {"prefill_attention_backend": "dsv4"},
            {"decode_attention_backend": "dsv4"},
            {"page_size": DSV4_PAGE_SIZE},
            {
                "attention_backend": "dsv4",
                "prefill_attention_backend": "dsv4",
                "decode_attention_backend": "dsv4",
                "page_size": DSV4_PAGE_SIZE,
            },
        )
        for fields in cases:
            with self.subTest(**fields):
                server_args = _make_server_args("cuda", **fields)
                _apply_dsv4_overrides(server_args)
                self.assertEqual(
                    resolution_result(server_args, "attention_backend"), "dsv4"
                )
                self.assertEqual(
                    attention_backends_of(resolved_view(server_args)), ("dsv4", "dsv4")
                )
                self.assertEqual(
                    resolution_result(server_args, "page_size"), DSV4_PAGE_SIZE
                )

    def test_explicit_unsupported_attention_backends_are_rejected(self):
        cases = (
            ("attention_backend", "--attention-backend", "flashinfer"),
            ("attention_backend", "--attention-backend", "triton"),
            ("prefill_attention_backend", "--prefill-attention-backend", "fa3"),
            ("decode_attention_backend", "--decode-attention-backend", "flashmla"),
        )
        for field, flag, value in cases:
            with self.subTest(field=field, value=value):
                server_args = _make_server_args("cuda", **{field: value})
                with self.assertRaisesRegex(
                    ValueError,
                    rf"DeepSeek-V4.*{flag}.*'dsv4'.*{value!r}",
                ):
                    _apply_dsv4_overrides(server_args)

    def test_explicit_unsupported_page_size_is_rejected(self):
        for value in (0, 16, 64, 128, 512):
            with self.subTest(page_size=value):
                server_args = _make_server_args("cuda", page_size=value)
                with self.assertRaisesRegex(
                    ValueError,
                    rf"DeepSeek-V4.*--page-size {DSV4_PAGE_SIZE}.*got {value}",
                ):
                    _apply_dsv4_overrides(server_args)


class TestDeepSeekV4NpuOverrides(CustomTestCase):
    def test_unset_defaults_override_generic_ascend_to_dsv4_and_128(self):
        server_args = _make_server_args("npu")
        declared = _apply_npu_defaults_then_dsv4_overrides(server_args)
        self.assertEqual(declared["attention_backend"], "dsv4")
        self.assertEqual(declared["prefill_attention_backend"], "dsv4")
        self.assertEqual(declared["decode_attention_backend"], "dsv4")
        self.assertEqual(declared["page_size"], DSV4_NPU_PAGE_SIZE)
        self.assertEqual(resolution_result(server_args, "attention_backend"), "dsv4")
        self.assertEqual(
            resolution_result(server_args, "prefill_attention_backend"), "dsv4"
        )
        self.assertEqual(
            resolution_result(server_args, "decode_attention_backend"), "dsv4"
        )
        self.assertEqual(
            resolution_result(server_args, "page_size"), DSV4_NPU_PAGE_SIZE
        )
        # Raw inputs remain unset.
        self.assertIsNone(server_args.attention_backend)
        self.assertIsNone(server_args.prefill_attention_backend)
        self.assertIsNone(server_args.decode_attention_backend)
        self.assertIsNone(server_args.page_size)

    def test_explicit_supported_values_are_accepted(self):
        cases = (
            {"attention_backend": "dsv4"},
            {"prefill_attention_backend": "dsv4"},
            {"decode_attention_backend": "dsv4"},
            {"page_size": DSV4_NPU_PAGE_SIZE},
            {
                "attention_backend": "dsv4",
                "prefill_attention_backend": "dsv4",
                "decode_attention_backend": "dsv4",
                "page_size": DSV4_NPU_PAGE_SIZE,
            },
        )
        for fields in cases:
            with self.subTest(**fields):
                server_args = _make_server_args("npu", **fields)
                _apply_npu_defaults_then_dsv4_overrides(server_args)
                self.assertEqual(
                    resolution_result(server_args, "attention_backend"), "dsv4"
                )
                self.assertEqual(
                    resolution_result(server_args, "prefill_attention_backend"), "dsv4"
                )
                self.assertEqual(
                    resolution_result(server_args, "decode_attention_backend"), "dsv4"
                )
                self.assertEqual(
                    resolution_result(server_args, "page_size"), DSV4_NPU_PAGE_SIZE
                )

    def test_explicit_unsupported_attention_backends_are_rejected(self):
        cases = (
            ("attention_backend", "--attention-backend", "ascend"),
            ("attention_backend", "--attention-backend", "triton"),
            ("prefill_attention_backend", "--prefill-attention-backend", "ascend"),
            ("decode_attention_backend", "--decode-attention-backend", "ascend"),
        )
        for field, flag, value in cases:
            with self.subTest(field=field, value=value):
                server_args = _make_server_args("npu", **{field: value})
                with self.assertRaisesRegex(
                    ValueError,
                    rf"DeepSeek-V4.*{flag}.*'dsv4'.*{value!r}",
                ):
                    _apply_npu_defaults_then_dsv4_overrides(server_args)

    def test_explicit_unsupported_page_size_is_rejected(self):
        for value in (0, 64, 256):
            with self.subTest(page_size=value):
                server_args = _make_server_args("npu", page_size=value)
                with self.assertRaisesRegex(
                    ValueError,
                    rf"DeepSeek-V4.*--page-size {DSV4_NPU_PAGE_SIZE}.*got {value}",
                ):
                    _apply_npu_defaults_then_dsv4_overrides(server_args)


if __name__ == "__main__":
    unittest.main()
