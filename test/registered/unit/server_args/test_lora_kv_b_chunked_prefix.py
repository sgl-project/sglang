"""LoRA on kv_b_proj keeps cached prefixes off the MHA prefix paths.

BUG REGRESSION. The DeepSeek/Kimi MHA prefix paths re-project the cached latent
through kv_b_proj on prefix rows while the wrapped layer runs the adapter
kernels with the batch's extend-token segments; on trtllm_mla every prefix hit
took that path and faulted (flashinfer fmhaKernels.cuh:1338). Only the absorbed
MLA path adds the kv_b_proj delta per token, so the LoRA hook disables the
chunked prefix cache for kv_b_proj targets and rejects the backends that ignore
the flag.

    python -m pytest test/registered/unit/server_args/test_lora_kv_b_chunked_prefix.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups import lora_hook
from sglang.srt.arg_groups.lora_hook import check_lora_server_args
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.lora.utils import kv_b_lora_prefix_cache_error
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

_KIMI_TARGETS = [
    "language_model.model.layers.*.self_attn.q_a_proj",
    "language_model.model.layers.*.self_attn.q_b_proj",
    "language_model.model.layers.*.self_attn.kv_a_proj_with_mqa",
    "language_model.model.layers.*.self_attn.kv_b_proj",
    "language_model.model.layers.*.self_attn.o_proj",
]


def _args(**fields):
    fields.setdefault("enable_lora", True)
    fields.setdefault("max_lora_rank", 16)
    fields.setdefault("lora_target_modules", _KIMI_TARGETS)
    return ServerArgs(model_path="dummy", **fields)


def _platform(is_sm100_or_sm110: bool):
    return SimpleNamespace(is_sm100_or_sm110=is_sm100_or_sm110, is_hip=False)


class TestKvBLoraPrefixCacheHook(CustomTestCase):
    def test_kv_b_proj_target_disables_chunked_prefix_cache(self):
        for backend in ("trtllm_mla", "fa3", "flashinfer", "flashmla", None):
            with self.subTest(backend=backend):
                args = _args(attention_backend=backend)
                check_lora_server_args(args)
                self.assertTrue(resolution_result(args, "disable_chunked_prefix_cache"))

    def test_explicit_flag_is_left_alone(self):
        args = _args(attention_backend="trtllm_mla", disable_chunked_prefix_cache=True)
        check_lora_server_args(args)
        self.assertTrue(resolution_result(args, "disable_chunked_prefix_cache"))
        self.assertFalse(
            any(
                "disable_chunked_prefix_cache" in declared
                for _, declared in getattr(args, "_resolved_overrides", [])
            )
        )

    def test_other_targets_keep_chunked_prefix_cache(self):
        for targets in (
            ["q_proj", "k_proj", "v_proj", "o_proj"],
            _KIMI_TARGETS[:3] + _KIMI_TARGETS[4:],
        ):
            with self.subTest(targets=targets):
                args = _args(
                    attention_backend="trtllm_mla", lora_target_modules=targets
                )
                check_lora_server_args(args)
                self.assertFalse(
                    resolution_result(args, "disable_chunked_prefix_cache")
                )

    def test_no_lora_keeps_chunked_prefix_cache(self):
        args = ServerArgs(model_path="dummy", attention_backend="trtllm_mla")
        check_lora_server_args(args)
        self.assertFalse(resolution_result(args, "disable_chunked_prefix_cache"))

    def test_all_targets_follow_the_model_attention_arch(self):
        for is_mla, expected in ((True, True), (False, False)):
            with (
                self.subTest(is_mla=is_mla),
                patch.object(lora_hook, "use_mla_backend", return_value=is_mla),
            ):
                args = _args(
                    attention_backend="trtllm_mla", lora_target_modules=["all"]
                )
                check_lora_server_args(args)
                self.assertEqual(
                    bool(resolution_result(args, "disable_chunked_prefix_cache")),
                    expected,
                )

    def test_inferred_targets_are_left_to_the_manager(self):
        # --lora-paths without --lora-target-modules: targets come from the
        # adapter configs at load time, where LoRAManager.init_lora_shapes checks.
        args = ServerArgs(
            model_path="dummy",
            attention_backend="trtllm_mla",
            lora_paths=["adapter=/tmp/adapter"],
        )
        check_lora_server_args(args)
        self.assertFalse(resolution_result(args, "disable_chunked_prefix_cache"))

    def test_backends_that_ignore_the_flag_are_rejected(self):
        for backend in ("dsa", "nsa", "aiter"):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(ValueError, backend):
                    check_lora_server_args(_args(attention_backend=backend))
        with (
            self.subTest(backend="fa4 off sm100"),
            patch.object(lora_hook, "get_platform", return_value=_platform(False)),
        ):
            with self.assertRaisesRegex(ValueError, "fa4"):
                check_lora_server_args(_args(attention_backend="fa4"))
        with (
            self.subTest(backend="fa4 on sm100"),
            patch.object(lora_hook, "get_platform", return_value=_platform(True)),
        ):
            args = _args(attention_backend="fa4")
            check_lora_server_args(args)
            self.assertTrue(resolution_result(args, "disable_chunked_prefix_cache"))

    def test_split_prefill_backend_decides(self):
        args = _args(attention_backend="fa3", prefill_attention_backend="dsa")
        with self.assertRaisesRegex(ValueError, "dsa"):
            check_lora_server_args(args)

    def test_kv_cache_sharding_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "kv-cache-sharding"):
            check_lora_server_args(
                _args(attention_backend="trtllm_mla", enable_kv_cache_sharding=True)
            )


class TestKvBLoraPrefixCacheError(CustomTestCase):
    """The helper is shared by the hook and LoRAManager.init_lora_shapes."""

    def test_flag_still_on_names_the_flag(self):
        error = kv_b_lora_prefix_cache_error(
            "trtllm_mla",
            disable_chunked_prefix_cache=False,
            enable_kv_cache_sharding=False,
            fa4_absorbed_mla=True,
        )
        self.assertIn("--disable-chunked-prefix-cache", error)

    def test_safe_configuration(self):
        for backend in ("trtllm_mla", "fa3", "flashinfer", "flashmla", "fa4", None):
            with self.subTest(backend=backend):
                self.assertIsNone(
                    kv_b_lora_prefix_cache_error(
                        backend,
                        disable_chunked_prefix_cache=True,
                        enable_kv_cache_sharding=False,
                        fa4_absorbed_mla=True,
                    )
                )


if __name__ == "__main__":
    unittest.main()
