"""LoRA on kv_b_proj and the MHA cached-prefix paths.

BUG REGRESSION. The DeepSeek/Kimi MHA prefix paths re-project the cached latent
through kv_b_proj on prefix rows; run with the batch's extend-token segments the
adapter kernels faulted on trtllm_mla (flashinfer fmhaKernels.cuh:1338) and put
the delta on the wrong rows elsewhere. LoRA backends that route those rows to
their requests' adapters serve every attention backend unchanged. Without that
routing (other LoRA backends, Full prefill CUDA graphs, ROCm) the hook keeps
cached prefixes on the absorbed MLA path: it disables the chunked prefix cache
for kv_b_proj targets and rejects the backends that ignore the flag.

    python -m pytest test/registered/unit/server_args/test_lora_kv_b_chunked_prefix.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups import lora_hook
from sglang.srt.arg_groups.lora_hook import check_lora_server_args
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.lora.backend import lora_registry
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
    # cannot route cached-prefix rows
    fields.setdefault("lora_backend", "torch_native")
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

    def test_two_batch_overlap_is_rejected_with_lora(self):
        for targets in (_KIMI_TARGETS, ["q_proj", "o_proj"]):
            with self.subTest(targets=targets):
                with self.assertRaisesRegex(ValueError, "two-batch-overlap"):
                    check_lora_server_args(
                        _args(
                            attention_backend="fa3",
                            lora_target_modules=targets,
                            enable_two_batch_overlap=True,
                        )
                    )

    def test_kv_cache_sharding_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "kv-cache-sharding"):
            check_lora_server_args(
                _args(attention_backend="trtllm_mla", enable_kv_cache_sharding=True)
            )


class TestKvBLoraCachedRowsRouting(CustomTestCase):
    def test_routing_backends_leave_every_attention_backend_alone(self):
        for lora_backend in ("csgmv", "triton"):
            for backend in ("trtllm_mla", "fa3", "dsa", "nsa", "fa4"):
                with self.subTest(lora_backend=lora_backend, backend=backend):
                    args = _args(lora_backend=lora_backend, attention_backend=backend)
                    check_lora_server_args(args)
                    self.assertFalse(
                        resolution_result(args, "disable_chunked_prefix_cache")
                    )

    def test_full_prefill_graph_keeps_the_absorbed_path_rule(self):
        args = _args(
            lora_backend="csgmv",
            attention_backend="trtllm_mla",
            cuda_graph_backend_prefill="full",
        )
        check_lora_server_args(args)
        self.assertTrue(resolution_result(args, "disable_chunked_prefix_cache"))
        with self.assertRaisesRegex(ValueError, "nsa"):
            check_lora_server_args(
                _args(
                    lora_backend="csgmv",
                    attention_backend="nsa",
                    cuda_graph_backend_prefill="full",
                )
            )

    def test_rocm_keeps_the_absorbed_path_rule(self):
        with patch.object(lora_registry, "is_hip", return_value=True):
            args = _args(lora_backend="triton", attention_backend="trtllm_mla")
            check_lora_server_args(args)
        self.assertTrue(resolution_result(args, "disable_chunked_prefix_cache"))


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
