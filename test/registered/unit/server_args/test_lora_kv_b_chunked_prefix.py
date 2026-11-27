"""LoRA on kv_b_proj keeps trtllm_mla cached prefixes on the absorbed MLA path.

BUG REGRESSION. The DeepSeek/Kimi MHA cached-prefix paths re-project the cached
latent through kv_b_proj on prefix rows while the wrapped layer runs the adapter
kernels with the batch's extend-token segments; on trtllm_mla every prefix hit
took that path and faulted (flashinfer fmhaKernels.cuh:1338). The LoRA hook
resolves --disable-chunked-prefix-cache for kv_b_proj targets on trtllm_mla
only, and leaves every other configuration as it was (no new rejections).

    python -m pytest test/registered/unit/server_args/test_lora_kv_b_chunked_prefix.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups import lora_hook
from sglang.srt.arg_groups.lora_hook import check_lora_server_args
from sglang.srt.arg_groups.overrides import resolution_result
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


def _declared(args) -> bool:
    return any(
        "disable_chunked_prefix_cache" in declared
        for _, declared in getattr(args, "_resolved_overrides", None) or ()
    )


class TestKvBLoraPrefixCacheHook(CustomTestCase):
    def test_trtllm_mla_disables_chunked_prefix_cache(self):
        args = _args(attention_backend="trtllm_mla")
        check_lora_server_args(args)
        self.assertTrue(resolution_result(args, "disable_chunked_prefix_cache"))

    def test_split_prefill_backend_decides(self):
        args = _args(attention_backend="dsa", prefill_attention_backend="trtllm_mla")
        check_lora_server_args(args)
        self.assertTrue(resolution_result(args, "disable_chunked_prefix_cache"))
        args = _args(attention_backend="trtllm_mla", prefill_attention_backend="fa3")
        check_lora_server_args(args)
        self.assertFalse(_declared(args))

    def test_other_backends_are_left_alone(self):
        # fa3/flashinfer/flashmla/fa4 reroute only above a batch prefix threshold,
        # dsa/nsa (GLM-5.x, DeepSeek-V3.2) and aiter route cached prefixes on their
        # own: all of them launch and route exactly as before the rule existed.
        for on_sm100 in (True, False):
            with patch.object(
                lora_hook, "get_platform", return_value=_platform(on_sm100)
            ):
                for backend in (
                    "fa3",
                    "flashinfer",
                    "flashmla",
                    "fa4",
                    "dsa",
                    "nsa",
                    "aiter",
                    "triton",
                    None,
                ):
                    with self.subTest(backend=backend, on_sm100=on_sm100):
                        args = _args(attention_backend=backend)
                        check_lora_server_args(args)
                        self.assertFalse(_declared(args))

    def test_kv_cache_sharding_is_left_alone(self):
        args = _args(attention_backend="trtllm_mla", enable_kv_cache_sharding=True)
        check_lora_server_args(args)
        self.assertFalse(_declared(args))

    def test_two_batch_overlap_still_launches_with_lora(self):
        check_lora_server_args(
            _args(attention_backend="fa3", enable_two_batch_overlap=True)
        )

    def test_explicit_flag_is_left_alone(self):
        args = _args(attention_backend="trtllm_mla", disable_chunked_prefix_cache=True)
        check_lora_server_args(args)
        self.assertTrue(resolution_result(args, "disable_chunked_prefix_cache"))
        self.assertFalse(_declared(args))

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
                self.assertFalse(_declared(args))

    def test_no_lora_keeps_chunked_prefix_cache(self):
        args = ServerArgs(model_path="dummy", attention_backend="trtllm_mla")
        check_lora_server_args(args)
        self.assertFalse(_declared(args))

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
                self.assertEqual(_declared(args), expected)

    def test_inferred_targets_are_not_covered(self):
        args = ServerArgs(
            model_path="dummy",
            attention_backend="trtllm_mla",
            lora_paths=["adapter=/tmp/adapter"],
        )
        check_lora_server_args(args)
        self.assertFalse(_declared(args))


if __name__ == "__main__":
    unittest.main()
