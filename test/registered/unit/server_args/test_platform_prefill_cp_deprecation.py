"""Validate platform-specific prefill CP policy."""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.arg_groups.deepseek_v4_hook import (
    validate_deepseek_v4_cp,
    validate_deepseek_v41_features,
)
from sglang.srt.arg_groups.parallel_hook import (
    handle_context_parallelism,
    validate_prefill_cp_platform,
)
from sglang.srt.layers.cp.base import init_cp_strategy
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    default_cuda_graph_config,
    with_phase,
)
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_ENV_GATE = "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate"


def _cp_args(arch="DeepseekV4ForCausalLM", model_type="deepseek_v4", **overrides):
    args = ServerArgs(
        **{
            "model_path": "local-deepseek-v4",
            "enable_prefill_cp": True,
            "cp_strategy": "interleave",
            "tp_size": 2,
            "attention_backend": "dsv4",
            **overrides,
        }
    )
    args._model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=[arch], model_type=model_type),
        hf_text_config=SimpleNamespace(model_type=model_type),
        is_multimodal=False,
    )
    return args


class TestPlatformPrefillCPPolicy(CustomTestCase):
    def tearDown(self):
        init_cp_strategy(enable_prefill_cp=False, cp_size=1, cp_strategy="interleave")

    def test_platform_cp_rejected_before_model_lookup(self):
        for platform in ("is_musa",):
            facts = dict(is_hip=False, is_npu=False, is_musa=False)
            facts[platform] = True
            for strategy in (None, "zigzag", "interleave"):
                with self.subTest(platform=platform, strategy=strategy):
                    with override_platform(**facts):
                        args = ServerArgs(
                            model_path="missing-model-must-not-be-loaded",
                            enable_prefill_cp=True,
                            cp_strategy=strategy,
                        )
                        with self.assertRaisesRegex(ValueError, "deprecated.*refactor"):
                            validate_prefill_cp_platform(args)

    def test_context_parallel_handler_rejects_before_model_lookup(self):
        for platform in ("is_musa",):
            facts = dict(is_hip=False, is_npu=False, is_musa=False)
            facts[platform] = True
            with self.subTest(platform=platform), override_platform(**facts):
                args = ServerArgs(
                    model_path="missing-model-must-not-be-loaded",
                    enable_prefill_cp=True,
                    cp_strategy="interleave",
                )
                with self.assertRaisesRegex(ValueError, "deprecated.*refactor"):
                    handle_context_parallelism(args)

    def test_resolution_rejects_even_dummy_models(self):
        for platform in ("is_musa",):
            facts = dict(is_hip=False, is_npu=False, is_musa=False)
            facts[platform] = True
            for model_path in ("dummy", "none", "missing-model-must-not-be-loaded"):
                with self.subTest(platform=platform, model_path=model_path):
                    with override_platform(**facts):
                        args = ServerArgs(
                            model_path=model_path,
                            enable_prefill_cp=True,
                            cp_strategy="interleave",
                        )
                        with self.assertRaisesRegex(ValueError, "deprecated.*refactor"):
                            args.resolve_once()

    @override_platform(is_hip=True, is_npu=False, is_musa=False)
    def test_hip_context_parallel_is_deepseek_v4_only(self):
        handle_context_parallelism(_cp_args(attn_cp_size=2))
        args = _cp_args("DeepseekV3ForCausalLM", "deepseek_v3")
        with self.assertRaisesRegex(ValueError, "only supported.*DeepseekV4"):
            handle_context_parallelism(args)

    @override_platform(is_hip=True, is_npu=False, is_musa=False)
    def test_hip_deepseek_v4_cp_rejections(self):
        cases = (
            ("dsv4.*both phases", dict(prefill_attention_backend="flashinfer")),
            ("support multiple nodes", dict(nnodes=2)),
            ("support DeepSeek-V4.1", dict(model_type="deepseek_v41")),
            ("bounded-replay", dict(enable_decoder_swa_bounded_replay=True)),
            ("two-batch-overlap", dict(enable_two_batch_overlap=True)),
            ("fp8 unified_kv", dict(fp8=True)),
        )
        for regex, overrides in cases:
            overrides = dict(overrides)
            fp8 = overrides.pop("fp8", False)
            args = _cp_args(**overrides)
            with (
                self.subTest(regex=regex),
                mock.patch(f"{_ENV_GATE}.is_unified_kv_fp8", return_value=fp8),
                self.assertRaisesRegex(ValueError, regex),
            ):
                validate_deepseek_v4_cp(args)

    @override_platform(is_hip=False, is_npu=False, is_musa=False)
    def test_cuda_deepseek_v4_cp_allows_multiple_nodes(self):
        validate_deepseek_v4_cp(_cp_args(nnodes=2))

    def test_non_cp_and_decode_cp_are_not_rejected(self):
        for platform in ("is_hip", "is_npu", "is_musa"):
            facts = dict(is_hip=False, is_npu=False, is_musa=False)
            facts[platform] = True
            for dcp_size in (1, 2):
                with self.subTest(platform=platform, dcp_size=dcp_size):
                    with override_platform(**facts):
                        args = ServerArgs(model_path="dummy", dcp_size=dcp_size)
                        validate_prefill_cp_platform(args)

    def test_generic_cp_is_not_rejected_or_modified(self):
        for is_hip in (False, True):
            for strategy in ("zigzag", "interleave"):
                with (
                    self.subTest(is_hip=is_hip, strategy=strategy),
                    override_platform(is_hip=is_hip, is_npu=False, is_musa=False),
                ):
                    args = ServerArgs(
                        model_path="dummy", enable_prefill_cp=True, cp_strategy=strategy
                    )
                    validate_prefill_cp_platform(args)
                    self.assertTrue(args.enable_prefill_cp)
                    self.assertEqual(args.cp_strategy, strategy)


class TestDecoderBoundedReplayPrefillGraph(CustomTestCase):
    def _args(self, backend, **overrides):
        args = _cp_args(
            "DeepseekV41ForCausalLM",
            "deepseek_v41",
            enable_prefill_cp=False,
            enable_decoder_swa_bounded_replay=True,
            cuda_graph_config=with_phase(
                default_cuda_graph_config(),
                Phase.PREFILL,
                backend=backend,
                max_seq_len=16384,
            ),
            **overrides,
        )
        return args

    @override_platform(is_hip=False, is_npu=False, is_musa=False)
    def test_only_the_breakable_prefill_graph_is_admitted(self):
        """Only the breakable graph re-runs the switch onto the late-layer tail at
        replay; any other captured prefill would run the late layers on stale rows."""
        for backend in (Backend.DISABLED, Backend.BREAKABLE):
            with self.subTest(backend=backend):
                validate_deepseek_v41_features(self._args(backend))
        for backend in (Backend.FULL, Backend.TC_PIECEWISE):
            with (
                self.subTest(backend=backend),
                self.assertRaisesRegex(ValueError, f"{backend} prefill CUDA graph"),
            ):
                validate_deepseek_v41_features(self._args(backend))

    def test_breakable_prefill_graph_is_rejected_where_the_tail_is_not_captured(self):
        with (
            override_platform(is_hip=True, is_npu=False, is_musa=False),
            self.assertRaisesRegex(ValueError, "on ROCm"),
        ):
            validate_deepseek_v41_features(self._args(Backend.BREAKABLE))
        with (
            override_platform(is_hip=False, is_npu=False, is_musa=False),
            self.assertRaisesRegex(ValueError, "under context parallelism"),
        ):
            validate_deepseek_v41_features(
                self._args(Backend.BREAKABLE, attn_cp_size=2)
            )


if __name__ == "__main__":
    unittest.main()
