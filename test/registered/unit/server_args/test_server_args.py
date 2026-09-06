import argparse
import dataclasses
import json
import os
import socket
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import sglang.srt.server_args as server_args_module
from sglang.srt.arg_groups import parallel_hook, pd_disaggregation_hook, serving_hook
from sglang.srt.arg_groups.attention_hook import (
    handle_attention_backend_compatibility,
    handle_deterministic_inference,
)
from sglang.srt.arg_groups.cuda_graph_hook import (
    apply_cuda_graph_compatibility,
    disable_tc_piecewise_cudagraph_if_incompatible,
    handle_cuda_graph_config,
)
from sglang.srt.arg_groups.hicache_hook import (
    handle_hicache,
    handle_hicache_ratio_default,
)
from sglang.srt.arg_groups.hisparse_hook import (
    validate_hisparse_dsa_backend,
    validate_hisparse_kv_cache_dtype,
)
from sglang.srt.arg_groups.kv_cache_hook import (
    handle_cache_compatibility,
    validate_prefill_only_disable_kv_cache_args,
)
from sglang.srt.arg_groups.mamba_hook import handle_mamba_backend
from sglang.srt.arg_groups.memory_hook import handle_gpu_memory_settings
from sglang.srt.arg_groups.model_path_hook import handle_load_format
from sglang.srt.arg_groups.moe_hook import (
    handle_a2a_moe,
    validate_deepep_v2_dispatch_token_budget,
    validate_deepep_v2_speculative_draft,
)
from sglang.srt.arg_groups.overrides import (
    cutedsl_moe_max_num_tokens,
    max_speculative_num_draft_tokens,
    resolution_result,
)
from sglang.srt.arg_groups.parallel_hook import (
    handle_context_parallelism,
    handle_data_parallelism,
    handle_legacy_cp_runtime_compatibility,
    handle_platform_cp_compatibility,
)
from sglang.srt.arg_groups.pd_disaggregation_hook import handle_pd_disaggregation
from sglang.srt.arg_groups.serving_hook import (
    handle_crash_dump_env,
    handle_deprecated_args,
    handle_load_balance_method,
    handle_missing_default_values,
    handle_multimodal_feature_transport,
    handle_ssl_validation,
    handle_tokenizer_batching,
    ssl_verify_of,
)
from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.arg_groups.validation_hook import (
    check_two_batch_overlap,
)
from sglang.srt.entrypoints.sidecar import (
    SGLANG_GRPC_ENDPOINT_ENV,
    Sidecar,
    _run_sidecar,
    build_sidecar_endpoint,
    start_sidecar,
)
from sglang.srt.environ import envs
from sglang.srt.layers.cp.base import is_cp_enabled, is_interleave
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    Phase,
    PhaseConfig,
)
from sglang.srt.runtime_context import (
    describe_kv_events_publisher,
    get_context,
    get_serving,
    override_platform,
)
from sglang.srt.server_args import PortArgs, ServerArgs, prepare_server_args
from sglang.srt.utils.server_args_config_parser import ConfigArgumentMerger
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    CustomTestCase,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cpu_ci(est_time=11, suite="stage-b-test-cpu-intel")

# Mock get_device() so all tests run on CPU-only CI runners
_mock_device = patch(
    "sglang.srt.arg_groups.serving_hook.get_device", return_value="cuda"
)
_mock_device.start()


class TestPrepareServerArgs(CustomTestCase):
    def test_weight_cache_daemon_allows_static_eplb(self):
        args = ServerArgs(
            model_path="dummy",
            weight_cache_mode="daemon",
            enable_eplb=True,
        )

        # This validation runs before model construction and should allow the
        # daemon to build the same static EPLB layout as the engine.
        handle_load_format(args)

    def test_enable_w4a4_mxfp4_megamoe_preserves_legacy_deepgemm_env(self):
        deepgemm_env = {
            "DG_USE_FP4_ACTS": "0",
            "DG_USE_MXF4_KIND": "0",
        }
        with patch.dict(os.environ, deepgemm_env, clear=False):
            try:
                args = prepare_server_args(
                    ["--model-path", "dummy", "--enable-w4a4-mxfp4-megamoe"]
                )
            except SystemExit as exc:
                self.fail(
                    "--enable-w4a4-mxfp4-megamoe must be accepted by the CLI "
                    f"parser, got SystemExit({exc.code})"
                )

            args.resolve_once()

            self.assertTrue(resolution_result(args, "enable_w4a4_mxfp4_megamoe"))
            self.assertEqual(os.environ["DG_USE_FP4_ACTS"], "0")
            self.assertEqual(os.environ["DG_USE_MXF4_KIND"], "0")

    def test_w4a4_mxfp4_megamoe_disabled_preserves_deepgemm_env(self):
        deepgemm_env = {
            "DG_USE_FP4_ACTS": "0",
            "DG_USE_MXF4_KIND": "0",
        }
        with patch.dict(os.environ, deepgemm_env, clear=False):
            args = prepare_server_args(["--model-path", "dummy"])
            # Resolve, or the check that the environment stays untouched has
            # nothing to be untouched by.
            args.resolve_once()

            self.assertFalse(resolution_result(args, "enable_w4a4_mxfp4_megamoe"))
            self.assertEqual(os.environ["DG_USE_FP4_ACTS"], "0")
            self.assertEqual(os.environ["DG_USE_MXF4_KIND"], "0")

    def test_prefill_decode_interval(self):
        args = ServerArgs(model_path="dummy", prefill_decode_interval=16)
        args.resolve_once()
        self.assertEqual(resolution_result(args, "prefill_decode_interval"), 16)

        with self.assertRaisesRegex(
            ValueError, "--prefill-decode-interval must be non-negative"
        ):
            ServerArgs(model_path="dummy", prefill_decode_interval=-1).resolve_once()

    def test_dsv4_prefill_backend_cli_choices(self):
        parser = server_args_module.argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        base_args = ["--model-path", "dummy-model"]

        default_args = parser.parse_args(base_args)
        self.assertEqual(default_args.dsv4_prefill_backend, "auto")

        q8_args = parser.parse_args(
            base_args + ["--dsv4-prefill-backend", "flashmla_sparse_q8"]
        )
        self.assertEqual(q8_args.dsv4_prefill_backend, "flashmla_sparse_q8")

        with self.assertRaises(SystemExit):
            parser.parse_args(base_args + ["--dsv4-prefill-backend", "flashmla_kv"])

    def test_return_hidden_states_mode_configuration(self):
        def _resolved(**kwargs):
            server_args = ServerArgs(**kwargs)
            server_args.resolve_once()
            return server_args

        disabled = _resolved(model_path="dummy")
        self.assertFalse(resolution_result(disabled, "enable_return_hidden_states"))
        self.assertIsNone(resolution_result(disabled, "return_hidden_states_mode"))

        last = _resolved(
            model_path="dummy",
            return_hidden_states_mode="last",
        )
        self.assertTrue(resolution_result(last, "enable_return_hidden_states"))
        self.assertEqual(resolution_result(last, "return_hidden_states_mode"), "last")

        legacy_full = _resolved(
            model_path="dummy",
            enable_return_hidden_states=True,
        )
        self.assertTrue(resolution_result(legacy_full, "enable_return_hidden_states"))
        self.assertEqual(
            resolution_result(legacy_full, "return_hidden_states_mode"), "full"
        )

        parsed_last = prepare_server_args(
            [
                "--model-path",
                "dummy",
                "--return-hidden-states-mode",
                "last",
            ]
        )
        parsed_last.resolve_once()
        self.assertTrue(resolution_result(parsed_last, "enable_return_hidden_states"))
        self.assertEqual(
            resolution_result(parsed_last, "return_hidden_states_mode"), "last"
        )

        # The rejection is resolution's, not the constructor's.
        with self.assertRaisesRegex(
            ValueError,
            "return_hidden_states_mode must be one of",
        ):
            _resolved(
                model_path="dummy",
                return_hidden_states_mode="lst",
            )

    def test_draft_quantization_explicitness_survives_asdict_round_trip(self):
        inherited = ServerArgs(model_path="dummy", quantization="modelopt_fp4")
        handle_missing_default_values(inherited)
        self.assertEqual(
            resolution_result(inherited, "speculative_draft_model_quantization"),
            "modelopt_fp4",
        )
        self.assertFalse(
            resolution_result(
                inherited, "_speculative_draft_quantization_explicitly_set"
            )
        )

        reconstructed = ServerArgs(**dataclasses.asdict(inherited))
        handle_missing_default_values(reconstructed)

        self.assertFalse(
            resolution_result(
                reconstructed, "_speculative_draft_quantization_explicitly_set"
            )
        )

    def test_config_nested_dict_args_are_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("mm-process-config:\n  image:\n    resize: 128\n")
            config_file = f.name

        try:
            parser = server_args_module.argparse.ArgumentParser()
            ServerArgs.add_cli_args(parser)
            merged = ConfigArgumentMerger(parser).merge_config_with_args(
                [
                    "--config",
                    config_file,
                    "--model-path",
                    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
                ]
            )
            value = merged[merged.index("--mm-process-config") + 1]
            parsed = parser.parse_args(merged)

            self.assertEqual(json.loads(value), {"image": {"resize": 128}})
            self.assertEqual(parsed.mm_process_config, {"image": {"resize": 128}})
        finally:
            os.unlink(config_file)


class TestMmEncoderDataParallelLogging(CustomTestCase):
    def test_logs_when_encoder_dp_has_no_parallelism(self):
        server_args = ServerArgs(
            model_path="dummy", mm_enable_dp_encoder=True, tp_size=1
        )

        with self.assertLogs(parallel_hook.logger, level="WARNING") as logs:
            handle_data_parallelism(server_args)

        self.assertIn("TP=1", logs.output[0])
        self.assertIn("no data-parallel work", logs.output[0])

    def test_logs_encoder_dp_tradeoff_for_tp(self):
        server_args = ServerArgs(
            model_path="dummy", mm_enable_dp_encoder=True, tp_size=4
        )

        with self.assertLogs(parallel_hook.logger, level="INFO") as logs:
            handle_data_parallelism(server_args)

        self.assertIn("TP=4", logs.output[0])
        self.assertIn("high-resolution or multi-image", logs.output[0])


class TestImageProcessorBackend(CustomTestCase):
    def test_new_backend_does_not_set_legacy_flag(self):
        server_args = ServerArgs(model_path="dummy", image_processor_backend="pil")

        handle_deprecated_args(server_args)

        self.assertEqual(
            resolution_result(server_args, "image_processor_backend"), "pil"
        )
        self.assertFalse(resolution_result(server_args, "disable_fast_image_processor"))

    def test_legacy_flag_maps_to_pil_with_one_warning(self):
        server_args = ServerArgs(model_path="dummy", disable_fast_image_processor=True)

        with self.assertLogs(serving_hook.logger, level="WARNING") as logs:
            handle_deprecated_args(server_args)

        self.assertEqual(
            resolution_result(server_args, "image_processor_backend"), "pil"
        )
        self.assertTrue(resolution_result(server_args, "disable_fast_image_processor"))
        self.assertEqual(
            sum(
                "--disable-fast-image-processor is deprecated" in x for x in logs.output
            ),
            1,
        )

    def test_legacy_flag_rejects_torchvision_backend(self):
        server_args = ServerArgs(
            model_path="dummy",
            image_processor_backend="torchvision",
            disable_fast_image_processor=True,
        )

        with self.assertRaisesRegex(ValueError, "conflicts.*torchvision"):
            handle_deprecated_args(server_args)


class TestMultimodalFeatureTransport(CustomTestCase):
    @staticmethod
    def _set_model_type(server_args, *, is_multimodal):
        server_args._model_config = SimpleNamespace(is_multimodal=is_multimodal)

    @override_platform(is_cuda=True)
    def test_cuda_ipc_is_explicit_and_bounded(self):
        server_args = ServerArgs(
            model_path="dummy",
            mm_feature_transport="cuda_ipc",
            tokenizer_worker_num=4,
            base_gpu_id=2,
        )

        with patch.dict(os.environ, {"SGLANG_USE_CUDA_IPC_TRANSPORT": "0"}):
            with self.assertLogs(serving_hook.logger, level="INFO") as logs:
                handle_multimodal_feature_transport(server_args)

            self.assertEqual(
                resolution_result(server_args, "mm_feature_transport"), "cuda_ipc"
            )
            self.assertTrue(envs.SGLANG_USE_CUDA_IPC_TRANSPORT.get())

        output = "\n".join(logs.output)
        self.assertIn("base GPU 2", output)
        self.assertIn("4 tokenizer worker", output)

    @override_platform(is_cuda=True)
    def test_legacy_keep_flag_maps_to_cuda_ipc(self):
        server_args = ServerArgs(model_path="dummy", keep_mm_feature_on_device=True)

        with patch.dict(os.environ, {"SGLANG_USE_CUDA_IPC_TRANSPORT": "0"}):
            with self.assertLogs(serving_hook.logger, level="WARNING") as logs:
                handle_multimodal_feature_transport(server_args)

            self.assertEqual(
                resolution_result(server_args, "mm_feature_transport"), "cuda_ipc"
            )
            self.assertFalse(
                resolution_result(server_args, "keep_mm_feature_on_device")
            )
            self.assertTrue(envs.SGLANG_USE_CUDA_IPC_TRANSPORT.get())

        self.assertIn("deprecated", logs.output[0])

    def test_legacy_keep_flag_rejects_explicit_cuda_vmm(self):
        server_args = ServerArgs(
            model_path="dummy",
            keep_mm_feature_on_device=True,
            mm_feature_transport="cuda_vmm",
        )

        with self.assertRaisesRegex(ValueError, "conflicts.*cuda_vmm"):
            handle_multimodal_feature_transport(server_args)

    @override_platform(is_cuda=True)
    def test_explicit_cpu_overrides_legacy_environment(self):
        server_args = ServerArgs(model_path="dummy", mm_feature_transport="cpu")

        with patch.dict(os.environ, {"SGLANG_USE_CUDA_IPC_TRANSPORT": "1"}):
            with self.assertLogs(serving_hook.logger, level="WARNING") as logs:
                handle_multimodal_feature_transport(server_args)

            self.assertEqual(
                resolution_result(server_args, "mm_feature_transport"), "cpu"
            )
            self.assertFalse(envs.SGLANG_USE_CUDA_IPC_TRANSPORT.get())

        self.assertIn("overrides", logs.output[0])

    def test_default_transport_is_cpu(self):
        server_args = ServerArgs(model_path="dummy")

        with patch.dict(os.environ, {"SGLANG_USE_CUDA_IPC_TRANSPORT": "0"}):
            handle_multimodal_feature_transport(server_args)

            self.assertEqual(
                resolution_result(server_args, "mm_feature_transport"), "cpu"
            )
            self.assertFalse(envs.SGLANG_USE_CUDA_IPC_TRANSPORT.get())

    @override_platform(is_cuda=True)
    def test_default_transport_is_cpu_for_text_only_model(self):
        server_args = ServerArgs(model_path="dummy")
        self._set_model_type(server_args, is_multimodal=False)

        with patch.dict(os.environ, {}, clear=False):
            envs.SGLANG_USE_CUDA_IPC_TRANSPORT.clear()
            with self.assertNoLogs(server_args_module.logger, level="INFO"):
                handle_multimodal_feature_transport(server_args)

            self.assertEqual(
                resolution_result(server_args, "mm_feature_transport"), "cpu"
            )
            self.assertFalse(envs.SGLANG_USE_CUDA_IPC_TRANSPORT.get())

    @override_platform(is_cuda=True)
    def test_default_transport_is_cpu_for_multimodal_model(self):
        server_args = ServerArgs(model_path="dummy")
        self._set_model_type(server_args, is_multimodal=True)

        with patch.dict(os.environ, {}, clear=False):
            envs.SGLANG_USE_CUDA_IPC_TRANSPORT.clear()
            with self.assertNoLogs(server_args_module.logger, level="INFO"):
                handle_multimodal_feature_transport(server_args)

            self.assertEqual(
                resolution_result(server_args, "mm_feature_transport"), "cpu"
            )
            self.assertFalse(envs.SGLANG_USE_CUDA_IPC_TRANSPORT.get())

    @patch("sglang.srt.arg_groups.serving_hook.os.path.exists", return_value=True)
    @patch(
        "sglang.srt.arg_groups.serving_hook.is_mnnvl_fabric_device", return_value=True
    )
    @override_platform(is_cuda=True)
    @patch(
        "sglang.srt.model_loader.utils.supports_cuda_vmm_feature_transport",
        return_value=True,
    )
    def test_default_transport_is_cuda_vmm_for_supported_multinode_mnnvl(
        self, _mock_supports_cuda_vmm, _mock_is_cuda, _mock_is_mnnvl
    ):
        server_args = ServerArgs(model_path="dummy", nnodes=2)
        self._set_model_type(server_args, is_multimodal=True)

        with patch.dict(os.environ, {}, clear=False):
            envs.SGLANG_USE_CUDA_IPC_TRANSPORT.clear()
            with self.assertLogs(serving_hook.logger, level="INFO") as logs:
                handle_multimodal_feature_transport(server_args)

            self.assertEqual(
                resolution_result(server_args, "mm_feature_transport"), "cuda_vmm"
            )
            self.assertFalse(envs.SGLANG_USE_CUDA_IPC_TRANSPORT.get())

        output = "\n".join(logs.output)
        self.assertIn("auto-resolved to cuda_vmm", output)
        self.assertIn("CUDA FABRIC", output)

    @patch("sglang.srt.arg_groups.serving_hook.os.path.exists", return_value=True)
    @patch(
        "sglang.srt.arg_groups.serving_hook.is_mnnvl_fabric_device", return_value=True
    )
    @override_platform(is_cuda=True)
    @patch(
        "sglang.srt.model_loader.utils.supports_cuda_vmm_feature_transport",
        return_value=False,
    )
    def test_default_transport_is_cpu_for_unsupported_multinode_model(
        self, _mock_supports_cuda_vmm, _mock_is_cuda, _mock_is_mnnvl
    ):
        server_args = ServerArgs(model_path="dummy", nnodes=2)
        self._set_model_type(server_args, is_multimodal=True)

        with self.assertLogs(serving_hook.logger, level="INFO") as logs:
            handle_multimodal_feature_transport(server_args)

        self.assertEqual(resolution_result(server_args, "mm_feature_transport"), "cpu")
        self.assertIn("has not opted into CUDA VMM", "\n".join(logs.output))

    @patch("sglang.srt.arg_groups.serving_hook.os.path.exists", return_value=False)
    @patch(
        "sglang.srt.arg_groups.serving_hook.is_mnnvl_fabric_device", return_value=True
    )
    @override_platform(is_cuda=True)
    def test_default_transport_is_cpu_without_imex_channel(
        self, _mock_is_cuda, _mock_is_mnnvl
    ):
        server_args = ServerArgs(model_path="dummy", nnodes=2)
        self._set_model_type(server_args, is_multimodal=True)

        with patch.dict(os.environ, {}, clear=False):
            envs.SGLANG_USE_CUDA_IPC_TRANSPORT.clear()
            with self.assertLogs(serving_hook.logger, level="INFO") as logs:
                handle_multimodal_feature_transport(server_args)

            self.assertEqual(
                resolution_result(server_args, "mm_feature_transport"), "cpu"
            )

        self.assertIn("no IMEX channel", "\n".join(logs.output))

    @patch(
        "sglang.srt.arg_groups.serving_hook.is_mnnvl_fabric_device", return_value=False
    )
    @override_platform(is_cuda=True)
    def test_default_transport_is_cpu_for_multinode_non_mnnvl(self, _mock_is_cuda):
        server_args = ServerArgs(model_path="dummy", nnodes=2)
        self._set_model_type(server_args, is_multimodal=True)

        with patch.dict(os.environ, {}, clear=False):
            envs.SGLANG_USE_CUDA_IPC_TRANSPORT.clear()
            handle_multimodal_feature_transport(server_args)

            self.assertEqual(
                resolution_result(server_args, "mm_feature_transport"), "cpu"
            )
            self.assertFalse(envs.SGLANG_USE_CUDA_IPC_TRANSPORT.get())

    @override_platform(is_cuda=True)
    def test_default_transport_is_cpu_for_language_only_model(self):
        server_args = ServerArgs(model_path="dummy", language_only=True)
        self._set_model_type(server_args, is_multimodal=True)

        with patch.dict(os.environ, {}, clear=False):
            envs.SGLANG_USE_CUDA_IPC_TRANSPORT.clear()
            handle_multimodal_feature_transport(server_args)

            self.assertEqual(
                resolution_result(server_args, "mm_feature_transport"), "cpu"
            )
            self.assertFalse(envs.SGLANG_USE_CUDA_IPC_TRANSPORT.get())

    @override_platform(is_cuda=False)
    def test_cuda_ipc_rejects_non_nvidia_platforms(self):
        server_args = ServerArgs(model_path="dummy", mm_feature_transport="cuda_ipc")

        with self.assertRaisesRegex(ValueError, "requires NVIDIA CUDA"):
            handle_multimodal_feature_transport(server_args)

    @override_platform(is_cuda=True)
    def test_cuda_ipc_rejects_multi_node(self):
        server_args = ServerArgs(
            model_path="dummy", mm_feature_transport="cuda_ipc", nnodes=2
        )

        with self.assertRaisesRegex(ValueError, "single node"):
            handle_multimodal_feature_transport(server_args)

    @override_platform(is_cuda=True)
    def test_cuda_vmm_is_explicit_and_uses_shared_budget(self):
        server_args = ServerArgs(
            model_path="dummy",
            mm_feature_transport="cuda_vmm",
            nnodes=2,
            tokenizer_worker_num=2,
        )

        with (
            patch.dict(os.environ, {"SGLANG_USE_CUDA_IPC_TRANSPORT": "1"}),
            envs.SGLANG_MM_FEATURE_CACHE_MB.override(256),
        ):
            with self.assertLogs(serving_hook.logger, level="INFO") as logs:
                handle_multimodal_feature_transport(server_args)

            self.assertEqual(
                resolution_result(server_args, "mm_feature_transport"), "cuda_vmm"
            )
            self.assertFalse(envs.SGLANG_USE_CUDA_IPC_TRANSPORT.get())

        output = "\n".join(logs.output)
        self.assertIn("CUDA FABRIC", output)
        self.assertIn("256 MiB", output)
        self.assertIn("2 tokenizer worker", output)
        self.assertIn("falls back to inline CPU", output)

    @override_platform(is_cuda=False)
    def test_cuda_vmm_rejects_non_nvidia_platforms(self):
        server_args = ServerArgs(model_path="dummy", mm_feature_transport="cuda_vmm")

        with self.assertRaisesRegex(ValueError, "requires NVIDIA CUDA"):
            handle_multimodal_feature_transport(server_args)

    @override_platform(is_cuda=True)
    def test_cuda_vmm_rejects_rust_server(self):
        server_args = ServerArgs(model_path="dummy", mm_feature_transport="cuda_vmm")

        with (
            envs.SGLANG_RUST_SERVER.override(True),
            self.assertRaisesRegex(ValueError, "SGLANG_RUST_SERVER"),
        ):
            handle_multimodal_feature_transport(server_args)

    @override_platform(is_cuda=True)
    def test_cuda_vmm_rejects_pipeline_parallelism(self):
        server_args = ServerArgs(
            model_path="dummy", mm_feature_transport="cuda_vmm", pp_size=2
        )

        with self.assertRaisesRegex(ValueError, "pipeline parallelism"):
            handle_multimodal_feature_transport(server_args)


class TestMambaCacheStochasticRounding(unittest.TestCase):
    def test_rejects_fp32_ssm_cache(self):
        server_args = ServerArgs(
            model_path="dummy",
            mamba_ssm_dtype="float32",
            enable_mamba_cache_stochastic_rounding=True,
        )

        with self.assertRaisesRegex(ValueError, "--mamba-ssm-dtype float16"):
            handle_mamba_backend(server_args)

    @override_platform(is_cuda=False)
    def test_rejects_non_cuda(self):
        server_args = ServerArgs(
            model_path="dummy",
            mamba_ssm_dtype="float16",
            enable_mamba_cache_stochastic_rounding=True,
        )

        with self.assertRaisesRegex(ValueError, "NVIDIA CUDA"):
            handle_mamba_backend(server_args)

    @override_platform(is_cuda=True)
    @override_platform(is_sm100=False)
    def test_rejects_triton_without_sm100(self):
        server_args = ServerArgs(
            model_path="dummy",
            mamba_ssm_dtype="float16",
            mamba_backend="triton",
            enable_mamba_cache_stochastic_rounding=True,
        )

        with self.assertRaisesRegex(ValueError, "requires SM100"):
            handle_mamba_backend(server_args)


class TestLoadBalanceMethod(unittest.TestCase):
    def _load_balance_args(self, **kwargs):
        server_args = ServerArgs(model_path="dummy", **kwargs)
        handle_pd_disaggregation(server_args)
        handle_load_balance_method(server_args)
        return server_args

    def test_non_pd_defaults_to_round_robin(self):
        server_args = self._load_balance_args(disaggregation_mode="null")
        self.assertEqual(
            resolution_result(server_args, "load_balance_method"), "round_robin"
        )

    def test_pd_prefill_defaults_to_follow_bootstrap_room(self):
        server_args = self._load_balance_args(disaggregation_mode="prefill")
        self.assertEqual(
            resolution_result(server_args, "load_balance_method"),
            "follow_bootstrap_room",
        )

    def test_pd_decode_defaults_to_round_robin(self):
        server_args = self._load_balance_args(disaggregation_mode="decode")
        self.assertEqual(
            resolution_result(server_args, "load_balance_method"), "round_robin"
        )

    def test_pd_prefill_dcp_warns_about_performance(self):
        server_args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="prefill",
            dcp_size=4,
        )
        with self.assertLogs(pd_disaggregation_hook.logger, level="WARNING") as logs:
            handle_pd_disaggregation(server_args)
        self.assertIn("without improving prefill performance", "\n".join(logs.output))

    def test_pd_decode_dcp_forces_chunk_cache(self):
        server_args = self._load_balance_args(
            disaggregation_mode="decode",
            disaggregation_transfer_backend="mooncake",
            dcp_size=4,
        )
        self.assertTrue(resolution_result(server_args, "disable_radix_cache"))

    def test_pd_decode_dcp_rejects_unsupported_transfer_backend(self):
        server_args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="decode",
            disaggregation_transfer_backend="mori",
            dcp_size=4,
        )
        with self.assertRaisesRegex(
            ValueError, "mooncake, nixl, or fake for synthetic benchmarking"
        ):
            handle_pd_disaggregation(server_args)

    def test_pd_decode_dcp_allows_fake_transfer_backend(self):
        server_args = self._load_balance_args(
            disaggregation_mode="decode",
            disaggregation_transfer_backend="fake",
            dcp_size=4,
        )
        self.assertTrue(resolution_result(server_args, "disable_radix_cache"))

    def test_pd_decode_dcp_rejects_radix_cache(self):
        server_args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="decode",
            disaggregation_transfer_backend="nixl",
            disaggregation_decode_enable_radix_cache=True,
            dcp_size=4,
        )
        with self.assertRaisesRegex(ValueError, "currently requires chunk cache"):
            handle_pd_disaggregation(server_args)

    def test_pd_decode_dcp_rejects_hierarchical_cache(self):
        server_args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="decode",
            disaggregation_transfer_backend="nixl",
            enable_hierarchical_cache=True,
            dcp_size=4,
        )
        with self.assertRaisesRegex(ValueError, "--enable-hierarchical-cache"):
            handle_pd_disaggregation(server_args)

    def test_pd_decode_radix_cache_rejects_hisparse(self):
        server_args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="decode",
            disaggregation_decode_enable_radix_cache=True,
            disaggregation_transfer_backend="nixl",
            enable_hisparse=True,
        )
        with self.assertRaises(ValueError) as context:
            handle_pd_disaggregation(server_args)

        self.assertIn(
            "--disaggregation-decode-enable-radix-cache is incompatible with "
            "--enable-hisparse",
            str(context.exception),
        )

    def test_pd_decode_radix_cache_rejects_fake_backend(self):
        server_args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="decode",
            disaggregation_decode_enable_radix_cache=True,
            disaggregation_transfer_backend="fake",
        )
        with self.assertRaises(ValueError) as context:
            handle_pd_disaggregation(server_args)

        self.assertIn(
            "--disaggregation-decode-enable-radix-cache is incompatible "
            "with --disaggregation-transfer-backend fake",
            str(context.exception),
        )

    def test_pd_decode_radix_cache_allows_mooncake_tcp(self):
        server_args = self._load_balance_args(
            disaggregation_mode="decode",
            disaggregation_decode_enable_radix_cache=True,
            disaggregation_transfer_backend="mooncake_tcp",
        )

        self.assertFalse(resolution_result(server_args, "disable_radix_cache"))
        self.assertEqual(
            resolution_result(server_args, "disaggregation_transfer_backend"),
            "mooncake",
        )

    def test_pd_decode_hicache_allows_rust_tree_core(self):
        server_args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="decode",
            disaggregation_decode_enable_radix_cache=True,
            disaggregation_transfer_backend="nixl",
            enable_hierarchical_cache=True,
        )
        with envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("rust"):
            handle_pd_disaggregation(server_args)

        self.assertFalse(resolution_result(server_args, "disable_radix_cache"))


class TestSkipTokenizerInit(unittest.TestCase):
    def test_skip_tokenizer_worker_counts(self):
        server_args = ServerArgs(
            model_path="dummy",
            skip_tokenizer_init=True,
            tokenizer_worker_num=4,
            detokenizer_worker_num=3,
        )

        handle_tokenizer_batching(server_args)

        # Tokenizer fanout preserved; detokenizer coerced to 1 (no decode work).
        self.assertEqual(resolution_result(server_args, "tokenizer_worker_num"), 4)
        self.assertEqual(resolution_result(server_args, "detokenizer_worker_num"), 1)


class TestHiSparseDsaBackendPolicy(unittest.TestCase):
    # The backend selection moved to the resolution pipeline; these policy
    # tests drive the pass through its read-only view.
    @staticmethod
    def _resolve(kv_cache_dtype, **kw):
        from types import SimpleNamespace

        from sglang.srt.arg_groups.overrides import (
            ResolvedView,
            _dsa_split_backend_resolution,
        )

        hf = SimpleNamespace(architectures=["DeepseekV32ForCausalLM"])
        defaults = dict(
            kv_cache_dtype=kv_cache_dtype,
            dsa_prefill_backend=None,
            dsa_decode_backend=None,
            enable_hisparse=True,
        )
        defaults.update(kw)
        view = ResolvedView(
            SimpleNamespace(_model_config=SimpleNamespace(hf_config=hf), **defaults)
        )
        with (
            patch("sglang.srt.configs.model_config.is_deepseek_dsa", return_value=True),
            override_platform(is_npu=False),
            override_platform(is_xpu=False),
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
        ):
            declared = _dsa_split_backend_resolution(view)
        return {
            "dsa_prefill_backend": declared.get(
                "dsa_prefill_backend", defaults["dsa_prefill_backend"]
            ),
            "dsa_decode_backend": declared.get(
                "dsa_decode_backend", defaults["dsa_decode_backend"]
            ),
        }

    @override_platform(is_hip=False)
    def test_hisparse_defaults_to_flashmla_sparse_on_cuda_bfloat16(self):
        resolved = self._resolve("bfloat16")

        self.assertEqual(resolved["dsa_prefill_backend"], "flashmla_sparse")
        self.assertEqual(resolved["dsa_decode_backend"], "flashmla_sparse")

    @override_platform(is_hip=False)
    def test_hisparse_defaults_to_flashmla_kv_on_cuda_fp8(self):
        resolved = self._resolve("fp8_e4m3")

        self.assertEqual(resolved["dsa_prefill_backend"], "flashmla_kv")
        self.assertEqual(resolved["dsa_decode_backend"], "flashmla_kv")

    @override_platform(is_hip=False)
    def test_hisparse_accepts_flashinfer_sparse_mla_on_cuda_fp8(self):
        """SM120 GLM DSA resolves both DSA backends to flashinfer_sparse_mla, so
        the fp8 hisparse allow-set must admit it or --enable-hisparse cannot
        start there at all. The device/arch narrowing happens later, in
        _validate_flashinfer_sparse_mla_backend."""
        server_args = ServerArgs(
            model_path="dummy",
            enable_hisparse=True,
            kv_cache_dtype="fp8_e4m3",
            dsa_prefill_backend="flashinfer_sparse_mla",
            dsa_decode_backend="flashinfer_sparse_mla",
        )

        validate_hisparse_dsa_backend(server_args, "dsa_prefill_backend", "prefill")
        validate_hisparse_dsa_backend(server_args, "dsa_decode_backend", "decode")

    @override_platform(is_hip=True)
    def test_hisparse_defaults_to_tilelang_on_rocm(self):
        resolved = self._resolve("bfloat16")

        self.assertEqual(resolved["dsa_prefill_backend"], "tilelang")
        self.assertEqual(resolved["dsa_decode_backend"], "tilelang")

    @override_platform(is_hip=True)
    def test_hisparse_preserves_rocm_user_backend_and_defaults_missing_side(self):
        resolved = self._resolve("bfloat16", dsa_prefill_backend="tilelang")

        self.assertEqual(resolved["dsa_prefill_backend"], "tilelang")
        self.assertEqual(resolved["dsa_decode_backend"], "tilelang")

    @override_platform(is_hip=True)
    def test_hisparse_accepts_aiter_backend_on_rocm(self):
        server_args = ServerArgs(
            model_path="dummy",
            enable_hisparse=True,
            kv_cache_dtype="bfloat16",
            dsa_prefill_backend="aiter",
            dsa_decode_backend="aiter",
        )

        validate_hisparse_dsa_backend(server_args, "dsa_prefill_backend", "prefill")
        validate_hisparse_dsa_backend(server_args, "dsa_decode_backend", "decode")

    @override_platform(is_hip=True)
    def test_hisparse_rejects_cuda_backend_on_rocm(self):
        server_args = ServerArgs(
            model_path="dummy",
            enable_hisparse=True,
            kv_cache_dtype="bfloat16",
            dsa_prefill_backend="flashmla_sparse",
        )

        with self.assertRaisesRegex(ValueError, "tilelang"):
            validate_hisparse_dsa_backend(server_args, "dsa_prefill_backend", "prefill")

    @override_platform(is_hip=False)
    def test_hisparse_rejects_rocm_backend_on_cuda(self):
        server_args = ServerArgs(
            model_path="dummy",
            enable_hisparse=True,
            kv_cache_dtype="bfloat16",
            dsa_decode_backend="tilelang",
        )

        with self.assertRaisesRegex(ValueError, "flashmla_sparse"):
            validate_hisparse_dsa_backend(server_args, "dsa_decode_backend", "decode")

    def test_hisparse_accepts_bfloat16_kv_cache_dtype(self):
        server_args = ServerArgs(
            model_path="dummy",
            enable_hisparse=True,
            kv_cache_dtype="bfloat16",
        )

        validate_hisparse_kv_cache_dtype(server_args)

    def test_hisparse_accepts_fp8_e4m3_kv_cache_dtype(self):
        server_args = ServerArgs(
            model_path="dummy",
            enable_hisparse=True,
            kv_cache_dtype="fp8_e4m3",
        )

        validate_hisparse_kv_cache_dtype(server_args)

    def test_hisparse_rejects_unsupported_kv_cache_dtype(self):
        server_args = ServerArgs(
            model_path="dummy",
            enable_hisparse=True,
            kv_cache_dtype="float16",
        )

        with self.assertRaisesRegex(ValueError, r"fp8_e4m3"):
            validate_hisparse_kv_cache_dtype(server_args)


class TestFa4PageSizeAutoForce(CustomTestCase):
    """FA4 requires page_size 128 for non-MLA models on SM100. The auto-force
    must trigger for `--attention-backend fa4` (combined) too, not only for the
    explicit `--prefill-attention-backend fa4` path."""

    def _make_args(self, attention_backend, prefill=None, decode=None, page_size=1):
        args = ServerArgs(model_path="dummy")
        args.attention_backend = attention_backend
        args.prefill_attention_backend = prefill
        args.decode_attention_backend = decode
        args.page_size = page_size
        # Short-circuit model_config_of(): the fa4 page_size branch only needs
        # use_mla_backend() (mocked) and override_platform(is_sm100=...), not a
        # real model_config. Pre-set the attribute so get_model_config returns
        # early without touching ModelConfig.from_server_args.
        args._model_config = MagicMock()
        args._model_config.hf_config.dual_chunk_attention_config = None
        return args

    @override_platform(is_sm100=True)
    def test_combined_attention_backend_fa4_forces_page_size_128(self):
        # `--attention-backend fa4` (combined): prefill/decode fields stay None.
        args = self._make_args(attention_backend="fa4")

        handle_attention_backend_compatibility(args)

        from sglang.srt.arg_groups.overrides import resolved_view

        self.assertEqual(args.page_size, 1)  # the field stays pristine
        self.assertEqual(resolved_view(args).page_size, 128)

    @override_platform(is_sm100=True)
    def test_explicit_prefill_fa4_forces_page_size_128(self):
        # `--prefill-attention-backend fa4`: the previously-covered path.
        args = self._make_args(attention_backend=None, prefill="fa4", page_size=1)

        handle_attention_backend_compatibility(args)

        from sglang.srt.arg_groups.overrides import resolved_view

        self.assertEqual(args.page_size, 1)  # the field stays pristine
        self.assertEqual(resolved_view(args).page_size, 128)


class TestContextParallelServerArgs(CustomTestCase):
    def setUp(self):
        self.parser = server_args_module.argparse.ArgumentParser()
        ServerArgs.add_cli_args(self.parser)

    def _new_cp_args(self, **overrides):
        server_args = object.__new__(ServerArgs)
        defaults = dict(
            enable_prefill_context_parallel=False,
            enable_dsa_prefill_context_parallel=False,
            enable_prefill_cp=False,
            cp_strategy=None,
            model_path="instance://127.0.0.1:8000/dummy",
            dsa_prefill_cp_mode="round-robin-split",
            prefill_cp_mode="in-seq-split",
            attn_cp_size=1,
            tp_size=1,
            dp_size=1,
            moe_dp_size=1,
            ep_size=1,
            pp_size=1,
            dcp_size=1,
            enable_aiter_allreduce_fusion=False,
        )
        defaults.update(overrides)
        for key, value in defaults.items():
            setattr(server_args, key, value)
        return server_args

    def test_canonical_prefill_cp_requires_strategy(self):
        args = self.parser.parse_args(["--model", "dummy", "--enable-prefill-cp"])

        self.assertTrue(resolution_result(args, "enable_prefill_cp"))
        self.assertIsNone(resolution_result(args, "cp_strategy"))

        server_args = self._new_cp_args(
            enable_prefill_cp=resolution_result(args, "enable_prefill_cp"),
            cp_strategy=resolution_result(args, "cp_strategy"),
        )
        with self.assertRaisesRegex(ValueError, "--cp-strategy"):
            handle_context_parallelism(server_args)

    @override_platform(is_hip=False, is_npu=False)
    def test_deepseek_v32_prefill_cp_rejects_zigzag(self):
        server_args = self._new_cp_args(
            model_path="deepseek-ai/DeepSeek-V3.2",
            enable_prefill_cp=True,
            cp_strategy="zigzag",
        )
        server_args._model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["DeepseekV32ForCausalLM"]),
            is_multimodal=False,
        )

        with self.assertRaisesRegex(ValueError, "DeepSeek V3.2.*interleave"):
            handle_context_parallelism(server_args)

    @override_platform(is_hip=False, is_npu=False)
    def test_generic_canonical_cp_mirrors_to_transitional_runtime_fields(self):
        cases = (
            (
                "zigzag_mla_or_gqa",
                "zigzag",
                "fa3",
                True,
                False,
                "in-seq-split",
            ),
            (
                "interleave_dsa",
                "interleave",
                "dsa",
                False,
                True,
                "round-robin-split",
            ),
        )

        for name, strategy, backend, expect_generic, expect_dsa, mode in cases:
            with self.subTest(name=name):
                server_args = self._new_cp_args(
                    enable_prefill_cp=True,
                    cp_strategy=strategy,
                    attention_backend=backend,
                )

                handle_platform_cp_compatibility(server_args)

                self.assertFalse(
                    resolution_result(server_args, "enable_prefill_context_parallel")
                )
                self.assertFalse(
                    resolution_result(
                        server_args, "enable_dsa_prefill_context_parallel"
                    )
                )

                handle_legacy_cp_runtime_compatibility(server_args)

                self.assertEqual(
                    resolution_result(server_args, "enable_prefill_context_parallel"),
                    expect_generic,
                )
                self.assertEqual(
                    resolution_result(
                        server_args, "enable_dsa_prefill_context_parallel"
                    ),
                    expect_dsa,
                )
                self.assertEqual(
                    resolution_result(server_args, "dsa_prefill_cp_mode"), mode
                )
                self.assertEqual(
                    resolution_result(server_args, "prefill_cp_mode"), mode
                )

    @override_platform(is_hip=False, is_npu=False)
    def test_non_platform_legacy_prefill_cp_is_rejected(self):
        server_args = ServerArgs(
            model_path="instance://127.0.0.1:8000/dummy",
            enable_prefill_context_parallel=True,
        )
        with self.assertRaisesRegex(ValueError, "HIP or Ascend NPU"):
            handle_platform_cp_compatibility(server_args)

    def test_generic_v1_cp_options_are_not_public_cli(self):
        removed_options = (
            ("--enable-dsa-prefill-context-parallel", []),
            ("--dsa-prefill-cp-mode", ["round-robin-split"]),
            ("--prefill-cp-mode", ["in-seq-split"]),
        )

        for option, values in removed_options:
            with self.subTest(option=option), self.assertRaises(SystemExit):
                self.parser.parse_args(["--model", "dummy", option, *values])

    def test_npu_cp_compatibility_options_remain_public_cli(self):
        args = self.parser.parse_args(
            [
                "--model",
                "dummy",
                "--enable-prefill-context-parallel",
                "--enable-nsa-prefill-context-parallel",
                "--nsa-prefill-cp-mode",
                "round-robin-split",
            ]
        )

        self.assertTrue(resolution_result(args, "enable_prefill_context_parallel"))
        self.assertTrue(resolution_result(args, "enable_dsa_prefill_context_parallel"))
        self.assertEqual(
            resolution_result(args, "dsa_prefill_cp_mode"), "round-robin-split"
        )

    def test_canonical_interleave_cp_mirrors_to_dsa_runtime_aliases(self):
        server_args = self._new_cp_args(
            enable_prefill_cp=True,
            cp_strategy="interleave",
            attention_backend="dsa",
        )

        handle_legacy_cp_runtime_compatibility(server_args)
        handle_context_parallelism(server_args)

        self.assertTrue(
            resolution_result(server_args, "enable_dsa_prefill_context_parallel")
        )
        self.assertFalse(
            resolution_result(server_args, "enable_prefill_context_parallel")
        )
        self.assertEqual(
            resolution_result(server_args, "dsa_prefill_cp_mode"), "round-robin-split"
        )
        self.assertEqual(
            resolution_result(server_args, "prefill_cp_mode"), "round-robin-split"
        )

    def test_context_parallel_handler_initializes_cp_strategy(self):
        server_args = self._new_cp_args(
            enable_prefill_cp=True,
            cp_strategy="interleave",
            attn_cp_size=2,
            tp_size=2,
        )

        handle_context_parallelism(server_args)

        self.assertTrue(is_cp_enabled())
        self.assertTrue(is_interleave())


class TestPortArgs(unittest.TestCase):
    @patch("sglang.srt.server_args.tempfile.NamedTemporaryFile")
    def test_init_new_standard_case(self, mock_temp_file):
        mock_temp_file.return_value.name = "temp_file"

        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = False

        port_args = PortArgs.init_new(server_args)

        self.assertTrue(port_args.tokenizer_ipc_name.startswith("ipc://"))
        self.assertTrue(port_args.scheduler_input_ipc_name.startswith("ipc://"))
        self.assertTrue(port_args.detokenizer_ipc_name.startswith("ipc://"))
        self.assertIsInstance(port_args.nccl_port, int)

    @patch("sglang.srt.server_args.tempfile.NamedTemporaryFile")
    def test_init_new_builds_decoupled_spec_ipc_config(self, mock_temp_file):
        mock_temp_file.return_value.name = "temp_file"

        server_args = ServerArgs(model_path="dummy")
        server_args.nccl_port = None
        server_args.enable_dp_attention = False
        server_args.decoupled_spec_role = "verifier"
        server_args.decoupled_spec_bind_endpoint = "ipc:///tmp/v"
        server_args.decoupled_spec_connect_endpoints = ["ipc:///tmp/d"]
        server_args.decoupled_spec_rank = 0

        port_args = PortArgs.init_new(server_args)

        self.assertIsNotNone(port_args.decoupled_spec_ipc_config)
        self.assertEqual(port_args.decoupled_spec_ipc_config.rank, 0)
        self.assertEqual(
            port_args.decoupled_spec_ipc_config.bind_endpoint, "ipc:///tmp/v"
        )
        self.assertEqual(
            port_args.decoupled_spec_ipc_config.connect_endpoints, ("ipc:///tmp/d",)
        )

    @patch("sglang.srt.server_args.tempfile.NamedTemporaryFile")
    def test_init_new_no_decoupled_config_when_role_null(self, mock_temp_file):
        mock_temp_file.return_value.name = "temp_file"

        server_args = ServerArgs(model_path="dummy")
        server_args.nccl_port = None
        server_args.enable_dp_attention = False
        # decoupled_spec_role defaults to "null"

        port_args = PortArgs.init_new(server_args)

        self.assertIsNone(port_args.decoupled_spec_ipc_config)

    def test_init_new_decoupled_role_requires_endpoints(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.nccl_port = None
        server_args.enable_dp_attention = False
        server_args.decoupled_spec_role = "drafter"
        # endpoints intentionally left as their None defaults

        with self.assertRaises(ValueError):
            PortArgs.init_new(server_args)

    def test_init_new_with_single_node_dp_attention(self):

        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = True
        server_args.nnodes = 1
        server_args.dist_init_addr = None

        port_args = PortArgs.init_new(server_args)

        self.assertTrue(port_args.tokenizer_ipc_name.startswith("tcp://127.0.0.1:"))
        self.assertTrue(
            port_args.scheduler_input_ipc_name.startswith("tcp://127.0.0.1:")
        )
        self.assertTrue(port_args.detokenizer_ipc_name.startswith("tcp://127.0.0.1:"))
        self.assertIsInstance(port_args.nccl_port, int)

    def test_init_new_with_dp_rank(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = True
        server_args.nnodes = 1
        server_args.dist_init_addr = "192.168.1.1:25000"

        worker_ports = [25006, 25007, 25008, 25009]
        port_args = PortArgs.init_new(server_args, dp_rank=2, worker_ports=worker_ports)

        self.assertTrue(port_args.scheduler_input_ipc_name.endswith(":25008"))

        self.assertTrue(port_args.tokenizer_ipc_name.startswith("tcp://192.168.1.1:"))
        self.assertTrue(port_args.detokenizer_ipc_name.startswith("tcp://192.168.1.1:"))
        self.assertIsInstance(port_args.nccl_port, int)

    def test_init_new_with_ipv4_address(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None

        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "192.168.1.1:25000"

        port_args = PortArgs.init_new(server_args)

        self.assertTrue(port_args.tokenizer_ipc_name.startswith("tcp://192.168.1.1:"))
        self.assertTrue(
            port_args.scheduler_input_ipc_name.startswith("tcp://192.168.1.1:")
        )
        self.assertTrue(port_args.detokenizer_ipc_name.startswith("tcp://192.168.1.1:"))
        self.assertIsInstance(port_args.nccl_port, int)

    def test_init_new_with_malformed_ipv4_address(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None

        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "192.168.1.1"

        with self.assertRaises(ValueError) as context:
            PortArgs.init_new(server_args)

        self.assertIn("Missing port", str(context.exception))

    def test_init_new_with_malformed_ipv4_address_invalid_port(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None

        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "192.168.1.1:abc"

        with self.assertRaises(ValueError):
            PortArgs.init_new(server_args)


class TestSSLArgs(unittest.TestCase):
    def _validate_ssl(self, **kwargs):
        server_args = ServerArgs(model_path="dummy", **kwargs)
        handle_ssl_validation(server_args)
        return server_args

    def test_ssl_keyfile_without_certfile_raises(self):
        with self.assertRaises(ValueError) as context:
            self._validate_ssl(ssl_keyfile="key.pem")
        self.assertIn("--ssl-certfile", str(context.exception))

    def test_ssl_certfile_without_keyfile_raises(self):
        with self.assertRaises(ValueError) as context:
            self._validate_ssl(ssl_certfile="cert.pem")
        self.assertIn("--ssl-keyfile", str(context.exception))

    def test_url_returns_http_without_ssl(self):
        server_args = ServerArgs(model_path="dummy")
        self.assertTrue(server_args.url().startswith("http://"))

    def test_url_rewrites_all_interfaces_to_loopback(self):
        server_args = ServerArgs(model_path="dummy", host="0.0.0.0")
        self.assertEqual(server_args.url(), "http://127.0.0.1:30000")

    def test_url_rewrites_empty_host_to_loopback(self):
        server_args = ServerArgs(model_path="dummy", host="")
        self.assertEqual(server_args.url(), "http://127.0.0.1:30000")

    @patch("os.path.isfile", return_value=True)
    def test_url_returns_https_with_ssl(self, _mock_isfile):
        server_args = self._validate_ssl(ssl_keyfile="key.pem", ssl_certfile="cert.pem")
        self.assertTrue(server_args.url().startswith("https://"))

    def test_ssl_verify_without_ssl(self):
        # the derived read lives with the rest of the SSL handling now

        server_args = ServerArgs(model_path="dummy")
        self.assertIs(ssl_verify_of(server_args), True)

    @patch("os.path.isfile", return_value=True)
    def test_ssl_verify_with_ssl_no_ca(self, _mock_isfile):
        server_args = self._validate_ssl(ssl_keyfile="key.pem", ssl_certfile="cert.pem")
        self.assertIs(ssl_verify_of(server_args), False)

    @patch("os.path.isfile", return_value=True)
    def test_ssl_verify_with_ssl_and_ca(self, _mock_isfile):
        server_args = self._validate_ssl(
            ssl_keyfile="key.pem",
            ssl_certfile="cert.pem",
            ssl_ca_certs="ca.pem",
        )
        self.assertEqual(ssl_verify_of(server_args), "ca.pem")

    def test_ssl_ca_certs_without_certfile_raises(self):
        with self.assertRaises(ValueError) as context:
            self._validate_ssl(ssl_ca_certs="ca.pem")
        self.assertIn("--ssl-ca-certs", str(context.exception))

    def test_ssl_keyfile_password_without_certfile_raises(self):
        with self.assertRaises(ValueError) as context:
            self._validate_ssl(ssl_keyfile_password="secret")
        self.assertIn("--ssl-keyfile-password", str(context.exception))

    def test_ssl_keyfile_not_found_raises(self):
        with self.assertRaises(ValueError) as context:
            self._validate_ssl(
                ssl_keyfile="/nonexistent/key.pem",
                ssl_certfile="/nonexistent/cert.pem",
            )
        self.assertIn("not found", str(context.exception))

    def test_ssl_certfile_not_found_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".pem") as keyfile:
            with self.assertRaises(ValueError) as context:
                self._validate_ssl(
                    ssl_keyfile=keyfile.name,
                    ssl_certfile="/nonexistent/cert.pem",
                )
            self.assertIn("SSL certificate file not found", str(context.exception))

    def test_ssl_ca_certs_not_found_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".pem") as keyfile:
            with tempfile.NamedTemporaryFile(suffix=".pem") as certfile:
                with self.assertRaises(ValueError) as context:
                    self._validate_ssl(
                        ssl_keyfile=keyfile.name,
                        ssl_certfile=certfile.name,
                        ssl_ca_certs="/nonexistent/ca.pem",
                    )
                self.assertIn(
                    "SSL CA certificates file not found", str(context.exception)
                )

    def test_enable_ssl_refresh_without_ssl_raises(self):
        with self.assertRaises(ValueError) as context:
            self._validate_ssl(enable_ssl_refresh=True)
        self.assertIn("--enable-ssl-refresh", str(context.exception))
        self.assertIn("--ssl-certfile", str(context.exception))

    @patch("os.path.isfile", return_value=True)
    def test_enable_ssl_refresh_with_ssl_accepted(self, _mock_isfile):
        server_args = self._validate_ssl(
            ssl_keyfile="key.pem",
            ssl_certfile="cert.pem",
            enable_ssl_refresh=True,
        )
        self.assertTrue(resolution_result(server_args, "enable_ssl_refresh"))


class TestHiCacheArgs(unittest.TestCase):
    def _make_args(self, **overrides) -> ServerArgs:
        # Not resolved: a dummy model path takes the pipeline's early return,
        # so `_handle_hicache` would never run. Its one prerequisite (the
        # host/device ratio default) is run by hand.
        args = ServerArgs(model_path="dummy", **overrides)
        handle_hicache_ratio_default(args)
        return args

    def _assert_hicache_fields(
        self,
        args: ServerArgs,
        *,
        expected_io_backend: str,
        expected_mem_layout: str,
        expected_decode_backend: str | None = None,
    ):
        self.assertEqual(
            resolution_result(args, "hicache_io_backend"), expected_io_backend
        )
        self.assertEqual(
            resolution_result(args, "hicache_mem_layout"), expected_mem_layout
        )
        if expected_decode_backend is not None:
            self.assertEqual(
                resolution_result(args, "decode_attention_backend"),
                expected_decode_backend,
            )

    def test_buffer_only_accepts_both_tree_cores(self):
        for backend in ("python", "rust"):
            args = self._make_args(
                enable_hierarchical_cache=True,
                hicache_host_memory_mode="buffer_only",
                hicache_storage_backend="file",
            )
            with envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override(backend):
                handle_hicache(args)

    def test_hicache_io_backend_and_mem_layout_compatibility(self):
        cases = [
            {
                "name": "default_kernel_page_first",
                "overrides": {
                    "enable_hierarchical_cache": True,
                },
                "expected_io_backend": "kernel",
                "expected_mem_layout": "page_first",
            },
            {
                "name": "kernel_with_page_first_direct",
                "overrides": {
                    "enable_hierarchical_cache": True,
                    "hicache_io_backend": "kernel",
                    "hicache_mem_layout": "page_first_direct",
                },
                "expected_io_backend": "direct",
                "expected_mem_layout": "page_first_direct",
            },
            {
                "name": "direct_with_page_first",
                "overrides": {
                    "enable_hierarchical_cache": True,
                    "hicache_io_backend": "direct",
                    "hicache_mem_layout": "page_first",
                },
                "expected_io_backend": "direct",
                "expected_mem_layout": "page_first_direct",
            },
            {
                "name": "mooncake_with_layer_first",
                "overrides": {
                    "enable_hierarchical_cache": True,
                    "hicache_storage_backend": "mooncake",
                    "hicache_io_backend": "direct",
                    "hicache_mem_layout": "layer_first",
                },
                "expected_io_backend": "direct",
                "expected_mem_layout": "page_first_direct",
            },
            {
                "name": "fa3_kernel_with_explicit_decode_backend",
                "overrides": {
                    "enable_hierarchical_cache": True,
                    "hicache_io_backend": "kernel",
                    "hicache_mem_layout": "page_first",
                    "attention_backend": "triton",
                    "decode_attention_backend": "fa3",
                },
                "expected_io_backend": "kernel",
                "expected_mem_layout": "page_first",
                "expected_decode_backend": "fa3",
            },
        ]

        for case in cases:
            with self.subTest(case=case["name"]):
                args = self._make_args(**case["overrides"])
                handle_hicache(args)
                self._assert_hicache_fields(
                    args,
                    expected_io_backend=case["expected_io_backend"],
                    expected_mem_layout=case["expected_mem_layout"],
                    expected_decode_backend=case.get("expected_decode_backend"),
                )

    def test_hicache_kernel_keeps_implicit_fa3_decode_backend(self):
        args = self._make_args(
            enable_hierarchical_cache=True,
            hicache_io_backend="kernel",
            attention_backend="fa3",
            decode_attention_backend=None,
        )
        handle_hicache(args)

        self.assertEqual(resolution_result(args, "hicache_io_backend"), "kernel")
        self.assertEqual(resolution_result(args, "hicache_mem_layout"), "page_first")
        self.assertIsNone(resolution_result(args, "decode_attention_backend"))

    def test_decode_offload_rejects_host_pool_retraction(self):
        args = self._make_args(
            disaggregation_mode="decode",
            disaggregation_decode_enable_offload_kvcache=True,
            hicache_storage_backend="file",
            disaggregation_decode_retraction_backup="host_pool",
        )

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            handle_cache_compatibility(args)

    def test_decode_offload_allows_cpu_tensor_retraction(self):
        args = self._make_args(
            disaggregation_mode="decode",
            disaggregation_decode_enable_offload_kvcache=True,
            hicache_storage_backend="file",
            disaggregation_decode_retraction_backup="cpu_tensor",
        )

        handle_cache_compatibility(args)


class TestNgramExternalSamArgs(CustomTestCase):
    def _make_dummy_ngram_args(self, **overrides):
        args = ServerArgs(model_path="dummy")
        args.speculative_algorithm = "NGRAM"
        args.speculative_num_draft_tokens = 12
        args.device = "cuda"
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_external_sam_budget_must_fit_draft_budget(self):
        args = self._make_dummy_ngram_args(
            speculative_num_draft_tokens=4,
            speculative_ngram_external_corpus_path="/tmp/ngram-corpus.jsonl",
            speculative_ngram_external_sam_budget=4,
        )
        with self.assertRaises(ValueError) as context:
            handle_speculative_decoding(args)
        self.assertIn("speculative_num_draft_tokens - 1", str(context.exception))

    def test_external_corpus_max_tokens_must_be_positive(self):
        args = self._make_dummy_ngram_args(
            speculative_ngram_external_corpus_path="/tmp/ngram-corpus.jsonl",
            speculative_ngram_external_sam_budget=2,
            speculative_ngram_external_corpus_max_tokens=0,
        )
        with self.assertRaises(ValueError) as context:
            handle_speculative_decoding(args)
        self.assertIn("external-corpus-max-tokens", str(context.exception))


class TestDecoupledSpecArgs(CustomTestCase):
    """Decoupled speculative-decoding CLI flags.

    These flags are auto-derived from the ``A[...]`` field metadata on
    ``ServerArgs``; a bare annotation is silently skipped by
    ``add_cli_args_from_dataclass``. This guards against the regression where
    the flags went missing (e.g. after rebasing onto the auto-gen
    ``add_cli_args``), which the direct-attribute ``PortArgs`` tests cannot
    catch because they never exercise the CLI.
    """

    def test_decoupled_spec_cli_flags_round_trip(self):
        server_args = prepare_server_args(
            [
                "--model-path",
                "dummy",
                "--decoupled-spec-role",
                "verifier",
                "--decoupled-spec-bind-endpoint",
                "ipc:///tmp/v",
                "--decoupled-spec-connect-endpoints",
                '["ipc:///tmp/d"]',
                "--decoupled-spec-rank",
                "0",
                "--spec-trace-dir",
                "/tmp/tr",
            ]
        )
        self.assertEqual(
            resolution_result(server_args, "decoupled_spec_role"), "verifier"
        )
        self.assertEqual(
            resolution_result(server_args, "decoupled_spec_bind_endpoint"),
            "ipc:///tmp/v",
        )
        self.assertEqual(
            resolution_result(server_args, "decoupled_spec_connect_endpoints"),
            ["ipc:///tmp/d"],
        )
        self.assertEqual(resolution_result(server_args, "decoupled_spec_rank"), 0)
        self.assertEqual(resolution_result(server_args, "spec_trace_dir"), "/tmp/tr")

    def test_decoupled_spec_role_rejects_invalid_choice(self):
        with self.assertRaises(SystemExit):
            prepare_server_args(
                ["--model-path", "dummy", "--decoupled-spec-role", "bogus"]
            )


class TestAdaptiveSpecArgs(CustomTestCase):
    def test_adaptive_defaults_to_config_step_when_spec_params_omitted(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "1": {"candidate_steps": [1, 3, 5]},
                    "8": {"candidate_steps": [1]},
                },
                f,
            )
            f.flush()

            args = ServerArgs(model_path="dummy")
            args.speculative_algorithm = "EAGLE"
            args.speculative_adaptive = True
            args.speculative_adaptive_config = f.name
            args.device = "cuda"
            args._model_config = SimpleNamespace(
                hf_config=SimpleNamespace(
                    architectures=["LlamaForCausalLM"],
                    get_text_config=lambda: SimpleNamespace(),
                )
            )

            handle_speculative_decoding(args)
            self.assertEqual(max_speculative_num_draft_tokens(args), 6)

        self.assertTrue(resolution_result(args, "speculative_adaptive"))
        self.assertEqual(resolution_result(args, "speculative_eagle_topk"), 1)
        self.assertEqual(resolution_result(args, "speculative_num_steps"), 3)
        self.assertEqual(resolution_result(args, "speculative_num_draft_tokens"), 4)


class TestWaterfillArgs(CustomTestCase):
    def test_waterfill_enforces_shared_experts_fusion(self):
        server_args = ServerArgs(
            model_path="dummy",
            moe_a2a_backend="deepep",
            enable_waterfill=True,
            disable_shared_experts_fusion=True,
        )
        # dummy-model path short-circuits __post_init__; invoke the handler directly.
        handle_a2a_moe(server_args)

        from sglang.srt.arg_groups.overrides import resolved_view

        self.assertTrue(server_args.disable_shared_experts_fusion)
        self.assertFalse(resolved_view(server_args).disable_shared_experts_fusion)
        self.assertTrue(resolution_result(server_args, "enforce_shared_experts_fusion"))

    def test_waterfill_overrides_moe_a2a_backend_to_deepep(self):
        server_args = ServerArgs(
            model_path="dummy",
            moe_a2a_backend="none",
            enable_waterfill=True,
        )
        # dummy-model path short-circuits __post_init__; invoke the handler directly.
        handle_a2a_moe(server_args)

        from sglang.srt.arg_groups.overrides import resolved_view

        self.assertEqual(server_args.moe_a2a_backend, "none")  # pristine
        self.assertEqual(resolved_view(server_args).moe_a2a_backend, "deepep")
        self.assertTrue(resolution_result(server_args, "enforce_shared_experts_fusion"))

    def test_waterfill_keeps_megamoe_backend(self):
        server_args = ServerArgs(
            model_path="dummy",
            moe_a2a_backend="megamoe",
            enable_waterfill=True,
            disable_shared_experts_fusion=True,
        )
        # dummy-model path short-circuits __post_init__; invoke the handler directly.
        handle_a2a_moe(server_args)

        from sglang.srt.arg_groups.overrides import resolved_view

        self.assertEqual(resolved_view(server_args).moe_a2a_backend, "megamoe")
        self.assertFalse(resolved_view(server_args).disable_shared_experts_fusion)
        self.assertTrue(resolution_result(server_args, "enforce_shared_experts_fusion"))

    def test_waterfill_supports_deepep_low_latency_mode(self):
        server_args = ServerArgs(
            model_path="dummy",
            moe_a2a_backend="deepep",
            enable_waterfill=True,
            deepep_mode="low_latency",
        )
        # dummy-model path short-circuits __post_init__; invoke the handler directly.
        handle_a2a_moe(server_args)

        self.assertEqual(resolution_result(server_args, "deepep_mode"), "low_latency")
        self.assertFalse(resolution_result(server_args, "disable_cuda_graph"))
        self.assertTrue(resolution_result(server_args, "enforce_shared_experts_fusion"))


class TestPrefillOnlyDisableKvCache(unittest.TestCase):
    """Validation for --prefill-only-disable-kv-cache.

    The flag wires NoOpMHATokenToKVPool, which is only safe when:
      - the engine is in embedding mode (fa_skip_kv_cache active in FA backend),
      - chunked_prefill_size == -1 (no inter-chunk K/V reuse),
      - disable_radix_cache (radix cache otherwise indexes empty pool slots),
      - no context-parallel attention (CP writes to the pool via set_kv_buffer),
      - no HiSparse (uses a different pool family),
      - kv_cache_dtype is not nvfp4/fp4_mx_block16 (FP4 pool is a separate allocation path).
    All other configurations must be rejected before model load.
    """

    def _base_kwargs(self, **overrides):
        kwargs = dict(
            model_path="dummy",
            is_embedding=True,
            chunked_prefill_size=-1,
            disable_radix_cache=True,
            prefill_only_disable_kv_cache=True,
        )
        kwargs.update(overrides)
        return kwargs

    def _validate_prefill_only_args(self, **overrides):
        sa = ServerArgs(**self._base_kwargs(**overrides))
        validate_prefill_only_disable_kv_cache_args(sa)
        return sa

    def test_valid_minimal_config_constructs(self):
        sa = self._validate_prefill_only_args()
        self.assertTrue(resolution_result(sa, "prefill_only_disable_kv_cache"))

    def test_rejects_when_not_embedding(self):
        with self.assertRaisesRegex(ValueError, "requires --is-embedding"):
            self._validate_prefill_only_args(is_embedding=False)

    def test_rejects_when_chunked_prefill_size_not_minus_one(self):
        with self.assertRaisesRegex(ValueError, "--chunked-prefill-size=-1"):
            self._validate_prefill_only_args(chunked_prefill_size=8192)

    def test_rejects_when_radix_cache_enabled(self):
        with self.assertRaisesRegex(ValueError, "--disable-radix-cache"):
            self._validate_prefill_only_args(disable_radix_cache=False)

    def test_rejects_attn_cp_size_greater_than_one(self):
        with self.assertRaisesRegex(ValueError, "--attn-cp-size"):
            self._validate_prefill_only_args(attn_cp_size=2, tp_size=2)

    def test_rejects_prefill_context_parallel(self):
        with self.assertRaisesRegex(ValueError, "--enable-prefill-cp"):
            self._validate_prefill_only_args(
                enable_prefill_cp=True,
                cp_strategy="zigzag",
            )

    def test_rejects_hisparse(self):
        with self.assertRaisesRegex(ValueError, "--enable-hisparse"):
            self._validate_prefill_only_args(enable_hisparse=True)

    def test_rejects_fp4_kv_cache(self):
        for kv_cache_dtype in ("nvfp4", "fp4_mx_block16"):
            with self.subTest(kv_cache_dtype=kv_cache_dtype):
                with self.assertRaisesRegex(ValueError, "nvfp4.*fp4_mx_block16"):
                    self._validate_prefill_only_args(kv_cache_dtype=kv_cache_dtype)


class TestCudaGraphConfigDataclassAccess(CustomTestCase):
    @patch(
        "sglang.srt.model_executor.runner_backend."
        "tc_piecewise_cuda_graph_backend.get_moe_a2a_backend"
    )
    def test_tc_piecewise_build_config_reads_phase_config_dataclass(
        self, mock_get_moe_a2a_backend
    ):
        from sglang.srt.model_executor.runner_backend.tc_piecewise_cuda_graph_backend import (
            TcPiecewiseCudaGraphBackend,
        )

        mock_backend = mock_get_moe_a2a_backend.return_value
        mock_backend.is_deepep.return_value = False
        mock_backend.is_mooncake.return_value = False
        from sglang.srt.runtime_context import get_context

        # The graph configuration is a bag leaf; the debug switch is raw input
        # and stays on the argument.
        override = get_context().override_server_args(
            cuda_graph_config=CudaGraphConfig(
                prefill=PhaseConfig(
                    backend=Backend.TC_PIECEWISE,
                    bs=[32, 64],
                    tc_compiler="eager",
                )
            )
        )
        override.install()
        self.addCleanup(override.restore)
        server_args = SimpleNamespace(enable_torch_compile_debug_mode=False)

        config = TcPiecewiseCudaGraphBackend.build_compilation_config(server_args)

        self.assertEqual(config.get_capture_sizes(), [32, 64])
        self.assertEqual(config.compiler, "eager")


class TestPipelineParallelPrefillCudaGraphPolicy(CustomTestCase):
    def test_pp_prefill_graph_is_opt_in(self):
        cases = (
            (set(), Backend.DISABLED),
            ({(Phase.PREFILL, "backend")}, Backend.BREAKABLE),
        )
        for locked, expected in cases:
            with self.subTest(locked=locked):
                args = ServerArgs(
                    model_path="dummy",
                    pp_size=4,
                    cuda_graph_config=CudaGraphConfig(
                        prefill=PhaseConfig(backend=Backend.BREAKABLE)
                    ),
                )
                args._cuda_graph_config_locked = locked
                apply_cuda_graph_compatibility(args)
                self.assertEqual(
                    resolution_result(args, "cuda_graph_config").prefill.backend,
                    expected,
                )

    def test_pp_prefill_capture_limit_policy(self):
        cases = (
            (4096, None, 4096),
            (32768, None, 8192),
            (32768, 16384, 16384),
        )
        for chunked_prefill_size, max_bs, expected in cases:
            with self.subTest(chunked_prefill_size=chunked_prefill_size, max_bs=max_bs):
                args = ServerArgs(
                    model_path="dummy",
                    pp_size=4,
                    chunked_prefill_size=chunked_prefill_size,
                    mem_fraction_static=0.8,
                    cuda_graph_config=CudaGraphConfig(
                        decode=PhaseConfig(backend=Backend.DISABLED, max_bs=1, bs=[1]),
                        prefill=PhaseConfig(backend=Backend.BREAKABLE, max_bs=max_bs),
                    ),
                )
                args._cuda_graph_config_locked = {(Phase.PREFILL, "backend")} | (
                    {(Phase.PREFILL, "max_bs")} if max_bs is not None else set()
                )
                with patch(
                    "sglang.srt.arg_groups.memory_hook.use_mla_backend",
                    return_value=False,
                ):
                    handle_gpu_memory_settings(args, gpu_mem=None)
                prefill = resolution_result(args, "cuda_graph_config").prefill
                self.assertEqual((prefill.max_bs, prefill.bs[-1]), (expected, expected))


class TestCudaGraphDisaggregationRoles(CustomTestCase):
    def _handled_args(self, **overrides):
        args = ServerArgs(model_path="dummy", **overrides)
        args._model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
            is_piecewise_cuda_graph_disabled_model=False,
            is_multimodal=False,
            is_multimodal_piecewise_cuda_graph_supported=False,
        )
        with patch("sglang.srt.utils.is_cuda", return_value=True):
            handle_cuda_graph_config(args)
        return args

    def test_cuda_graph_prefill_role_defaults_disable_decode_graph(self):
        args = self._handled_args(disaggregation_mode="prefill")

        self.assertFalse(resolution_result(args, "disable_cuda_graph"))
        self.assertEqual(
            resolution_result(args, "cuda_graph_config").decode.backend,
            Backend.DISABLED,
        )
        self.assertEqual(
            resolution_result(args, "cuda_graph_config").prefill.backend,
            Backend.BREAKABLE,
        )

    def test_cuda_graph_decode_role_defaults_disable_prefill_graph(self):
        args = self._handled_args(disaggregation_mode="decode")

        self.assertEqual(
            resolution_result(args, "cuda_graph_config").prefill.backend,
            Backend.DISABLED,
        )
        self.assertNotEqual(
            resolution_result(args, "cuda_graph_config").decode.backend,
            Backend.DISABLED,
        )

    def test_cuda_graph_global_disable_still_disables_both_phases_for_all_roles(self):
        for disaggregation_mode in ("prefill", "decode", "null"):
            with self.subTest(disaggregation_mode=disaggregation_mode):
                args = self._handled_args(
                    disaggregation_mode=disaggregation_mode,
                    disable_cuda_graph=True,
                )

                self.assertEqual(
                    resolution_result(args, "cuda_graph_config").decode.backend,
                    Backend.DISABLED,
                )
                self.assertEqual(
                    resolution_result(args, "cuda_graph_config").prefill.backend,
                    Backend.DISABLED,
                )

    def test_cuda_graph_explicit_decode_backend_survives_prefill_role(self):
        args = self._handled_args(
            disaggregation_mode="prefill",
            cuda_graph_backend_decode=Backend.FULL,
        )

        self.assertEqual(
            resolution_result(args, "cuda_graph_config").decode.backend, Backend.FULL
        )
        self.assertIn((Phase.DECODE, "backend"), args._cuda_graph_config_locked)


class TestPrefillCudaGraphLoRACompatibility(CustomTestCase):
    """LoRA no longer auto-disables the breakable prefill CUDA graph; guards
    test_bcg_with_lora.py against a rule re-disabling it (vacuous pass)."""

    def _handled_args(self, **overrides):
        args = ServerArgs(model_path="dummy", **overrides)
        args._model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
            is_piecewise_cuda_graph_disabled_model=False,
            is_multimodal=False,
            is_multimodal_piecewise_cuda_graph_supported=False,
        )
        with patch("sglang.srt.utils.is_cuda", return_value=True):
            handle_cuda_graph_config(args)
        return args

    def test_enable_lora_keeps_breakable_prefill_graph(self):
        args = self._handled_args(enable_lora=True)

        self.assertEqual(
            resolution_result(args, "cuda_graph_config").prefill.backend,
            Backend.BREAKABLE,
        )

    def test_lora_paths_keep_breakable_prefill_graph(self):
        args = self._handled_args(lora_paths=["dummy/lora-adapter"])

        self.assertEqual(
            resolution_result(args, "cuda_graph_config").prefill.backend,
            Backend.BREAKABLE,
        )

    def test_lora_still_disables_tc_piecewise_prefill_graph(self):
        # Pin the tc_piecewise LoRA rule itself, with the hardware rule
        # neutralized so this runs on CPU-only CI.
        args = ServerArgs(model_path="dummy", enable_lora=True)
        args._model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
            is_piecewise_cuda_graph_disabled_model=False,
            is_multimodal=False,
            is_multimodal_piecewise_cuda_graph_supported=False,
        )
        args.cuda_graph_config = CudaGraphConfig(
            prefill=PhaseConfig(backend=Backend.TC_PIECEWISE)
        )
        with (
            override_platform(is_hip=False),
            override_platform(is_npu=False),
            patch("sglang.srt.arg_groups.cuda_graph_hook.is_cpu", return_value=False),
            patch("sglang.srt.arg_groups.cuda_graph_hook.is_mps", return_value=False),
            override_platform(is_xpu=False),
        ):
            disable_tc_piecewise_cudagraph_if_incompatible(args)

        self.assertEqual(
            resolution_result(args, "cuda_graph_config").prefill.backend,
            Backend.DISABLED,
        )


class TestBreakableCudaGraphMultimodalAllowlist(CustomTestCase):
    """The BCG "multimodal model" rule exempts archs on the BCG multimodal
    opt-in allowlist (multimodal_breakable_cuda_graph_supported_model_archs)."""

    def _handled_args(self, *, architectures, is_multimodal, allowlisted):
        args = ServerArgs(model_path="dummy")
        args._model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=architectures),
            is_piecewise_cuda_graph_disabled_model=False,
            is_multimodal=is_multimodal,
            is_multimodal_piecewise_cuda_graph_supported=False,
            is_multimodal_breakable_cuda_graph_supported=allowlisted,
        )
        with patch("sglang.srt.utils.is_cuda", return_value=True):
            handle_cuda_graph_config(args)
        return args

    def test_multimodal_arch_disables_prefill_breakable(self):
        args = self._handled_args(
            architectures=["Qwen3VLForConditionalGeneration"],
            is_multimodal=True,
            allowlisted=False,
        )
        self.assertEqual(
            resolution_result(args, "cuda_graph_config").prefill.backend,
            Backend.DISABLED,
        )

    def test_allowlisted_multimodal_arch_keeps_prefill_breakable(self):
        args = self._handled_args(
            architectures=["Qwen3_5MoeForConditionalGeneration"],
            is_multimodal=True,
            allowlisted=True,
        )
        self.assertEqual(
            resolution_result(args, "cuda_graph_config").prefill.backend,
            Backend.BREAKABLE,
        )

    def test_allowlist_membership(self):
        from sglang.srt.configs.model_config import (
            is_multimodal_breakable_cuda_graph_supported,
        )

        self.assertTrue(
            is_multimodal_breakable_cuda_graph_supported(
                ["Qwen3_5MoeForConditionalGeneration"]
            )
        )
        self.assertTrue(
            is_multimodal_breakable_cuda_graph_supported(
                ["Qwen3_5ForConditionalGeneration"]
            )
        )
        self.assertFalse(
            is_multimodal_breakable_cuda_graph_supported(
                ["Qwen3VLForConditionalGeneration"]
            )
        )


class TestCutedslMoeMaxNumTokens(CustomTestCase):
    """The shared CuteDSL MoE per-forward token bound. Fields are set directly
    to exercise the math independently of __post_init__ resolution.

    cg-refactor: the legacy disable_piecewise_cuda_graph /
    piecewise_cuda_graph_max_tokens / cuda_graph_max_bs fields were
    consolidated into cuda_graph_config; the helper accepts the legacy
    kwarg names for test readability and translates them to the per-phase
    dataclasses.
    """

    def _args(self, **overrides):
        server_args = ServerArgs(model_path="dummy")
        fields = dict(
            speculative_algorithm=None,
            speculative_num_draft_tokens=None,
            max_prefill_tokens=16384,
            disable_piecewise_cuda_graph=False,
            piecewise_cuda_graph_max_tokens=2048,
            cuda_graph_max_bs=512,
        )
        fields.update(overrides)
        disable_piecewise = fields.pop("disable_piecewise_cuda_graph")
        piecewise_max = fields.pop("piecewise_cuda_graph_max_tokens")
        cg_max_bs = fields.pop("cuda_graph_max_bs")
        for key, value in fields.items():
            setattr(server_args, key, value)
        server_args.cuda_graph_config = CudaGraphConfig(
            decode=PhaseConfig(backend=Backend.FULL, max_bs=cg_max_bs),
            prefill=PhaseConfig(
                backend=(
                    Backend.DISABLED if disable_piecewise else Backend.TC_PIECEWISE
                ),
                max_bs=piecewise_max,
                tc_compiler="eager",
            ),
        )
        return server_args

    def test_prefill_dominates_in_default_config(self):
        self.assertEqual(cutedsl_moe_max_num_tokens(self._args()), 16384)

    def test_speculative_decoding_scales_decode_bound(self):
        # decode bound 512 * 8 dominates the small prefill/piecewise bounds
        args = self._args(
            max_prefill_tokens=512,
            piecewise_cuda_graph_max_tokens=512,
            speculative_algorithm="EAGLE",
            speculative_num_draft_tokens=8,
        )
        self.assertEqual(cutedsl_moe_max_num_tokens(args), 4096)

    def test_piecewise_bound_excluded_when_disabled(self):
        args = self._args(
            max_prefill_tokens=512,
            disable_piecewise_cuda_graph=True,
            cuda_graph_max_bs=64,
        )
        self.assertEqual(cutedsl_moe_max_num_tokens(args), 512)


class TestSamplingBackendTokenOracleEnvGate(CustomTestCase):
    """The 'token_oracle' choice is gated on SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE.

    The choice set is finalized when CLI arguments are registered, so each
    parser must reflect the environment at construction time.
    """

    def test_token_oracle_rejected_when_env_disabled(self):
        with patch.dict(os.environ, {"SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE": "0"}):
            with self.assertRaises(SystemExit):
                server_args_module.prepare_server_args(
                    [
                        "--model-path",
                        DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
                        "--sampling-backend",
                        "token_oracle",
                    ]
                )

    def test_token_oracle_accepted_when_env_enabled(self):
        with patch.dict(os.environ, {"SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE": "1"}):
            parsed = server_args_module.prepare_server_args(
                [
                    "--model-path",
                    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
                    "--sampling-backend",
                    "token_oracle",
                    # Explicit device so ServerArgs.__post_init__ does not call
                    # get_device() (fails on CPU-only CI runners) and does not run
                    # _handle_cpu_backends (which would override sampling_backend
                    # to "pytorch", masking what we want to verify).
                    "--device",
                    "cuda",
                ]
            )
        self.assertEqual(parsed.sampling_backend, "token_oracle")

    def test_gate_is_recomputed_for_each_parser(self):
        with patch.dict(os.environ, {"SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE": "1"}):
            enabled_parser = argparse.ArgumentParser()
            ServerArgs.add_cli_args(enabled_parser)

        with patch.dict(os.environ, {"SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE": "0"}):
            disabled_parser = argparse.ArgumentParser()
            ServerArgs.add_cli_args(disabled_parser)

        enabled_action = next(
            action
            for action in enabled_parser._actions
            if action.dest == "sampling_backend"
        )
        disabled_action = next(
            action
            for action in disabled_parser._actions
            if action.dest == "sampling_backend"
        )
        self.assertIn("token_oracle", enabled_action.choices)
        self.assertNotIn("token_oracle", disabled_action.choices)


class TestDeepEPv2Args(CustomTestCase):
    """DeepEP v2 server-argument resolution and validation."""

    def _args(self, **overrides):
        server_args = ServerArgs(model_path="dummy", moe_a2a_backend="deepep_v2")
        server_args._model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
        )
        # The dummy path does not initialize phase configs.
        server_args.cuda_graph_config = CudaGraphConfig(
            decode=PhaseConfig(backend=Backend.FULL, max_bs=512),
            prefill=PhaseConfig(backend=Backend.FULL, max_bs=512),
        )
        server_args._resolved_overrides = []
        valid = {f.name for f in dataclasses.fields(ServerArgs)}
        for key, value in overrides.items():
            # Reject stale field names before setattr silently accepts them.
            assert key in valid, f"{key} is not a ServerArgs field"
            setattr(server_args, key, value)
        return server_args

    def test_validated_architectures_allowed(self):
        for architecture in (
            "DeepseekV3ForCausalLM",
            "DeepseekV4ForCausalLM",
            "Qwen3MoeForCausalLM",
        ):
            args = self._args(moe_runner_backend="deep_gemm")
            args._model_config.hf_config.architectures = [architecture]
            handle_a2a_moe(args)

    def test_unvalidated_and_missing_architectures_rejected(self):
        for architectures in (
            ["Qwen2MoeForCausalLM"],
            ["Qwen3_5MoeForCausalLM"],
            [],
            None,
        ):
            args = self._args(moe_runner_backend="deep_gemm")
            args._model_config.hf_config.architectures = architectures
            with self.assertRaisesRegex(ValueError, "not validated"):
                handle_a2a_moe(args)

    def test_instance_connector_rejected(self):
        args = self._args(
            model_path="instance://worker/model",
            moe_runner_backend="deep_gemm",
        )
        with self.assertRaisesRegex(ValueError, "instance connector"):
            handle_a2a_moe(args)

    def test_deterministic_inference_rejected(self):
        args = self._args(
            moe_runner_backend="deep_gemm",
            enable_deterministic_inference=True,
        )
        with self.assertRaisesRegex(ValueError, "deterministic sorting"):
            handle_a2a_moe(args)

    def test_rl_on_policy_deterministic_inference_rejected(self):
        args = self._args(
            moe_runner_backend="deep_gemm",
            rl_on_policy_target="fsdp",
        )
        args._model_config.hf_config.architectures = ["Qwen3MoeForCausalLM"]
        with (
            envs.SGLANG_VLM_CACHE_SIZE_MB.override(envs.SGLANG_VLM_CACHE_SIZE_MB.get()),
            envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.override(
                envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get()
            ),
        ):
            handle_deterministic_inference(args)
        with self.assertRaisesRegex(ValueError, "deterministic sorting"):
            handle_a2a_moe(args)

    def test_deterministic_inference_does_not_affect_legacy_deepep(self):
        args = self._args(
            moe_a2a_backend="deepep",
            moe_runner_backend="deep_gemm",
            enable_deterministic_inference=True,
        )
        handle_a2a_moe(args)

    def test_runner_restored_by_declaration_fails_fast(self):
        # Validate the declaration-resolved runner rather than the raw field.
        args = self._args(moe_runner_backend="auto")
        args._resolved_overrides = [
            ("test_mxfp8", {"moe_runner_backend": "flashinfer_trtllm"})
        ]
        with self.assertRaises(ValueError):
            handle_a2a_moe(args)

    def test_declarations_resolve_ep_size_and_fusion(self):
        from sglang.srt.arg_groups.overrides import resolved_view

        args = self._args(moe_runner_backend="auto", tp_size=2)
        handle_a2a_moe(args)
        self.assertEqual(resolved_view(args).ep_size, args.tp_size)
        self.assertTrue(resolved_view(args).disable_shared_experts_fusion)

    def test_auto_runner_defaults_to_deep_gemm(self):
        from sglang.srt.arg_groups.overrides import resolved_view

        args = self._args(moe_runner_backend="auto")
        handle_a2a_moe(args)
        self.assertEqual(resolved_view(args).moe_runner_backend, "deep_gemm")

    def test_unsupported_runner_rejected(self):
        args = self._args(moe_runner_backend="flashinfer_trtllm")
        with self.assertRaises(ValueError):
            handle_a2a_moe(args)

    def test_triton_runner_rejected(self):
        args = self._args(moe_runner_backend="triton")
        with self.assertRaises(ValueError):
            handle_a2a_moe(args)

    def test_decode_graph_stays_enabled_in_both_comm_modes(self):
        for mode in ("direct", "hybrid"):
            args = self._args(moe_runner_backend="deep_gemm", deepep_v2_mode=mode)
            handle_a2a_moe(args)
            declared = resolution_result(args, "cuda_graph_config")
            self.assertEqual(declared.decode.backend, Backend.FULL)
            self.assertEqual(declared.prefill.backend, Backend.DISABLED)

    def test_two_batch_overlap_rejected(self):
        args = self._args(moe_runner_backend="deep_gemm", enable_two_batch_overlap=True)
        with self.assertRaises(ValueError):
            handle_a2a_moe(args)

    def test_speculative_draft_backend_rejected(self):
        for main_backend in ("none", "deepep", "deepep_v2"):
            args = self._args(
                moe_a2a_backend=main_backend,
                moe_runner_backend="deep_gemm",
                speculative_moe_a2a_backend="deepep_v2",
            )
            with self.assertRaisesRegex(ValueError, "speculative draft backend"):
                validate_deepep_v2_speculative_draft(args)

    def test_inherited_speculative_draft_backend_rejected(self):
        args = self._args(
            moe_runner_backend="deep_gemm",
            speculative_algorithm="EAGLE",
        )
        with self.assertRaisesRegex(ValueError, "speculative draft backend"):
            validate_deepep_v2_speculative_draft(args)

    def test_ngram_does_not_inherit_a_draft_backend(self):
        args = self._args(
            moe_runner_backend="deep_gemm",
            speculative_algorithm="NGRAM",
        )
        validate_deepep_v2_speculative_draft(args)

    def test_explicit_legacy_speculative_backend_allowed(self):
        args = self._args(
            moe_runner_backend="deep_gemm",
            speculative_algorithm="EAGLE",
            speculative_moe_a2a_backend="deepep",
        )
        validate_deepep_v2_speculative_draft(args)

    def test_resolved_legacy_speculative_backend_allowed(self):
        args = self._args(
            moe_runner_backend="deep_gemm",
            speculative_algorithm="EAGLE",
        )
        args._resolved_overrides = [
            (
                "test_speculative_backend",
                {"speculative_moe_a2a_backend": "deepep"},
            )
        ]
        validate_deepep_v2_speculative_draft(args)

    def test_prefill_chunk_exceeding_cap_rejected(self):
        args = self._args(moe_runner_backend="deep_gemm", chunked_prefill_size=2048)
        with envs.SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(1024):
            with self.assertRaisesRegex(ValueError, "NUM_MAX_DISPATCH_TOKENS_PER_RANK"):
                validate_deepep_v2_dispatch_token_budget(args)

    def test_prefill_chunk_at_cap_boundary_accepted(self):
        args = self._args(moe_runner_backend="deep_gemm", chunked_prefill_size=1024)
        with envs.SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(1024):
            validate_deepep_v2_dispatch_token_budget(args)

    def test_dynamic_chunking_probe_is_included(self):
        args = self._args(
            chunked_prefill_size=1024,
            max_prefill_tokens=1024,
            enable_dynamic_chunking=True,
            pp_size=2,
            disaggregation_mode="prefill",
        )
        with envs.SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(1024):
            with self.assertRaisesRegex(ValueError, "required=1280"):
                validate_deepep_v2_dispatch_token_budget(args)

    def test_disabled_chunking_uses_max_prefill_tokens(self):
        for disabled in (None, 0, -1):
            args = self._args(
                chunked_prefill_size=disabled,
                max_prefill_tokens=1024,
                disaggregation_mode="prefill",
            )
            with envs.SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(128):
                with self.assertRaisesRegex(ValueError, "required=1024"):
                    validate_deepep_v2_dispatch_token_budget(args)

    def test_decode_role_skips_prefill_capacity(self):
        args = self._args(
            chunked_prefill_size=4096,
            disaggregation_mode="decode",
            max_running_requests=32,
            dp_size=1,
        )
        with envs.SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(128):
            validate_deepep_v2_dispatch_token_budget(args)

    def test_decode_graph_capacity_boundaries(self):
        for max_bs, raises in ((128, False), (129, True)):
            args = self._args(
                disaggregation_mode="decode",
                max_running_requests=None,
            )
            args.cuda_graph_config.decode.max_bs = max_bs
            with envs.SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(128):
                if raises:
                    with self.assertRaisesRegex(ValueError, "decode CUDA graph"):
                        validate_deepep_v2_dispatch_token_budget(args)
                else:
                    validate_deepep_v2_dispatch_token_budget(args)

    def test_dp_attention_divides_max_running_requests_per_rank(self):
        args = self._args(
            disaggregation_mode="decode",
            max_running_requests=256,
            tp_size=8,
            dp_size=8,
            enable_dp_attention=True,
        )
        with envs.SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(128):
            validate_deepep_v2_dispatch_token_budget(args)

    def test_tp_only_max_running_requests_is_not_divided(self):
        args = self._args(
            disaggregation_mode="decode",
            max_running_requests=256,
            tp_size=8,
            dp_size=1,
            enable_dp_attention=False,
        )
        with envs.SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(128):
            with self.assertRaisesRegex(ValueError, "decode CUDA graph"):
                validate_deepep_v2_dispatch_token_budget(args)

    def test_memory_derived_eager_pool_remains_runtime_validated(self):
        args = self._args(
            disaggregation_mode="decode",
            max_running_requests=None,
        )
        args.cuda_graph_config.decode.backend = Backend.DISABLED
        with envs.SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(1):
            validate_deepep_v2_dispatch_token_budget(args)

    def test_speculative_decode_width_is_included(self):
        args = self._args(
            disaggregation_mode="decode",
            speculative_algorithm="EAGLE",
            speculative_num_draft_tokens=8,
            max_running_requests=256,
            dp_size=8,
            enable_dp_attention=True,
        )
        with envs.SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(128):
            with self.assertRaisesRegex(ValueError, "tokens/request=8"):
                validate_deepep_v2_dispatch_token_budget(args)

    def test_adaptive_speculative_uses_widest_candidate(self):
        args = self._args(
            disaggregation_mode="decode",
            speculative_algorithm="EAGLE",
            speculative_num_draft_tokens=4,
            speculative_adaptive=True,
            max_running_requests=128,
            dp_size=8,
            enable_dp_attention=True,
        )
        with patch(
            "sglang.srt.arg_groups.moe_hook.max_speculative_num_draft_tokens",
            return_value=16,
        ):
            with envs.SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(128):
                with self.assertRaisesRegex(ValueError, "tokens/request=16"):
                    validate_deepep_v2_dispatch_token_budget(args)

    def test_prefill_role_skips_decode_capacity(self):
        args = self._args(
            disaggregation_mode="prefill",
            chunked_prefill_size=64,
            max_running_requests=8192,
        )
        with envs.SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(128):
            validate_deepep_v2_dispatch_token_budget(args)

    def test_other_backend_skips_capacity_validation(self):
        args = self._args(
            moe_a2a_backend="deepep",
            chunked_prefill_size=4096,
            max_running_requests=4096,
        )
        with envs.SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(1):
            validate_deepep_v2_dispatch_token_budget(args)

    def test_capacity_validation_uses_resolved_backend(self):
        args = self._args(chunked_prefill_size=4096)
        args._resolved_overrides = [("test", {"moe_a2a_backend": "deepep"})]
        with envs.SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(1):
            validate_deepep_v2_dispatch_token_budget(args)


class TestHandleCrashDumpEnv(CustomTestCase):
    _COREDUMP_ENV_KEYS = (
        "CUDA_ENABLE_COREDUMP_ON_EXCEPTION",
        "CUDA_ENABLE_USER_TRIGGERED_COREDUMP",
        "CUDA_COREDUMP_SHOW_PROGRESS",
        "CUDA_COREDUMP_GENERATION_FLAGS",
        "CUDA_COREDUMP_FILE",
        "CUDA_COREDUMP_PIPE",
    )

    def _run_handler(self, crash_dump_folder, preset_env=None):
        server_args = ServerArgs.__new__(ServerArgs)
        server_args.crash_dump_folder = crash_dump_folder
        with patch.dict(os.environ, preset_env or {}):
            for key in self._COREDUMP_ENV_KEYS:
                if key not in (preset_env or {}):
                    os.environ.pop(key, None)
            handle_crash_dump_env(server_args)

    def test_creates_coredump_dir_when_auto_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run_handler(tmp)
            self.assertTrue(
                os.path.isdir(os.path.join(tmp, socket.gethostname())),
                "coredump dir not created for auto-set CUDA_COREDUMP_FILE",
            )

    def test_creates_coredump_dir_when_env_preset(self):
        # Regression test: when CUDA_COREDUMP_FILE is preset, the coredump
        # directory must still be created up front.
        with tempfile.TemporaryDirectory() as tmp:
            preset_dir = os.path.join(tmp, "preset-location")
            self._run_handler(
                tmp,
                preset_env={"CUDA_COREDUMP_FILE": f"{preset_dir}/%h/core.cuda.%t.%p"},
            )
            self.assertTrue(
                os.path.isdir(os.path.join(preset_dir, socket.gethostname())),
                "coredump dir not created for preset CUDA_COREDUMP_FILE",
            )


class TestGrpcServerArgs(CustomTestCase):
    """Native gRPC is enabled by --grpc-port (or SGLANG_GRPC_PORT) and runs
    alongside HTTP; --smg-grpc-mode (and the deprecated --grpc-mode) select the
    legacy SMG server. Worker-threads / max-prefill-tokens are env-only knobs.

    The gRPC setup lives in `serving_hook.handle_deprecated_args`, which
    __post_init__ skips for dummy models, so these tests build a dummy
    ServerArgs and invoke that handler directly (mirroring the real flow for a
    concrete model path).
    """

    @staticmethod
    def _args(**kwargs):
        return ServerArgs(model_path="dummy", **kwargs)

    def test_http_only_high_port_does_not_derive_grpc_port(self):
        sa = self._args(port=56000)
        handle_deprecated_args(sa)
        self.assertIsNone(resolution_result(sa, "grpc_port"))

    def test_grpc_port_enables_native_and_env_knobs(self):
        sa = self._args(grpc_port=50051)
        with envs.SGLANG_GRPC_WORKER_THREADS.override(8):
            handle_deprecated_args(sa)
        self.assertEqual(resolution_result(sa, "grpc_port"), 50051)
        self.assertEqual(resolution_result(sa, "grpc_worker_threads"), 8)

    def test_env_grpc_port_enables_native(self):
        sa = self._args(port=30000)
        with envs.SGLANG_GRPC_PORT.override(45000):
            handle_deprecated_args(sa)
        self.assertEqual(resolution_result(sa, "grpc_port"), 45000)

    @staticmethod
    def _sidecar_parser():
        parser = server_args_module.argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        return parser

    def test_sidecar_builds_loopback_grpc_endpoints(self):
        self.assertEqual(
            build_sidecar_endpoint("0.0.0.0", 50051),
            "http://127.0.0.1:50051",
        )
        self.assertEqual(
            build_sidecar_endpoint("::", 50051),
            "http://[::1]:50051",
        )
        self.assertEqual(
            build_sidecar_endpoint("[::]", 50051),
            "http://[::1]:50051",
        )

    def test_sidecar_args_parse_as_exact_json_argv(self):
        argv = ["--flag", "value"]
        parsed = self._sidecar_parser().parse_args(
            ["--model-path", "dummy", "--sidecar-args", json.dumps(argv)]
        )
        self.assertEqual(parsed.sidecar_args, argv)

    def test_start_sidecar_passes_endpoint_and_provider_argv_separately(self):
        from sglang.srt.runtime_context import get_context as get_context_for_config

        server_args = SimpleNamespace(
            sidecar="example.sidecar",
            sidecar_args=[
                "--sidecar-shutdown-timeout",
                "42",
                "--grpc-connections",
                "2",
            ],
            host="127.0.0.1",
        )
        # Every value the sidecar reads is resolved config, so the case states
        # them all through the context rather than half here and half in a
        # stand-in the readers no longer consult.
        override = get_context_for_config().override_server_args(
            grpc_port=50051,
            sidecar="example.sidecar",
            sidecar_args=[
                "--sidecar-shutdown-timeout",
                "42",
                "--grpc-connections",
                "2",
            ],
            host="127.0.0.1",
        )
        override.install()
        self.addCleanup(override.restore)
        with (
            patch("sglang.srt.entrypoints.sidecar.mp.get_context") as get_context,
            patch("sglang.srt.entrypoints.sidecar.Sidecar") as sidecar_class,
        ):
            start_sidecar()

        process_kwargs = get_context.return_value.Process.call_args.kwargs
        self.assertEqual(process_kwargs["name"], "sglang_sidecar_example.sidecar")
        self.assertEqual(process_kwargs["target"], _run_sidecar)
        self.assertEqual(
            process_kwargs["args"],
            (
                "example.sidecar",
                ["--grpc-connections", "2"],
                "http://127.0.0.1:50051",
            ),
        )
        sidecar_class.assert_called_once_with(
            get_context.return_value.Process.return_value,
            "example.sidecar",
            shutdown_timeout=42.0,
        )

    def test_sidecar_requires_native_grpc(self):
        sa = self._args(sidecar="example.sidecar")
        with self.assertRaisesRegex(ValueError, "requires --grpc-port"):
            handle_deprecated_args(sa)

    def test_sidecar_rejects_legacy_grpc(self):
        sa = self._args(sidecar="example.sidecar", smg_grpc_mode=True)
        with self.assertRaisesRegex(ValueError, "native gRPC server"):
            handle_deprecated_args(sa)

    def test_sidecar_rejects_empty_value(self):
        sa = self._args(sidecar="", grpc_port=50051)
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            handle_deprecated_args(sa)

    def test_sidecar_sets_endpoint_env_before_import_and_calls_main(self):
        main = MagicMock()

        def import_module(module_name):
            self.assertEqual(module_name, "example.sidecar")
            self.assertEqual(
                os.environ[SGLANG_GRPC_ENDPOINT_ENV],
                "http://127.0.0.1:50051",
            )
            self.assertEqual(os.environ["DYN_NAMESPACE"], "pluh")
            return SimpleNamespace(main=main)

        with (
            patch.dict(
                os.environ,
                {
                    SGLANG_GRPC_ENDPOINT_ENV: "http://stale.example:1",
                    "DYN_NAMESPACE": "pluh",
                },
            ),
            patch("sglang.srt.entrypoints.sidecar.kill_itself_when_parent_died"),
            patch(
                "sglang.srt.entrypoints.sidecar.importlib.import_module",
                side_effect=import_module,
            ),
        ):
            _run_sidecar(
                "example.sidecar",
                ["--provider-flag", "value"],
                "http://127.0.0.1:50051",
            )

        main.assert_called_once_with(["--provider-flag", "value"])

    def test_sidecar_stop_uses_configured_shutdown_timeout(self):
        proc = MagicMock(pid=1234)
        proc.is_alive.side_effect = [True, True]
        sidecar = Sidecar(
            proc,
            "example.sidecar",
            shutdown_timeout=42.0,
        )

        with patch("sglang.srt.entrypoints.sidecar.kill_process_tree") as kill_tree:
            sidecar.stop()

        proc.terminate.assert_called_once_with()
        proc.join.assert_called_once_with(timeout=42.0)
        kill_tree.assert_called_once_with(1234, wait_timeout=42.0)

    def test_legacy_smg_derives_grpc_port_from_http_port(self):
        sa = self._args(port=30000, smg_grpc_mode=True)
        handle_deprecated_args(sa)
        self.assertEqual(resolution_result(sa, "grpc_port"), 40000)

    def test_grpc_mode_is_deprecated_alias_for_smg_grpc_mode(self):
        sa = self._args(grpc_mode=True)
        with self.assertLogs(serving_hook.logger, level="WARNING") as cm:
            handle_deprecated_args(sa)
        self.assertTrue(resolution_result(sa, "smg_grpc_mode"))
        self.assertTrue(any("--grpc-mode is deprecated" in line for line in cm.output))

    def test_legacy_smg_takes_precedence_over_grpc_port(self):
        sa = self._args(grpc_port=50051, smg_grpc_mode=True)
        handle_deprecated_args(sa)
        self.assertTrue(resolution_result(sa, "smg_grpc_mode"))
        self.assertEqual(resolution_result(sa, "grpc_port"), 50051)

    def test_native_grpc_rejects_multi_tokenizer(self):
        sa = self._args(grpc_port=40000, tokenizer_worker_num=2)
        with self.assertRaises(ValueError):
            handle_deprecated_args(sa)

    def test_native_grpc_rejects_http_auth(self):
        sa = self._args(grpc_port=40000, api_key="secret")
        with self.assertRaises(ValueError):
            handle_deprecated_args(sa)

    def test_invalid_grpc_worker_threads_rejected(self):
        sa = self._args(grpc_port=40000)
        with envs.SGLANG_GRPC_WORKER_THREADS.override(0):
            with self.assertRaises(ValueError):
                handle_deprecated_args(sa)

    def test_start_server_call_site_matches_native_signature(self):
        """Regression for the startup blocker: the native start_server binding
        only accepts (host, port, runtime_handle, worker_threads, ...). The
        arg-parsing tests above never call start_server, so a stray kwarg (e.g.
        the removed max_prefill_tokens) would only surface as a TypeError at
        launch. This mocks the native extension and locks the kwarg set."""
        from sglang.srt.entrypoints import http_server

        fake_core = SimpleNamespace(start_server=MagicMock(return_value="handle"))
        fake_bridge = SimpleNamespace(RuntimeHandle=MagicMock(return_value="rt"))
        override = get_context().override_server_args(
            host="127.0.0.1", grpc_port=50051, grpc_worker_threads=4
        )
        server_args = override.install()
        self.addCleanup(override.restore)
        with (
            patch(
                "sglang.srt.rust_extensions.load_rust_extension",
                return_value=fake_core,
            ) as load_rust_extension,
            patch.dict(
                "sys.modules", {"sglang.srt.entrypoints.grpc_bridge": fake_bridge}
            ),
        ):
            handle = http_server._start_native_grpc_server_for_runtime(
                server_args=server_args,
                tokenizer_manager=MagicMock(),
                template_manager=MagicMock(),
                scheduler_info={},
                grpc_port=get_serving().grpc_port,
            )

        self.assertEqual(handle, "handle")
        load_rust_extension.assert_called_once_with("sglang.srt.rust_extensions._grpc")
        _, kwargs = fake_core.start_server.call_args
        self.assertEqual(
            set(kwargs), {"host", "port", "runtime_handle", "worker_threads"}
        )
        self.assertEqual(kwargs["worker_threads"], 4)
        self.assertNotIn("max_prefill_tokens", kwargs)


class TestTwoBatchOverlapBackend(CustomTestCase):
    """Non-EP DP two-batch-overlap backend requirement.

    With no EP a2a backend (moe_a2a_backend='none'), --enable-two-batch-overlap
    is only valid on the DeepSeek-V4 non-EP DP TP-MoE path (overlapping the DP
    all_gatherv / reduce_scatterv with the other ubatch's compute), which
    requires --enable-dp-attention. This replaced the removed opt-in
    SGLANG_ENABLE_DP_TBO env: enabling DP TBO now needs no extra flag.

    dummy-model short-circuits __post_init__, so the guard handler is invoked
    directly (same pattern as TestWaterfillArgs)."""

    def _args(self, **overrides):
        args = ServerArgs(model_path="dummy")
        args.enable_two_batch_overlap = True
        args.moe_a2a_backend = "none"
        args.enable_dp_attention = False
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_no_a2a_without_dp_attention_raises(self):
        args = self._args(enable_dp_attention=False)
        with self.assertRaisesRegex(ValueError, "enable-dp-attention"):
            check_two_batch_overlap(args)

    def test_no_a2a_with_dp_attention_ok(self):
        # DP TBO path is valid: --enable-dp-attention + --enable-two-batch-overlap
        # with a2a backend 'none' must NOT raise (no SGLANG_ENABLE_DP_TBO needed).
        args = self._args(enable_dp_attention=True)
        check_two_batch_overlap(args)

    def test_ep_a2a_backend_ok_without_dp_attention(self):
        # EP a2a path (e.g. deepep) overlaps dispatch/combine; the guard does not
        # require dp-attention there.
        args = self._args(moe_a2a_backend="deepep", enable_dp_attention=False)
        check_two_batch_overlap(args)


class TestDcpKvEventContract(CustomTestCase):
    """DCP widens the radix-tree page to page_size * dcp_size, which the
    advertised KV-event block size must reflect."""

    KV_EVENTS = '{"publisher":"zmq","topic":"kv","endpoint":"tcp://*:5557"}'

    def test_kv_events_descriptor_reports_logical_block_size(self):
        """Advertising the physical page_size made every KV-aware router hash
        prompts at a width no emitted block can match, silently pinning its
        hit rate to zero while stores kept applying cleanly."""
        args = ServerArgs(
            model_path="dummy",
            tp_size=4,
            dcp_size=4,
            page_size=64,
            kv_events_config=self.KV_EVENTS,
        )
        self.assertEqual(describe_kv_events_publisher(args)["block_size"], 256)
        args = ServerArgs(
            model_path="dummy", page_size=64, kv_events_config=self.KV_EVENTS
        )
        self.assertEqual(describe_kv_events_publisher(args)["block_size"], 64)

    def test_kv_event_block_size_widens_a_single_token_page(self):
        # page_size=1 + DCP is a real deployment shape: the allocator is still
        # paged, at dcp_size.
        from sglang.srt.arg_groups.overrides import (
            kv_event_block_size_of,
            resolving_view,
        )

        args = ServerArgs(model_path="dummy", tp_size=8, dcp_size=8, page_size=1)
        self.assertEqual(kv_event_block_size_of(resolving_view(args)), 8)


if __name__ == "__main__":
    unittest.main()
