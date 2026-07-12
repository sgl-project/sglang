import importlib
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import sglang.srt.server_args as server_args_module
from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.environ import envs
from sglang.srt.layers.cp.base import is_cp_enabled, is_interleave
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    Phase,
    PhaseConfig,
)
from sglang.srt.server_args import PortArgs, ServerArgs, prepare_server_args
from sglang.srt.server_args_config_parser import ConfigArgumentMerger
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    CustomTestCase,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cpu_ci(est_time=12, suite="base-c-test-cpu")

# Mock get_device() so all tests run on CPU-only CI runners
_mock_device = patch("sglang.srt.server_args.get_device", return_value="cuda")
_mock_device.start()


class TestPrepareServerArgs(CustomTestCase):
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


class TestMambaCacheStochasticRounding(unittest.TestCase):
    def test_rejects_fp32_ssm_cache(self):
        server_args = ServerArgs(
            model_path="dummy",
            mamba_ssm_dtype="float32",
            enable_mamba_cache_stochastic_rounding=True,
        )

        with self.assertRaisesRegex(ValueError, "--mamba-ssm-dtype float16"):
            server_args._handle_mamba_backend()

    @patch("sglang.srt.server_args.is_cuda", return_value=False)
    def test_rejects_non_cuda(self, _mock_is_cuda):
        server_args = ServerArgs(
            model_path="dummy",
            mamba_ssm_dtype="float16",
            enable_mamba_cache_stochastic_rounding=True,
        )

        with self.assertRaisesRegex(ValueError, "NVIDIA CUDA"):
            server_args._handle_mamba_backend()

    @patch("sglang.srt.server_args.is_cuda", return_value=True)
    @patch("sglang.srt.server_args.is_sm100_supported", return_value=False)
    def test_rejects_triton_without_sm100(self, _mock_sm100, _mock_is_cuda):
        server_args = ServerArgs(
            model_path="dummy",
            mamba_ssm_dtype="float16",
            mamba_backend="triton",
            enable_mamba_cache_stochastic_rounding=True,
        )

        with self.assertRaisesRegex(ValueError, "requires SM100"):
            server_args._handle_mamba_backend()


class TestLoadBalanceMethod(unittest.TestCase):
    def _load_balance_args(self, **kwargs):
        server_args = ServerArgs(model_path="dummy", **kwargs)
        server_args._handle_pd_disaggregation()
        server_args._handle_load_balance_method()
        return server_args

    def test_non_pd_defaults_to_round_robin(self):
        server_args = self._load_balance_args(disaggregation_mode="null")
        self.assertEqual(server_args.load_balance_method, "round_robin")

    def test_pd_prefill_defaults_to_follow_bootstrap_room(self):
        server_args = self._load_balance_args(disaggregation_mode="prefill")
        self.assertEqual(server_args.load_balance_method, "follow_bootstrap_room")

    def test_pd_decode_defaults_to_round_robin(self):
        server_args = self._load_balance_args(disaggregation_mode="decode")
        self.assertEqual(server_args.load_balance_method, "round_robin")

    def test_pd_decode_radix_cache_rejects_hisparse(self):
        server_args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="decode",
            disaggregation_decode_enable_radix_cache=True,
            disaggregation_transfer_backend="nixl",
            enable_hisparse=True,
        )
        with self.assertRaises(ValueError) as context:
            server_args._handle_pd_disaggregation()

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
            server_args._handle_pd_disaggregation()

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

        self.assertFalse(server_args.disable_radix_cache)
        self.assertEqual(server_args.disaggregation_transfer_backend, "mooncake")


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
            SimpleNamespace(
                get_model_config=lambda: SimpleNamespace(hf_config=hf), **defaults
            )
        )
        with (
            patch("sglang.srt.configs.model_config.is_deepseek_dsa", return_value=True),
            patch("sglang.srt.arg_groups.overrides.is_npu", return_value=False),
            patch("sglang.srt.arg_groups.overrides.is_xpu", return_value=False),
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

    @patch("sglang.srt.server_args.is_hip", return_value=False)
    def test_hisparse_defaults_to_flashmla_sparse_on_cuda_bfloat16(self, _mock_is_hip):
        resolved = self._resolve("bfloat16")

        self.assertEqual(resolved["dsa_prefill_backend"], "flashmla_sparse")
        self.assertEqual(resolved["dsa_decode_backend"], "flashmla_sparse")

    @patch("sglang.srt.server_args.is_hip", return_value=False)
    def test_hisparse_defaults_to_flashmla_kv_on_cuda_fp8(self, _mock_is_hip):
        resolved = self._resolve("fp8_e4m3")

        self.assertEqual(resolved["dsa_prefill_backend"], "flashmla_kv")
        self.assertEqual(resolved["dsa_decode_backend"], "flashmla_kv")

    @patch("sglang.srt.server_args.is_hip", return_value=True)
    def test_hisparse_defaults_to_tilelang_on_rocm(self, _mock_is_hip):
        resolved = self._resolve("bfloat16")

        self.assertEqual(resolved["dsa_prefill_backend"], "tilelang")
        self.assertEqual(resolved["dsa_decode_backend"], "tilelang")

    @patch("sglang.srt.server_args.is_hip", return_value=True)
    def test_hisparse_preserves_rocm_user_backend_and_defaults_missing_side(
        self, _mock_is_hip
    ):
        resolved = self._resolve("bfloat16", dsa_prefill_backend="tilelang")

        self.assertEqual(resolved["dsa_prefill_backend"], "tilelang")
        self.assertEqual(resolved["dsa_decode_backend"], "tilelang")

    @patch("sglang.srt.server_args.is_hip", return_value=True)
    def test_hisparse_accepts_aiter_backend_on_rocm(self, _mock_is_hip):
        server_args = ServerArgs(
            model_path="dummy",
            enable_hisparse=True,
            kv_cache_dtype="bfloat16",
            dsa_prefill_backend="aiter",
            dsa_decode_backend="aiter",
        )

        server_args._validate_hisparse_dsa_backend("dsa_prefill_backend", "prefill")
        server_args._validate_hisparse_dsa_backend("dsa_decode_backend", "decode")

    @patch("sglang.srt.server_args.is_hip", return_value=True)
    def test_hisparse_rejects_cuda_backend_on_rocm(self, _mock_is_hip):
        server_args = ServerArgs(
            model_path="dummy",
            enable_hisparse=True,
            kv_cache_dtype="bfloat16",
            dsa_prefill_backend="flashmla_sparse",
        )

        with self.assertRaisesRegex(ValueError, "tilelang"):
            server_args._validate_hisparse_dsa_backend("dsa_prefill_backend", "prefill")

    @patch("sglang.srt.server_args.is_hip", return_value=False)
    def test_hisparse_rejects_rocm_backend_on_cuda(self, _mock_is_hip):
        server_args = ServerArgs(
            model_path="dummy",
            enable_hisparse=True,
            kv_cache_dtype="bfloat16",
            dsa_decode_backend="tilelang",
        )

        with self.assertRaisesRegex(ValueError, "flashmla_sparse"):
            server_args._validate_hisparse_dsa_backend("dsa_decode_backend", "decode")

    def test_hisparse_accepts_bfloat16_kv_cache_dtype(self):
        server_args = ServerArgs(
            model_path="dummy",
            enable_hisparse=True,
            kv_cache_dtype="bfloat16",
        )

        server_args._validate_hisparse_kv_cache_dtype()

    def test_hisparse_accepts_fp8_e4m3_kv_cache_dtype(self):
        server_args = ServerArgs(
            model_path="dummy",
            enable_hisparse=True,
            kv_cache_dtype="fp8_e4m3",
        )

        server_args._validate_hisparse_kv_cache_dtype()

    def test_hisparse_rejects_unsupported_kv_cache_dtype(self):
        server_args = ServerArgs(
            model_path="dummy",
            enable_hisparse=True,
            kv_cache_dtype="float16",
        )

        with self.assertRaisesRegex(ValueError, r"fp8_e4m3"):
            server_args._validate_hisparse_kv_cache_dtype()


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
        # Short-circuit get_model_config(): the fa4 page_size branch only needs
        # use_mla_backend() (mocked) and is_sm100_supported() (mocked), not a
        # real model_config. Pre-set the attribute so get_model_config returns
        # early without touching ModelConfig.from_server_args.
        args.model_config = MagicMock()
        args.model_config.hf_config.dual_chunk_attention_config = None
        return args

    @patch("sglang.srt.arg_groups.overrides.is_sm100_supported", return_value=True)
    @patch("sglang.srt.server_args.ServerArgs.use_mla_backend", return_value=False)
    def test_combined_attention_backend_fa4_forces_page_size_128(
        self, _mock_mla, _mock_sm100
    ):
        # `--attention-backend fa4` (combined): prefill/decode fields stay None.
        args = self._make_args(attention_backend="fa4")

        args._handle_attention_backend_compatibility()

        from sglang.srt.arg_groups.overrides import resolved_view

        self.assertEqual(args.page_size, 1)  # dual-apply retired: pristine
        self.assertEqual(resolved_view(args).page_size, 128)

    @patch("sglang.srt.arg_groups.overrides.is_sm100_supported", return_value=True)
    @patch("sglang.srt.server_args.ServerArgs.use_mla_backend", return_value=False)
    def test_explicit_prefill_fa4_forces_page_size_128(self, _mock_mla, _mock_sm100):
        # `--prefill-attention-backend fa4`: the previously-covered path.
        args = self._make_args(attention_backend=None, prefill="fa4", page_size=1)

        args._handle_attention_backend_compatibility()

        from sglang.srt.arg_groups.overrides import resolved_view

        self.assertEqual(args.page_size, 1)  # dual-apply retired: pristine
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
            enable_aiter_allreduce_fusion=False,
        )
        defaults.update(overrides)
        for key, value in defaults.items():
            setattr(server_args, key, value)
        return server_args

    def test_canonical_prefill_cp_requires_strategy(self):
        args = self.parser.parse_args(["--model", "dummy", "--enable-prefill-cp"])

        self.assertTrue(args.enable_prefill_cp)
        self.assertIsNone(args.cp_strategy)

        server_args = self._new_cp_args(
            enable_prefill_cp=args.enable_prefill_cp,
            cp_strategy=args.cp_strategy,
        )
        with self.assertRaisesRegex(ValueError, "--cp-strategy"):
            server_args._handle_context_parallelism()

    def test_deprecated_dsa_cp_mode_maps_to_unified_strategy(self):
        args = self.parser.parse_args(
            [
                "--model",
                "dummy",
                "--enable-dsa-prefill-context-parallel",
                "--dsa-prefill-cp-mode",
                "round-robin-split",
            ]
        )
        server_args = self._new_cp_args(
            enable_dsa_prefill_context_parallel=(
                args.enable_dsa_prefill_context_parallel
            ),
            dsa_prefill_cp_mode=args.dsa_prefill_cp_mode,
        )

        server_args._handle_legacy_cp_arguments()

        self.assertTrue(server_args.enable_prefill_cp)
        self.assertEqual(server_args.cp_strategy, "interleave")
        self.assertEqual(server_args.dsa_prefill_cp_mode, "round-robin-split")

    def test_canonical_interleave_cp_mirrors_to_dsa_runtime_aliases(self):
        server_args = self._new_cp_args(
            enable_prefill_cp=True,
            cp_strategy="interleave",
            attention_backend="dsa",
        )

        server_args._handle_legacy_cp_arguments()
        server_args._handle_context_parallelism()

        self.assertTrue(server_args.enable_dsa_prefill_context_parallel)
        self.assertFalse(server_args.enable_prefill_context_parallel)
        self.assertEqual(server_args.dsa_prefill_cp_mode, "round-robin-split")
        self.assertEqual(server_args.prefill_cp_mode, "round-robin-split")

    def test_context_parallel_handler_initializes_cp_strategy(self):
        server_args = self._new_cp_args(
            enable_prefill_cp=True,
            cp_strategy="interleave",
            attn_cp_size=2,
            tp_size=2,
        )

        server_args._handle_context_parallelism()

        self.assertTrue(is_cp_enabled())
        self.assertTrue(is_interleave())

    def test_registered_cp_legacy_args_map_to_unified_strategy(self):
        cases = [
            (
                "deepseek_v3_mla_cp",
                dict(enable_prefill_context_parallel=True),
                "zigzag",
                "in-seq-split",
                False,
                True,
            ),
            (
                "qwen3_gqa_cp",
                dict(
                    enable_prefill_context_parallel=True,
                    tp_size=4,
                    attn_cp_size=2,
                ),
                "zigzag",
                "in-seq-split",
                False,
                True,
            ),
            (
                "deepseek_v32_dsa_in_seq_split",
                dict(
                    enable_dsa_prefill_context_parallel=True,
                    dsa_prefill_cp_mode="in-seq-split",
                    tp_size=8,
                    dp_size=2,
                    attn_cp_size=4,
                ),
                "zigzag",
                "in-seq-split",
                True,
                False,
            ),
            (
                "deepseek_v32_dsa_round_robin_split",
                dict(
                    enable_dsa_prefill_context_parallel=True,
                    tp_size=8,
                    attn_cp_size=8,
                ),
                "interleave",
                "round-robin-split",
                True,
                False,
            ),
            (
                "deepseek_v4_flash_fp4_b200_dsa_round_robin_split",
                dict(
                    enable_dsa_prefill_context_parallel=True,
                    dsa_prefill_cp_mode="round-robin-split",
                    tp_size=4,
                    attn_cp_size=4,
                ),
                "interleave",
                "round-robin-split",
                True,
                False,
            ),
        ]

        for name, overrides, strategy, mode, expect_dsa, expect_generic in cases:
            with self.subTest(name=name):
                server_args = self._new_cp_args(**overrides)

                server_args._handle_legacy_cp_arguments()
                server_args._handle_context_parallelism()

                self.assertTrue(server_args.enable_prefill_cp)
                self.assertEqual(server_args.cp_strategy, strategy)
                self.assertEqual(server_args.dsa_prefill_cp_mode, mode)
                self.assertEqual(server_args.prefill_cp_mode, mode)
                self.assertEqual(
                    server_args.enable_dsa_prefill_context_parallel, expect_dsa
                )
                self.assertEqual(
                    server_args.enable_prefill_context_parallel, expect_generic
                )


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
        server_args._handle_ssl_validation()
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
        server_args = ServerArgs(model_path="dummy")
        self.assertIs(server_args.ssl_verify(), True)

    @patch("os.path.isfile", return_value=True)
    def test_ssl_verify_with_ssl_no_ca(self, _mock_isfile):
        server_args = self._validate_ssl(ssl_keyfile="key.pem", ssl_certfile="cert.pem")
        self.assertIs(server_args.ssl_verify(), False)

    @patch("os.path.isfile", return_value=True)
    def test_ssl_verify_with_ssl_and_ca(self, _mock_isfile):
        server_args = self._validate_ssl(
            ssl_keyfile="key.pem",
            ssl_certfile="cert.pem",
            ssl_ca_certs="ca.pem",
        )
        self.assertEqual(server_args.ssl_verify(), "ca.pem")

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
        self.assertTrue(server_args.enable_ssl_refresh)


class TestHiCacheArgs(unittest.TestCase):
    def _make_args(self, **overrides) -> ServerArgs:
        args = ServerArgs(model_path="dummy")
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def _assert_hicache_fields(
        self,
        args: ServerArgs,
        *,
        expected_io_backend: str,
        expected_mem_layout: str,
        expected_decode_backend: str | None = None,
    ):
        self.assertEqual(args.hicache_io_backend, expected_io_backend)
        self.assertEqual(args.hicache_mem_layout, expected_mem_layout)
        if expected_decode_backend is not None:
            self.assertEqual(args.decode_attention_backend, expected_decode_backend)

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
                args._handle_hicache()
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

        args._handle_hicache()

        self.assertEqual(args.hicache_io_backend, "kernel")
        self.assertEqual(args.hicache_mem_layout, "page_first")
        self.assertIsNone(args.decode_attention_backend)


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
        self.assertEqual(server_args.decoupled_spec_role, "verifier")
        self.assertEqual(server_args.decoupled_spec_bind_endpoint, "ipc:///tmp/v")
        self.assertEqual(server_args.decoupled_spec_connect_endpoints, ["ipc:///tmp/d"])
        self.assertEqual(server_args.decoupled_spec_rank, 0)
        self.assertEqual(server_args.spec_trace_dir, "/tmp/tr")

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
            args.get_model_config = lambda: SimpleNamespace(
                hf_config=SimpleNamespace(
                    architectures=["LlamaForCausalLM"],
                    get_text_config=lambda: SimpleNamespace(),
                )
            )

            handle_speculative_decoding(args)

        self.assertTrue(args.speculative_adaptive)
        self.assertEqual(args.speculative_eagle_topk, 1)
        self.assertEqual(args.speculative_num_steps, 3)
        self.assertEqual(args.speculative_num_draft_tokens, 4)


class TestDeepEPWaterfillArgs(CustomTestCase):
    def test_waterfill_enforces_shared_experts_fusion(self):
        server_args = ServerArgs(
            model_path="dummy",
            moe_a2a_backend="deepep",
            enable_deepep_waterfill=True,
            disable_shared_experts_fusion=True,
        )
        # dummy-model path short-circuits __post_init__; invoke the handler directly.
        server_args._handle_a2a_moe()

        from sglang.srt.arg_groups.overrides import resolved_view

        # dual-apply retired: the fields stay pristine, the declarations win
        self.assertTrue(server_args.disable_shared_experts_fusion)
        self.assertFalse(resolved_view(server_args).disable_shared_experts_fusion)
        self.assertTrue(server_args.enforce_shared_experts_fusion)

    def test_waterfill_overrides_moe_a2a_backend_to_deepep(self):
        server_args = ServerArgs(
            model_path="dummy",
            moe_a2a_backend="none",
            enable_deepep_waterfill=True,
        )
        # dummy-model path short-circuits __post_init__; invoke the handler directly.
        server_args._handle_a2a_moe()

        from sglang.srt.arg_groups.overrides import resolved_view

        self.assertEqual(server_args.moe_a2a_backend, "none")  # pristine
        self.assertEqual(resolved_view(server_args).moe_a2a_backend, "deepep")
        self.assertTrue(server_args.enforce_shared_experts_fusion)

    def test_waterfill_supports_deepep_low_latency_mode(self):
        server_args = ServerArgs(
            model_path="dummy",
            moe_a2a_backend="deepep",
            enable_deepep_waterfill=True,
            deepep_mode="low_latency",
        )
        # dummy-model path short-circuits __post_init__; invoke the handler directly.
        server_args._handle_a2a_moe()

        self.assertEqual(server_args.deepep_mode, "low_latency")
        self.assertFalse(server_args.disable_cuda_graph)
        self.assertTrue(server_args.enforce_shared_experts_fusion)


class TestPrefillOnlyDisableKvCache(unittest.TestCase):
    """Validation for --prefill-only-disable-kv-cache.

    The flag wires NoOpMHATokenToKVPool, which is only safe when:
      - the engine is in embedding mode (fa_skip_kv_cache active in FA backend),
      - chunked_prefill_size == -1 (no inter-chunk K/V reuse),
      - disable_radix_cache (radix cache otherwise indexes empty pool slots),
      - no context-parallel attention (CP writes to the pool via set_kv_buffer),
      - no HiSparse (uses a different pool family),
      - kv_cache_dtype != fp4_e2m1 (FP4 pool is a separate allocation path).
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
        sa._handle_legacy_cp_arguments()
        sa._validate_prefill_only_disable_kv_cache_args()
        return sa

    def test_valid_minimal_config_constructs(self):
        sa = self._validate_prefill_only_args()
        self.assertTrue(sa.prefill_only_disable_kv_cache)

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
            self._validate_prefill_only_args(enable_prefill_context_parallel=True)

    def test_rejects_hisparse(self):
        with self.assertRaisesRegex(ValueError, "--enable-hisparse"):
            self._validate_prefill_only_args(enable_hisparse=True)

    def test_rejects_fp4_kv_cache(self):
        with self.assertRaisesRegex(ValueError, "fp4_e2m1"):
            self._validate_prefill_only_args(kv_cache_dtype="fp4_e2m1")


class TestSessionRadixCacheServerArgs(unittest.TestCase):
    def test_requires_priority_radix_eviction_policy(self):
        server_args = ServerArgs(
            model_path="dummy",
            enable_session_radix_cache=True,
            radix_eviction_policy="lru",
        )
        with self.assertRaisesRegex(ValueError, "--radix-eviction-policy priority"):
            server_args._handle_cache_compatibility()


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
        server_args = SimpleNamespace(
            cuda_graph_config=CudaGraphConfig(
                prefill=PhaseConfig(
                    backend=Backend.TC_PIECEWISE,
                    bs=[32, 64],
                    tc_compiler="eager",
                )
            ),
            enable_torch_compile_debug_mode=False,
        )

        config = TcPiecewiseCudaGraphBackend.build_compilation_config(server_args)

        self.assertEqual(config.get_capture_sizes(), [32, 64])
        self.assertEqual(config.compiler, "eager")


class TestCudaGraphDisaggregationRoles(CustomTestCase):
    def _handled_args(self, **overrides):
        args = ServerArgs(model_path="dummy", **overrides)
        args.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
            is_piecewise_cuda_graph_disabled_model=False,
            is_multimodal=False,
            is_multimodal_piecewise_cuda_graph_supported=False,
        )
        with (
            patch("sglang.srt.utils.is_cuda", return_value=True),
            patch.object(ServerArgs, "use_mla_backend", return_value=False),
        ):
            args._handle_cuda_graph_config()
        return args

    def test_cuda_graph_prefill_role_defaults_disable_decode_graph(self):
        args = self._handled_args(disaggregation_mode="prefill")

        self.assertFalse(args.disable_cuda_graph)
        self.assertEqual(args.cuda_graph_config.decode.backend, Backend.DISABLED)
        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.BREAKABLE)

    def test_cuda_graph_decode_role_defaults_disable_prefill_graph(self):
        args = self._handled_args(disaggregation_mode="decode")

        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.DISABLED)
        self.assertNotEqual(args.cuda_graph_config.decode.backend, Backend.DISABLED)

    def test_cuda_graph_global_disable_still_disables_both_phases_for_all_roles(self):
        for disaggregation_mode in ("prefill", "decode", "null"):
            with self.subTest(disaggregation_mode=disaggregation_mode):
                args = self._handled_args(
                    disaggregation_mode=disaggregation_mode,
                    disable_cuda_graph=True,
                )

                self.assertEqual(
                    args.cuda_graph_config.decode.backend, Backend.DISABLED
                )
                self.assertEqual(
                    args.cuda_graph_config.prefill.backend, Backend.DISABLED
                )

    def test_cuda_graph_explicit_decode_backend_survives_prefill_role(self):
        args = self._handled_args(
            disaggregation_mode="prefill",
            cuda_graph_backend_decode=Backend.FULL,
        )

        self.assertEqual(args.cuda_graph_config.decode.backend, Backend.FULL)
        self.assertIn((Phase.DECODE, "backend"), args._cuda_graph_config_locked)


class TestBreakableCudaGraphMultimodalAllowlist(CustomTestCase):
    """The BCG "multimodal model" rule exempts archs on the BCG multimodal
    opt-in allowlist (multimodal_breakable_cuda_graph_supported_model_archs)."""

    def _handled_args(self, *, architectures, is_multimodal, allowlisted):
        args = ServerArgs(model_path="dummy")
        args.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=architectures),
            is_piecewise_cuda_graph_disabled_model=False,
            is_multimodal=is_multimodal,
            is_multimodal_piecewise_cuda_graph_supported=False,
            is_multimodal_breakable_cuda_graph_supported=allowlisted,
        )
        with (
            patch("sglang.srt.utils.is_cuda", return_value=True),
            patch.object(ServerArgs, "use_mla_backend", return_value=False),
        ):
            args._handle_cuda_graph_config()
        return args

    def test_multimodal_arch_disables_prefill_breakable(self):
        args = self._handled_args(
            architectures=["Qwen3VLForConditionalGeneration"],
            is_multimodal=True,
            allowlisted=False,
        )
        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.DISABLED)

    def test_allowlisted_multimodal_arch_keeps_prefill_breakable(self):
        args = self._handled_args(
            architectures=["Qwen3_5MoeForConditionalGeneration"],
            is_multimodal=True,
            allowlisted=True,
        )
        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.BREAKABLE)

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
        self.assertEqual(self._args().cutedsl_moe_max_num_tokens(), 16384)

    def test_speculative_decoding_scales_decode_bound(self):
        # decode bound 512 * 8 dominates the small prefill/piecewise bounds
        args = self._args(
            max_prefill_tokens=512,
            piecewise_cuda_graph_max_tokens=512,
            speculative_algorithm="EAGLE",
            speculative_num_draft_tokens=8,
        )
        self.assertEqual(args.cutedsl_moe_max_num_tokens(), 4096)

    def test_piecewise_bound_excluded_when_disabled(self):
        args = self._args(
            max_prefill_tokens=512,
            disable_piecewise_cuda_graph=True,
            cuda_graph_max_bs=64,
        )
        self.assertEqual(args.cutedsl_moe_max_num_tokens(), 512)


class TestSamplingBackendTokenOracleEnvGate(CustomTestCase):
    """The 'token_oracle' choice is gated on SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE.

    The choice set is built once at server_args.py import time, so each subtest
    reloads the module with the env var set to the desired value.
    """

    def _reload_server_args_with_env(self, *, enabled: bool):
        previous = os.environ.get("SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE")
        os.environ["SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE"] = "1" if enabled else "0"
        try:
            return importlib.reload(server_args_module)
        finally:
            if previous is None:
                os.environ.pop("SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE", None)
            else:
                os.environ["SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE"] = previous

    def test_token_oracle_rejected_when_env_disabled(self):
        reloaded = self._reload_server_args_with_env(enabled=False)
        self.assertNotIn("token_oracle", reloaded.SAMPLING_BACKEND_CHOICES)

        with self.assertRaises(SystemExit):
            reloaded.prepare_server_args(
                [
                    "--model-path",
                    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
                    "--sampling-backend",
                    "token_oracle",
                ]
            )

    def test_token_oracle_accepted_when_env_enabled(self):
        reloaded = self._reload_server_args_with_env(enabled=True)
        self.assertIn("token_oracle", reloaded.SAMPLING_BACKEND_CHOICES)

        parsed = reloaded.prepare_server_args(
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


class TestGrpcServerArgs(CustomTestCase):
    """Native gRPC is enabled by --grpc-port (or SGLANG_GRPC_PORT) and runs
    alongside HTTP; --smg-grpc-mode (and the deprecated --grpc-mode) select the
    legacy SMG server. Worker-threads / max-prefill-tokens are env-only knobs.

    The gRPC setup lives in ServerArgs._handle_deprecated_args, which
    __post_init__ skips for dummy models, so these tests build a dummy
    ServerArgs and invoke that handler directly (mirroring the real flow for a
    concrete model path).
    """

    @staticmethod
    def _args(**kwargs):
        return ServerArgs(model_path="dummy", **kwargs)

    def test_http_only_high_port_does_not_derive_grpc_port(self):
        sa = self._args(port=56000)
        sa._handle_deprecated_args()
        self.assertIsNone(sa.grpc_port)

    def test_grpc_port_enables_native_and_env_knobs(self):
        sa = self._args(grpc_port=50051)
        with envs.SGLANG_GRPC_WORKER_THREADS.override(8):
            sa._handle_deprecated_args()
        self.assertEqual(sa.grpc_port, 50051)
        self.assertEqual(sa.grpc_worker_threads, 8)

    def test_env_grpc_port_enables_native(self):
        sa = self._args(port=30000)
        with envs.SGLANG_GRPC_PORT.override(45000):
            sa._handle_deprecated_args()
        self.assertEqual(sa.grpc_port, 45000)

    def test_legacy_smg_derives_grpc_port_from_http_port(self):
        sa = self._args(port=30000, smg_grpc_mode=True)
        sa._handle_deprecated_args()
        self.assertEqual(sa.grpc_port, 40000)

    def test_grpc_mode_is_deprecated_alias_for_smg_grpc_mode(self):
        sa = self._args(grpc_mode=True)
        with self.assertLogs(server_args_module.logger, level="WARNING") as cm:
            sa._handle_deprecated_args()
        self.assertTrue(sa.smg_grpc_mode)
        self.assertTrue(any("--grpc-mode is deprecated" in line for line in cm.output))

    def test_legacy_smg_takes_precedence_over_grpc_port(self):
        sa = self._args(grpc_port=50051, smg_grpc_mode=True)
        sa._handle_deprecated_args()
        self.assertTrue(sa.smg_grpc_mode)
        self.assertEqual(sa.grpc_port, 50051)

    def test_native_grpc_rejects_multi_tokenizer(self):
        sa = self._args(grpc_port=40000, tokenizer_worker_num=2)
        with self.assertRaises(ValueError):
            sa._handle_deprecated_args()

    def test_native_grpc_rejects_http_auth(self):
        sa = self._args(grpc_port=40000, api_key="secret")
        with self.assertRaises(ValueError):
            sa._handle_deprecated_args()

    def test_invalid_grpc_worker_threads_rejected(self):
        sa = self._args(grpc_port=40000)
        with envs.SGLANG_GRPC_WORKER_THREADS.override(0):
            with self.assertRaises(ValueError):
                sa._handle_deprecated_args()

    def test_start_server_call_site_matches_native_signature(self):
        """Regression for the startup blocker: the native start_server binding
        only accepts (host, port, runtime_handle, worker_threads, ...). The
        arg-parsing tests above never call start_server, so a stray kwarg (e.g.
        the removed max_prefill_tokens) would only surface as a TypeError at
        launch. This mocks the native extension and locks the kwarg set."""
        import sys

        from sglang.srt.entrypoints import http_server

        fake_core = SimpleNamespace(start_server=MagicMock(return_value="handle"))
        fake_bridge = SimpleNamespace(RuntimeHandle=MagicMock(return_value="rt"))
        server_args = SimpleNamespace(
            host="127.0.0.1", grpc_port=50051, grpc_worker_threads=4
        )
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.grpc": SimpleNamespace(_core=fake_core),
                "sglang.srt.grpc._core": fake_core,
                "sglang.srt.entrypoints.grpc_bridge": fake_bridge,
            },
        ):
            handle = http_server._start_native_grpc_server_for_runtime(
                server_args=server_args,
                tokenizer_manager=MagicMock(),
                template_manager=MagicMock(),
                scheduler_info={},
            )

        self.assertEqual(handle, "handle")
        _, kwargs = fake_core.start_server.call_args
        self.assertEqual(
            set(kwargs), {"host", "port", "runtime_handle", "worker_threads"}
        )
        self.assertNotIn("max_prefill_tokens", kwargs)


class TestTwoBatchOverlapBackend(CustomTestCase):
    """Non-EP DP two-batch-overlap backend requirement.

    With no EP a2a backend (moe_a2a_backend='none'), --enable-two-batch-overlap
    is only valid on the DeepSeek-V4 non-EP DP TP-MoE path (overlapping the DP
    all_gatherv / reduce_scatterv with the other ubatch's compute), which
    requires --enable-dp-attention. This replaced the removed opt-in
    SGLANG_ENABLE_DP_TBO env: enabling DP TBO now needs no extra flag.

    dummy-model short-circuits __post_init__, so the guard handler is invoked
    directly (same pattern as TestDeepEPWaterfillArgs)."""

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
            args._check_two_batch_overlap()

    def test_no_a2a_with_dp_attention_ok(self):
        # DP TBO path is valid: --enable-dp-attention + --enable-two-batch-overlap
        # with a2a backend 'none' must NOT raise (no SGLANG_ENABLE_DP_TBO needed).
        args = self._args(enable_dp_attention=True)
        args._check_two_batch_overlap()

    def test_ep_a2a_backend_ok_without_dp_attention(self):
        # EP a2a path (e.g. deepep) overlaps dispatch/combine; the guard does not
        # require dp-attention there.
        args = self._args(moe_a2a_backend="deepep", enable_dp_attention=False)
        args._check_two_batch_overlap()


if __name__ == "__main__":
    unittest.main()
