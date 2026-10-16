"""CPU-only tests of the actual SM80 launch validator, isolated via its AST.

No CUDA, model download, or import of the serving stack is needed. Only external
configuration/runtime dependencies are stubbed; the validation code is compiled
unchanged from the production source.
"""

import ast
import json
import os
import runpy
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[4]
HOOK = ROOT / "python/sglang/srt/arg_groups/deepseek_v4_hook.py"
BASE = ROOT / "python/sglang/srt/arg_groups/model_override_base.py"
register_cpu_ci = runpy.run_path(
    str(ROOT / "python/sglang/test/ci/ci_register.py")
)["register_cpu_ci"]
register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _load_functions(path, names, namespace):
    source = ast.parse(path.read_text())
    functions = [
        node
        for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if len(functions) != len(names):
        raise AssertionError(f"Missing production functions: {names}")
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    isolated = ast.fix_missing_locations(
        ast.Module(body=[future, *functions], type_ignores=[])
    )
    exec(compile(isolated, str(path), "exec"), namespace)  # noqa: S102


class TestDeepseekV41SM80Experimental(unittest.TestCase):
    def setUp(self):
        self.cfg = SimpleNamespace(
            device="cuda",
            base_gpu_id=3,
            dsv4_attn_backend="auto",
            dsv4_prefill_backend="auto",
            enable_deepseek_v4_fp4_indexer=False,
            enable_prefill_cp=False,
            attn_cp_size=1,
            dcp_size=1,
            attn_dp_size=1,
            pp_size=1,
            nnodes=1,
            disaggregation_mode="null",
            speculative_algorithm=None,
            speculative_draft_model_path=None,
            decoupled_spec_role="null",
            enable_mixed_chunk=False,
            enable_decoder_swa_bounded_replay=False,
            enable_encoder_swa_bounded_replay=False,
            enable_hierarchical_cache=False,
            enable_hisparse=False,
            enable_lora=False,
            enable_two_batch_overlap=False,
            language_model_only=False,
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(backend="disabled", bs=None, max_bs=None),
                prefill=SimpleNamespace(backend="disabled"),
            ),
            attention_backend="dsv4",
            prefill_attention_backend=None,
            decode_attention_backend=None,
            kv_cache_dtype="fp8_e4m3",
            tp_size=8,
            max_running_requests=1,
            context_length=512,
            disable_cuda_graph=True,
            disable_prefill_cuda_graph=True,
        )
        self.model = SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="deepseek_v41"),
            hf_config=SimpleNamespace(model_type="outer_wrapper"),
            is_multimodal=False,
        )
        actual_envs = runpy.run_path(str(ROOT / "python/sglang/srt/environ.py"))["envs"]
        self.envs = SimpleNamespace(
            SGLANG_DSV41_SM80_EXPERIMENTAL=actual_envs.SGLANG_DSV41_SM80_EXPERIMENTAL,
            SGLANG_DSV41_SM80_FULL_GRAPH=actual_envs.SGLANG_DSV41_SM80_FULL_GRAPH,
            SGLANG_DSV41_SM80_FFN_GRAPH=actual_envs.SGLANG_DSV41_SM80_FFN_GRAPH,
            SGLANG_HACK_FLASHMLA_BACKEND=SimpleNamespace(get=Mock(return_value="auto")),
            SGLANG_DSV4_UNIFIED_KV_FP8=SimpleNamespace(get=Mock(return_value=False)),
        )
        self.torch = ModuleType("torch")
        self.capability = Mock(return_value=(8, 0))
        self.torch.cuda = SimpleNamespace(get_device_capability=self.capability)
        self.declare = Mock()
        self.view = Mock(side_effect=lambda args: args)
        self.namespace = {
            "os": os,
            "resolving_view": self.view,
            "resolved_view": lambda args: args,
            "model_config_of": Mock(side_effect=lambda args: self.model),
            "envs": self.envs,
            "Backend": SimpleNamespace(DISABLED="disabled", FULL="full"),
            "attn_dp_enabled_of": lambda cfg: cfg.attn_dp_size > 1,
            "declare_resolution": self.declare,
        }
        _load_functions(BASE, {"attention_backends_of"}, self.namespace)
        _load_functions(
            HOOK,
            {"validate_deepseek_v41_sm80_experimental"},
            self.namespace,
        )
        self.validate = self.namespace["validate_deepseek_v41_sm80_experimental"]
        self.enterContext(patch.dict(sys.modules, {"torch": self.torch}))
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        os.environ["SGLANG_DSV41_SM80_EXPERIMENTAL"] = "1"

    def test_inactive_matrix_has_no_effect(self):
        # Invalid active-mode settings must not be inspected when inactive.
        self.cfg.dsv4_attn_backend = "trtllm"
        cases = [
            (None, "deepseek_v41", "cuda", (8, 0)),
            ("0", "deepseek_v41", "cuda", (8, 0)),
            ("true", "deepseek_v41", "cuda", (8, 0)),
            ("01", "deepseek_v41", "cuda", (8, 0)),
            ("1", "deepseek_v4", "cuda", (8, 0)),
            ("1", "deepseek_v3", "cuda", (8, 0)),
            ("1", "deepseek_v41", "cpu", (8, 0)),
            ("1", "deepseek_v41", "npu", (8, 0)),
            ("1", "deepseek_v41", "cuda", (8, 6)),
            ("1", "deepseek_v41", "cuda", (9, 0)),
            ("1", "deepseek_v41", "cuda", (10, 0)),
        ]
        for opt_in, model_type, device, capability in cases:
            with self.subTest(case=(opt_in, model_type, device, capability)):
                self.capability.reset_mock()
                self.view.reset_mock()
                if opt_in is None:
                    os.environ.pop("SGLANG_DSV41_SM80_EXPERIMENTAL", None)
                else:
                    os.environ["SGLANG_DSV41_SM80_EXPERIMENTAL"] = opt_in
                self.model.hf_text_config.model_type = model_type
                self.cfg.device = device
                self.capability.return_value = capability
                before = vars(self.cfg).copy()
                self.validate(self.cfg)
                self.assertEqual(vars(self.cfg), before)
                self.declare.assert_not_called()
                if opt_in != "1":
                    self.view.assert_not_called()
                if opt_in != "1" or model_type != "deepseek_v41" or device != "cuda":
                    self.capability.assert_not_called()
                else:
                    self.capability.assert_called_once_with(3)

    def test_valid_single_node_tp8_uses_actual_device_and_only_resolves_selector(self):
        before = vars(self.cfg).copy()
        self.validate(self.cfg)
        self.capability.assert_called_once_with(3)
        self.declare.assert_called_once_with(
            self.cfg,
            "validate_deepseek_v41_sm80_experimental",
            dsv4_attn_backend="flashmla",
        )
        self.assertEqual(vars(self.cfg), before)
        self.assertEqual(self.cfg.tp_size, 8)
        self.assertEqual(self.cfg.max_running_requests, 1)
        self.assertEqual(self.cfg.context_length, 512)

    def test_explicit_flashmla_is_accepted_without_declarations(self):
        self.cfg.dsv4_attn_backend = "flashmla"
        self.validate(self.cfg)
        self.declare.assert_not_called()

    def test_enabled_unsupported_features_fail_explicitly(self):
        cases = [
            ("dsv4_attn_backend", "trtllm", "TRT-LLM"),
            ("dsv4_prefill_backend", "flashmla_sparse_q8", "Q8"),
            ("enable_deepseek_v4_fp4_indexer", True, "FP4 indexer"),
            ("enable_prefill_cp", True, "CP/DCP"),
            ("attn_cp_size", 2, "CP/DCP"),
            ("dcp_size", 2, "CP/DCP"),
            ("attn_dp_size", 2, "DP attention"),
            ("pp_size", 2, "pipeline parallelism"),
            ("nnodes", 2, "multiple nodes"),
            ("disaggregation_mode", "prefill", "PD disaggregation"),
            ("disaggregation_mode", "decode", "PD disaggregation"),
            ("speculative_algorithm", "EAGLE", "MTP"),
            ("speculative_algorithm", "DSPARK", "MTP"),
            ("speculative_draft_model_path", "mtp-draft", "MTP"),
            ("decoupled_spec_role", "drafter", "MTP"),
            ("enable_mixed_chunk", True, "mixed prefill/decode"),
            ("enable_decoder_swa_bounded_replay", True, "decoder SWA"),
            ("enable_encoder_swa_bounded_replay", True, "encoder SWA"),
            ("enable_hierarchical_cache", True, "HiCache"),
            ("enable_hisparse", True, "HiSparse"),
            ("enable_lora", True, "LoRA"),
            ("enable_two_batch_overlap", True, "two-batch overlap"),
            ("dsv4_attn_backend", "other", "flashmla"),
            ("kv_cache_dtype", "bfloat16", "fp8_e4m3"),
            ("kv_cache_dtype", "auto", "fp8_e4m3"),
        ]
        for field, value, error in cases:
            with self.subTest(field=field, value=value):
                old = getattr(self.cfg, field)
                setattr(self.cfg, field, value)
                try:
                    with self.assertRaises(ValueError) as caught:
                        self.validate(self.cfg)
                    self.assertIn(error, str(caught.exception))
                    self.declare.assert_not_called()
                finally:
                    setattr(self.cfg, field, old)

    def test_rejects_unified_layout_env_even_when_hip_gate_would_be_false(self):
        for name, value in (
            ("SGLANG_HACK_FLASHMLA_BACKEND", "unified_kv_triton"),
            ("SGLANG_DSV4_UNIFIED_KV_FP8", True),
        ):
            with self.subTest(name=name):
                getter = getattr(self.envs, name).get
                old = getter.return_value
                getter.return_value = value
                with self.assertRaisesRegex(ValueError, "unified KV layout"):
                    self.validate(self.cfg)
                getter.return_value = old
        self.declare.assert_not_called()

    def test_vision_pipeline_is_rejected(self):
        self.model.is_multimodal = True
        with self.assertRaisesRegex(ValueError, "vision / multimodal"):
            self.validate(self.cfg)
        self.declare.assert_not_called()

    def test_vision_checkpoint_accepts_standalone_language_model_only(self):
        self.model.is_multimodal = True
        self.cfg.language_model_only = True
        self.validate(self.cfg)
        self.declare.assert_called_once()

    def test_resolved_graph_overrides_fail_despite_legacy_disable_flags(self):
        for phase in ("prefill", "decode"):
            for backend in ("full", "breakable"):
                with self.subTest(phase=phase, backend=backend):
                    graph = getattr(self.cfg.cuda_graph_config, phase)
                    graph.backend = backend
                    with self.assertRaisesRegex(ValueError, "eager decode and prefill"):
                        self.validate(self.cfg)
                    graph.backend = "disabled"
        self.declare.assert_not_called()

    def test_resolved_disabled_graph_selectors_are_sufficient(self):
        self.cfg.disable_cuda_graph = False
        self.cfg.disable_prefill_cuda_graph = False
        self.validate(self.cfg)
        self.declare.assert_called_once()

    def test_split_attention_backends_must_both_be_dsv4(self):
        for field in ("prefill_attention_backend", "decode_attention_backend"):
            with self.subTest(field=field):
                setattr(self.cfg, field, "flashinfer")
                with self.assertRaisesRegex(
                    ValueError, "dsv4 attention for both phases"
                ):
                    self.validate(self.cfg)
                setattr(self.cfg, field, None)
        self.declare.assert_not_called()

    def _enable_full_graph(self):
        os.environ["SGLANG_DSV41_SM80_FULL_GRAPH"] = "1"
        # Model the resolved JSON override, despite legacy disable booleans.
        override = json.loads(
            '{"decode":{"backend":"full","bs":[1],"max_bs":1},'
            '"prefill":{"backend":"disabled"}}'
        )
        self.cfg.cuda_graph_config = SimpleNamespace(
            **{phase: SimpleNamespace(**values) for phase, values in override.items()}
        )

    def test_full_graph_accepts_resolved_json_and_only_resolves_selector(self):
        self._enable_full_graph()
        graph = self.cfg.cuda_graph_config
        self.validate(self.cfg)
        self.capability.assert_called_once_with(3)
        self.declare.assert_called_once_with(
            self.cfg,
            "validate_deepseek_v41_sm80_experimental",
            dsv4_attn_backend="flashmla",
        )
        self.assertIs(self.cfg.cuda_graph_config, graph)
        self.assertEqual(vars(graph.decode), dict(backend="full", bs=[1], max_bs=1))
        self.assertTrue(self.cfg.disable_cuda_graph)
        self.assertTrue(self.cfg.disable_prefill_cuda_graph)

    def test_full_graph_stays_behind_all_existing_activation_gates(self):
        self._enable_full_graph()
        os.environ["SGLANG_DSV41_SM80_FFN_GRAPH"] = "1"
        self.test_inactive_matrix_has_no_effect()

    def test_nonliteral_full_graph_flag_preserves_eager_only_guard(self):
        self._enable_full_graph()
        for value in (None, "", "0", "true", "True", "01", " 1", "1 "):
            with self.subTest(value=value):
                if value is None:
                    os.environ.pop("SGLANG_DSV41_SM80_FULL_GRAPH", None)
                else:
                    os.environ["SGLANG_DSV41_SM80_FULL_GRAPH"] = value
                with self.assertRaisesRegex(ValueError, "eager decode and prefill"):
                    self.validate(self.cfg)
                self.cfg.cuda_graph_config.decode.backend = "disabled"
                self.cfg.dsv4_attn_backend = "flashmla"
                self.validate(self.cfg)
                self.cfg.cuda_graph_config.decode.backend = "full"
        self.declare.assert_not_called()

    def test_full_graph_requires_full_decode_and_disabled_prefill(self):
        self._enable_full_graph()
        for decode, prefill in (
            ("disabled", "disabled"),
            ("breakable", "disabled"),
            ("cuda", "disabled"),
            ("full", "full"),
            ("full", "breakable"),
        ):
            with self.subTest(decode=decode, prefill=prefill):
                self.cfg.cuda_graph_config.decode.backend = decode
                self.cfg.cuda_graph_config.prefill.backend = prefill
                with self.assertRaisesRegex(
                    ValueError, "decode backend full and prefill backend disabled"
                ):
                    self.validate(self.cfg)
        self.declare.assert_not_called()

    def test_full_graph_rejects_missing_or_wide_resolved_batch_limits(self):
        self._enable_full_graph()
        for requests, bs, max_bs in (
            (None, [1], 1),
            (0, [1], 1),
            (2, [1], 1),
            (1, None, 1),
            (1, [], 1),
            (1, [2], 1),
            (1, [1, 2], 1),
            (1, [1, 1], 1),
            (1, [1], None),
            (1, [1], 0),
            (1, [1], 2),
            (1, None, None),
        ):
            with self.subTest(requests=requests, bs=bs, max_bs=max_bs):
                self.cfg.max_running_requests = requests
                self.cfg.cuda_graph_config.decode.bs = bs
                self.cfg.cuda_graph_config.decode.max_bs = max_bs
                with self.assertRaisesRegex(
                    ValueError, "resolved decode bs=\\[1\\], max_bs=1"
                ):
                    self.validate(self.cfg)
        self.declare.assert_not_called()

    def test_full_graph_rejects_nested_ffn_capture(self):
        self._enable_full_graph()
        os.environ["SGLANG_DSV41_SM80_FFN_GRAPH"] = "1"
        with self.assertRaisesRegex(ValueError, "FFN_GRAPH=1.*nested capture"):
            self.validate(self.cfg)
        self.declare.assert_not_called()

    def test_full_graph_ffn_guard_is_literal_and_does_not_change_eager(self):
        self._enable_full_graph()
        self.cfg.dsv4_attn_backend = "flashmla"
        for value in (None, "0", "true", "01"):
            with self.subTest(value=value):
                if value is None:
                    os.environ.pop("SGLANG_DSV41_SM80_FFN_GRAPH", None)
                else:
                    os.environ["SGLANG_DSV41_SM80_FFN_GRAPH"] = value
                self.validate(self.cfg)
        os.environ.pop("SGLANG_DSV41_SM80_FULL_GRAPH")
        os.environ["SGLANG_DSV41_SM80_FFN_GRAPH"] = "1"
        self.cfg.cuda_graph_config.decode.backend = "disabled"
        self.validate(self.cfg)
        self.declare.assert_not_called()

    def test_full_graph_retains_all_other_unsupported_feature_guards(self):
        self._enable_full_graph()
        self.test_enabled_unsupported_features_fail_explicitly()
        self.test_rejects_unified_layout_env_even_when_hip_gate_would_be_false()
        self.test_split_attention_backends_must_both_be_dsv4()
        self.test_vision_pipeline_is_rejected()

    def test_apply_defaults_runs_validation_immediately_after_dtype_resolution(self):
        calls = []
        namespace = {
            "resolving_view": lambda args: args,
            "get_platform": lambda: SimpleNamespace(is_hip=False),
            "run_post_process_pass": lambda args, fn: calls.append("resolve_dtype"),
            "_deepseek_v4_kv_cache_dtype": object(),
            "validate_deepseek_v41_sm80_experimental": lambda args: calls.append(
                "validate_sm80"
            ),
            "declare_resolution": Mock(),
        }
        _load_functions(HOOK, {"apply_deepseek_v4_defaults"}, namespace)
        self.cfg.dsv4_attn_backend = "flashmla"
        namespace["apply_deepseek_v4_defaults"](self.cfg, "DeepseekV4ForCausalLM")
        self.assertEqual(calls, ["resolve_dtype", "validate_sm80"])

    def test_validator_uses_only_real_cfg_schema_fields(self):
        fields = set()
        for path in (ROOT / "python/sglang/srt/arg_groups/fields").glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.AnnAssign) and isinstance(
                    node.target, ast.Name
                ):
                    fields.add(node.target.id)
        hook_tree = ast.parse(HOOK.read_text())
        validator = next(
            node
            for node in hook_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "validate_deepseek_v41_sm80_experimental"
        )
        used = {
            node.attr
            for node in ast.walk(validator)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "cfg"
        }
        self.assertFalse(used - fields, f"Invented config fields: {used - fields}")


if __name__ == "__main__":
    unittest.main()
