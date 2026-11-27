import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.io_struct import GetInternalStateReq
from sglang.srt.managers.scheduler_components import internal_state
from sglang.srt.managers.scheduler_components.internal_state import (
    SchedulerInternalStateController,
)
from sglang.srt.true_on_policy import (
    TRUE_ON_POLICY_V1,
    get_on_policy_rms_norm_kwargs,
    get_rl_on_policy_target,
    is_true_on_policy_enabled,
    patch_prefill_only_deterministic_inference_for_cuda_graph,
    resolve_true_on_policy_runtime_policy,
    should_disable_flashinfer_allreduce_fusion,
    should_disable_fused_qk_norm_mrope,
    should_disable_mlp_allreduce_fusion_for_on_policy,
    should_disable_reduce_scatter_for_on_policy,
    should_force_bfloat16_dense_tensor_math,
    should_force_bfloat16_lm_head,
    validate_true_on_policy_contract,
)
from sglang.srt.true_on_policy.device import DEVICE_STATE_KEY
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

_PATCH_TARGET = "sglang.srt.true_on_policy.config._get_global_server_args"


def _mock_args(**kwargs):
    defaults = dict(
        rl_on_policy_target=None,
        true_on_policy_contract=None,
        tp_size=1,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _contract_args(*, tp_size: int = 1):
    return _mock_args(
        true_on_policy_contract=TRUE_ON_POLICY_V1,
        tp_size=tp_size,
    )


def _contract_layout(**kwargs):
    fields = dict(
        true_on_policy_contract=TRUE_ON_POLICY_V1,
        tp_size=1,
        ep_size=1,
        dp_size=1,
        attn_dp_size=1,
        enable_dp_attention=False,
        moe_dense_tp_size=None,
        enable_dp_lm_head=False,
        attn_cp_size=1,
        dcp_size=1,
        enable_lora=None,
        lora_paths=None,
        enable_torch_compile=False,
        enable_mixed_chunk=False,
    )
    fields.update(kwargs)
    return SimpleNamespace(**fields)


class TestDefaultPathUnchanged(unittest.TestCase):
    """Default serving must not enter true-on-policy policy paths."""

    def setUp(self):
        self.default_args = _mock_args()

    def test_default_args_no_on_policy(self):
        with patch(_PATCH_TARGET, return_value=self.default_args):
            self.assertIsNone(get_rl_on_policy_target())
            self.assertFalse(is_true_on_policy_enabled())

    def test_default_args_reduce_scatter_available(self):
        with patch(_PATCH_TARGET, return_value=self.default_args):
            self.assertFalse(should_disable_reduce_scatter_for_on_policy())

    def test_default_args_mlp_fusion_available(self):
        with patch(_PATCH_TARGET, return_value=self.default_args):
            self.assertFalse(should_disable_mlp_allreduce_fusion_for_on_policy())

    def test_default_args_dense_math_helpers_are_inactive(self):
        with patch(_PATCH_TARGET, return_value=self.default_args):
            self.assertFalse(should_force_bfloat16_dense_tensor_math())
            self.assertFalse(should_force_bfloat16_lm_head(use_fp32_lm_head=False))
            self.assertEqual(get_on_policy_rms_norm_kwargs(), {})

    def test_default_args_flashinfer_fusion_available(self):
        with patch(_PATCH_TARGET, return_value=self.default_args):
            self.assertFalse(should_disable_flashinfer_allreduce_fusion())


class TestOnPolicyHelpers(unittest.TestCase):
    def test_contract_resolver_ignores_legacy_target_without_contract(self):
        policy = resolve_true_on_policy_runtime_policy(
            _mock_args(rl_on_policy_target="fsdp_tp", tp_size=2)
        )

        self.assertIsNone(policy.contract_name)
        self.assertFalse(policy.enabled)

    def test_contract_resolver_accepts_explicit_v1_contract(self):
        policy = resolve_true_on_policy_runtime_policy(_contract_args(tp_size=1))

        self.assertTrue(policy.enabled)
        self.assertTrue(policy.force_bfloat16_dense_tensor_math)

    def test_reduce_scatter_and_fusion_are_disabled_for_contract(self):
        with patch(_PATCH_TARGET, return_value=_contract_args(tp_size=1)):
            self.assertTrue(should_disable_reduce_scatter_for_on_policy())
        with patch(_PATCH_TARGET, return_value=_contract_args(tp_size=2)):
            self.assertTrue(should_disable_mlp_allreduce_fusion_for_on_policy())
        with patch(_PATCH_TARGET, return_value=_mock_args()):
            self.assertFalse(should_disable_reduce_scatter_for_on_policy())

    def test_contract_refuses_split_reductions(self):
        layout = _contract_layout
        validate_true_on_policy_contract(layout())
        validate_true_on_policy_contract(
            layout(
                tp_size=8,
                ep_size=8,
                attn_dp_size=8,
                moe_dense_tp_size=1,
                enable_dp_lm_head=True,
            )
        )
        for split in (
            dict(tp_size=2, ep_size=2),
            dict(tp_size=2, ep_size=2, attn_dp_size=2, enable_dp_lm_head=True),
            dict(attn_cp_size=2),
            dict(dcp_size=2),
        ):
            with self.subTest(**split), self.assertRaisesRegex(ValueError, "full K"):
                validate_true_on_policy_contract(layout(**split))

    def test_contract_refuses_lora(self):
        """Serving LoRA adds the adapter with its own kernels, which the trainer's adapters do not repeat."""
        validate_true_on_policy_contract(
            _contract_layout(enable_lora=False, lora_paths=["adapter=/path"])
        )
        for lora in (dict(enable_lora=True), dict(lora_paths=["adapter=/path"])):
            with self.subTest(**lora), self.assertRaisesRegex(ValueError, "LoRA"):
                validate_true_on_policy_contract(_contract_layout(**lora))

    def test_contract_refuses_torch_compile_and_mixed_chunk(self):
        for flag in ("enable_torch_compile", "enable_mixed_chunk"):
            refusal = "--" + flag.replace("_", "-")
            with self.subTest(flag), self.assertRaisesRegex(ValueError, refusal):
                validate_true_on_policy_contract(_contract_layout(**{flag: True}))

    def test_prefill_only_cuda_graph_patch_only_scopes_attention_splits(self):
        server_args = SimpleNamespace(
            enable_prefill_only_deterministic_inference=True,
            enable_deterministic_inference=True,
            enable_flashinfer_allreduce_fusion=False,
            rl_on_policy_target="fsdp_tp",
            true_on_policy_contract=TRUE_ON_POLICY_V1,
            disable_custom_all_reduce=True,
        )
        attn_backend = SimpleNamespace(num_splits=1)

        with patch.dict(
            os.environ,
            {
                "SGLANG_ENABLE_DETERMINISTIC_INFERENCE": "1",
                "SGLANG_DISABLE_CUSTOM_ALL_REDUCE": "1",
                "NCCL_ALGO": "allreduce:tree",
            },
            clear=False,
        ):
            with patch_prefill_only_deterministic_inference_for_cuda_graph(
                server_args,
                attn_backend=attn_backend,
            ) as patched:
                self.assertTrue(patched)
                self.assertTrue(server_args.enable_deterministic_inference)
                self.assertFalse(server_args.enable_flashinfer_allreduce_fusion)
                self.assertEqual(server_args.rl_on_policy_target, "fsdp_tp")
                self.assertEqual(
                    server_args.true_on_policy_contract,
                    TRUE_ON_POLICY_V1,
                )
                self.assertEqual(attn_backend.num_splits, 0)
                self.assertEqual(
                    os.environ["SGLANG_ENABLE_DETERMINISTIC_INFERENCE"], "1"
                )
                self.assertEqual(os.environ["SGLANG_DISABLE_CUSTOM_ALL_REDUCE"], "1")
                self.assertEqual(os.environ["NCCL_ALGO"], "allreduce:tree")

            self.assertTrue(server_args.enable_deterministic_inference)
            self.assertFalse(server_args.enable_flashinfer_allreduce_fusion)
            self.assertEqual(server_args.rl_on_policy_target, "fsdp_tp")
            self.assertEqual(
                server_args.true_on_policy_contract,
                TRUE_ON_POLICY_V1,
            )
            self.assertTrue(server_args.disable_custom_all_reduce)
            self.assertEqual(attn_backend.num_splits, 1)
            self.assertEqual(os.environ["SGLANG_ENABLE_DETERMINISTIC_INFERENCE"], "1")
            self.assertEqual(os.environ["SGLANG_DISABLE_CUSTOM_ALL_REDUCE"], "1")
            self.assertEqual(os.environ["NCCL_ALGO"], "allreduce:tree")

    def test_contract_dense_math_helpers_force_bfloat16_and_rms_norm_kwargs(self):
        with patch(_PATCH_TARGET, return_value=_contract_args()):
            kwargs = get_on_policy_rms_norm_kwargs(
                weight_dtype=torch.float32,
                override_orig_dtype=torch.float32,
                fp32_residual=True,
            )
            self.assertTrue(should_force_bfloat16_dense_tensor_math())
            self.assertTrue(should_force_bfloat16_lm_head(use_fp32_lm_head=False))
            self.assertFalse(should_force_bfloat16_lm_head(use_fp32_lm_head=True))
        self.assertEqual(kwargs["weight_dtype"], torch.float32)
        self.assertEqual(kwargs["override_orig_dtype"], torch.float32)
        self.assertNotIn("cast_x_before_out_mul", kwargs)
        self.assertTrue(kwargs["fp32_residual"])

    def test_flashinfer_allreduce_fusion_helpers(self):
        with patch(_PATCH_TARGET, return_value=_contract_args(tp_size=1)):
            self.assertTrue(should_disable_flashinfer_allreduce_fusion())
        with patch(_PATCH_TARGET, return_value=_mock_args()):
            self.assertFalse(should_disable_flashinfer_allreduce_fusion())

    def test_fused_qk_norm_mrope_helper_follows_true_on_policy_contract(self):
        with patch(_PATCH_TARGET, return_value=_contract_args(tp_size=1)):
            self.assertTrue(should_disable_fused_qk_norm_mrope())
        with patch(_PATCH_TARGET, return_value=_mock_args()):
            self.assertFalse(should_disable_fused_qk_norm_mrope())

    def test_get_rl_on_policy_target_returns_correct_value(self):
        with patch(
            _PATCH_TARGET, return_value=_mock_args(rl_on_policy_target="fsdp_tp")
        ):
            self.assertEqual(get_rl_on_policy_target(), "fsdp_tp")
        with patch(_PATCH_TARGET, return_value=_mock_args(rl_on_policy_target="fsdp")):
            self.assertEqual(get_rl_on_policy_target(), "fsdp")
        with patch(_PATCH_TARGET, return_value=_mock_args()):
            self.assertIsNone(get_rl_on_policy_target())

    def test_is_true_on_policy_enabled_for_both_targets(self):
        with patch(_PATCH_TARGET, return_value=_contract_args(tp_size=1)):
            self.assertTrue(is_true_on_policy_enabled())
        with patch(_PATCH_TARGET, return_value=_contract_args(tp_size=2)):
            self.assertTrue(is_true_on_policy_enabled())
        with patch(_PATCH_TARGET, return_value=_mock_args()):
            self.assertFalse(is_true_on_policy_enabled())

    def test_cuda_graph_patch_noop_when_disabled(self):
        server_args = SimpleNamespace(
            enable_prefill_only_deterministic_inference=False,
            enable_deterministic_inference=True,
            rl_on_policy_target="fsdp_tp",
        )
        with patch_prefill_only_deterministic_inference_for_cuda_graph(
            server_args,
        ) as patched:
            self.assertFalse(patched)
            self.assertTrue(server_args.enable_deterministic_inference)
            self.assertEqual(server_args.rl_on_policy_target, "fsdp_tp")

    def test_cuda_graph_patch_noop_when_dvr_verify(self):
        server_args = SimpleNamespace(
            enable_prefill_only_deterministic_inference=True,
            enable_deterministic_inference=True,
            enable_flashinfer_allreduce_fusion=False,
            rl_on_policy_target="fsdp_tp",
            disable_custom_all_reduce=True,
        )
        with patch_prefill_only_deterministic_inference_for_cuda_graph(
            server_args,
            dvr_target_verify_cuda_graph=True,
        ) as patched:
            self.assertFalse(patched)
            self.assertTrue(server_args.enable_deterministic_inference)
            self.assertEqual(server_args.rl_on_policy_target, "fsdp_tp")

    def test_tp_invariant_ops_import_is_available(self):
        import sglang.srt.tp_invariant_ops as tp_invariant_ops

        self.assertTrue(hasattr(tp_invariant_ops, "matmul_tp_inv"))

    def test_legacy_on_policy_utils_import_matches_true_on_policy_namespace(self):
        from sglang.srt import true_on_policy
        from sglang.srt.layers import on_policy_utils as legacy

        self.assertIs(
            legacy.get_on_policy_rms_norm_kwargs,
            true_on_policy.get_on_policy_rms_norm_kwargs,
        )
        self.assertIs(
            legacy.patch_prefill_only_deterministic_inference_for_cuda_graph,
            true_on_policy.patch_prefill_only_deterministic_inference_for_cuda_graph,
        )
        self.assertTrue(hasattr(torch.ops, "tp_inv_ops"))


class TestDeviceReport(unittest.TestCase):
    _OTHER_CONTRIBUTORS = (
        "_add_topology",
        "_add_memory_usage",
        "_add_disaggregation",
        "_add_elastic_ep",
        "_add_speculative",
        "_add_step_times",
        "_add_rust_server",
        "_add_env_vars",
    )

    def _internal_state(self, contract):
        scheduler = SimpleNamespace(
            tp_worker=SimpleNamespace(model_runner=SimpleNamespace(gpu_id=3))
        )
        controller = SchedulerInternalStateController(scheduler, record_step_time=False)
        context = SimpleNamespace(
            resolved_server_args_dict=lambda: {"true_on_policy_contract": contract}
        )
        with (
            patch.object(internal_state, "get_context", return_value=context),
            patch(
                "torch.cuda.get_device_capability", return_value=(10, 0)
            ) as capability,
            patch("torch.cuda.get_device_name", return_value="NVIDIA B200"),
            patch.multiple(
                SchedulerInternalStateController,
                **{attr: lambda self, state: None for attr in self._OTHER_CONTRIBUTORS},
            ),
        ):
            state = controller.get_internal_state(GetInternalStateReq()).internal_state
        return state, capability

    def test_contract_scheduler_reports_its_own_gpu(self):
        """The TOP trainer reads this field from /server_info to refuse an engine on another GPU architecture."""
        state, capability = self._internal_state(TRUE_ON_POLICY_V1)
        self.assertEqual(
            state[DEVICE_STATE_KEY], {"capability": [10, 0], "name": "NVIDIA B200"}
        )
        capability.assert_called_once_with(3)

    def test_no_device_report_without_a_contract(self):
        state, capability = self._internal_state(None)
        self.assertNotIn(DEVICE_STATE_KEY, state)
        capability.assert_not_called()


class TestContractModelTypeLookup(unittest.TestCase):
    """The attention-backend pick reads the model type as handle_deterministic_inference does,
    so a connector model path does not fail argument resolution."""

    _MODEL_CONFIG_OF = "sglang.srt.arg_groups.model_override_base.model_config_of"

    def test_instance_connector_reads_no_config(self):
        from sglang.srt.true_on_policy.contracts import _model_type

        with patch(self._MODEL_CONFIG_OF, side_effect=AssertionError("read")):
            self.assertIsNone(_model_type(SimpleNamespace(model_path="instance://h:1")))

    def test_unreadable_config_keeps_the_contract_backend(self):
        from sglang.srt.true_on_policy.contracts import _model_type

        with patch(self._MODEL_CONFIG_OF, side_effect=OSError("remote")):
            self.assertIsNone(_model_type(SimpleNamespace(model_path="s3://b/m")))
        config = SimpleNamespace(hf_config=SimpleNamespace(model_type="glm_moe_dsa"))
        with patch(self._MODEL_CONFIG_OF, return_value=config):
            self.assertEqual(
                _model_type(SimpleNamespace(model_path="/models/glm")), "glm_moe_dsa"
            )

    def test_deterministic_inference_keeps_radix_for_glm53_only(self):
        """GLM-5.3-Flash keeps radix reuse on the DSA backend it shares with GLM-5, whose TOP program does not."""
        from sglang.srt.true_on_policy.contracts import top_keeps_radix_cache

        def keeps(model_type, backend, contract="true_on_policy_v1"):
            config = SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type))
            args = SimpleNamespace(
                model_path="/models/m", true_on_policy_contract=contract
            )
            with patch(self._MODEL_CONFIG_OF, return_value=config):
                return top_keeps_radix_cache(args, backend)

        self.assertTrue(keeps("glm5_next", "dsa"))
        self.assertTrue(keeps("glm5_next_text", "dsa"))
        self.assertFalse(keeps("glm_moe_dsa", "dsa"))
        self.assertFalse(keeps("glm5_next", "dsa", contract=None))
        self.assertTrue(keeps("deepseek_v4", "dsv4"))


if __name__ == "__main__":
    unittest.main()
