"""Unit tests for srt/model_loader/utils.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

import concurrent.futures
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.srt.configs.model_config import ModelImpl
from sglang.srt.model_loader.utils import (
    _get_transformers_backend_arch,
    _is_moe_model,
    _is_sequence_classification_model,
    _model_impl_from_architecture,
    maybe_executor_submit,
    set_default_torch_dtype,
    should_async_load,
)
from sglang.test.test_utils import CustomTestCase


def _make_model_config(**overrides):
    """Create a minimal mock ModelConfig for testing."""
    cfg = MagicMock()
    cfg.is_generation = overrides.get("is_generation", True)
    cfg.is_multimodal = overrides.get("is_multimodal", False)

    text_config = MagicMock()
    # Default: no MoE attributes
    for attr in (
        "num_local_experts",
        "num_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "n_routed_experts",
    ):
        setattr(text_config, attr, None)

    # Apply overrides to text_config
    for k, v in overrides.get("text_config_attrs", {}).items():
        setattr(text_config, k, v)

    cfg.hf_text_config = text_config
    cfg.hf_config = overrides.get("hf_config", text_config)
    return cfg


# ---------------------------------------------------------------------------
# set_default_torch_dtype
# ---------------------------------------------------------------------------
class TestSetDefaultTorchDtype(CustomTestCase):
    def test_restores_original_dtype(self):
        original = torch.get_default_dtype()
        with set_default_torch_dtype(torch.float16):
            self.assertEqual(torch.get_default_dtype(), torch.float16)
        self.assertEqual(torch.get_default_dtype(), original)

    def test_nested_context_managers(self):
        original = torch.get_default_dtype()
        with set_default_torch_dtype(torch.float16):
            with set_default_torch_dtype(torch.bfloat16):
                self.assertEqual(torch.get_default_dtype(), torch.bfloat16)
            self.assertEqual(torch.get_default_dtype(), torch.float16)
        self.assertEqual(torch.get_default_dtype(), original)

    def test_restores_dtype_on_exception(self):
        original = torch.get_default_dtype()
        try:
            with set_default_torch_dtype(torch.float16):
                raise ValueError("test error")
        except ValueError:
            pass
        self.assertEqual(torch.get_default_dtype(), original)


# ---------------------------------------------------------------------------
# _is_moe_model
# ---------------------------------------------------------------------------
class TestIsMoeModel(CustomTestCase):
    def test_moe_in_architecture_name(self):
        cfg = _make_model_config()
        self.assertTrue(_is_moe_model(cfg, ["MixtralForCausalLM"]))
        self.assertTrue(_is_moe_model(cfg, ["SomeMoeModel"]))

    def test_mixtral_in_architecture_name(self):
        cfg = _make_model_config()
        self.assertTrue(_is_moe_model(cfg, ["MixtralForCausalLM"]))

    def test_no_moe_architecture(self):
        cfg = _make_model_config()
        self.assertFalse(_is_moe_model(cfg, ["LlamaForCausalLM"]))

    def test_num_local_experts_gt_1(self):
        cfg = _make_model_config(text_config_attrs={"num_local_experts": 8})
        self.assertTrue(_is_moe_model(cfg, ["LlamaForCausalLM"]))

    def test_num_experts_gt_1(self):
        cfg = _make_model_config(text_config_attrs={"num_experts": 4})
        self.assertTrue(_is_moe_model(cfg, ["LlamaForCausalLM"]))

    def test_num_experts_eq_1_not_moe(self):
        cfg = _make_model_config(text_config_attrs={"num_experts": 1})
        self.assertFalse(_is_moe_model(cfg, ["LlamaForCausalLM"]))

    def test_num_experts_eq_0_not_moe(self):
        cfg = _make_model_config(text_config_attrs={"num_experts": 0})
        self.assertFalse(_is_moe_model(cfg, ["LlamaForCausalLM"]))

    def test_moe_intermediate_size_gt_0(self):
        cfg = _make_model_config(text_config_attrs={"moe_intermediate_size": 256})
        self.assertTrue(_is_moe_model(cfg, ["LlamaForCausalLM"]))

    def test_moe_intermediate_size_eq_0_not_moe(self):
        cfg = _make_model_config(text_config_attrs={"moe_intermediate_size": 0})
        self.assertFalse(_is_moe_model(cfg, ["LlamaForCausalLM"]))

    def test_n_routed_experts_gt_1(self):
        cfg = _make_model_config(text_config_attrs={"n_routed_experts": 16})
        self.assertTrue(_is_moe_model(cfg, ["DeepseekV3ForCausalLM"]))

    def test_bool_true_is_moe(self):
        cfg = _make_model_config(text_config_attrs={"num_local_experts": True})
        self.assertTrue(_is_moe_model(cfg, ["LlamaForCausalLM"]))

    def test_bool_false_not_moe(self):
        cfg = _make_model_config(text_config_attrs={"num_local_experts": False})
        self.assertFalse(_is_moe_model(cfg, ["LlamaForCausalLM"]))

    def test_list_attribute_non_empty_is_moe(self):
        cfg = _make_model_config(text_config_attrs={"num_experts_per_tok": [2, 3]})
        self.assertTrue(_is_moe_model(cfg, ["LlamaForCausalLM"]))

    def test_list_attribute_empty_not_moe(self):
        cfg = _make_model_config(text_config_attrs={"num_experts_per_tok": []})
        self.assertFalse(_is_moe_model(cfg, ["LlamaForCausalLM"]))

    def test_empty_string_attribute_not_moe(self):
        cfg = _make_model_config(text_config_attrs={"num_experts": ""})
        self.assertFalse(_is_moe_model(cfg, ["LlamaForCausalLM"]))

    def test_dict_non_empty_is_moe(self):
        cfg = _make_model_config(text_config_attrs={"num_experts": {"a": 1}})
        self.assertTrue(_is_moe_model(cfg, ["LlamaForCausalLM"]))

    def test_dict_empty_not_moe(self):
        cfg = _make_model_config(text_config_attrs={"num_experts": {}})
        self.assertFalse(_is_moe_model(cfg, ["LlamaForCausalLM"]))

    def test_case_insensitive_architecture_match(self):
        cfg = _make_model_config()
        self.assertTrue(_is_moe_model(cfg, ["MIXTRALMODEL"]))
        self.assertTrue(_is_moe_model(cfg, ["SomeMOEArch"]))

    def test_multiple_architectures(self):
        cfg = _make_model_config()
        self.assertTrue(_is_moe_model(cfg, ["LlamaForCausalLM", "MixtralForCausalLM"]))

    def test_no_moe_attributes_all_none(self):
        cfg = _make_model_config()
        self.assertFalse(_is_moe_model(cfg, ["LlamaForCausalLM"]))


# ---------------------------------------------------------------------------
# _is_sequence_classification_model
# ---------------------------------------------------------------------------
class TestIsSequenceClassificationModel(CustomTestCase):
    def test_sequence_classification_arch(self):
        self.assertTrue(
            _is_sequence_classification_model(["BertForSequenceClassification"])
        )

    def test_reward_model_arch(self):
        self.assertTrue(_is_sequence_classification_model(["GPTRewardModel"]))

    def test_causal_lm_not_classification(self):
        self.assertFalse(_is_sequence_classification_model(["LlamaForCausalLM"]))

    def test_case_insensitive(self):
        self.assertTrue(
            _is_sequence_classification_model(["BERTFORSEQUENCECLASSIFICATION"])
        )

    def test_empty_list(self):
        self.assertFalse(_is_sequence_classification_model([]))

    def test_multiple_archs_one_matches(self):
        self.assertTrue(
            _is_sequence_classification_model(
                ["LlamaForCausalLM", "BertForSequenceClassification"]
            )
        )


# ---------------------------------------------------------------------------
# _get_transformers_backend_arch
# ---------------------------------------------------------------------------
class TestGetTransformersBackendArch(CustomTestCase):
    def test_basic_causal_lm(self):
        cfg = _make_model_config(is_generation=True, is_multimodal=False)
        result = _get_transformers_backend_arch(cfg, ["LlamaForCausalLM"])
        self.assertEqual(result, "TransformersForCausalLM")

    def test_multimodal_causal_lm(self):
        cfg = _make_model_config(is_generation=True, is_multimodal=True)
        result = _get_transformers_backend_arch(cfg, ["LlamaForCausalLM"])
        self.assertEqual(result, "TransformersMultiModalForCausalLM")

    def test_moe_causal_lm(self):
        cfg = _make_model_config(
            is_generation=True,
            is_multimodal=False,
            text_config_attrs={"num_local_experts": 8},
        )
        result = _get_transformers_backend_arch(cfg, ["LlamaForCausalLM"])
        self.assertEqual(result, "TransformersMoEForCausalLM")

    def test_multimodal_moe_causal_lm(self):
        cfg = _make_model_config(
            is_generation=True,
            is_multimodal=True,
            text_config_attrs={"num_local_experts": 8},
        )
        result = _get_transformers_backend_arch(cfg, ["LlamaForCausalLM"])
        self.assertEqual(result, "TransformersMultiModalMoEForCausalLM")

    def test_pooling_embedding_model(self):
        cfg = _make_model_config(is_generation=False, is_multimodal=False)
        result = _get_transformers_backend_arch(cfg, ["LlamaForCausalLM"])
        self.assertEqual(result, "TransformersEmbeddingModel")

    def test_pooling_sequence_classification(self):
        cfg = _make_model_config(is_generation=False, is_multimodal=False)
        result = _get_transformers_backend_arch(
            cfg, ["BertForSequenceClassification"]
        )
        self.assertEqual(result, "TransformersForSequenceClassification")

    def test_pooling_reward_model(self):
        cfg = _make_model_config(is_generation=False, is_multimodal=False)
        result = _get_transformers_backend_arch(cfg, ["GPTRewardModel"])
        self.assertEqual(result, "TransformersForSequenceClassification")

    def test_multimodal_detected_from_config_mismatch(self):
        """Multimodal is true when hf_config != hf_text_config."""
        text_config = MagicMock()
        hf_config = MagicMock()  # different object
        for attr in (
            "num_local_experts",
            "num_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "n_routed_experts",
        ):
            setattr(text_config, attr, None)
        cfg = MagicMock()
        cfg.is_generation = True
        cfg.is_multimodal = False
        cfg.hf_text_config = text_config
        cfg.hf_config = hf_config
        result = _get_transformers_backend_arch(cfg, ["LlamaForCausalLM"])
        self.assertEqual(result, "TransformersMultiModalForCausalLM")


# ---------------------------------------------------------------------------
# _model_impl_from_architecture
# ---------------------------------------------------------------------------
class TestModelImplFromArchitecture(CustomTestCase):
    def test_transformers_prefix(self):
        self.assertEqual(
            _model_impl_from_architecture("TransformersForCausalLM"),
            ModelImpl.TRANSFORMERS,
        )

    def test_transformers_multimodal(self):
        self.assertEqual(
            _model_impl_from_architecture("TransformersMultiModalForCausalLM"),
            ModelImpl.TRANSFORMERS,
        )

    def test_mindspore_prefix(self):
        self.assertEqual(
            _model_impl_from_architecture("MindSporeForCausalLM"),
            ModelImpl.MINDSPORE,
        )

    def test_sglang_native(self):
        self.assertEqual(
            _model_impl_from_architecture("LlamaForCausalLM"),
            ModelImpl.SGLANG,
        )

    def test_sglang_for_unknown(self):
        self.assertEqual(
            _model_impl_from_architecture("SomeCustomModel"),
            ModelImpl.SGLANG,
        )


# ---------------------------------------------------------------------------
# should_async_load
# ---------------------------------------------------------------------------
class TestShouldAsyncLoad(CustomTestCase):
    def test_cpu_tensor_returns_true(self):
        weight = torch.zeros(2, 2, device="cpu")
        self.assertTrue(should_async_load(weight))

    def test_no_device_attribute_returns_false(self):
        weight = MagicMock(spec=[])
        # No device attribute
        del weight.device
        self.assertFalse(should_async_load(weight))

    def test_none_device_returns_false(self):
        weight = MagicMock()
        weight.device = None
        self.assertFalse(should_async_load(weight))

    def test_meta_device_returns_false(self):
        weight = torch.zeros(2, 2, device="meta")
        self.assertFalse(should_async_load(weight))


# ---------------------------------------------------------------------------
# maybe_executor_submit
# ---------------------------------------------------------------------------
class TestMaybeExecutorSubmit(CustomTestCase):
    def test_sync_mode_runs_inline(self):
        result = []

        def func(val):
            result.append(val)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            futures = []
            maybe_executor_submit(
                executor=executor,
                futures=futures,
                use_async=False,
                func=func,
                func_args=(42,),
            )
        self.assertEqual(result, [42])
        self.assertEqual(len(futures), 0)

    def test_async_mode_submits_future(self):
        result = []

        def func(val):
            result.append(val)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            futures = []
            maybe_executor_submit(
                executor=executor,
                futures=futures,
                use_async=True,
                func=func,
                func_args=(99,),
            )
            # Wait for future to complete
            concurrent.futures.wait(futures)
        self.assertEqual(result, [99])
        self.assertEqual(len(futures), 1)

    def test_sync_mode_with_kwargs(self):
        result = {}

        def func(key, value=None):
            result[key] = value

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            futures = []
            maybe_executor_submit(
                executor=executor,
                futures=futures,
                use_async=False,
                func=func,
                func_args=("k",),
                func_kwargs={"value": "v"},
            )
        self.assertEqual(result, {"k": "v"})

    def test_async_mode_with_kwargs(self):
        result = {}

        def func(key, value=None):
            result[key] = value

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            futures = []
            maybe_executor_submit(
                executor=executor,
                futures=futures,
                use_async=True,
                func=func,
                func_args=("k",),
                func_kwargs={"value": "v"},
            )
            concurrent.futures.wait(futures)
        self.assertEqual(result, {"k": "v"})

    def test_default_kwargs_is_empty(self):
        """func_kwargs defaults to empty dict when not provided."""
        called_with = []

        def func():
            called_with.append(True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            futures = []
            maybe_executor_submit(
                executor=executor,
                futures=futures,
                use_async=False,
                func=func,
            )
        self.assertEqual(called_with, [True])

    def test_multiple_async_submissions(self):
        result = []

        def func(val):
            result.append(val)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for i in range(5):
                maybe_executor_submit(
                    executor=executor,
                    futures=futures,
                    use_async=True,
                    func=func,
                    func_args=(i,),
                )
            concurrent.futures.wait(futures)
        self.assertEqual(sorted(result), [0, 1, 2, 3, 4])
        self.assertEqual(len(futures), 5)


if __name__ == "__main__":
    unittest.main()
