import importlib
import sys
import types
import unittest
from types import ModuleType
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

NUM_LAYERS = 4


def import_model(name):
    if importlib.util.find_spec("vllm") is not None:
        return importlib.import_module(f"sglang.srt.models.{name}")
    # CPU CI omits vLLM; the Bailing modules import an AWQ kernel from it.
    vllm = ModuleType("vllm")
    vllm.__path__ = []
    custom_ops = ModuleType("vllm._custom_ops")
    custom_ops.awq_dequantize = MagicMock()
    with patch.dict(sys.modules, {"vllm": vllm, "vllm._custom_ops": custom_ops}):
        return importlib.import_module(f"sglang.srt.models.{name}")


class StubModule(torch.nn.Module):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return MagicMock()


def stub(*args, **kwargs):
    return StubModule()


PARALLEL = types.SimpleNamespace(
    tp_size=1,
    tp_rank=0,
    attn_tp_size=1,
    attn_tp_rank=0,
    attn_dp_size=1,
    attn_dp_rank=0,
    moe_ep_size=1,
    moe_tp_size=1,
    moe_dp_size=1,
)


def bailing_hybrid_config(num_layers):
    from sglang.srt.configs.bailing_hybrid import BailingHybridConfig

    return BailingHybridConfig(num_hidden_layers=num_layers, attention_type=1)


def bailing_moe_config(num_layers):
    from sglang.srt.configs.bailing_moe_v2 import BailingMoeV2Config

    return BailingMoeV2Config(num_hidden_layers=num_layers)


# name: (module, decoder layer class, submodules to stub)
CASES = {
    "bailing_moe_linear": (
        "bailing_moe_linear",
        "BailingMoELinearDecoderLayer",
        [
            "BailingMLP",
            "BailingMoE",
            "BailingMoEAttention",
            "BailingMoELinearAttention",
            "DsV3MLA",
        ],
    ),
    "bailing_moe": (
        "bailing_moe",
        "BailingMoEBlock",
        ["BailingMoEAttention", "BailingMoEMLP", "BailingMoESparseMoeBlock"],
    ),
}


def build_draft(case, config):
    """Construct a NextN draft layer with stubbed compute; return its FFN
    declaration and the compute submodules it built."""
    module_name, class_name, stubs = CASES[case]
    module = import_model(module_name)
    append_stages = MagicMock(return_value=(MagicMock(), MagicMock()))
    built = []

    def recording_stub(name):
        def make(*args, **kwargs):
            built.append(name)
            return StubModule()

        return make

    from sglang.srt.layers.layer_boundary import declare_attn, declare_ffn

    patches = dict(
        declare_attn=declare_attn,
        declare_ffn=declare_ffn,
        append_stages=append_stages,
    )
    patches.update({name: recording_stub(name) for name in stubs})
    if hasattr(module, "get_parallel"):
        patches["get_parallel"] = lambda: PARALLEL
    with patch.multiple(module, **patches):
        getattr(module, class_name)(config, layer_id=0, is_nextn=True)
    append_stages.assert_called_once()
    return append_stages.call_args.args[1][0], built


class TestDraftLayerDeclarations(CustomTestCase):
    def test_bailing_draft_layer_builds_an_moe(self):
        """The Bailing V2 NextN checkpoint holds expert weights, so the NextN
        layer builds the sparse MoE block, also when the model's first layers
        are dense."""
        config = bailing_moe_config(NUM_LAYERS)
        config.first_k_dense_replace = 1
        ffn, built = build_draft("bailing_moe", config)
        self.assertIn("BailingMoESparseMoeBlock", built)
        self.assertNotIn("BailingMoEMLP", built)
        self.assertTrue(ffn.sparse)

    def test_bailing_hybrid_draft_layer_is_planned_as_sparse(self):
        """The Bailing hybrid NextN layer builds an MoE, so it declares a sparse
        FFN, also when the model's first layers are dense."""
        config = bailing_hybrid_config(NUM_LAYERS)
        config.first_k_dense_replace = 1
        ffn, built = build_draft("bailing_moe_linear", config)
        self.assertIn("BailingMoE", built)
        self.assertTrue(ffn.sparse)


if __name__ == "__main__":
    unittest.main()
