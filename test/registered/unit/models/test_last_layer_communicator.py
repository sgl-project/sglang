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


def qwen3_next_config(num_layers):
    from sglang.srt.configs.qwen3_next import Qwen3NextConfig

    return Qwen3NextConfig(num_hidden_layers=num_layers)


def bailing_hybrid_config(num_layers):
    from sglang.srt.configs.bailing_hybrid import BailingHybridConfig

    return BailingHybridConfig(num_hidden_layers=num_layers, attention_type=1)


def bailing_moe_config(num_layers):
    from sglang.srt.configs.bailing_moe_v2 import BailingMoeV2Config

    return BailingMoeV2Config(num_hidden_layers=num_layers)


def step3p5_config(num_layers):
    from sglang.srt.configs.step3p5 import Step3p5Config

    # Per-layer lists also cover the MTP layer, whose id is num_layers.
    return Step3p5Config(
        num_hidden_layers=num_layers,
        layer_types=["full_attention"] * (num_layers + 1),
        moe_layers_enum=",".join(str(i) for i in range(1, num_layers + 1)),
        swiglu_limits_shared=None,
        rope_theta=[10000.0] * (num_layers + 1),
        partial_rotary_factors=[1.0] * (num_layers + 1),
        use_head_wise_attn_gate=False,
        share_expert_dim=16,
    )


def minimax_m3_config(num_layers):
    from transformers import PretrainedConfig

    return PretrainedConfig(
        num_hidden_layers=num_layers,
        hidden_size=16,
        rms_norm_eps=1e-6,
        dense_intermediate_size=32,
    )


# name: (module, decoder layer class, config factory, submodules to stub,
#        (config layer count, layer id) of the draft model's layer or None)
CASES = {
    "qwen3_next_linear": (
        "qwen3_next",
        "Qwen3HybridLinearDecoderLayer",
        qwen3_next_config,
        ["Qwen3GatedDeltaNet", "Qwen2MoeSparseMoeBlock", "Qwen2MoeMLP"],
        (1, 0),
    ),
    "qwen3_next_attention": (
        "qwen3_next",
        "Qwen3HybridAttentionDecoderLayer",
        qwen3_next_config,
        [
            "Qwen2MoeSparseMoeBlock",
            "Qwen2MoeMLP",
            "QKVParallelLinear",
            "RowParallelLinear",
            "RadixAttention",
            "get_rope",
        ],
        (1, 0),
    ),
    "bailing_moe_linear": (
        "bailing_moe_linear",
        "BailingMoELinearDecoderLayer",
        bailing_hybrid_config,
        [
            "BailingMLP",
            "BailingMoE",
            "BailingMoEAttention",
            "BailingMoELinearAttention",
            "DsV3MLA",
        ],
        (NUM_LAYERS, 0),
    ),
    "bailing_moe": (
        "bailing_moe",
        "BailingMoEBlock",
        bailing_moe_config,
        ["BailingMoEAttention", "BailingMoEMLP", "BailingMoESparseMoeBlock"],
        (NUM_LAYERS, 0),
    ),
    "step3p5": (
        "step3p5",
        "Step3p5DecoderLayer",
        step3p5_config,
        ["Step3p5Attention", "Step3p5MLP", "Step3p5MoEMLP"],
        (NUM_LAYERS, NUM_LAYERS),
    ),
    "minimax_m3": (
        "minimax_m3",
        "MiniMaxM3DecoderLayer",
        minimax_m3_config,
        ["MiniMaxM3Attention", "MiniMaxM3MLP", "MiniMaxM3MoE"],
        None,
    ),
}


def is_last_layer_passed(case, num_layers, layer_id, **kwargs):
    module_name, class_name, make_config, stubs, _ = CASES[case]
    module = import_model(module_name)
    communicator = MagicMock()
    patches = dict(LayerCommunicator=communicator, LayerScatterModes=MagicMock())
    patches.update({name: stub for name in stubs})
    if hasattr(module, "get_parallel"):
        patches["get_parallel"] = lambda: PARALLEL
    with patch.multiple(module, **patches):
        getattr(module, class_name)(
            make_config(num_layers), layer_id=layer_id, **kwargs
        )
    communicator.assert_called_once()
    return communicator.call_args.kwargs.get("is_last_layer", False)


class TestLastLayerCommunicator(CustomTestCase):
    def test_only_the_last_layer_is_last(self):
        """With all-reduce fusion on, only the model's last layer is marked last,
        so its FFN all-reduce is never left to a next layer that does not exist."""
        for case in CASES:
            for layer_id in range(NUM_LAYERS):
                with self.subTest(case=case, layer_id=layer_id):
                    self.assertEqual(
                        is_last_layer_passed(case, NUM_LAYERS, layer_id),
                        layer_id == NUM_LAYERS - 1,
                    )

    def test_draft_model_layer_is_last(self):
        """The single decoder layer of a NextN / MTP draft model is marked last,
        whatever layer id and layer count it is built with."""
        for case, (*_, draft) in CASES.items():
            if draft is None:
                continue
            num_layers, layer_id = draft
            with self.subTest(case=case):
                self.assertTrue(
                    is_last_layer_passed(case, num_layers, layer_id, is_nextn=True)
                )


if __name__ == "__main__":
    unittest.main()
