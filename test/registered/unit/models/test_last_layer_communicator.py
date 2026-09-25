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


def build(case, num_layers, layer_id, config=None, **kwargs):
    """Construct one decoder layer with its submodules stubbed. Returns the
    LayerCommunicator kwargs, the LayerScatterModes.init_new kwargs, and the
    names of the stubbed submodules it built."""
    module_name, class_name, make_config, stubs, _ = CASES[case]
    module = import_model(module_name)
    communicator = MagicMock()
    scatter_modes = MagicMock()
    built = []

    def recording_stub(name):
        def make(*args, **kwargs):
            built.append(name)
            return StubModule()

        return make

    patches = dict(LayerCommunicator=communicator, LayerScatterModes=scatter_modes)
    patches.update({name: recording_stub(name) for name in stubs})
    if hasattr(module, "get_parallel"):
        patches["get_parallel"] = lambda: PARALLEL
    with patch.multiple(module, **patches):
        getattr(module, class_name)(
            config or make_config(num_layers), layer_id=layer_id, **kwargs
        )
    communicator.assert_called_once()
    return communicator.call_args.kwargs, scatter_modes.init_new.call_args.kwargs, built


def planned_as_last(case, num_layers, layer_id, **kwargs):
    """Whether the layer's layout plan makes it the model's last layer, the one
    fact the communicator reads the last layer from."""
    passed, planned, _ = build(case, num_layers, layer_id, **kwargs)
    assert "is_last_layer" not in passed, "the plan is the only source"
    return planned["layer_id"] == planned["num_layers"] - 1


class TestLastLayerCommunicator(CustomTestCase):
    def test_only_the_last_layer_is_last(self):
        """With all-reduce fusion on, only the model's last layer is marked last,
        so its FFN all-reduce is never left to a next layer that does not exist."""
        for case in CASES:
            for layer_id in range(NUM_LAYERS):
                with self.subTest(case=case, layer_id=layer_id):
                    self.assertEqual(
                        planned_as_last(case, NUM_LAYERS, layer_id),
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
                    planned_as_last(case, num_layers, layer_id, is_nextn=True)
                )

    def test_draft_model_layer_is_planned_as_a_one_layer_model(self):
        """The layout plan of a NextN / MTP draft layer treats it as the first and
        the last layer, so it takes the model's input layout and returns the
        model's output layout even when its MLP runs on SCATTERED tokens."""
        for case, (*_, draft) in CASES.items():
            if draft is None:
                continue
            num_layers, layer_id = draft
            with self.subTest(case=case):
                _, planned, _ = build(case, num_layers, layer_id, is_nextn=True)
                self.assertEqual((planned["layer_id"], planned["num_layers"]), (0, 1))

    def test_bailing_draft_layer_builds_an_moe(self):
        """The Bailing V2 NextN checkpoint holds expert weights, so the NextN
        layer builds the sparse MoE block, also when the model's first layers
        are dense."""
        config = bailing_moe_config(NUM_LAYERS)
        config.first_k_dense_replace = 1
        _, planned, built = build(
            "bailing_moe", NUM_LAYERS, 0, config=config, is_nextn=True
        )
        self.assertIn("BailingMoESparseMoeBlock", built)
        self.assertNotIn("BailingMoEMLP", built)
        self.assertTrue(planned["is_layer_sparse"])

    def test_bailing_hybrid_draft_layer_is_planned_as_sparse(self):
        """The Bailing hybrid NextN layer builds an MoE, so its layout plan is
        that of a sparse layer, also when the model's first layers are dense."""
        config = bailing_hybrid_config(NUM_LAYERS)
        config.first_k_dense_replace = 1
        _, planned, built = build(
            "bailing_moe_linear", NUM_LAYERS, 0, config=config, is_nextn=True
        )
        self.assertIn("BailingMoE", built)
        self.assertTrue(planned["is_layer_sparse"])


class TestLayerScatterModesLastLayer(CustomTestCase):
    def test_the_plan_marks_the_last_layer(self):
        from sglang.srt.layers import communicator as comm

        with (
            patch.object(comm, "enable_moe_dense_fully_dp", return_value=False),
            patch.object(comm, "_generic_prefill_cp_shards_tokens", return_value=False),
            patch.object(comm, "is_dsa_enable_prefill_cp", return_value=False),
            patch.object(comm, "is_mla_cp_enabled", return_value=False),
        ):
            for num_layers, layer_id in (
                (NUM_LAYERS, 0),
                (NUM_LAYERS, 2),
                (NUM_LAYERS, 3),
                (1, 0),
            ):
                with self.subTest(num_layers=num_layers, layer_id=layer_id):
                    modes = comm.LayerScatterModes.init_new(
                        layer_id=layer_id,
                        num_layers=num_layers,
                        is_layer_sparse=False,
                        is_previous_layer_sparse=False,
                        is_next_layer_sparse=False,
                    )
                    self.assertEqual(modes.is_last_layer, layer_id == num_layers - 1)


if __name__ == "__main__":
    unittest.main()
