"""Entry-class methods the runner calls by name answer for the model they are on."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Exaone4Config

from sglang.srt.model_executor.model_runner_components import load_model_utils
from sglang.srt.models.exaone4 import Exaone4ForCausalLM
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sglang.srt.runtime_context import get_parallel, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import parallel_scope, publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

HIDDEN = 32
HEADS = 4
KV_HEADS = 2
HEAD_DIM = HIDDEN // HEADS
INTERMEDIATE = 48


def _exaone4():
    config = Exaone4Config(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=1,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        vocab_size=64,
        max_position_embeddings=32,
        sliding_window=None,
        sliding_window_pattern=None,
        layer_types=["full_attention"],
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = Exaone4ForCausalLM(config)
    for param in model.parameters():
        torch.nn.init.normal_(param)
    return model


def _glm5_next_without_kv_scale_loader():
    """The entry class over a backbone that, like Glm5NextModel, has no
    load_kv_cache_scales."""
    model = object.__new__(Glm5NextForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.model = torch.nn.Module()
    return model


class _ContextTestCase(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        # CPU CI does not enable the CPU engine; keep RoPE on the native path.
        self.enterContext(
            patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True)
        )
        publish(ServerArgs(model_path="dummy", device="cuda", tp_size=1), role="test")
        get_parallel().override_permanently(
            pp_group=SimpleNamespace(
                is_first_rank=True, is_last_rank=True, rank_in_group=0, world_size=1
            )
        )


class TestExaone4GetWeightsByName(_ContextTestCase):
    def test_tp_replicated_kv_heads_return_the_local_rows(self):
        # LogitsProcessor reads the TP group's cpu_group to chunk input logprobs.
        with parallel_scope(
            tp_size=4,
            attn_tp_size=4,
            moe_tp_size=4,
            tp_group=SimpleNamespace(cpu_group=None),
            attn_tp_group=SimpleNamespace(cpu_group=None),
        ):
            model = _exaone4()
            qkv = model.model.layers[0].self_attn.qkv_proj.weight.data
            q, k, v = qkv.split(HEAD_DIM, dim=0)
            for proj, rows in (("q", q), ("k", k), ("v", v)):
                with self.subTest(proj=proj):
                    name = f"model.layers.0.self_attn.{proj}_proj.weight"
                    self.assertEqual(
                        model.get_weights_by_name(name, tp_size=4), rows.tolist()
                    )

    def test_checkpoint_names_resolve_to_their_slice_of_the_packed_tensor(self):
        """A lookup by checkpoint name (q_proj, up_proj, ...) returns that
        tensor's rows of the packed parameter, not None."""
        model = _exaone4()
        layer = model.model.layers[0]
        qkv = layer.self_attn.qkv_proj.weight.data
        gate_up = layer.mlp.gate_up_proj.weight.data
        q_rows = HEADS * HEAD_DIM
        kv_rows = KV_HEADS * HEAD_DIM
        expected = {
            "self_attn.q_proj": qkv[:q_rows],
            "self_attn.k_proj": qkv[q_rows : q_rows + kv_rows],
            "self_attn.v_proj": qkv[q_rows + kv_rows :],
            "mlp.gate_proj": gate_up[:INTERMEDIATE],
            "mlp.up_proj": gate_up[INTERMEDIATE:],
            "mlp.down_proj": layer.mlp.down_proj.weight.data,
        }
        for name, rows in expected.items():
            with self.subTest(name=name):
                got = model.get_weights_by_name(f"model.layers.0.{name}.weight")
                self.assertIsNotNone(got)
                self.assertEqual(got, rows.to(torch.float32).numpy().tolist())


class TestKvScaleFileRefused(_ContextTestCase):
    def test_models_without_a_kv_scale_loader_refuse_the_scale_file(self):
        """An FP8 KV cache with --quantization-param-path on a model that
        cannot read the file stops with the unsupported-model error; it neither
        crashes inside the model nor reports the scales as loaded."""
        models = {
            "exaone4": _exaone4,
            "glm5_next": _glm5_next_without_kv_scale_loader,
        }
        for kind, build in models.items():
            with (
                self.subTest(kind=kind),
                patch.object(
                    load_model_utils,
                    "get_model",
                    return_value=SimpleNamespace(quantization_param_path="scales.json"),
                ),
                self.assertRaisesRegex(
                    RuntimeError, "does not support loading scaling factors"
                ),
            ):
                load_model_utils.load_kv_cache_scales(
                    model=build(), kv_cache_dtype="fp8_e4m3", device="cpu"
                )


if __name__ == "__main__":
    unittest.main()
