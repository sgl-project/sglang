"""What small real models hand their attention and linear layers at construction."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Gemma2Config, GPT2Config, LlamaConfig

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models.gemma2 import Gemma2Model
from sglang.srt.models.gpt2 import GPT2Attention
from sglang.srt.models.llama import LlamaDecoderLayer, LlamaModel
from sglang.srt.models.ministral3 import (
    Ministral3Attention,
    Ministral3DecoderLayer,
    Ministral3Model,
)
from sglang.srt.runtime_context import get_parallel, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import parallel_scope, publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

SLIDING_WINDOW = 4


class _PrefixRecorder(QuantizationConfig):
    """Builds every linear unquantized and records the name it was given."""

    def __init__(self):
        super().__init__()
        self.linear_prefixes = []

    def get_name(self):
        return "prefix_recorder"

    def get_supported_act_dtypes(self):
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        return 0

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config):
        return cls()

    def get_scaled_act_names(self):
        return []

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase):
            self.linear_prefixes.append(prefix)
            return UnquantizedLinearMethod()
        return None


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


class TestGemma2LayerPrefix(_ContextTestCase):
    def test_linears_are_named_under_their_layer(self):
        """Quantization configs match per-module targets and ignore lists
        against the checkpoint name, so each layer's linears carry
        `layers.<i>.`; without it every layer answers to the same name."""
        config = Gemma2Config(
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=64,
            max_position_embeddings=32,
            sliding_window=SLIDING_WINDOW,
            hidden_act="gelu_pytorch_tanh",
            hidden_activation="gelu_pytorch_tanh",
            architectures=["Gemma2ForCausalLM"],
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        recorder = _PrefixRecorder()
        Gemma2Model(config, quant_config=recorder, prefix="model")
        self.assertEqual(
            sorted(recorder.linear_prefixes),
            sorted(
                f"model.layers.{i}.{name}"
                for i in range(2)
                for name in (
                    "self_attn.qkv_proj",
                    "self_attn.o_proj",
                    "mlp.gate_up_proj",
                    "mlp.down_proj",
                )
            ),
        )


class TestGpt2KvHeadsUnderTp(_ContextTestCase):
    def test_kv_heads_follow_the_rank_share_of_c_attn(self):
        """Each TP rank's c_attn emits its share of the K/V heads; the
        attention layer splits K and V by the same per-rank count."""
        config = GPT2Config(n_embd=32, n_head=4, n_layer=1, vocab_size=64)
        for tp_size in (1, 2):
            with (
                self.subTest(tp_size=tp_size),
                parallel_scope(
                    tp_size=tp_size, attn_tp_size=tp_size, moe_tp_size=tp_size
                ),
            ):
                attn = GPT2Attention(layer_id=0, config=config)
                self.assertEqual(attn.attn.tp_k_head_num, attn.c_attn.num_kv_heads)
                self.assertEqual(attn.attn.tp_v_head_num, attn.c_attn.num_kv_heads)


class TestMinistral3LayerConstruction(_ContextTestCase):
    @staticmethod
    def config():
        return LlamaConfig(
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=64,
            max_position_embeddings=32,
            original_max_position_embeddings=32,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        )

    def test_decoder_layers_are_built_once(self):
        # A superclass model followed by replacement layers temporarily keeps
        # both stacks' weights alive. Count real decoder constructions, rather
        # than only inspecting the final ModuleList, which hides the duplicate.
        original_init = LlamaDecoderLayer.__init__
        for tp_size in (1, 2):
            for pp_size in (1, 2):
                for pp_rank in range(pp_size):
                    built = []
                    tp_group = SimpleNamespace(rank_in_group=0, world_size=tp_size)

                    def record_init(layer, *args, **kwargs):
                        built.append(type(layer))
                        original_init(layer, *args, **kwargs)

                    with (
                        self.subTest(tp=tp_size, pp=pp_size, rank=pp_rank),
                        parallel_scope(
                            tp_size=tp_size,
                            tp_group=tp_group,
                            attn_tp_group=tp_group,
                            moe_tp_group=tp_group,
                            attn_tp_size=tp_size,
                            moe_tp_size=tp_size,
                            pp_rank=pp_rank,
                            pp_size=pp_size,
                            pp_group=SimpleNamespace(
                                is_first_rank=pp_rank == 0,
                                is_last_rank=pp_rank == pp_size - 1,
                                rank_in_group=pp_rank,
                                world_size=pp_size,
                            ),
                        ),
                        patch.object(LlamaDecoderLayer, "__init__", record_init),
                    ):
                        model = Ministral3Model(self.config(), prefix="model")
                        start, end = model.start_layer, model.end_layer
                        self.assertEqual(
                            built, [Ministral3DecoderLayer] * (end - start)
                        )
                        self.assertEqual(
                            (start, end),
                            (pp_rank * 4 // pp_size, (pp_rank + 1) * 4 // pp_size),
                        )
                        for layer in model.layers[start:end]:
                            self.assertIsInstance(layer.self_attn, Ministral3Attention)

    def test_llama_still_builds_llama_layers(self):
        model = LlamaModel(self.config(), prefix="model")
        self.assertEqual(
            [type(layer) for layer in model.layers], [LlamaDecoderLayer] * 4
        )

    def test_ministral_mlp_weights_are_allocated_once(self):
        recorder = _PrefixRecorder()
        Ministral3Model(self.config(), quant_config=recorder, prefix="model")
        self.assertEqual(
            sorted(name for name in recorder.linear_prefixes if ".mlp." in name),
            sorted(
                f"model.layers.{i}.mlp.{name}"
                for i in range(4)
                for name in ("gate_up_proj", "down_proj")
            ),
        )


if __name__ == "__main__":
    unittest.main()
