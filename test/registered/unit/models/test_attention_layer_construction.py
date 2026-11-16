"""What small real models hand their attention and linear layers at construction."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Gemma2Config, Gemma3nTextConfig, GPT2Config
from transformers.masking_utils import (
    and_masks,
    causal_mask_function,
    sliding_window_overlay,
)

from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models.exaone_moe import ExaoneMoEAttention
from sglang.srt.models.gemma2 import Gemma2Model
from sglang.srt.models.gemma3n_causal import Gemma3nAttention, Gemma3nForCausalLM
from sglang.srt.models.gpt2 import GPT2Attention
from sglang.srt.runtime_context import get_parallel, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import parallel_scope, publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

SLIDING_WINDOW = 4
SEQ_LEN = 12


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


def _hf_sliding_mask(sliding_window: int) -> torch.Tensor:
    """HF's sliding-window causal mask, [SEQ_LEN, SEQ_LEN] (True = attended)."""
    mask = and_masks(sliding_window_overlay(sliding_window), causal_mask_function)
    q_idx = torch.arange(SEQ_LEN).unsqueeze(1)
    kv_idx = torch.arange(SEQ_LEN).unsqueeze(0)
    return mask(0, 0, q_idx, kv_idx)


def _layer_mask(attn) -> torch.Tensor:
    return TorchNativeAttnBackend._make_sliding_window_mask(
        q_len=SEQ_LEN,
        kv_len=SEQ_LEN,
        sliding_window_size=attn.sliding_window_size,
        device=torch.device("cpu"),
    )


class TestSlidingLayerWindowMatchesHf(_ContextTestCase):
    """HF's `sliding_window` counts the current token; a layer's window is the
    number of earlier tokens. Handing the layer HF's value lets it see one
    token more than the reference model."""

    def test_gemma3n(self):
        config = Gemma3nTextConfig(
            hidden_size=32,
            intermediate_size=[48, 48],
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=64,
            vocab_size_per_layer_input=64,
            hidden_size_per_layer_input=8,
            max_position_embeddings=32,
            sliding_window=SLIDING_WINDOW,
            layer_types=["sliding_attention", "full_attention"],
            num_kv_shared_layers=0,
            pad_token_id=0,
        )
        sliding = Gemma3nAttention(
            layer_id=0, config=config, max_position_embeddings=32
        )
        full = Gemma3nAttention(layer_id=1, config=config, max_position_embeddings=32)

        self.assertTrue(
            torch.equal(_layer_mask(sliding.attn), _hf_sliding_mask(SLIDING_WINDOW))
        )
        # The runner sizes the window from the model-level value.
        model = object.__new__(Gemma3nForCausalLM)
        object.__setattr__(model, "config", config)
        self.assertEqual(
            sliding.attn.sliding_window_size, model.get_attention_sliding_window_size()
        )
        self.assertEqual(full.attn.sliding_window_size, -1)

    def test_k_exaone(self):
        config = SimpleNamespace(
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=SLIDING_WINDOW,
            rms_norm_eps=1e-5,
        )
        sliding, full = (
            ExaoneMoEAttention(
                config=config,
                hidden_size=32,
                num_heads=4,
                num_kv_heads=2,
                layer_id=layer_id,
                max_position_embeddings=32,
            )
            for layer_id in (0, 1)
        )

        self.assertTrue(
            torch.equal(_layer_mask(sliding.attn), _hf_sliding_mask(SLIDING_WINDOW))
        )
        self.assertEqual(full.attn.sliding_window_size, -1)


if __name__ == "__main__":
    unittest.main()
