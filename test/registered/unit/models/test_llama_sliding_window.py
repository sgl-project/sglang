"""Phi-3 and Mistral models built from llama.py keep their config's sliding window."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from transformers import LlamaConfig, MistralConfig, Phi3Config

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.models import llama
from sglang.srt.models.llama import LlamaForCausalLM, Phi3ForCausalLM
from sglang.srt.models.mistral import MistralForCausalLM
from sglang.srt.runtime_context import reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import parallel_scope, publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

TINY = dict(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
)


class TestLlamaSlidingWindow(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy", device="cpu"), role="test")
        pp_group = SimpleNamespace(
            is_first_rank=True, is_last_rank=True, rank_in_group=0, world_size=1
        )
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(parallel_scope(pp_group=pp_group))
        # The rotary kernels are irrelevant here and need a GPU build to import.
        stack.enter_context(patch.object(llama, "get_rope"))

    def test_window_follows_config(self):
        """Phi-3-mini-4k (W = 2047) and Mistral-7B-v0.1 (W = 4096) attend to the last W
        keys as in HF; a W at or past the context length (Phi-3.5, Phi-4-mini) means none."""
        cases = [
            (Phi3ForCausalLM, Phi3Config(sliding_window=2047, **TINY), 2046),
            (MistralForCausalLM, MistralConfig(sliding_window=4096, **TINY), 4095),
            (
                Phi3ForCausalLM,
                Phi3Config(
                    sliding_window=262144, max_position_embeddings=131072, **TINY
                ),
                None,
            ),
            (Phi3ForCausalLM, Phi3Config(sliding_window=None, **TINY), None),
            (MistralForCausalLM, MistralConfig(sliding_window=None, **TINY), None),
            (LlamaForCausalLM, LlamaConfig(**TINY), None),
        ]
        for model_cls, config, window in cases:
            with self.subTest(model=model_cls.__name__, sliding_window=window):
                model = model_cls(config)
                layer_windows = {
                    m.sliding_window_size
                    for m in model.modules()
                    if isinstance(m, RadixAttention)
                }
                self.assertEqual(layer_windows, {-1 if window is None else window})
                if model_cls is not LlamaForCausalLM:
                    self.assertEqual(model.get_attention_sliding_window_size(), window)


if __name__ == "__main__":
    unittest.main()
