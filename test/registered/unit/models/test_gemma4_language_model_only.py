"""Gemma4 --language-model-only drops the vision stack and nothing else."""

import socket
import unittest

import torch
from transformers import Gemma4Config, Gemma4TextConfig, Gemma4VisionConfig

from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.models.gemma4_mm import Gemma4ForConditionalGeneration
from sglang.srt.runtime_context import SpawnRanks, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

VISION_PREFIXES = ("vision_tower.", "embed_vision.")


def build_config(*, language_model_only):
    text = Gemma4TextConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=16,
        global_head_dim=16,
        vocab_size=512,
        vocab_size_per_layer_input=512,
        hidden_size_per_layer_input=8,
        max_position_embeddings=32,
        sliding_window=16,
        layer_types=["sliding_attention", "full_attention"],
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config = Gemma4Config(
        text_config=text,
        vision_config=Gemma4VisionConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            patch_size=16,
            position_embedding_size=32,
        ),
        audio_config=None,
    )
    # ModelConfig writes this field onto every hf_config it builds.
    config.language_model_only = language_model_only
    return config


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestGemma4LanguageModelOnly(CustomTestCase):
    def setUp(self):
        if torch.distributed.is_initialized():
            self.skipTest("requires an isolated distributed test process")
        reset_context()
        self.addCleanup(reset_context)
        original = torch.get_default_dtype()
        self.addCleanup(torch.set_default_dtype, original)
        torch.set_default_dtype(torch.bfloat16)
        server = ServerArgs(model_path="dummy", device="cuda", tp_size=1)
        publish(server, role="test", ranks=SpawnRanks(world_rank=0, gpu_id=0))
        torch.cuda.set_device(0)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
        )
        self.addCleanup(destroy_distributed_environment)
        initialize_model_parallel()
        self.addCleanup(destroy_model_parallel)
        initialize_dp_attention(server)

    def _param_names(self, *, language_model_only):
        with torch.device("cuda"):
            model = Gemma4ForConditionalGeneration(
                build_config(language_model_only=language_model_only)
            )
        return {name for name, _ in model.named_parameters()}

    def test_text_parameters_match_the_full_model(self):
        full = self._param_names(language_model_only=False)
        lm_only = self._param_names(language_model_only=True)
        self.assertTrue(any(n.startswith(VISION_PREFIXES) for n in full))
        self.assertEqual(
            lm_only, {n for n in full if not n.startswith(VISION_PREFIXES)}
        )


if __name__ == "__main__":
    unittest.main()
