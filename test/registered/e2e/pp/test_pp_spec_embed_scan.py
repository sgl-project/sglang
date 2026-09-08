"""
Usage:
python3 -m unittest test_pp_spec_embed_scan.TestDraftEmbedScan
"""

import os
import unittest

import torch
from transformers import MistralConfig, PretrainedConfig

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

# Shrunk from inclusionAI/Ling-mini-2.0; the field names are the real
# checkpoint's, only the sizes are cut down for test speed.
_BAILING_CONFIG = {
    "architectures": ["BailingMoeV2ForCausalLM"],
    "model_type": "bailing_moe",
    "num_hidden_layers": 1,
    "num_nextn_predict_layers": 1,
    "hidden_size": 256,
    "intermediate_size": 512,
    "moe_intermediate_size": 128,
    "moe_shared_expert_intermediate_size": 128,
    "num_shared_experts": 1,
    "num_experts": 16,
    "num_experts_per_tok": 4,
    "n_group": 4,
    "topk_group": 2,
    "norm_topk_prob": True,
    "moe_router_enable_expert_bias": True,
    "routed_scaling_factor": 2.5,
    "score_function": "sigmoid",
    "router_dtype": "fp32",
    "first_k_dense_replace": 0,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "partial_rotary_factor": 0.5,
    "use_qk_norm": True,
    "use_qkv_bias": False,
    "use_bias": False,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 4096,
    "rope_parameters": {"rope_theta": 600000, "rope_type": "default"},
    "vocab_size": 1024,
    "tie_word_embeddings": False,
}


class TestDraftEmbedScan(CustomTestCase):
    """PP+spec loads the draft input embedding via a type scan plus a
    checkpoint-name table; a draft family whose embedding hangs under a new
    attribute name (or grows a second VocabParallelEmbedding) would misload
    silently. Pin the scan on the families the runtime GLM/DSv4 tests miss.
    """

    @classmethod
    def setUpClass(cls):
        from sglang.srt.distributed.parallel_state import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        from sglang.srt.runtime_context import get_context
        from sglang.srt.server_args import ServerArgs

        server_args = ServerArgs(model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST)
        server_args.resolve_once()
        get_context().set_server_args(server_args)

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29611")
        init_distributed_environment(
            world_size=1, rank=0, local_rank=0, distributed_init_method="env://"
        )
        initialize_model_parallel(tensor_model_parallel_size=1)
        torch.set_default_dtype(torch.bfloat16)
        torch.cuda.set_device(0)

    def test_bailing_nextn_embedding_is_found(self):
        from sglang.srt.models.bailing_moe_nextn import BailingMoeForCausalLMNextN
        from sglang.srt.speculative.eagle_worker_v2 import _find_draft_input_embedding

        config = PretrainedConfig.from_dict(dict(_BAILING_CONFIG))
        with torch.device("cuda"):
            model = BailingMoeForCausalLMNextN(config)
        self.assertIs(_find_draft_input_embedding(model), model.model.word_embeddings)

    def test_mistral_eagle_embedding_is_found(self):
        from sglang.srt.models.mistral_eagle import MistralForCausalLMEagle
        from sglang.srt.speculative.eagle_worker_v2 import _find_draft_input_embedding

        config = MistralConfig(
            vocab_size=1024,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        with torch.device("cuda"):
            model = MistralForCausalLMEagle(config)
        self.assertIs(_find_draft_input_embedding(model), model.model.embed_tokens)

    def test_name_table_matches_published_checkpoints(self):
        from sglang.srt.speculative.eagle_worker_v2 import _EMBED_TENSOR_NAMES

        # External-source literals: the embedding tensor's spelling in each
        # family's published target checkpoint (inclusionAI/Ling-*-2.0
        # model.safetensors.index.json; Mistral-Large-3 consolidated index).
        self.assertIn("model.word_embeddings.weight", _EMBED_TENSOR_NAMES)
        self.assertIn("tok_embeddings.weight", _EMBED_TENSOR_NAMES)


if __name__ == "__main__":
    unittest.main()
