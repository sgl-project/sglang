"""Load a small dense DSpark checkpoint through the production CPU loader.

The synthetic file has the same 64-tensor layout as a five-layer BF16 export.
It verifies loading and shared-module contracts, not accelerator inference.
"""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.environ import envs
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.models.dspark import DSparkDraftModel
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _checkpoint():
    config = {
        "architectures": ["DSparkDraftModel"],
        "dtype": "bfloat16",
        "block_size": 8,
        "aux_hidden_state_layer_ids": [2, 4, 6],
        "mask_token_id": 31,
        "markov_rank": 4,
        "markov_head_type": "vanilla",
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
        "transformer_layer_config": {
            "model_type": "qwen3",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 5,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            # Keep head_dim independent of hidden_size / num_attention_heads,
            # and three captured features independent of the five draft layers.
            "head_dim": 16,
            "vocab_size": 32,
            "max_position_embeddings": 128,
            "tie_word_embeddings": False,
        },
    }
    weights = {}

    def add(name, shape):
        # Distinct values by source tensor and position reveal shard swaps and
        # accidental transposes; all comparisons use the serialized BF16 data.
        value = torch.arange(torch.Size(shape).numel()).reshape(shape) % 23
        weights[name] = (value.float() / 32 + len(weights)).to(torch.bfloat16)

    for index in range(5):
        for name, shape in {
            "self_attn.q_proj.weight": (32, 16),
            "self_attn.k_proj.weight": (32, 16),
            "self_attn.v_proj.weight": (32, 16),
            "self_attn.o_proj.weight": (16, 32),
            "self_attn.q_norm.weight": (16,),
            "self_attn.k_norm.weight": (16,),
            "mlp.gate_proj.weight": (32, 16),
            "mlp.up_proj.weight": (32, 16),
            "mlp.down_proj.weight": (16, 32),
            "input_layernorm.weight": (16,),
            "post_attention_layernorm.weight": (16,),
        }.items():
            add(f"layers.{index}.{name}", shape)
    for name, shape in {
        "fc.weight": (16, 48),
        "hidden_norm.weight": (16,),
        "norm.weight": (16,),
        "markov_head.markov_w1.weight": (32, 4),
        "markov_head.markov_w2.weight": (32, 4),
        "confidence_head.proj.weight": (1, 20),
        "confidence_head.proj.bias": (1,),
        "embed_tokens.weight": (32, 16),
        "lm_head.weight": (32, 16),
    }.items():
        add(name, shape)
    return config, weights


class TestDSparkWeightLoading(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.group_dir = tempfile.TemporaryDirectory()
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"file://{cls.group_dir.name}/gloo",
            backend="gloo",
        )
        initialize_model_parallel(tensor_model_parallel_size=1, backend="gloo")

    @classmethod
    def tearDownClass(cls):
        destroy_model_parallel()
        destroy_distributed_environment()
        cls.group_dir.cleanup()
        super().tearDownClass()

    def setUp(self):
        super().setUp()
        self.enterContext(envs.SGLANG_RAGGED_VERIFY_MODE.override("static"))
        self.enterContext(
            get_context().override_server_args(
                speculative_algorithm="DSpark", device="cpu"
            )
        )
        self.model_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.config, self.weights = _checkpoint()

    def _load(self):
        (self.model_dir / "config.json").write_text(json.dumps(self.config))
        save_file(self.weights, str(self.model_dir / "model.safetensors"))
        config = ModelConfig(
            str(self.model_dir),
            dtype="bfloat16",
            is_draft_model=True,
            speculative_algorithm="DSpark",
        )
        self.assertIsNone(config.quantization)
        loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
        return loader.load_model(model_config=config, device_config=DeviceConfig("cpu"))

    def test_all_static_weights_and_shards_are_loaded_in_bfloat16(self):
        model = self._load()
        self.assertIsInstance(model, DSparkDraftModel)
        self.assertEqual(len(self.weights), 64)
        self.assertIsNone(model.confidence_head)
        self.assertIsNone(model.embed_tokens)
        self.assertIsNone(model.lm_head)
        params = dict(model.named_parameters())
        checked = set()
        checked_source_count = 0
        for name, source in self.weights.items():
            if name.startswith(("confidence_head.", "embed_tokens.", "lm_head.")):
                continue
            loaded = None
            for export_name, packed_name, shard in (
                ("q_proj", "qkv_proj", 0),
                ("k_proj", "qkv_proj", 1),
                ("v_proj", "qkv_proj", 2),
                ("gate_proj", "gate_up_proj", 0),
                ("up_proj", "gate_up_proj", 1),
            ):
                if f".{export_name}." in name:
                    destination = name.replace(export_name, packed_name)
                    loaded = params[destination].narrow(0, shard * 32, 32)
                    break
            else:
                destination = name
                loaded = params[name]
            self.assertEqual(loaded.dtype, torch.bfloat16)
            torch.testing.assert_close(loaded, source, rtol=0, atol=0)
            checked.add(destination)
            checked_source_count += 1
        self.assertEqual(checked_source_count, 60)
        self.assertEqual(checked, set(params))
        self.assertEqual(model.fc.in_features, 48)
        self.assertEqual(len(model.layers), 5)
        self.assertEqual(model.layers[0].self_attn.head_dim, 16)

    def test_shared_target_modules_are_preserved_and_markov_is_used(self):
        model = self._load()
        with set_default_torch_dtype(torch.bfloat16), torch.device("cpu"):
            embedding = VocabParallelEmbedding(32, 16)
            head = ParallelLMHead(32, 16)
        with torch.no_grad():
            embedding.weight.fill_(0.25)
            head.weight.copy_(
                (torch.arange(head.weight.numel()).reshape_as(head.weight) % 7) / 16
            )
        embedding_before, head_before = (
            embedding.weight.detach().clone(),
            head.weight.detach().clone(),
        )
        model.attach_shared_modules(embed_tokens=embedding, lm_head=head)
        # A reload must not replace target weights with the draft export's copy.
        model.load_weights(self.weights.items())
        self.assertIs(model.embed_tokens, embedding)
        self.assertIs(model.lm_head, head)
        torch.testing.assert_close(embedding.weight, embedding_before, rtol=0, atol=0)
        torch.testing.assert_close(head.weight, head_before, rtol=0, atol=0)
        hidden = torch.full((2, 16), 0.125, dtype=torch.bfloat16)
        logits, _ = model.compute_base_logits(hidden)
        expected = hidden @ head_before.T
        torch.testing.assert_close(logits, expected[:, :32], rtol=0, atol=0)
        previous = torch.tensor([1, 3])
        corrected = model.markov_head.apply_step_logits(
            logits, token_ids=previous, hidden_states=None
        )
        expected_bias = (
            self.weights["markov_head.markov_w1.weight"][previous]
            @ self.weights["markov_head.markov_w2.weight"].T
        )
        torch.testing.assert_close(corrected, logits + expected_bias, rtol=0, atol=0)
        self.assertFalse(torch.equal(corrected, logits))

    def test_fc_feature_count_mismatch_is_rejected(self):
        self.weights["fc.weight"] = torch.ones(16, 80, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "fc.weight shape mismatch"):
            self._load()

    def test_native_config_packed_weights_and_backbone_aliases(self):
        # Exercise the existing flat HF export, fused projections, backbone
        # model prefix, and encoder aliases through the production loader.
        from sglang.srt.configs.speculators import normalize_speculators_dspark_config

        self.config = normalize_speculators_dspark_config(self.config)
        packed_expected = {}
        for index in range(5):
            for packed, sources in (
                ("self_attn.qkv_proj", ("q_proj", "k_proj", "v_proj")),
                ("mlp.gate_up_proj", ("gate_proj", "up_proj")),
            ):
                family = packed.split(".")[0]
                name = f"layers.{index}.{packed}.weight"
                weight = torch.cat(
                    [
                        self.weights.pop(f"layers.{index}.{family}.{source}.weight")
                        for source in sources
                    ]
                )
                packed_expected[name] = weight
                self.weights[name] = weight
        self.weights["encoder.fc.weight"] = self.weights.pop("fc.weight")
        self.weights["encoder.output_norm_enc.weight"] = self.weights.pop(
            "hidden_norm.weight"
        )
        self.weights = {
            f"model.{name}" if name.startswith("layers.") else name: value
            for name, value in self.weights.items()
        }
        self.weights["model.layers.0.self_attn.rotary_emb.inv_freq"] = torch.ones(8)
        model = self._load()
        params = dict(model.named_parameters())
        for name, weight in packed_expected.items():
            torch.testing.assert_close(params[name], weight, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
