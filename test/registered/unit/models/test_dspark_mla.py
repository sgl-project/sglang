import json
import tempfile
import unittest
from pathlib import Path

import torch

from sglang.srt.configs.model_config import AttentionArch, ModelConfig
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.utils.hf_transformers_utils import get_config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

HEADS, NOPE, ROPE, V, RANK, HIDDEN = 4, 8, 8, 12, 16, 64


def _draft_config(**overrides) -> dict:
    config = {
        "architectures": ["MLADSparkModel"],
        "model_type": "deepseek_v3",
        "hidden_size": HIDDEN,
        "num_hidden_layers": 5,
        "num_attention_heads": HEADS,
        "num_key_value_heads": 1,
        "q_lora_rank": None,
        "kv_lora_rank": RANK,
        "qk_nope_head_dim": NOPE,
        "qk_rope_head_dim": ROPE,
        "v_head_dim": V,
        "intermediate_size": 32,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 4096,
        "vocab_size": 100,
        "torch_dtype": "float32",
        "layer_types": ["sliding_attention"] * 5,
        "sliding_window": 2048,
        "is_causal": False,
        "block_size": 4,
        "markov_rank": 8,
        "markov_head_type": "rnn",
        "dflash_config": {
            "num_target_layers": 72,
            "target_layer_ids": [1, 36, 70],
            "mask_token": "<SPECIAL_99>",
            "mask_token_id": 99,
        },
    }
    config.update(overrides)
    return config


class TestMLADSparkDraft(CustomTestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.path = Path(self._dir.name)

    def _write(self, **overrides) -> str:
        path = Path(tempfile.mkdtemp(dir=self.path))
        (path / "config.json").write_text(json.dumps(_draft_config(**overrides)))
        return str(path)

    def _build(self, **overrides):
        from sglang.srt.models.dspark_mla import MLADSparkModel

        hf_config = get_config(self._write(**overrides), trust_remote_code=False)
        with published_topology():
            return MLADSparkModel(hf_config)

    def test_draft_pool_is_sized_for_expanded_mla_kv(self):
        # The checkpoint's num_key_value_heads describes the latent, not the cache.
        config = ModelConfig(self._write())
        self.assertEqual(config.attention_arch, AttentionArch.MHA)
        self.assertEqual(
            (config.head_dim, config.v_head_dim), (NOPE + ROPE, NOPE + ROPE)
        )
        self.assertEqual(config.get_num_kv_heads(2), HEADS // 2)
        self.assertFalse(config.is_hybrid_swa)

    def test_layers_are_bidirectional_sliding_mla_with_rnn_head(self):
        for block_size in (3, 4, 5):
            with self.subTest(block_size=block_size):
                model = self._build(block_size=block_size)
                self.assertEqual(model.gamma, block_size)
                self.assertEqual(type(model.markov_head).__name__, "RNNHead")
                for layer in model.layers:
                    attn = layer.self_attn.attn
                    self.assertEqual(attn.sliding_window_size, 2047)
                    self.assertEqual(attn.attn_type, AttentionType.ENCODER_ONLY)
                    self.assertEqual(attn.tp_k_head_num, HEADS)
                    self.assertEqual(attn.head_dim, NOPE + ROPE)

    def test_cached_kv_reproduces_latent_mla_attention(self):
        torch.manual_seed(0)
        model = self._build()
        attn = model.layers[0].self_attn
        with torch.no_grad():
            for param in attn.parameters():
                param.normal_()
        tokens = 6
        hidden = torch.randn(tokens, HIDDEN)
        positions = torch.arange(100, 100 + tokens)

        k, v = attn.context_kv(positions, hidden)
        q = attn.project_q(hidden)
        q_pe = q[..., NOPE:].contiguous()
        q_pe, _ = attn.rotary_emb(positions, q_pe, q_pe.clone())
        q = torch.cat([q[..., :NOPE], q_pe], dim=-1)
        probs = torch.einsum("thd,shd->hts", q, k).mul(attn.scaling).softmax(-1)
        cached = torch.einsum("hts,shd->thd", probs, v)

        # DeepSeek MLA layout: kv_a output is [latent | k_pe]; kv_b rows are per
        # head [k_nope | v]; attention runs on the latent with W_UK / W_UV absorbed.
        w_kva = attn.kv_a_proj_with_mqa.weight
        latent = attn.kv_a_layernorm(hidden @ w_kva[:RANK].T)
        k_pe = (hidden @ w_kva[RANK:].T).unsqueeze(1)
        _, k_pe = attn.rotary_emb(positions, k_pe.clone(), k_pe)
        w_kvb = attn.kv_b_proj.weight.view(HEADS, NOPE + V, RANK)
        q_latent = torch.einsum("tqn,qnr->tqr", q[..., :NOPE], w_kvb[:, :NOPE])
        scores = torch.einsum("tqr,sr->qts", q_latent, latent) + torch.einsum(
            "thd,sd->hts", q[..., NOPE:], k_pe.squeeze(1)
        )
        latent_out = torch.einsum(
            "hts,sr->thr", scores.mul(attn.scaling).softmax(-1), latent
        )
        expected = torch.einsum("thr,hvr->thv", latent_out, w_kvb[:, NOPE:])

        torch.testing.assert_close(cached[..., :V], expected, rtol=1e-4, atol=1e-4)
        self.assertEqual(cached[..., V:].abs().max().item(), 0.0)

    def _global_checkpoint(self, model) -> dict:
        checkpoint = {}
        for name, param in model.named_parameters():
            weight = torch.randn_like(param)
            if ".gate_up_proj." in name:
                gate, up = weight.chunk(2)
                checkpoint[name.replace("gate_up_proj", "gate_proj")] = gate
                checkpoint[name.replace("gate_up_proj", "up_proj")] = up
            else:
                checkpoint[name] = weight
        return checkpoint

    def test_checkpoint_must_match_the_model_exactly(self):
        torch.manual_seed(0)
        model = self._build()
        complete = self._global_checkpoint(model)
        model.load_weights(complete.items())
        gate_up = model.layers[0].mlp.gate_up_proj.weight
        torch.testing.assert_close(
            gate_up,
            torch.cat(
                [
                    complete["layers.0.mlp.gate_proj.weight"],
                    complete["layers.0.mlp.up_proj.weight"],
                ]
            ),
        )

        name = "layers.0.self_attn.kv_b_proj.weight"
        cases = {
            "unexpected": {
                **complete,
                "layers.0.self_attn.q_norm.weight": torch.ones(1),
            },
            "missing backbone": {k: w for k, w in complete.items() if k != name},
            "missing gate half": {
                k: w
                for k, w in complete.items()
                if k != "layers.1.mlp.gate_proj.weight"
            },
            "missing up half": {
                k: w for k, w in complete.items() if k != "layers.1.mlp.up_proj.weight"
            },
            "missing markov": {
                k: w for k, w in complete.items() if "joint_proj" not in k
            },
            "wrong shape": {**complete, name: complete[name][:, :-1]},
        }
        for case, weights in cases.items():
            with self.subTest(case=case), self.assertRaises(ValueError):
                model.load_weights(weights.items())

    def test_tp_ranks_load_their_slice_of_a_global_checkpoint(self):
        from sglang.srt.models.dspark_mla import MLADSparkModel

        torch.manual_seed(0)
        hf_config = get_config(self._write(), trust_remote_code=False)
        with published_topology():
            full = MLADSparkModel(hf_config)
        checkpoint = self._global_checkpoint(full)
        full.load_weights(checkpoint.items())
        full_params = dict(full.named_parameters())

        for rank in (0, 1):
            with published_topology(tp_size=2, ranks={"world_rank": rank}):
                shard = MLADSparkModel(hf_config)
                shard.load_weights(checkpoint.items())
            sharded = set()
            for name, param in shard.named_parameters():
                expected = full_params[name]
                if param.shape != expected.shape:
                    sharded.add(name.split(".")[-2])
                    if ".gate_up_proj." in name:
                        gate, up = expected.chunk(2)
                        expected = torch.cat([gate.chunk(2)[rank], up.chunk(2)[rank]])
                    else:
                        dim = 0 if param.shape[0] != expected.shape[0] else 1
                        expected = expected.chunk(2, dim)[rank]
                with self.subTest(rank=rank, name=name):
                    torch.testing.assert_close(param, expected, rtol=0, atol=0)
            self.assertEqual(
                sharded, {"q_proj", "kv_b_proj", "o_proj", "gate_up_proj", "down_proj"}
            )

    def test_attention_cp_shards_heads_like_the_draft_pool(self):
        # Prefill CP: TP4 is attention TP2 x CP2, and the draft pool holds the
        # heads of one attention-TP rank.
        from sglang.srt.models.dspark_mla import MLADSparkModel
        from sglang.srt.runtime_context import get_parallel

        torch.manual_seed(0)
        path = self._write()
        hf_config = get_config(path, trust_remote_code=False)
        with published_topology():
            full = MLADSparkModel(hf_config)
        checkpoint = self._global_checkpoint(full)
        full.load_weights(checkpoint.items())
        full_params = dict(full.named_parameters())
        cp = dict(tp_size=4, attn_cp_size=2, enable_prefill_cp=True)

        for rank in range(4):
            attn_tp_rank, tp_rank = rank % 2, rank
            with published_topology(
                cp_strategy="interleave", ranks={"world_rank": rank}, **cp
            ):
                pool_heads = ModelConfig(path).get_num_kv_heads(
                    get_parallel().attn_tp_size
                )
                shard = MLADSparkModel(hf_config)
                shard.load_weights(checkpoint.items())
            attn = shard.layers[0].self_attn
            k, v = attn.context_kv(torch.arange(3), torch.randn(3, HIDDEN))
            with self.subTest(rank=rank):
                self.assertEqual(pool_heads, HEADS // 2)
                self.assertEqual(attn.attn.tp_k_head_num, pool_heads)
                self.assertEqual(tuple(k.shape), (3, pool_heads, NOPE + ROPE))
                self.assertEqual(tuple(v.shape), (3, pool_heads, NOPE + ROPE))
            for name, param in shard.named_parameters():
                expected = full_params[name]
                if ".gate_up_proj." in name:
                    gate, up = expected.chunk(2)
                    expected = torch.cat([gate.chunk(4)[tp_rank], up.chunk(4)[tp_rank]])
                elif ".down_proj." in name:
                    expected = expected.chunk(4, 1)[tp_rank]
                elif ".o_proj." in name:
                    expected = expected.chunk(2, 1)[attn_tp_rank]
                elif ".q_proj." in name or ".kv_b_proj." in name:
                    expected = expected.chunk(2)[attn_tp_rank]
                with self.subTest(rank=rank, name=name):
                    torch.testing.assert_close(param, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
