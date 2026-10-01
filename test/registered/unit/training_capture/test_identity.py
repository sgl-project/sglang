"""Global identity must agree with every rank's actual local KV placement."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import torch
from sglang.srt.training_capture.identity import (
    RankTargetContract,
    assemble_target_contract,
    bind_rank_target_contract,
    bind_target_contract,
)
from sglang.srt.training_capture.protocol import ContractError, canonical_bytes
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestTargetIdentity(CustomTestCase):
    def setUp(self):
        runtime = patch(
            "sglang.srt.layers.rotary_embedding.base.get_exec",
            return_value=SimpleNamespace(
                deterministic=SimpleNamespace(rl_on_policy_target=None)
            ),
        )
        runtime.start()
        self.addCleanup(runtime.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        (root / "model.safetensors").write_bytes(b"synthetic-weight-artifact")
        (root / "config.json").write_text('{"num_hidden_layers":6}')
        (root / "tokenizer.json").write_text('{"test_vocab":256}')
        self.root = root

    def model_args(self, *, tp_rank=0, tp_size=4, pp_rank=0, pp_size=3):
        from sglang.srt.layers.linear import QKVParallelLinear
        from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        start, end = pp_rank * (6 // pp_size), (pp_rank + 1) * (6 // pp_size)
        model = type("Qwen3ForCausalLM", (), {})()
        model.pp_group = SimpleNamespace(world_size=pp_size, rank_in_group=pp_rank)
        model.logits_processor = SimpleNamespace(
            logit_scale=None,
            final_logit_softcapping=None,
            do_tensor_parallel_all_gather=tp_size > 1,
        )
        layers, buffers = [object() for _ in range(6)], {}
        for layer in range(start, end):
            heads = 2 if layer in (2, 3) else 8
            projection = QKVParallelLinear(
                2,
                head_size=4,
                total_num_heads=8,
                total_num_kv_heads=heads,
                bias=False,
                params_dtype=torch.float32,
                tp_rank=tp_rank,
                tp_size=tp_size,
            )
            layers[layer] = SimpleNamespace(
                self_attn=SimpleNamespace(
                    qkv_proj=projection,
                    total_num_kv_heads=heads,
                    num_kv_heads=projection.num_kv_heads,
                    head_dim=4,
                    rotary_emb=RotaryEmbedding(4, 4, 8, 10000, True, torch.bfloat16),
                    k_norm=SimpleNamespace(variance_epsilon=1e-6),
                )
            )
            buffers[layer] = torch.empty(
                8, projection.num_kv_heads, 4, dtype=torch.bfloat16
            )
        model.model = SimpleNamespace(layers=layers, start_layer=start, end_layer=end)
        pool = MagicMock(spec=MHATokenToKVPool)
        pool.is_quantized_kv_cache = pool.use_hnd = False
        pool.dtype, pool.v_head_dim, pool.page_size = torch.bfloat16, 4, 1
        pool.start_layer, pool.layer_num = start, end - start
        pool.get_key_buffer.side_effect = lambda layer: buffers[layer]
        pool.get_value_buffer.side_effect = lambda layer: buffers[layer]
        config = SimpleNamespace(
            hf_text_config=SimpleNamespace(to_dict=lambda: {"num_hidden_layers": 6}),
            is_multimodal=False,
            quantization=None,
            vocab_size=256,
            model_path=str(self.root),
            dtype=torch.bfloat16,
        )
        return {
            "model_id": "test-target",
            "selected_layer_ids": [3, 1],
            "storage_chunk_tokens": 2,
            "model": model,
            "model_config": config,
            "tokenizer_path": str(self.root),
            "pool": pool,
        }

    def bind(self, *, tp_rank=0, tp_size=4, pp_rank=0, pp_size=3, **kwargs):
        return bind_rank_target_contract(
            **self.model_args(
                tp_rank=tp_rank, tp_size=tp_size, pp_rank=pp_rank, pp_size=pp_size
            ),
            tp_rank=tp_rank,
            tp_size=tp_size,
            pp_rank=pp_rank,
            pp_size=pp_size,
            **kwargs,
        )

    def ranks(self):
        return [self.bind(tp_rank=tp, pp_rank=pp) for pp in range(3) for tp in range(4)]

    def test_last_stage_requires_global_teacher_logits(self):
        args = self.model_args(tp_rank=1, tp_size=4, pp_rank=2, pp_size=3)
        args["model"].logits_processor.do_tensor_parallel_all_gather = False
        with self.assertRaisesRegex(ContractError, "global TP logits"):
            bind_rank_target_contract(
                **args, tp_rank=1, tp_size=4, pp_rank=2, pp_size=3
            )
        # An earlier stage exports KV without producing vocabulary logits.
        args = self.model_args(tp_rank=1, tp_size=4, pp_rank=0, pp_size=3)
        args["model"].logits_processor.do_tensor_parallel_all_gather = False
        bind_rank_target_contract(**args, tp_rank=1, tp_size=4, pp_rank=0, pp_size=3)

    def test_global_contract_matches_single_rank_across_tp_and_pp(self):
        ranks = [
            msgspec.json.decode(canonical_bytes(rank), type=RankTargetContract)
            for rank in reversed(self.ranks())
        ]
        teacher, kv, layout = assemble_target_contract(
            ranks, tp_size=4, pp_size=3, aux_tp_rank=1
        )
        single_teacher, single_kv = bind_target_contract(
            **self.model_args(tp_size=1, pp_size=1)
        )
        self.assertEqual(teacher, single_teacher)
        self.assertEqual(kv, single_kv)
        self.assertEqual(
            [(g.layer_id, g.num_kv_heads) for g in kv.layers], [(3, 2), (1, 8)]
        )
        self.assertEqual(layout.topology.aux_owner, "dp0-pp2-tp1")
        self.assertTrue(layout.partition("dp0-pp2-tp1").include_aux)
        self.assertFalse(layout.partition("dp0-pp2-tp1").heads)
        self.assertFalse(layout.partition("dp0-pp1-tp1").active)
        self.assertFalse(layout.partition("dp0-pp1-tp3").active)

    def test_missing_inactive_rank_and_conflicting_common_identity_are_rejected(self):
        ranks = self.ranks()
        changes = [
            ranks[:5] + ranks[6:],
            ranks + ranks[:1],
        ]
        for fields in (
            {"dp_rank": 1},
            {"tp_rank": 4},
            {"tp_size": 2},
            {"selected_layer_ids": [1, 3]},
            {"source_page_size": 2},
            {"storage_chunk_tokens": 4},
            {"dtype": "float16"},
            {"num_attention_layers": 7},
            {
                "teacher": msgspec.structs.replace(
                    ranks[0].teacher, weights_revision="other"
                )
            },
            {
                "teacher": msgspec.structs.replace(
                    ranks[0].teacher, tokenizer_revision="other"
                )
            },
            {
                "teacher": msgspec.structs.replace(
                    ranks[0].teacher, output_transform="other"
                )
            },
        ):
            changes.append([msgspec.structs.replace(ranks[0], **fields), *ranks[1:]])
        for index, modified in enumerate(changes):
            with self.subTest(case=index), self.assertRaises(ContractError):
                assemble_target_contract(modified, tp_size=4, pp_size=3)

    def test_replica_head_geometry_and_cross_stage_codec_disagreement_are_rejected(
        self,
    ):
        ranks = self.ranks()
        replica = ranks[5]
        layer = replica.layers[0]
        changes = []
        for fields in (
            {"head_range": (1, 2)},
            {"geometry": msgspec.structs.replace(layer.geometry, num_kv_heads=4)},
            {"geometry": msgspec.structs.replace(layer.geometry, value_head_dim=8)},
            {"rope_config": layer.rope_config | {"theta": 20000.0}},
            {"source_k_norm": "none"},
        ):
            modified = list(ranks)
            modified[5] = msgspec.structs.replace(
                replica, layers=[msgspec.structs.replace(layer, **fields)]
            )
            changes.append(modified)
        changes.append(
            [
                msgspec.structs.replace(
                    rank,
                    layers=[
                        msgspec.structs.replace(x, source_k_norm="none")
                        for x in rank.layers
                    ],
                )
                if rank.pp_rank == 1
                else rank
                for rank in ranks
            ]
        )
        for index, modified in enumerate(changes):
            with self.subTest(case=index), self.assertRaises(ContractError):
                assemble_target_contract(modified, tp_size=4, pp_size=3)

    def test_empty_stage_cannot_hide_pp_gaps_or_inconsistent_stage_bounds(self):
        ranks = self.ranks()
        for region, all_ranks in (((5, 6), True), ((4, 5), True), ((3, 6), False)):
            changed = [
                msgspec.structs.replace(rank, pp_layer_range=region)
                if rank.pp_rank == 2 and (all_ranks or rank.tp_rank == 0)
                else rank
                for rank in ranks
            ]
            with self.subTest(region=region), self.assertRaises(ContractError):
                assemble_target_contract(changed, tp_size=4, pp_size=3)

    def test_local_rank_pool_and_projection_must_match_loaded_placement(self):
        for case in (
            "rank",
            "pool_stage",
            "pool_shape",
            "kv_tp",
            "heads",
            "empty",
            "duplicate",
        ):
            args = self.model_args()
            rank = 0
            if case == "rank":
                rank = 1
            elif case == "pool_stage":
                args["pool"].start_layer = 2
            elif case == "pool_shape":
                args["pool"].get_key_buffer.side_effect = lambda layer: torch.empty(
                    8, 1, 4, dtype=torch.bfloat16
                )
            elif case == "kv_tp":
                args["model"].model.layers[0].self_attn.qkv_proj.kv_tp_size = 2
            elif case == "heads":
                args["model"].model.layers[1].self_attn.num_kv_heads = 1
            elif case == "empty":
                args["selected_layer_ids"] = []
            else:
                args["selected_layer_ids"] = [1, 1]
            with self.subTest(case=case), self.assertRaises(ContractError):
                bind_rank_target_contract(
                    **args, tp_rank=rank, tp_size=4, pp_rank=0, pp_size=3
                )

    def test_artifact_and_output_transform_changes_cannot_join_existing_identity(self):
        ranks = self.ranks()
        before = ranks[-1].teacher
        for filename in ("model.safetensors", "tokenizer.json"):
            path = self.root / filename
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            with self.subTest(filename=filename), self.assertRaises(ContractError):
                self.bind(
                    tp_rank=3,
                    pp_rank=2,
                    expected_weights_revision=before.weights_revision,
                    expected_tokenizer_revision=before.tokenizer_revision,
                )
            replacement = self.bind(tp_rank=3, pp_rank=2)
            with self.assertRaises(ContractError):
                assemble_target_contract(
                    [*ranks[:-1], replacement], tp_size=4, pp_size=3
                )
            path.write_bytes(original)
        args = self.model_args(tp_rank=3, pp_rank=2)
        args["model"].logits_processor.logit_scale = 0.5
        changed = bind_rank_target_contract(
            **args, tp_rank=3, tp_size=4, pp_rank=2, pp_size=3
        )
        self.assertNotEqual(
            before.fingerprint_sha256, changed.teacher.fingerprint_sha256
        )
        with self.assertRaises(ContractError):
            assemble_target_contract([*ranks[:-1], changed], tp_size=4, pp_size=3)


if __name__ == "__main__":
    unittest.main()
