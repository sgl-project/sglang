import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import _handle_dflash
from sglang.srt.environ import envs
from sglang.srt.layers import linear, vocab_parallel_embedding
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative import dflash_info
from sglang.srt.speculative.dflash_dp import gather_draft_embedding
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestDFlashDPAdmission(CustomTestCase):
    def _args(self, **overrides):
        fields = dict(
            model_path="dummy",
            device="cuda",
            tp_size=4,
            attn_dp_size=2,
            attn_cp_size=1,
            enable_dp_lm_head=True,
            speculative_algorithm="DFLASH",
            speculative_draft_model_path="draft",
            speculative_num_draft_tokens=8,
            speculative_draft_attention_backend="flashinfer",
        )
        return ServerArgs(**(fields | overrides))

    def test_cuda_dp_keeps_explicit_placement_and_raw_input(self):
        args = self._args()
        _handle_dflash(args)
        self.assertTrue(args.enable_dp_lm_head)
        self.assertEqual(resolution_result(args, "attn_dp_size"), 2)
        self.assertIsNone(args.speculative_num_steps)
        self.assertEqual(resolution_result(args, "speculative_num_steps"), 1)

    def test_incompatible_cuda_dp_layouts_fail(self):
        for fields, message in [
            ({"enable_dp_lm_head": False}, "enable-dp-lm-head"),
            ({"attn_cp_size": 2}, "context parallel"),
            ({"pp_size": 2}, "pp_size"),
            ({"device": "xpu"}, "CUDA or NPU"),
        ]:
            with (
                self.subTest(fields=fields),
                self.assertRaisesRegex(ValueError, message),
            ):
                _handle_dflash(self._args(**fields))

    def test_unimplemented_mixed_prefill_coordination_fails_at_startup(self):
        with (
            envs.SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION.override(True),
            self.assertRaisesRegex(ValueError, "DP_SPEC_PREFILL_COORDINATION=0"),
        ):
            _handle_dflash(self._args())

    def test_non_dp_does_not_require_dp_head(self):
        args = self._args(attn_dp_size=1, enable_dp_lm_head=False)
        _handle_dflash(args)
        self.assertFalse(resolution_result(args, "enable_dp_lm_head"))


class _TableGroup:
    def __init__(self, shards):
        self.shards = shards

    def all_gather(self, weight, dim):
        if not any(torch.equal(weight, shard) for shard in self.shards):
            raise AssertionError("embedding used a different table's group")
        return torch.cat(self.shards, dim=dim)


def _embedding_shards(org_vocab, added_vocab, tp_size):
    embeddings = []
    for rank in range(tp_size):
        parallel = SimpleNamespace(tp_size=tp_size, tp_rank=rank, tp_group=None)
        with (
            patch.object(
                vocab_parallel_embedding, "get_parallel", return_value=parallel
            ),
            patch.object(linear, "get_parallel", return_value=parallel),
        ):
            embedding = VocabParallelEmbedding(
                num_embeddings=org_vocab + added_vocab,
                org_num_embeddings=org_vocab,
                embedding_dim=2,
                padding_size=8,
            )
        indices = embedding.shard_indices
        with torch.no_grad():
            embedding.weight.fill_(-999)
            for token in range(
                indices.org_vocab_start_index, indices.org_vocab_end_index
            ):
                embedding.weight[token - indices.org_vocab_start_index].fill_(token)
            for token in range(
                indices.added_vocab_start_index, indices.added_vocab_end_index
            ):
                offset = (
                    indices.num_org_elements_padded
                    + token
                    - indices.added_vocab_start_index
                )
                embedding.weight[offset].fill_(token)
        embeddings.append(embedding)
    group = _TableGroup([embedding.weight for embedding in embeddings])
    for embedding in embeddings:
        embedding.tp_group = group
    return embeddings


class TestDFlashDPEmbedding(CustomTestCase):
    def test_padding_and_added_vocabulary_keep_logical_token_ids(self):
        """Uneven and empty vocabulary shards must not shift token IDs after gathering."""
        for org_vocab, added_vocab, tp_size in [(5, 0, 4), (9, 3, 4), (9, 3, 1)]:
            with self.subTest(
                org_vocab=org_vocab, added_vocab=added_vocab, tp_size=tp_size
            ):
                expected = (
                    torch.arange(org_vocab + added_vocab)
                    .float()
                    .unsqueeze(1)
                    .expand(-1, 2)
                )
                for embedding in _embedding_shards(org_vocab, added_vocab, tp_size):
                    # The table's retained group wins over any ambient draft/target scope.
                    actual = gather_draft_embedding(embedding)
                    torch.testing.assert_close(actual, expected)

    def test_replicated_table_keeps_storage(self):
        embedding = _embedding_shards(9, 0, 1)[0]
        actual = gather_draft_embedding(embedding)
        self.assertEqual(actual.data_ptr(), embedding.weight.data_ptr())
        torch.testing.assert_close(actual[:, 0], torch.arange(9).float())

    def test_materialized_embedding_preserves_quant_method_and_output_dtype(self):
        embeddings = _embedding_shards(9, 3, 4)
        group = _TableGroup([e.weight.double() * 2 for e in embeddings])
        for embedding in embeddings:
            embedding.quant_method = SimpleNamespace(
                embedding=lambda layer, ids: layer.weight[ids].double() * 2
            )
            embedding.output_dtype = torch.float64
            embedding.tp_group = group
            actual = gather_draft_embedding(embedding)
            expected = (torch.arange(12).double() * 2).unsqueeze(1).expand(-1, 2)
            torch.testing.assert_close(actual, expected)

    def test_other_embedding_keeps_its_forward_semantics(self):
        class ScaledEmbedding(nn.Embedding):
            def forward(self, ids):
                return super().forward(ids) * 2

        embedding = ScaledEmbedding(8, 2)
        self.assertIsNone(gather_draft_embedding(embedding))


class TestDFlashDPMetadata(CustomTestCase):
    def test_deferred_verify_preserves_live_inputs_until_final_planning(self):
        """Early graph loading must not replace an idle or unequal DP batch's live rows."""
        for mode in (ForwardMode.IDLE, ForwardMode.DECODE):
            for graph_eligible in (False, True):
                with self.subTest(mode=mode, graph_eligible=graph_eligible):
                    n = 0 if mode.is_idle() else 4
                    ids = torch.arange(n)
                    forward = ForwardBatch(
                        forward_mode=mode,
                        batch_size=0 if mode.is_idle() else 1,
                        input_ids=ids,
                        req_pool_indices=torch.zeros(
                            0 if mode.is_idle() else 1, dtype=torch.long
                        ),
                        seq_lens=torch.ones(
                            0 if mode.is_idle() else 1, dtype=torch.long
                        ),
                        out_cache_loc=ids,
                        seq_lens_sum=0 if mode.is_idle() else 1,
                        positions=ids,
                    )

                    def preplan(batch):
                        batch.input_ids = torch.full((8,), -1)
                        batch.mark_forward_metadata_ready()

                    runner = SimpleNamespace(
                        decode_cuda_graph_runner=SimpleNamespace(
                            can_run_graph=lambda batch: graph_eligible,
                            load_batch=preplan,
                        ),
                        attn_backend=SimpleNamespace(init_forward_metadata=preplan),
                    )
                    batch = SimpleNamespace(forward_mode=mode)
                    spec = DFlashVerifyInput(
                        draft_token=ids, positions=ids, draft_token_num=4
                    )
                    with (
                        patch.object(dflash_info, "_is_npu", False),
                        patch.object(ForwardBatch, "init_new", return_value=forward),
                        patch(
                            "sglang.srt.speculative.spec_utils.prepare_mamba_track_for_verify"
                        ),
                    ):
                        result, can_graph = spec.prepare_for_verify(
                            batch,
                            SimpleNamespace(model_runner=runner),
                            defer_metadata_init=True,
                        )
                    torch.testing.assert_close(result.input_ids, ids)
                    self.assertTrue(result.needs_forward_metadata_init())
                    self.assertEqual(can_graph, graph_eligible)


if __name__ == "__main__":
    unittest.main()
