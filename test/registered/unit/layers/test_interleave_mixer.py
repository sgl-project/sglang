"""Sequence-order mixers and KPool metadata under interleave CP."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.cp import interleave
from sglang.srt.layers.cp.interleave import InterleaveContextParallelMetadata
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestInterleaveMixer(CustomTestCase):
    def test_mtp_prefill_shards_rotated_mm_embeddings_and_target_states_together(self):
        """Draft CP must fill MM request tails before sharding, without looking up
        image sentinel IDs, and restore target states before full-sequence logits.
        """
        from sglang.srt.layers.cp import base
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.model_executor.runner.eager_runner import EagerRunner
        from sglang.srt.models.deepseek_nextn import DeepseekModelNextN
        from sglang.srt.models.glm5_next_nextn import (
            Glm5NextForConditionalGenerationNextN,
        )
        from sglang.srt.runtime_context import get_context, get_parallel

        ids = torch.tensor([1, 900000, 900000, 2, 900000, 3, 4])
        positions = torch.tensor([1, 2, 3, 4, 1, 2, 3])
        embeddings = torch.arange(14).reshape(7, 2).float()
        target_states = embeddings + 100
        expected_embeddings = embeddings.clone()
        expected_embeddings[[3, 6]] = torch.tensor([[2.0, 2.0], [4.0, 4.0]])
        expected = expected_embeddings + target_states
        packed = torch.zeros(2, 4, 2)
        packed[0, :4] = expected[::2]
        packed[1, :3] = expected[1::2]

        class DraftBody(DeepseekModelNextN):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.embed_tokens = torch.nn.Embedding.from_pretrained(
                    torch.arange(8).float().repeat_interleave(2).reshape(8, 2)
                )

            def forward(self, input_ids, positions, forward_batch, input_embeds=None):
                logical = len(ids[rank::2])
                torch.testing.assert_close(input_ids[:logical], ids[rank::2])
                torch.testing.assert_close(
                    positions[:logical], batch.positions[rank::2]
                )
                return input_embeds + forward_batch.spec_info.hidden_states

        model = Glm5NextForConditionalGenerationNextN.__new__(
            Glm5NextForConditionalGenerationNextN
        )
        torch.nn.Module.__init__(model)
        model.model = DraftBody()
        model.lm_head = None
        model.pp_group = SimpleNamespace(is_last_rank=True)
        model.logits_processor = lambda ids, states, *args, **kw: states
        runner = SimpleNamespace(model_runner=SimpleNamespace(model=model))
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            input_ids=ids,
            positions=positions,
            mm_input_embeds=embeddings.clone(),
            contains_mm_inputs=lambda: True,
            extend_start_loc=torch.tensor([0, 4]),
            extend_seq_lens=torch.tensor([4, 3]),
            extend_seq_lens_cpu=[4, 3],
            spec_info=SimpleNamespace(hidden_states=target_states),
            attn_cp_metadata=InterleaveContextParallelMetadata(
                total_seq_lens=7,
                per_rank_actual_token=[4, 4],
                per_rank_logical_token=[4, 3],
            ),
        )
        for rank in range(2):

            def gather(output, local):
                torch.testing.assert_close(local, packed[rank])
                output.copy_(packed.flatten(0, 1))

            with (
                self.subTest(rank=rank),
                get_context().override_server_args(
                    tp_size=2,
                    attn_cp_size=2,
                    enable_prefill_cp=True,
                    cp_strategy="interleave",
                    cp_tp_group_sharing=True,
                ),
                get_parallel().override(
                    tp_rank=rank, attn_cp_rank=rank, attn_tp_rank=0, attn_cp_group=None
                ),
                patch.object(base, "_STRATEGY", interleave.InterleaveCPStrategy(2)),
                patch.object(interleave, "attn_cp_all_gather_into_tensor", gather),
                patch.object(interleave, "is_allocation_symmetric", return_value=False),
                patch("torch.cuda.current_stream", return_value=None),
            ):
                # Exercise the same model-boundary dispatch as the eager runner.
                if getattr(model, "supports_full_sequence_cp", False):
                    result = model(ids, positions, batch)
                else:
                    result = EagerRunner._execute_extend_cp(runner, batch, {})
                torch.testing.assert_close(result, expected)
                self.assertIs(batch.spec_info.hidden_states, target_states)
                self.assertFalse(hasattr(batch, "input_ids_global"))

    def test_wrapper_preserves_explicit_embedding_overrides(self):
        """Full-sequence CP must not replace caller embeddings with token lookup."""
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration

        ids = torch.tensor([1, 2, 3])
        embeddings = torch.arange(6).reshape(3, 2).float()

        class Decoder:
            def get_input_embeddings(self):
                return lambda ids: torch.zeros(len(ids), 2)

            def __call__(self, *, input_embeds, **kwargs):
                return input_embeds * 2

        model = SimpleNamespace(
            model=Decoder(),
            is_mrope_enabled=False,
            capture_aux_hidden_states=False,
            pp_group=SimpleNamespace(is_last_rank=False),
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            contains_mm_inputs=lambda: False,
            input_embeds=embeddings.clone(),
        )
        result = Glm5NextForConditionalGeneration.forward(
            model, ids, torch.arange(3), batch, input_embeds=embeddings
        )
        torch.testing.assert_close(result, embeddings * 2)
        torch.testing.assert_close(batch.input_embeds, embeddings)

    def test_kda_cache_uses_the_same_head_partition_as_linear_attention(self):
        from sglang.srt.configs.glm5_next import Glm5NextTextConfig
        from sglang.srt.runtime_context import (
            get_context,
            get_linear_attn_tp_rank,
            get_linear_attn_tp_size,
            get_parallel,
        )

        config = Glm5NextTextConfig(
            linear_attn_config={
                "num_heads": 32,
                "head_dim": 128,
                "short_conv_kernel_size": 4,
                "kda_layers": [0],
            }
        )
        for sharing, expected_heads, expected_rank in ((True, 8, 3), (False, 32, 0)):
            with (
                self.subTest(sharing=sharing),
                get_context().override_server_args(
                    tp_size=4,
                    attn_cp_size=4,
                    enable_prefill_cp=True,
                    cp_strategy="interleave",
                    cp_tp_group_sharing=sharing,
                ),
                get_parallel().override(tp_rank=3, attn_cp_rank=3, attn_tp_rank=0),
            ):
                self.assertEqual(get_linear_attn_tp_size(), 32 // expected_heads)
                self.assertEqual(get_linear_attn_tp_rank(), expected_rank)
                self.assertEqual(
                    config.mamba2_cache_params.shape.temporal,
                    (expected_heads, 128, 128),
                )

    def test_language_model_cp_boundary_consumes_completed_embeddings(self):
        from sglang.srt.layers.cp import base
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.models.glm5_next import Glm5NextModel
        from sglang.srt.runtime_context import get_context, get_parallel

        # Some IDs represent multimodal placeholders outside the vocabulary.
        # The model boundary must use the embeddings the wrapper already made.
        ids = torch.tensor([1, 900000, 900000, 2, 3])
        positions = torch.arange(5)
        embeddings = torch.arange(10).reshape(5, 2).float()
        metadata = InterleaveContextParallelMetadata(
            total_seq_lens=5,
            per_rank_actual_token=[3, 3],
            per_rank_logical_token=[3, 2],
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            attn_cp_metadata=metadata,
            input_ids=ids,
            spec_info=None,
            extend_seq_lens_cpu=[5],
        )
        expected = embeddings * 2
        packed = torch.zeros(2, 3, 2)
        packed[0, :3] = expected[::2]
        packed[1, :2] = expected[1::2]

        def gather(output, local):
            output.copy_(packed.flatten(0, 1))

        for rank in range(2):

            def body(
                input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors=None
            ):
                logical = 3 if rank == 0 else 2
                torch.testing.assert_close(input_embeds[:logical], embeddings[rank::2])
                torch.testing.assert_close(
                    positions[:logical], torch.arange(rank, 5, 2)
                )
                torch.testing.assert_close(input_ids[:logical], ids[rank::2])
                return input_embeds * 2

            model = SimpleNamespace(
                pp_group=SimpleNamespace(is_last_rank=True), _forward=body
            )
            with (
                self.subTest(rank=rank),
                get_context().override_server_args(
                    tp_size=2,
                    attn_cp_size=2,
                    enable_prefill_cp=True,
                    cp_strategy="interleave",
                ),
                get_parallel().override(
                    tp_rank=rank, attn_cp_rank=rank, attn_tp_rank=0, attn_cp_group=None
                ),
                patch.object(base, "_STRATEGY", interleave.InterleaveCPStrategy(2)),
                patch.object(interleave, "attn_cp_all_gather_into_tensor", gather),
                patch.object(interleave, "is_allocation_symmetric", return_value=False),
            ):
                result = Glm5NextModel.forward(
                    model, None, positions, batch, input_embeds=embeddings
                )
                torch.testing.assert_close(result, expected)
                self.assertFalse(hasattr(batch, "input_ids_global"))

    def test_pooling_gathers_keys_and_scores_but_keeps_queries_local(self):
        from sglang.srt.layers.cp import base, interleave, interleave_kpool

        keys = torch.arange(33).reshape(11, 3).bfloat16()
        scores = keys + 100
        positions = torch.arange(11)
        packed = torch.full((4, 4, 6), -999, dtype=torch.bfloat16)
        for token in range(11):
            packed[token % 4, token // 4] = torch.cat((keys[token], scores[token]))
        metadata = InterleaveContextParallelMetadata(
            total_seq_lens=11,
            per_rank_actual_token=[4] * 4,
            per_rank_logical_token=[3, 3, 3, 2],
        )
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_context_parallel_extend=lambda: True),
            attn_cp_metadata=metadata,
            positions=positions,
        )
        for rank in range(4):
            with self.subTest(rank=rank):

                def gather(output, local):
                    torch.testing.assert_close(
                        local[: len(keys[rank::4])], packed[rank, : len(keys[rank::4])]
                    )
                    output.copy_(packed.flatten(0, 1))

                parallel = SimpleNamespace(attn_cp_rank=rank, attn_cp_group=None)
                with (
                    patch.object(base, "get_parallel", return_value=parallel),
                    patch.object(interleave, "get_parallel", return_value=parallel),
                    patch.object(interleave, "attn_cp_all_gather_into_tensor", gather),
                    patch.object(
                        interleave, "is_allocation_symmetric", return_value=False
                    ),
                    patch.object(
                        interleave_kpool,
                        "get_cp_strategy",
                        return_value=interleave.InterleaveCPStrategy(4),
                    ),
                ):
                    full_key, full_score, full_positions = (
                        interleave_kpool.materialize_compression_inputs(
                            packed[rank, :, :3],
                            packed[rank, :, 3:],
                            positions[rank::4],
                            batch,
                        )
                    )
                torch.testing.assert_close(full_key, keys)
                torch.testing.assert_close(full_score, scores)
                self.assertIs(full_positions, positions)

    def test_sequence_order_and_padding_round_trip(self):
        for tokens, shard_rows in ((8, 2), (11, 3), (11, 8), (1, 1), (0, 0)):
            with self.subTest(tokens=tokens, shard_rows=shard_rows):
                sequence = torch.arange(tokens * 2).reshape(tokens, 2).float()
                shards = torch.zeros(4, shard_rows, 2)
                for token in range(tokens):
                    shards[token % 4, token // 4] = sequence[token]
                packed = shards.flatten(0, 1)
                batch = SimpleNamespace(
                    forward_mode=SimpleNamespace(
                        is_context_parallel_extend=lambda: True
                    ),
                    attn_cp_metadata=InterleaveContextParallelMetadata(
                        total_seq_lens=tokens,
                        per_rank_actual_token=[shard_rows] * 4,
                    ),
                )
                with patch.object(
                    interleave,
                    "get_parallel",
                    return_value=SimpleNamespace(attn_cp_size=4),
                ):
                    restored = interleave.cp_interleave_to_sequence_order(packed, batch)
                    torch.testing.assert_close(restored, sequence)
                    # A causal mixer must see the original order, not rank-major rows.
                    mixed = restored.cumsum(0)
                    output = interleave.cp_interleave_to_rank_order(
                        mixed, batch, packed.shape[0]
                    )
                    torch.testing.assert_close(
                        interleave.cp_interleave_to_rank_order(
                            sequence, batch, packed.shape[0]
                        ),
                        packed,
                    )
                for token in range(tokens):
                    torch.testing.assert_close(
                        output.reshape(4, shard_rows, 2)[token % 4, token // 4],
                        sequence[: token + 1].sum(0),
                    )

    def test_decode_does_not_reorder_rows(self):
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_context_parallel_extend=lambda: False),
            attn_cp_metadata=None,
        )
        hidden = torch.arange(6).reshape(3, 2)
        self.assertIs(interleave.cp_interleave_to_sequence_order(hidden, batch), hidden)
        self.assertIs(interleave.cp_interleave_to_rank_order(hidden, batch, 3), hidden)

    def test_kpool_local_queries_keep_full_compression_request_order(self):
        from sglang.srt.layers.cp import interleave_kpool

        batch = SimpleNamespace(
            extend_seq_lens_cpu=[1, 3, 5],
            req_pool_indices=torch.tensor([9, 2, 7]),
            forward_mode=SimpleNamespace(is_context_parallel_extend=lambda: True),
            attn_cp_metadata=InterleaveContextParallelMetadata(total_seq_lens=9),
        )
        metadata = SimpleNamespace(
            real_page_table=torch.tensor([[20], [70]]),
            dsa_seqlens_expanded=torch.tensor([6, 11]),
            indexer_seq_lens_cpu=torch.tensor([8, 14]),
            dsa_extend_seq_lens_list=[1, 1],
        )
        with patch.object(
            interleave_kpool,
            "get_parallel",
            return_value=SimpleNamespace(attn_cp_size=4, attn_cp_rank=2),
        ):
            local = interleave_kpool.local_query_inputs(batch, metadata)
        self.assertEqual(local["local_extend_seq_lens_cpu"], [1, 1])
        self.assertEqual(local["local_seq_lens_cpu"], [8, 14])
        self.assertEqual(local["local_req_pool_indices"].tolist(), [2, 7])
        self.assertIs(local["local_real_page_table"], metadata.real_page_table)
        self.assertIs(local["local_seqlens_expanded"], metadata.dsa_seqlens_expanded)


if __name__ == "__main__":
    unittest.main()
