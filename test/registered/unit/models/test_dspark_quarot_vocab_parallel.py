"""CPU construction, vocab sharding and collectives for the GLM draft path.

The production draft constructor creates both tables; their real weight loaders,
embedding forward and base-logit gather are compared with full-table arithmetic.
Two-rank Gloo covers TP2 and DP2 with effective draft TP1. It does not cover a
multi-rank attention subgroup inside DP, HCCL, attention, or model acceptance.
Only the constructor's NPU eligibility query is mocked; tensors stay on CPU.
"""

import tempfile
import time
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import torch
import torch.multiprocessing as mp
from transformers import Qwen3Config

from sglang.srt.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.dspark_quarot import (
    GlmDSparkQuaRotConfig,
    glm_dspark_quarot_scope,
)
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.models.dspark import DSparkDraftModel
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.srt.speculative.spec_utils import draft_tp_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


def _check_table_rows(module, source):
    indices = module.shard_indices
    expected = source[indices.org_vocab_start_index : indices.org_vocab_end_index]
    torch.testing.assert_close(module.weight[: len(expected)], expected, rtol=0, atol=0)
    assert torch.count_nonzero(module.weight[len(expected) :]) == 0


def _check_vocab_case(rank, *, dp_enabled, replicate):
    vocab, width = 97, 16
    # With the default padding, TP2 splits at 64. Both ranks own real rows.
    config = Qwen3Config(
        hidden_size=width,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=vocab,
        max_position_embeddings=128,
        aux_hidden_state_layer_ids=[0],
        block_size=8,
        mask_token_id=96,
        markov_rank=4,
        markov_head_type="vanilla",
        sample_from_anchor=True,
    )
    # Construction only stores this plan. File reading / FC folding are tested
    # by the separate loader tests, so no rotation file is needed here.
    plan = GlmDSparkQuaRotConfig("unused", width, "unused")
    parallel = get_parallel()
    original_tp_group = parallel.tp_group
    expected_tp = parallel.attn_tp_size if dp_enabled else parallel.tp_size
    context = draft_tp_context(parallel.attn_tp_group) if dp_enabled else nullcontext()
    with (
        context,
        envs.SGLANG_ENABLE_EMBED_REPLICATION.override(replicate),
        envs.SGLANG_NPU_GLM_DSPARK_QUAROT.override("original"),
        glm_dspark_quarot_scope(plan),
        patch("sglang.srt.models.dspark.is_npu", return_value=True),
        torch.device("cpu"),
    ):
        model = DSparkDraftModel(config)
        assert isinstance(model.embed_tokens, VocabParallelEmbedding)
        assert isinstance(model.lm_head, ParallelLMHead)
        assert model._glm_dspark_quarot_config is plan
        assert model.uses_own_vocab_modules
        assert model.embed_tokens.tp_size == (1 if replicate else expected_tp)
        assert model.embed_tokens.use_attn_tp_group == (dp_enabled and not replicate)
        assert model.lm_head.tp_size == expected_tp
        assert not model.lm_head.use_attn_tp_group
        assert parallel.tp_size == expected_tp

        embedding = (torch.arange(vocab * width).reshape(vocab, width) % 31) / 32
        head = (torch.arange(vocab * width).reshape(vocab, width) % 23) / 16
        model.embed_tokens.weight_loader(model.embed_tokens.weight, embedding)
        model.lm_head.weight_loader(model.lm_head.weight, head)
        _check_table_rows(model.embed_tokens, embedding)
        _check_table_rows(model.lm_head, head)
        if expected_tp == 2:
            assert model.lm_head.weight.shape == (64, width)
            assert model.lm_head.shard_indices.org_vocab_start_index == rank * 64

        tokens = torch.tensor([0, 63, 64, 96])
        hidden = (torch.arange(4 * width).reshape(4, width) % 7) / 8
        if dp_enabled:
            # Different requests in the singleton attention groups expose an
            # accidental gather/reduce through the original two-rank TP group.
            tokens = tokens.roll(rank)
            hidden = hidden + rank / 4
        actual_embedding = model.forward_embed(tokens)
        actual_logits, _ = model.compute_base_logits(hidden)
        torch.testing.assert_close(actual_embedding, embedding[tokens], rtol=0, atol=0)
        torch.testing.assert_close(actual_logits, hidden @ head.T, rtol=0, atol=0)
        assert actual_logits.shape == (4, vocab)

    assert parallel.tp_group is original_tp_group


def _run_vocab_worker(rank, world_size, init_path):
    torch.set_num_threads(1)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"file://{init_path}",
        backend="gloo",
        timeout=45,
    )
    try:
        for dp_size in (1, 2) if world_size == 2 else (1,):
            dp_enabled = dp_size > 1
            with (
                get_context().override_server_args(
                    speculative_algorithm="DSpark",
                    device="cpu",
                    tp_size=world_size,
                    dp_size=dp_size,
                    enable_dp_attention=dp_enabled,
                ),
                get_flags().dp.override(enabled=dp_enabled),
                envs.SGLANG_RAGGED_VERIFY_MODE.override("static"),
                # These tests use tensor collectives, not the scheduler's
                # shared-memory object-broadcast transport.
                envs.SGLANG_USE_MESSAGE_QUEUE_BROADCASTER.override(False),
            ):
                initialize_model_parallel(
                    tensor_model_parallel_size=world_size,
                    attention_data_parallel_size=dp_size,
                    backend="gloo",
                )
                try:
                    for replicate in (False, True):
                        _check_vocab_case(
                            rank, dp_enabled=dp_enabled, replicate=replicate
                        )
                finally:
                    destroy_model_parallel()
    finally:
        destroy_distributed_environment()


class TestDSparkQuaRotVocabParallel(CustomTestCase):
    def _run(self, world_size):
        with tempfile.TemporaryDirectory() as directory:
            processes = mp.spawn(
                _run_vocab_worker,
                args=(world_size, f"{directory}/gloo"),
                nprocs=world_size,
                join=False,
            )
            try:
                deadline = time.monotonic() + 75
                while not processes.join(timeout=1):
                    if time.monotonic() > deadline:
                        self.fail("CPU vocab process group exceeded 75 seconds")
            finally:
                for process in processes.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)

    def test_tp1_with_sharded_and_replicated_embedding(self):
        self._run(1)

    def test_tp2_and_dp2_with_sharded_and_replicated_embedding(self):
        self._run(2)


if __name__ == "__main__":
    unittest.main()
