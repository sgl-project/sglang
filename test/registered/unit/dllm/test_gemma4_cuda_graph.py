"""Regression coverage for DiffusionGemma graph metadata and vocabulary shards."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.dllm.algorithm.gemma4_renoise import Gemma4Renoise
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.models.gemma4_diffusion import DiffusionGemmaTextEmbedding
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _ReadStream:
    reads_are_translated = False

    def __init__(self):
        self.tokens = torch.arange(64).view(4, 16)

    def fill_packed_read_stream(
        self,
        *,
        req_pool_indices,
        seq_lens,
        indptr,
        total_tokens,
        out,
        kv_start_idx=None,
        sliding_window=False,
    ):
        for i, (req, length) in enumerate(zip(req_pool_indices, seq_lens)):
            start = 0 if kv_start_idx is None else int(kv_start_idx[i])
            out[indptr[i] : indptr[i + 1]] = self.tokens[req, start : start + length]
        return False


class TestGemma4GraphMetadata(unittest.TestCase):
    def _backend(self):
        backend = TritonAttnBackend.__new__(TritonAttnBackend)
        backend.device = "cpu"
        backend.dllm_block_size = 4
        backend.sliding_window_size = 6
        backend.kv_index_translator = _ReadStream()
        backend.token_to_kv_pool = None
        backend.kv_indptr = torch.zeros(4, dtype=torch.int32)
        backend.qo_indptr = torch.zeros(4, dtype=torch.int32)
        backend.window_kv_indptr = torch.zeros(4, dtype=torch.int32)
        backend.cuda_graph_kv_indices = torch.zeros(64, dtype=torch.int64)
        backend.cuda_graph_window_kv_indices = torch.zeros(64, dtype=torch.int64)
        return backend

    def test_graph_canvas_stays_bidirectional(self):
        backend = self._backend()
        backend._can_run_dense_fp8_chunked_mha = Mock(return_value=False)
        backend.allow_bidirectional_attention_in_extend = False
        backend.dcp_size = 1
        backend.enable_deterministic = True
        backend._forward_extend_unified = Mock()
        layer = SimpleNamespace(
            qk_head_dim=4,
            v_head_dim=4,
            logit_capping_method="tanh",
            logit_cap=0.0,
            is_cross_attention=False,
            attn_type=AttentionType.DECODER_BIDIRECTIONAL,
        )
        qkv = torch.zeros(4, 4)
        for mode, expected_causal in (
            (ForwardMode.DLLM_EXTEND, False),
            (ForwardMode.EXTEND, True),
        ):
            with self.subTest(mode=mode):
                batch = SimpleNamespace(forward_mode=mode, mha_one_shot=False)
                backend.forward_extend(qkv, qkv, qkv, layer, batch, save_kv_cache=False)
                self.assertIs(
                    backend._forward_extend_unified.call_args.args[4], expected_causal
                )

    def test_graph_reads_context_without_reading_canvas_twice(self):
        backend = self._backend()
        requests = torch.tensor([1, 2, 0])
        # Prefixes 3, 11, and an empty padded request; each has a four-token canvas.
        backend._apply_cuda_graph_metadata(
            bs=3,
            req_pool_indices=requests,
            seq_lens=torch.tensor([7, 15, 4]),
            forward_mode=ForwardMode.DLLM_EXTEND,
            spec_info=None,
        )
        metadata = backend._build_cuda_graph_forward_metadata(
            3, ForwardMode.DLLM_EXTEND, None
        )
        self.assertEqual(metadata.max_extend_len, 4)
        self.assertIsNone(metadata.custom_mask)
        torch.testing.assert_close(
            metadata.qo_indptr, torch.tensor([0, 4, 8, 12], dtype=torch.int32)
        )
        torch.testing.assert_close(
            metadata.kv_indptr, torch.tensor([0, 3, 14, 14], dtype=torch.int32)
        )
        torch.testing.assert_close(
            metadata.kv_indices[:14],
            torch.tensor(list(range(16, 19)) + list(range(32, 43))),
        )
        torch.testing.assert_close(
            metadata.window_kv_indptr, torch.tensor([0, 3, 9, 9], dtype=torch.int32)
        )
        torch.testing.assert_close(
            metadata.window_kv_indices[:9],
            torch.tensor(list(range(16, 19)) + list(range(37, 43))),
        )

        # The captured views must see different lengths and request slots on replay.
        pointers = (
            metadata.kv_indices.data_ptr(),
            metadata.window_kv_indices.data_ptr(),
        )
        backend._apply_cuda_graph_metadata(
            bs=3,
            req_pool_indices=torch.tensor([3, 1, 0]),
            seq_lens=torch.tensor([4, 9, 4]),
            forward_mode=ForwardMode.DLLM_EXTEND,
            spec_info=None,
        )
        torch.testing.assert_close(
            metadata.kv_indptr, torch.tensor([0, 0, 5, 5], dtype=torch.int32)
        )
        torch.testing.assert_close(metadata.kv_indices[:5], torch.arange(16, 21))
        torch.testing.assert_close(metadata.window_kv_indices[:5], torch.arange(16, 21))
        self.assertEqual(
            pointers,
            (metadata.kv_indices.data_ptr(), metadata.window_kv_indices.data_ptr()),
        )


class TestGemma4GraphInputEmbeddings(unittest.TestCase):
    def test_preplanned_replay_refreshes_self_conditioning(self):
        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.dllm_uses_input_embeds = True
        runner.ragged_verify_mode = False
        runner.deepep_adapter = Mock()
        runner.bs = 4
        runner.raw_num_token = 12
        runner.captured_req_width = 4
        runner.enable_pdmux = False
        runner._capture_graph_size = Mock(return_value=4)
        runner._resolve_lora_variant = Mock(return_value=None)
        runner._resolve_dsa_variant = Mock(return_value=None)
        runner._make_graph_key = Mock(return_value=4)
        runner.buffers = SimpleNamespace(
            input_ids=torch.zeros(16, dtype=torch.long),
            positions=torch.zeros(16, dtype=torch.long),
            input_embeds=torch.full((16, 7), -1.0),
        )
        batch = SimpleNamespace(
            needs_forward_metadata_init=lambda: False,
            input_ids=torch.arange(12),
            positions=torch.arange(12),
            input_embeds=torch.ones(12, 7),
        )
        for value in (1.0, 2.0):
            batch.input_embeds.fill_(value)
            runner.load_batch(batch)
            torch.testing.assert_close(
                runner.buffers.input_embeds[:12], batch.input_embeds
            )
            self.assertTrue(torch.all(runner.buffers.input_embeds[12:] == -1.0))
        batch.input_embeds = None
        with self.assertRaisesRegex(ValueError, "prepared input embeddings"):
            runner.load_batch(batch)


class TestGemma4VocabularyShards(unittest.TestCase):
    def test_padded_shards_load_and_reconstruct_soft_embeddings(self):
        torch.manual_seed(123)
        config = SimpleNamespace(vocab_size=73, hidden_size=4)
        weight = torch.randn(73, 4)
        probabilities = torch.randn(6, 73).softmax(dim=-1)
        expected = probabilities @ weight * 2.0
        for tp_size in (1, 2, 4):
            with self.subTest(tp_size=tp_size):
                partials = []
                for rank in range(tp_size):
                    with patch(
                        "sglang.srt.layers.vocab_parallel_embedding.get_parallel",
                        return_value=SimpleNamespace(tp_rank=rank, tp_size=tp_size),
                    ):
                        embedding = DiffusionGemmaTextEmbedding(config)
                    embedding.weight_loader(embedding.weight, weight)
                    shard = embedding.shard_indices
                    count = shard.org_vocab_end_index - shard.org_vocab_start_index
                    torch.testing.assert_close(
                        embedding.weight[:count],
                        weight[shard.org_vocab_start_index : shard.org_vocab_end_index],
                    )
                    self.assertTrue(torch.all(embedding.weight[count:] == 0))
                    algorithm = Gemma4Renoise.__new__(Gemma4Renoise)
                    algorithm.embed_tokens = embedding
                    with patch(
                        "sglang.srt.dllm.algorithm.gemma4_renoise.tensor_model_parallel_all_reduce",
                        side_effect=lambda tensor: tensor,
                    ) as reduce:
                        partials.append(algorithm._soft_embeddings(probabilities))
                    self.assertEqual(reduce.call_count, int(tp_size > 1))
                torch.testing.assert_close(torch.stack(partials).sum(dim=0), expected)


if __name__ == "__main__":
    unittest.main()
