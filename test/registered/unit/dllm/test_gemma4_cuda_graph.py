"""Regression coverage for DiffusionGemma graph metadata and vocabulary shards."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.dllm.algorithm.gemma4_renoise import Gemma4Renoise
from sglang.srt.layers.attention.flashattention_dense_backend import (
    FlashAttentionDenseBackend,
)
from sglang.srt.layers.attention.graph_variants import (
    DLLM_FULL_WINDOW,
    DLLM_VARLEN,
    DllmWindowGraphVariants,
)
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.models.gemma4_diffusion import DiffusionGemmaTextEmbedding
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _ReadStream:
    reads_are_translated = False
    is_translating = False

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
        token_mapping=None,
    ):
        for i, (req, length) in enumerate(zip(req_pool_indices, seq_lens)):
            start = 0 if kv_start_idx is None else int(kv_start_idx[i])
            out[indptr[i] : indptr[i + 1]] = self.tokens[req, start : start + length]
        return False


class TestGemma4GraphMetadata(CustomTestCase):
    def test_full_window_variant_selection(self):
        for canvas in (128, 256):
            variants = DllmWindowGraphVariants(1023, canvas)
            for prefixes, captured, expected in (
                ([1022], 1, DLLM_VARLEN),
                ([1023], 1, DLLM_FULL_WINDOW),
                ([1023, 8192], 2, DLLM_FULL_WINDOW),
                ([1023, 1022], 2, DLLM_VARLEN),
                ([8192, 8192], 4, DLLM_VARLEN),
            ):
                batch = SimpleNamespace(
                    batch_size=len(prefixes),
                    input_ids=torch.zeros(len(prefixes) * canvas),
                    forward_mode=ForwardMode.DLLM_EXTEND,
                    seq_lens_cpu=torch.tensor(prefixes) + canvas,
                )
                self.assertEqual(variants.select(batch, captured), expected)

    def _backend(self):
        backend = FlashAttentionDenseBackend.__new__(FlashAttentionDenseBackend)
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
        backend._prefill_graph_metadata = {}
        backend._decode_graph_metadata = {}
        backend.use_sliding_window_kv_pool = False
        return backend

    def test_context_replay_does_not_replace_denoise_buffers(self):
        backend = self._backend()
        batch = SimpleNamespace(
            batch_size=2,
            input_ids=torch.zeros(8, dtype=torch.int64),
            out_cache_loc=torch.arange(8),
            req_pool_indices=torch.tensor([1, 2]),
            seq_lens=torch.tensor([7, 15]),
            encoder_lens=None,
            spec_info=None,
            forward_mode=ForwardMode.DLLM_EXTEND,
        )
        backend.init_forward_metadata_out_graph(batch, in_capture=True)
        denoise = backend.forward_metadata
        denoise_lengths = denoise.kv_indptr.clone()
        batch.forward_mode = ForwardMode.EXTEND
        batch.max_seq_len_override = 16
        batch.extend_prefix_lens = torch.tensor([3, 11])
        batch.extend_seq_lens = torch.tensor([4, 4])
        batch.extend_seq_lens_cpu = [4, 4]
        backend.init_forward_metadata_out_graph(batch, in_capture=True)
        context = backend.forward_metadata
        self.assertEqual(context.max_extend_len, 4)
        ptr = context.kv_indices.data_ptr()
        self.assertNotEqual(ptr, denoise.kv_indices.data_ptr())
        batch.extend_prefix_lens = torch.tensor([1, 8])
        backend.init_forward_metadata_out_graph(batch)
        self.assertEqual(ptr, backend.forward_metadata.kv_indices.data_ptr())
        torch.testing.assert_close(
            context.kv_indptr, torch.tensor([0, 1, 9], dtype=torch.int32)
        )
        torch.testing.assert_close(
            context.kv_indices[:9], torch.tensor([16, *range(32, 40)])
        )
        torch.testing.assert_close(denoise.kv_indptr, denoise_lengths)
        batch.forward_mode = ForwardMode.DLLM_EXTEND
        backend.init_forward_metadata_out_graph(batch)
        self.assertIs(backend.forward_metadata, denoise)

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


class TestGemma4GraphInputEmbeddings(CustomTestCase):
    def test_preplanned_replay_refreshes_self_conditioning(self):
        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.dllm_uses_input_embeds = True
        runner.ragged_verify_mode = False
        runner.deepep_adapter = Mock()
        runner.bs = 4
        runner.raw_num_token = 12
        runner.captured_req_width = 4
        runner.enable_pdmux = False
        runner.attention_graph_variants = None
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

        runner.input_preparation = Gemma4Renoise.prepare_graph_inputs
        state = torch.full((16, 7), -1.0)
        runner.buffer_registry = Mock()
        runner.buffer_registry.get_slot.return_value.buffer = state
        batch.input_preparation_state = torch.ones(12, 7)
        runner.load_batch(batch)
        torch.testing.assert_close(state[:12], batch.input_preparation_state)
        torch.testing.assert_close(runner.buffers.input_ids[:12], batch.input_ids)
        self.assertEqual(torch.count_nonzero(state[12:]), 0)
        self.assertEqual(torch.count_nonzero(runner.buffers.input_ids[12:]), 0)


class TestGemma4VocabularyShards(CustomTestCase):
    def test_padded_shards_load_and_reconstruct_soft_embeddings(self):
        torch.manual_seed(123)
        config = SimpleNamespace(vocab_size=73, hidden_size=4)
        weight = torch.randn(73, 4)
        probabilities = torch.randn(6, 73).softmax(dim=-1)
        expected = probabilities @ weight * 2.0
        # TP3 does not divide the 64-padded vocab (128), so padding scales with TP.
        for tp_size in (1, 2, 3, 4):
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
