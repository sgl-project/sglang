import unittest
from types import SimpleNamespace

import sgl_kernel  # noqa: F401
import torch

from sglang.srt.layers.attention.deepseek_v4_cpu_backend import DeepseekV4CPUBackend
from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Compressor
from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="stage-a-test-cpu-intel")


class TestDeepseekV41CPUAttention(CustomTestCase):
    def test_sparse_attention_matches_fp32_reference(self):
        """Padded picks exclude cache rows; sinks consume probability but not values."""
        generator = torch.Generator().manual_seed(12)
        swa_slots = torch.tensor([[-1, -1, 1, 2], [-1, 8, 2, 7], [5, 6, 7, 8]])
        extra_slots = torch.tensor([[-1, 4, 0], [11, -1, 3], [-1, -1, -1]])
        sink = torch.tensor([-2.0, 0.0, 3.0, -9.0])
        for dim in (68, 512):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(dim=dim, dtype=dtype):
                    query = torch.randn(3, 4, dim, generator=generator).to(dtype)
                    swa = torch.randn(20, dim, generator=generator).to(dtype)
                    extra = torch.randn(12, dim, generator=generator).to(dtype)
                    result = torch.ops.sgl_kernel.dsv41_sparse_attention_cpu(
                        query, swa, extra, swa_slots, extra_slots, sink, dim**-0.5
                    )
                    expected = torch.empty_like(query)
                    for row in range(3):
                        sw, ex = swa_slots[row], extra_slots[row]
                        keys = torch.cat([swa[sw[sw >= 0]], extra[ex[ex >= 0]]]).float()
                        logits = query[row].float() @ keys.T * dim**-0.5
                        probabilities = torch.cat([logits, sink[:, None]], -1).softmax(
                            -1
                        )
                        expected[row] = (probabilities[:, :-1] @ keys).to(dtype)
                    torch.testing.assert_close(result, expected)
        empty_slots = torch.full((3, 1), -1, dtype=torch.long)
        result = torch.ops.sgl_kernel.dsv41_sparse_attention_cpu(
            query, swa, extra, empty_slots, empty_slots, sink, 1.0
        )
        self.assertEqual(result.count_nonzero(), 0)
        with self.assertRaisesRegex(RuntimeError, "slot out of range"):
            torch.ops.sgl_kernel.dsv41_sparse_attention_cpu(
                query, swa, extra, torch.full((3, 1), 20), empty_slots, sink, 1.0
            )

    def test_candidate_consumer_reuses_source_blocks_not_source_topk(self):
        """Later indexers rescore the published blocks, including the newest."""
        backend = DeepseekV4CPUBackend.__new__(DeepseekV4CPUBackend)
        backend.req_to_token = torch.arange(2, 8).view(1, 6)
        backend.topk = 2
        backend.selected = {}
        backend.candidates = None
        backend.token_to_kv_pool = SimpleNamespace(
            get_low_ratio_index_k_dequant=lambda _, slots: slots[:, None].float()
        )
        indexer = SimpleNamespace(
            queries=lambda q, freqs, positions: q,
            head_weights=lambda x: x,
            scores=lambda q, k, w: torch.tensor([[8, 7, 6, 5, -10, -20]]).float(),
            is_candidate_source=True,
            uses_candidates=False,
            candidate_topk_blocks=2,
            candidate_block_size=2,
        )
        layer = SimpleNamespace(
            layer_id=20, compress_ratio=1, indexer=indexer, freqs_cis=None
        )
        x = torch.zeros(1, 1)
        req, pos = torch.tensor([0]), torch.tensor([5])
        backend._index(layer, x, x, req, pos)
        self.assertEqual(set(backend.selected[1][0].tolist()), {2, 3})
        indexer.is_candidate_source = False
        indexer.uses_candidates = True
        indexer.scores = lambda q, k, w: torch.tensor([[1, 2, 100, 99, 3, 4]]).float()
        layer.layer_id = 24
        backend._index(layer, x, x, req, pos)
        self.assertEqual(set(backend.selected[1][0].tolist()), {6, 7})

    def test_ragged_attention_respects_window_mapping_and_sink(self):
        """Uniform logits expose off-by-one windows, dummy picks and sink mass."""
        swa = torch.arange(20).float()[:, None].repeat(1, 4)
        extra = torch.arange(10).float()[:, None].repeat(1, 4)
        pool = SimpleNamespace(
            get_swa_key_buffer_radix=lambda _: swa,
            get_extra_key_buffer=lambda _: extra,
            translate_loc_from_full_to_swa=lambda slots: slots + 10,
        )
        backend = DeepseekV4CPUBackend(
            SimpleNamespace(
                model_config=SimpleNamespace(
                    hf_text_config=SimpleNamespace(
                        model_type="deepseek_v41", sliding_window=2, index_topk=2
                    )
                ),
                spec_algorithm=SimpleNamespace(is_none=lambda: True),
                token_to_kv_pool=pool,
                req_to_token_pool=SimpleNamespace(
                    req_to_token=torch.tensor([[1, 2, 3], [4, 5, 6]])
                ),
            )
        )
        backend.selected[1] = torch.tensor([[7, -1], [8, 9]])
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=torch.tensor([0, 1]),
            positions=torch.tensor([1, 2]),
        )
        result = backend.forward(
            torch.zeros(2, 2, 4),
            None,
            None,
            SimpleNamespace(layer_id=0, scaling=0.5),
            batch,
            compress_ratio=1,
            attn_sink=torch.zeros(2),
            save_kv_cache=False,
        )
        expected = torch.tensor([30 / 4, 48 / 5])[:, None, None].expand(2, 2, 4)
        torch.testing.assert_close(result, expected)

    def test_ratio_two_carries_odd_ragged_prefixes_across_chunks(self):
        """Each request's even token must survive until its own odd partner."""
        state = CompressStatePool(
            size=4,
            ring_size=2,
            overlap=False,
            head_dim=4,
            dtype=torch.float32,
            device="cpu",
            enable_memory_saver=False,
            ratio=2,
            request_scoped=True,
        )
        writes = []

        def store(layer_id, slots, latent, freqs_cis):
            writes.append((slots.clone(), latent.clone(), freqs_cis.clone()))

        backend = DeepseekV4CPUBackend.__new__(DeepseekV4CPUBackend)
        backend.token_to_kv_pool = SimpleNamespace(
            get_attention_compress_states=lambda _: state,
            set_extra_key_buffer_fused=store,
        )
        freqs = torch.polar(torch.ones(8, 2), torch.arange(16).view(8, 2).float())
        layer = SimpleNamespace(
            layer_id=2,
            compress_ratio=2,
            freqs_cis=freqs,
            indexer=None,
            compressor=SimpleNamespace(
                project=lambda x: (x, torch.zeros_like(x)),
                pool_pairs=DeepseekV41Compressor.pool_pairs,
                finish=lambda x: x,
            ),
        )
        backend._compress(
            layer,
            torch.tensor([0, 2, 4, 100]).float()[:, None].expand(-1, 4),
            torch.tensor([0, 0, 0, 1]),
            torch.tensor([0, 1, 2, 0]),
            SimpleNamespace(out_cache_loc=torch.tensor([2, 3, 4, 8])),
        )
        backend._compress(
            layer,
            torch.tensor([6, 102]).float()[:, None].expand(-1, 4),
            torch.tensor([0, 1]),
            torch.tensor([3, 1]),
            SimpleNamespace(out_cache_loc=torch.tensor([5, 9])),
        )
        self.assertEqual(writes[0][0].tolist(), [1])
        torch.testing.assert_close(writes[0][1], torch.ones(1, 4))
        self.assertEqual(writes[1][0].tolist(), [2, 4])
        torch.testing.assert_close(
            writes[1][1], torch.tensor([5, 101]).float()[:, None].expand(-1, 4)
        )
        torch.testing.assert_close(writes[1][2], freqs[torch.tensor([2, 0])])


if __name__ == "__main__":
    unittest.main()
