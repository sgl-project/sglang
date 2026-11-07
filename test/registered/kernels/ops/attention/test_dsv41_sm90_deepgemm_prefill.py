"""SM90 DG prefill: explicit FP8 arithmetic, causal cleaning and CP ownership."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.dsv4.sm90_deepgemm_prefill import (
    cast_index_queries_to_fp8,
    deepgemm_fp8_index_logits_prefill,
    gather_fp4_index_keys_as_fp8,
)
from sglang.kernels.ops.attention.dsv4.topk import topk_transform_ragged_v2
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import DenseBlocksBackend
from sglang.srt.layers.attention.dsv4.v41_indexer.full_topk import FullTopKIndexer
from sglang.srt.layers.attention.dsv4.v41_indexer.types import PrefillInputs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.version.cuda is not None
    and torch.cuda.get_device_capability()[0] == 9,
    "requires SM90 CUDA",
)
class TestSM90DeepGEMMPrefill(CustomTestCase):
    def setUp(self):
        torch.manual_seed(47)

    def test_fp64_reference_signed_weights_scales_odd_rows_and_causal_topk(self):
        for heads in (32, 64):
            for width in (3, 259):
                with self.subTest(heads=heads, width=width):
                    q = torch.randn(7, heads, 128, device="cuda").to(
                        torch.float8_e4m3fn
                    )
                    keys = torch.randn(width, 128, device="cuda").to(
                        torch.float8_e4m3fn
                    )
                    weights = torch.randn(7, heads, device="cuda").bfloat16()
                    scales = torch.rand(width, device="cuda") + 0.5
                    lens = torch.tensor(
                        [0, 1, 2, width - 1, width, 0, width],
                        device="cuda",
                        dtype=torch.int32,
                    )
                    actual = deepgemm_fp8_index_logits_prefill(
                        q, weights, keys, lens, key_scales=scales
                    )
                    dot = torch.einsum("mhd,nd->mhn", q.double(), keys.double()).relu()
                    weighted = dot * weights.double()[:, :, None]
                    reference = weighted.sum(1) * scales.double()[None, :]
                    # Bound cancellation error by total absolute contribution,
                    # rather than a relative error that explodes near zero.
                    tolerance = (
                        weighted.abs().sum(1) * scales.double()[None, :] * 1e-3 + 1e-3
                    )
                    visible = (
                        torch.arange(width, device="cuda")[None, :] < lens[:, None]
                    )
                    self.assertTrue(
                        (
                            (actual.double() - reference).abs()[visible]
                            <= tolerance[visible]
                        ).all()
                    )
                    self.assertTrue(torch.isneginf(actual[~visible]).all())
                    self.assertEqual(actual.stride(0) % 4, 0)
                    k = min(5, width)
                    selected = torch.empty(7, k, device="cuda", dtype=torch.int32)
                    topk_transform_ragged_v2(
                        actual,
                        lens,
                        out_offsets=torch.zeros_like(lens),
                        out_indices=selected,
                    )
                    self.assertTrue(
                        torch.equal((selected >= 0).sum(1), lens.clamp(max=k).long())
                    )
                    self.assertTrue(((selected < lens[:, None]) | (selected < 0)).all())
                    for row in (1, 2, 3, 4, 6):
                        picked = selected[row][selected[row] >= 0].long()
                        expected = (
                            actual[row, : int(lens[row])]
                            .topk(min(k, int(lens[row])))
                            .values
                        )
                        torch.testing.assert_close(
                            actual[row, picked].sort().values, expected.sort().values
                        )

    def test_graph_replay_recleans_shrinking_live_lengths(self):
        q = torch.randn(5, 32, 128, device="cuda").to(torch.float8_e4m3fn)
        keys = torch.randn(259, 128, device="cuda").to(torch.float8_e4m3fn)
        weights = torch.randn(5, 32, device="cuda").bfloat16()
        lens = torch.full((5,), 259, device="cuda", dtype=torch.int32)
        scales = torch.ones(259, device="cuda")
        for _ in range(2):
            deepgemm_fp8_index_logits_prefill(q, weights, keys, lens, key_scales=scales)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            scores = deepgemm_fp8_index_logits_prefill(
                q, weights, keys, lens, key_scales=scales
            )
        for lengths in (
            [259, 259, 259, 259, 259],
            [0, 1, 3, 63, 0],
            [256, 0, 129, 1, 259],
        ):
            lens.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
            graph.replay()
            visible = torch.arange(259, device="cuda")[None, :] < lens[:, None]
            self.assertTrue(torch.isneginf(scores[~visible]).all())
            expected = deepgemm_fp8_index_logits_prefill(
                q, weights, keys, lens, key_scales=scales
            )
            torch.testing.assert_close(
                scores[visible], expected[visible], rtol=0, atol=0
            )

    def test_runtime_cp_empty_shard_source_tail_consumer_and_padding(self):
        # All FP4 codes are +/-1 with four different scales per row. This
        # exercises real gather/unpack without depending on a packer reference.
        page_size = 16
        table = torch.full(
            (256, page_size * 68), 0xAA, device="cuda", dtype=torch.uint8
        )
        table[:, page_size * 64 :] = torch.tensor(
            [125, 126, 127, 128], device="cuda", dtype=torch.uint8
        ).repeat(256, page_size)
        req_table = torch.arange(8 * 512, device="cuda").view(8, 512) * 2
        positions = torch.tensor([0, 4, 16, 0, 0], device="cuda")
        raw = torch.full((5, 16), 999, device="cuda", dtype=torch.int32)
        inputs = PrefillInputs(
            indexer=SimpleNamespace(
                index_topk=16,
                index_head_dim=128,
                n_local_heads=32,
                queries=lambda q, f: q,
                head_weights=lambda x: x,
            ),
            layer_id=0,
            compress_ratio=2,
            freqs_cis=torch.ones(512, 1, device="cuda"),
            x=torch.ones(5, 32, device="cuda", dtype=torch.bfloat16),
            q_lora=torch.randn(5, 32, 128, device="cuda").bfloat16(),
            positions=positions,
            req_rows=None,
            req_pool_indices=torch.tensor([1, 3, 5], device="cuda"),
            k_slots=torch.cat(
                (req_table[1, :32:2], req_table[3, :128:2], req_table[5, :16:2])
            )
            // 2,
            index_plans=None,
            kv_page_table=torch.empty(0, device="cuda"),
            seq_lens_cpu=[33, 128, 17],
            rows_per_request=[2, 0, 1],
            rows_per_request_device=torch.tensor([2, 0, 1], device="cuda"),
            out_raw_indices=raw,
            out_page_indices=raw.clone(),
        )
        pool = SimpleNamespace(get_index_k_with_scale_buffer=lambda _: table)
        backend = FullTopKIndexer(
            token_to_kv_pool=pool,
            req_to_token=req_table,
            use_deep_gemm_prefill=False,
            use_deep_gemm_decode=False,
        )
        with envs.SGLANG_OPT_DSV41_SM90_DEEPGEMM_PREFILL.override(True):
            backend.topk_prefill(inputs)
            for row, req in enumerate((1, 1, 5)):
                visible = (int(positions[row]) + 1) // 2
                self.assertEqual(set(raw[row].tolist()) - {-1}, set(range(visible)))
                picked = raw[row][raw[row] >= 0].long()
                self.assertTrue(
                    torch.equal(
                        inputs.out_page_indices[row, :visible].long(),
                        req_table[req, picked * 2] // 2,
                    )
                )
            self.assertTrue((raw[3:] == -1).all())
            candidate = DenseBlocksBackend(
                token_to_kv_pool=pool,
                req_to_token=req_table,
                candidate_topk_blocks=1,
                candidate_block_size=8,
                use_deep_gemm_prefill=False,
            )
            published = candidate.publish_prefill(inputs)
            self.assertEqual(published.rows_per_request, [2, 0, 1])
            self.assertEqual(published.tail([1, 0, 1]).blocks.shape[0], 2)
            published.blocks.fill_(-1)
            candidate.consume_prefill(inputs, published)
            self.assertTrue((raw == -1).all())
            self.assertTrue((inputs.out_page_indices == -1).all())

    def test_key_gather_saturates_finite_e4m3_range(self):
        table = torch.full((1, 68), 0xF7, device="cuda", dtype=torch.uint8)
        table[:, 64:] = torch.tensor([136, 127, 127, 136], device="cuda")
        keys = gather_fp4_index_keys_as_fp8(
            torch.zeros(1, device="cuda", dtype=torch.int64), table, 1
        ).float()
        self.assertTrue(torch.isfinite(keys).all().item())
        expected = torch.tensor(
            [448.0, -448.0, 6.0, -6.0, 6.0, -6.0, 448.0, -448.0], device="cuda"
        ).view(4, 2)
        torch.testing.assert_close(
            keys.view(4, 16, 2), expected[:, None, :].expand(4, 16, 2)
        )

    def test_key_gather_matches_all_finite_ue8m0_scales(self):
        device = "cuda"
        page_size = 16
        num_slots = 16 * page_size
        codes = torch.arange(16, device=device).repeat(8)
        payload = (codes[::2] | (codes[1::2] << 4)).to(torch.uint8)
        exponents = (
            torch.arange(num_slots, device=device)[:, None]
            + torch.arange(4, device=device)[None, :] * 71
        ) % 255
        table = torch.empty((16, 68 * page_size + 3), dtype=torch.uint8, device=device)
        table = table[:, : 68 * page_size]
        table[:, : 64 * page_size] = payload.repeat(16, page_size)
        table[:, 64 * page_size :] = exponents.reshape(16, -1)
        # Reverse the slot order, cross pages and exercise a strided int32 view.
        slots = torch.arange(num_slots - 1, 0, -1, dtype=torch.int32, device=device)
        slots = torch.stack((slots, slots), dim=1)[:, 0]
        actual = gather_fp4_index_keys_as_fp8(slots, table, page_size).float()
        lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device=device,
            dtype=torch.float64,
        )
        scale = torch.exp2(exponents[slots.long()].double() - 127)
        expected = lut[codes] * scale.repeat_interleave(32, dim=1)
        expected = expected.clamp(-448, 448).float().to(torch.float8_e4m3fn).float()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_query_cast_saturates_fake_fp4_values(self):
        magnitudes = torch.tensor([0, 0.5, 1.5, 6], device="cuda")
        scales = torch.tensor([2**-13, 1, 64, 128, 1024], device="cuda")
        values = (scales[:, None] * magnitudes).flatten()
        values = torch.cat((values, -values))
        q = values.repeat(103)[: 32 * 128].view(1, 32, 128).bfloat16()
        expected = q.float().clamp(-448, 448).to(torch.float8_e4m3fn)
        torch.testing.assert_close(
            cast_index_queries_to_fp8(q).float(), expected.float(), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
