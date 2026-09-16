import unittest

import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=40, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(is_hip() and torch.cuda.is_available(), "HIP compact attention")
class TestCompactAttention(unittest.TestCase):
    def test_backend_reads_mixed_pool_formats(self):
        from functools import partial
        from types import SimpleNamespace
        from unittest.mock import patch

        from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
            dequantize_k_cache_paged,
        )
        from sglang.srt.environ import envs
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
            select_dsv4_kv_layout,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.test.kits.attention_unittest.attention_methods import (
            dsv4_attention as kit,
        )

        with envs.SGLANG_DSV4_KV_LAYOUT.override("v41"):
            layout, option = select_dsv4_kv_layout()
        case = kit.DSV4AttentionCase(
            name="compact_backend",
            backend="dsv4",
            forward_mode=ForwardMode.DECODE,
            num_heads=16,
            page_size=256,
            prefix_lens=(32, 64),
            extend_lens=(1, 1),
        )
        with patch.object(
            kit,
            "DeepSeekV4TokenToKVPool",
            partial(
                DeepSeekV4TokenToKVPool, kv_layout=layout, compressed_kv_layout=option
            ),
        ):
            fixture = kit.build_dsv4_attention_fixture(
                self, case, compression_ratios=[0, 1, 2]
            )
        self.addCleanup(fixture.runner._server_args_override.restore)
        backend = fixture.backend
        pool = fixture.runner.token_to_kv_pool
        batch = fixture.forward_batch
        backend.init_forward_metadata(batch)
        backend.init_forward_metadata_in_graph(batch)
        core = backend.forward_metadata.core_attn_metadata
        q = torch.randn(2, 16, 512, device="cuda", dtype=torch.bfloat16) * 0.25
        sink = torch.randn(16, device="cuda")
        for layer_id, ratio in ((1, 1), (2, 2)):
            main = pool.get_swa_raw_buffer(layer_id)
            extra = pool.get_extra_key_buffer(layer_id)
            main_page = pool.swa_page_size
            extra_page = pool.get_extra_key_page_size(layer_id)
            main_loc = torch.arange(
                main.shape[0] * main_page, device="cuda", dtype=torch.int32
            )
            extra_loc = torch.arange(
                extra.shape[0] * extra_page, device="cuda", dtype=torch.int32
            )
            pool.set_swa_key_buffer_radix_fused(
                layer_id,
                main_loc,
                torch.randn(len(main_loc), 512, device="cuda", dtype=torch.bfloat16),
            )
            pool.set_extra_key_buffer_fused(
                layer_id,
                extra_loc,
                torch.randn(len(extra_loc), 512, device="cuda", dtype=torch.bfloat16),
            )
            extra_idx = core.sparse_page_indices(ratio)
            extra_idx.fill_(-1)
            extra_idx[:, :8] = torch.arange(3, 11, device="cuda", dtype=torch.int32)
            core.sparse_topk_lengths(ratio).fill_(8)
            actual = backend.forward(
                q,
                q,
                q,
                SimpleNamespace(layer_id=layer_id, v_head_dim=512),
                batch,
                compress_ratio=ratio,
                save_kv_cache=False,
                attn_sink=sink,
            )
            keys = []
            scores = []
            for raw, loc, page, kind, indices, lens in (
                (
                    main,
                    main_loc,
                    main_page,
                    layout,
                    core.swa_page_indices,
                    core.swa_topk_lengths,
                ),
                (
                    extra,
                    extra_loc,
                    extra_page,
                    pool.get_extra_key_layout(layer_id),
                    extra_idx,
                    core.sparse_topk_lengths(ratio),
                ),
            ):
                decoded = (
                    dequantize_k_cache_paged(raw, loc, page, layout=kind)
                    .squeeze(1)
                    .float()
                )
                k = decoded[indices.clamp_min(0).long()]
                valid = (indices >= 0) & (
                    torch.arange(indices.shape[1], device="cuda")[None, :]
                    < lens[:, None]
                )
                score = torch.einsum("nhd,nkd->nhk", q.float(), k) * 512**-0.5
                scores.append(score.masked_fill(~valid[:, None, :], -torch.inf))
                keys.append(k)
            logits = torch.cat([*scores, sink[None, :, None].expand(2, -1, -1)], -1)
            expected = torch.einsum(
                "nhk,nkd->nhd", logits.softmax(-1)[..., :-1], torch.cat(keys, 1)
            ).bfloat16()
            torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.03)

    def test_single_cache_large_offsets(self):
        from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache
        from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
            dequantize_k_cache_paged,
        )
        from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.hip_flash_mla import aiter_sparse_decode_fwd

        layout, page = KVLayout.V41, 16
        page_bytes = layout.page_bytes(page)
        for pages in (4, 2**31 // page_bytes + 4):
            raw = torch.empty(pages, page_bytes, device="cuda", dtype=torch.uint8)
            loc = torch.arange(
                (pages - 4) * page, pages * page, device="cuda", dtype=torch.int32
            )
            x = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
            fused_store_cache(
                x, raw, loc, page_size=page, type="flashmla", layout=layout
            )
            cache = raw[:, : page * layout.bytes_per_token].view(
                pages, page, 1, layout.bytes_per_token
            )
            q = torch.randn(3, 1, 16, 512, device="cuda", dtype=torch.bfloat16) * 0.25
            indices = loc[None, None, :].expand(3, 1, -1).contiguous()
            lengths = torch.full((3,), 64, device="cuda", dtype=torch.int32)
            decoded = (
                dequantize_k_cache_paged(raw, loc, page, layout=layout)
                .squeeze(1)
                .float()
            )
            logits = torch.einsum("nhd,kd->nhk", q[:, 0].float(), decoded) * 512**-0.5
            expected = torch.einsum(
                "nhk,kd->nhd", logits.softmax(-1), decoded
            ).bfloat16()
            for splits in (1, 4):
                with envs.SGLANG_OPT_HIP_ATTN_KV_SPLITS.override(splits):
                    actual, _ = aiter_sparse_decode_fwd(
                        q, cache, indices, None, 512**-0.5, topk_length=lengths
                    )
                torch.testing.assert_close(
                    actual[:, 0], expected, atol=0.003, rtol=0.03
                )
            del raw, cache

    def test_two_cache_graph_replay(self):
        from sglang.kernels.ops.attention.dsv4.attn import fused_store_cache
        from sglang.kernels.ops.attention.dsv4.compact_attention_hip import (
            compact_attention_hip,
        )
        from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
            dequantize_k_cache_paged,
        )
        from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout

        torch.manual_seed(32)
        for n, h, extra_layout in (
            (9, 17, KVLayout.V41),
            (9, 17, KVLayout.V41_FP4),
            (1025, 16, KVLayout.V41_FP4),
        ):
            pools = []
            for layout, page in ((KVLayout.V41, 16), (extra_layout, 32)):
                raw = torch.zeros(
                    8, layout.page_bytes(page), device="cuda", dtype=torch.uint8
                )
                x = torch.randn(8 * page, 512, device="cuda", dtype=torch.bfloat16)
                loc = torch.arange(len(x), device="cuda", dtype=torch.int32)
                cache = raw[:, : page * layout.bytes_per_token].view(
                    8, page, 1, layout.bytes_per_token
                )
                pools.append((layout, page, raw, x, loc, cache))
            q = torch.randn(n, h + 3, 512, device="cuda", dtype=torch.bfloat16)
            q.mul_(0.25)
            q = q[:, :h]
            self.assertFalse(q.is_contiguous())
            angles = torch.randn(n + 7, 32, device="cuda")
            freqs = torch.stack((angles.cos(), angles.sin()), dim=-1).flatten(1)
            positions = torch.arange(n, device="cuda") + 7
            indices = torch.randint(128, (n, 96), device="cuda", dtype=torch.int32)
            extra = torch.randint(256, (n, 192), device="cuda", dtype=torch.int32)
            lengths = torch.randint(0, 97, (n,), device="cuda", dtype=torch.int32)
            extra_lengths = torch.randint(
                0, 193, (n,), device="cuda", dtype=torch.int32
            )
            sink = torch.randn(h, device="cuda")

            def run():
                for layout, page, raw, x, loc, _ in pools:
                    fused_store_cache(
                        x, raw, loc, page_size=page, type="flashmla", layout=layout
                    )
                return compact_attention_hip(
                    q,
                    pools[0][-1],
                    indices,
                    lengths,
                    sink,
                    extra_cache=pools[1][-1],
                    extra_indices=extra,
                    extra_lengths=extra_lengths,
                    splits=1 if n >= 1024 else 4,
                    inv_rope=(freqs, positions),
                )

            for _ in range(2):
                run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = run()
            for replay in range(3):
                q.normal_(std=0.25)
                for _, _, _, x, _, _ in pools:
                    x.normal_()
                indices.random_(-1, 128)
                extra.random_(-1, 256)
                lengths.random_(0, 97)
                extra_lengths.random_(0, 193)
                lengths[0] = extra_lengths[0] = 0
                graph.replay()
                scores = []
                keys = []
                for (layout, page, raw, x, loc, _), idx, lens in zip(
                    pools, (indices, extra), (lengths, extra_lengths)
                ):
                    decoded = (
                        dequantize_k_cache_paged(raw, loc, page, layout=layout)
                        .squeeze(1)
                        .float()
                    )
                    k = decoded[idx.clamp_min(0).long()]
                    valid = (idx >= 0) & (
                        torch.arange(idx.shape[1], device="cuda")[None, :]
                        < lens[:, None]
                    )
                    score = torch.einsum("nhd,nkd->nhk", q.float(), k) * 512**-0.5
                    score.masked_fill_(~valid[:, None, :], -torch.inf)
                    scores.append(score)
                    keys.append(k)
                logits = torch.cat([*scores, sink[None, :, None].expand(n, -1, -1)], -1)
                ref = torch.einsum(
                    "nhk,nkd->nhd", logits.softmax(-1)[..., :-1], torch.cat(keys, 1)
                ).bfloat16()
                tail = ref[..., -64:].float().reshape(n, h, 32, 2)
                cos = freqs[positions, 0::2][:, None, :]
                sin = freqs[positions, 1::2][:, None, :]
                rotated = torch.stack(
                    (
                        tail[..., 0] * cos + tail[..., 1] * sin,
                        tail[..., 1] * cos - tail[..., 0] * sin,
                    ),
                    dim=-1,
                )
                ref[..., -64:] = rotated.flatten(-2).bfloat16()
                torch.testing.assert_close(out, ref, atol=0.004, rtol=0.03)
                self.assertTrue((out[0] == 0).all())


if __name__ == "__main__":
    unittest.main()
