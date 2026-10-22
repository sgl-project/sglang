"""FP8 MLA conversions and changing CUDA graph replay inputs, without a model."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.kvcache.mla_kv_gather import (
    gather_mla_kv_rows,
    refresh_mla_kv_shadow,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestMLAKVGather(CustomTestCase):
    def test_gather_and_shadow_match_full_pool_conversion(self):
        for src_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            for dst_dtype in (torch.bfloat16, torch.float16):
                for dim in (63, 576):
                    with self.subTest(src=src_dtype, dst=dst_dtype, dim=dim):
                        # All FP8 byte patterns, including subnormals and NaNs.
                        pool = (
                            torch.arange(257 * dim, device="cuda")
                            .remainder(256)
                            .to(torch.uint8)
                            .view(src_dtype)
                            .view(257, 1, dim)
                        )
                        indices = torch.tensor(
                            [0, 256, 7, 7] + list(range(15)) + [-1] * 13,
                            dtype=torch.int32,
                            device="cuda",
                        )
                        full = pool.to(dst_dtype)
                        for count in (0, 1, 19):
                            live = torch.tensor(
                                [count], dtype=torch.int32, device="cuda"
                            )
                            shadow = torch.full_like(full, -123)
                            refresh_mla_kv_shadow(
                                pool,
                                shadow,
                                indices,
                                live,
                                num_programs=2,
                                rows_per_program=2,
                            )
                            rows = indices[:count].long()
                            expected = torch.full_like(full, -123)
                            expected[rows] = full[rows]
                            torch.testing.assert_close(
                                shadow, expected, rtol=0, atol=0, equal_nan=True
                            )
                            compact = gather_mla_kv_rows(
                                pool, indices[:count], dst_dtype
                            )
                            torch.testing.assert_close(
                                compact, full[rows], rtol=0, atol=0, equal_nan=True
                            )

    def test_graph_replay_refreshes_new_rows_and_reuses_shadow_across_layers(self):
        from sglang.srt.layers.attention.flashinfer_mla_backend import _Fp8KvRowReader

        for page_size in (1, 64):
            with self.subTest(page_size=page_size):
                layers = [
                    torch.zeros((257, 1, 576), device="cuda", dtype=torch.float8_e4m3fn)
                    for _ in range(2)
                ]
                pool = SimpleNamespace(
                    dtype=torch.float8_e4m3fn,
                    size=257 - page_size,
                    page_size=page_size,
                    get_key_buffer=lambda layer_id: layers[layer_id],
                )
                reader = _Fp8KvRowReader(pool)
                indices = torch.zeros(32, dtype=torch.int32, device="cuda")
                live = torch.zeros(1, dtype=torch.int32, device="cuda")
                outputs = [
                    torch.empty_like(layer, dtype=torch.bfloat16) for layer in layers
                ]

                def step():
                    for layer_id in range(2):
                        outputs[layer_id].copy_(
                            reader.refresh(layer_id, indices, live, torch.bfloat16)
                        )

                # Warm the Triton specialization and allocate the persistent
                # shadow before capture. Every captured layer uses this buffer.
                step()
                shadow_ptr = reader._shadow.data_ptr()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    step()

                for iteration, rows in enumerate(
                    ([0, 256, 7, 7], [19], [], list(range(23)))
                ):
                    for layer_id, layer in enumerate(layers):
                        values = torch.arange(layer.numel(), device="cuda").remainder(
                            31
                        )
                        layer.copy_(
                            (values + iteration + layer_id)
                            .view(layer.shape)
                            .to(layer.dtype)
                        )
                    # Unused capacity deliberately contains invalid physical ids.
                    indices.fill_(-1)
                    indices[: len(rows)] = torch.tensor(
                        rows, dtype=torch.int32, device="cuda"
                    )
                    live.fill_(len(rows))
                    reader._shadow.fill_(-123)
                    graph.replay()
                    self.assertEqual(reader._shadow.data_ptr(), shadow_ptr)
                    for layer_id, layer in enumerate(layers):
                        expected = torch.full_like(outputs[layer_id], -123)
                        expected[rows] = layer.to(torch.bfloat16)[rows]
                        torch.testing.assert_close(
                            outputs[layer_id], expected, rtol=0, atol=0
                        )


if __name__ == "__main__":
    unittest.main()
