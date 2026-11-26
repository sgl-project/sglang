import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import BlockIds
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.models import deepseek_v4
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDeepSeekV4PP(unittest.TestCase):
    def test_incremental_source_pages(self):
        mapping = torch.arange(4, 36).reshape(2, 16)
        with (
            patch.object(
                deepseek_v4,
                "get_req_to_token_pool",
                return_value=NS(req_to_token=mapping),
            ),
            patch.object(
                deepseek_v4, "get_token_to_kv_pool", return_value=NS(page_size=4)
            ),
        ):
            for prefix, length, expected in (
                (0, 5, [5, 6]),
                (4, 9, [6, 7]),
                (5, 9, [6, 7]),
                (9, 9, []),
                (0, 1, [5]),
            ):
                with self.subTest(prefix=prefix, length=length):
                    batch = NS(
                        req_pool_indices=torch.tensor([1, 0]),
                        seq_lens_cpu=[length, length],
                        extend_prefix_lens_cpu=[prefix, prefix],
                    )
                    for offset in (0, 40):
                        mapping[1] = torch.arange(20, 36) + offset
                        pages = deepseek_v4.DeepseekV4Model._pp_source_page_ids(
                            None, batch
                        )
                        torch.testing.assert_close(
                            pages,
                            torch.tensor(
                                [page + offset // 4 for page in expected]
                                + [page - 4 for page in expected],
                                dtype=torch.int64,
                            ),
                        )
            mapping.resize_(1, 131072).copy_(torch.arange(131072).reshape(1, -1))
            batch.req_pool_indices = torch.tensor([0])
            total = 0
            for prefix in range(0, 131072, 512):
                batch.extend_prefix_lens_cpu, batch.seq_lens_cpu = (
                    [prefix],
                    [prefix + 512],
                )
                total += deepseek_v4.DeepseekV4Model._pp_source_page_ids(
                    None, batch
                ).numel()
            self.assertEqual(total, 131072 // 4)

    def test_handoff_remaps_logical_picks(self):
        raw = torch.tensor([[0, 3, -1], [1, 2, -1]], dtype=torch.int32)
        core = NS(
            c1_sparse_raw_indices=raw,
            c2_sparse_raw_indices=None,
            c1_sparse_topk_lengths=torch.tensor([2, 2]),
            page_table=torch.tensor([[7], [9]], dtype=torch.int32),
            c1_sparse_page_indices=torch.zeros_like(raw),
        )
        core.sparse_raw_indices = lambda ratio: getattr(
            core, f"c{ratio}_sparse_raw_indices"
        )
        core.sparse_page_indices = lambda ratio: getattr(
            core, f"c{ratio}_sparse_page_indices"
        )
        blocks = torch.tensor([[0], [0]], dtype=torch.int32)
        metadata = NS(core_metadata=core, candidate_metadata=BlockIds(blocks, [1, 1]))
        installed = []
        pool = NS(
            page_size=4,
            export_source_pages=lambda source, pages: {"kv": pages.clone()},
            install_source_pages=lambda *args: installed.append(args),
        )
        model = NS(
            end_layer=25,
            config=NS(
                compress_ratios=[0] * 20 + [1] * 20,
                kv_source_layer_ids=[20],
                index_source_layer_ids=[20, 24, 28],
                candidate_source_layer_id=20,
            ),
            _pp_source_page_ids=lambda batch: torch.tensor([1, 2]),
        )
        tensors, batch = {}, NS(extend_seq_lens_cpu=[1, 1])
        with (
            patch.object(
                deepseek_v4,
                "get_attn_backend",
                return_value=NS(forward_metadata=metadata),
            ),
            patch.object(deepseek_v4, "get_token_to_kv_pool", return_value=pool),
        ):
            deepseek_v4.DeepseekV4Model._export_pp_state(model, tensors, batch)
            model._pp_source_page_ids = lambda batch: torch.tensor([11, 12])
            core.c1_sparse_raw_indices, metadata.candidate_metadata = (
                torch.zeros_like(raw),
                None,
            )
            deepseek_v4.DeepseekV4Model._install_pp_state(
                model, PPProxyTensors(tensors), batch
            )
        torch.testing.assert_close(core.c1_sparse_raw_indices, raw)
        torch.testing.assert_close(
            core.c1_sparse_page_indices,
            torch.tensor([[28, 31, -1], [37, 38, -1]], dtype=torch.int32),
        )
        torch.testing.assert_close(metadata.candidate_metadata.blocks, blocks)
        self.assertEqual(installed[0][0], 20)
        torch.testing.assert_close(installed[0][1], torch.tensor([11, 12]))
        torch.testing.assert_close(installed[0][2]["kv"], torch.tensor([1, 2]))


if __name__ == "__main__":
    unittest.main()
