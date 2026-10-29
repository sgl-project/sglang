import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from sglang.srt.disaggregation.ascend.conn import AscendKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestAscendPageDcpSend(CustomTestCase):
    def test_rejects_incorrect_destination_geometry(self):
        mgr = object.__new__(AscendKVManager)
        mgr.is_hybrid_mla_backend = False
        mgr.kv_args = SimpleNamespace(
            kv_buf_groups=3,
            kv_data_ptrs=[1_000, 2_000, 3_000],
            kv_item_lens=[20, 8, 4],
            page_size=2,
            mla_compression_ratios=None,
        )
        mgr._dcp_remote_decode_layout = [False, True, True]
        mgr._transfer_data = Mock(return_value=0)

        with self.assertRaisesRegex(RuntimeError, "destination geometry"):
            mgr.send_kvcache_dcp(
                "decode",
                np.array([5], dtype=np.int32),
                [10_000, 20_000, 30_000],
                np.array([7], dtype=np.int32),
                dcp_token_item_lens=[10, 4, 2],
                dst_dcp_size=4,
                dst_dcp_rank=0,
                src_page_offset=0,
                decode_prefix_len=0,
                num_kv_tokens=2,
                executor=None,
                dst_layer_ids=[],
                dst_kv_item_lens=[20, 8, 16],
            )

    def _hybrid_case(self, stage_ids, with_draft):
        model_ids = [3, 7, 11, 15]
        widths = [20] * len(stage_ids) + [8] * len(stage_ids)
        src_ids = stage_ids * 2
        dst_ids = model_ids * 2
        dst_widths = [20] * len(model_ids) + [8] * len(model_ids)
        if with_draft:
            src_ids += [16, 16]
            dst_ids += [16, 16]
            widths += [20, 8]
            dst_widths += [40, 16]
        src_ptrs = [10_000 * (i + 1) for i in range(len(src_ids))]
        dst_ptrs = [100_000 * (i + 1) for i in range(len(dst_ids))]
        mgr = object.__new__(AscendKVManager)
        mgr.is_hybrid_mla_backend = True
        # Hybrid pairing must not need the legacy NPU group/layer counts.
        mgr.kv_args = SimpleNamespace(
            kv_data_ptrs=src_ptrs,
            kv_layer_ids=src_ids,
            kv_item_lens=widths,
            page_size=2,
        )
        mgr._dcp_remote_decode_layout = [False] * (2 * len(stage_ids)) + (
            [True, True] if with_draft else []
        )
        mgr.get_mla_kv_ptrs_with_pp = Mock(side_effect=AssertionError("legacy pairing"))
        mgr._transfer_data = Mock(return_value=0)
        kwargs = dict(
            mooncake_session_id="decode",
            prefill_kv_indices=np.array([5, 6, 7], dtype=np.int32),
            dst_kv_ptrs=dst_ptrs,
            dst_kv_indices=np.array([10, 11], dtype=np.int32),
            dcp_token_item_lens=[width // 2 for width in widths],
            dst_dcp_size=2,
            dst_dcp_rank=1,
            src_page_offset=0,
            decode_prefix_len=0,
            num_kv_tokens=6,
            executor=None,
            dst_layer_ids=dst_ids,
            dst_kv_item_lens=dst_widths,
        )
        expected = []
        for group, width in enumerate((20, 8)):
            for i, layer_id in enumerate(stage_ids):
                src = src_ptrs[group * len(stage_ids) + i]
                dst = dst_ptrs[group * len(model_ids) + model_ids.index(layer_id)]
                # Rank 1 owns source page 6, stored at local decode page 10.
                expected.append((src + 6 * width, dst + 10 * width, width))
        if with_draft:
            for i, width in enumerate((20, 8)):
                # Replicated draft keeps all three pages at global slots 20..22.
                expected.append(
                    (
                        src_ptrs[-2 + i] + 5 * width,
                        dst_ptrs[-2 + i] + 20 * width,
                        3 * width,
                    )
                )
        return mgr, kwargs, expected

    def test_hybrid_pairs_pages_by_layer_without_group_counts(self):
        for stage_ids in ([3, 7, 11, 15], [3, 7], [11, 15], [7, 11, 15], []):
            for with_draft in (False, True):
                for with_widths in (False, True):
                    with self.subTest(
                        stage_ids=stage_ids, draft=with_draft, widths=with_widths
                    ):
                        mgr, kwargs, expected = self._hybrid_case(stage_ids, with_draft)
                        if not with_widths:
                            kwargs.pop("dst_kv_item_lens")
                        self.assertEqual(mgr.send_kvcache_dcp(**kwargs), 0)
                        mgr._transfer_data.assert_called_once_with("decode", expected)
                        mgr.get_mla_kv_ptrs_with_pp.assert_not_called()

    def test_hybrid_rejects_incomplete_layer_metadata_before_transfer(self):
        for missing_metadata in (False, True):
            with self.subTest(missing_metadata=missing_metadata):
                mgr, kwargs, _ = self._hybrid_case([11, 15], True)
                if missing_metadata:
                    kwargs["dst_layer_ids"] = []
                else:
                    # Keep latent layer 11 but remove its RoPE entry.
                    for name in ("dst_layer_ids", "dst_kv_ptrs", "dst_kv_item_lens"):
                        kwargs[name].pop(6)
                with self.assertRaises(RuntimeError):
                    mgr.send_kvcache_dcp(**kwargs)
                mgr._transfer_data.assert_not_called()


if __name__ == "__main__":
    unittest.main()
