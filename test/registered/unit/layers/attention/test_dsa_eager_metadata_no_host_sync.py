import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import dsa_backend as module
from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDsaEagerMetadataNoHostSync(unittest.TestCase):
    def check_metadata(self, mode, host_mirror):
        table = torch.arange(2 * 32, dtype=torch.int32).reshape(2, 32)
        seq_lens = torch.tensor([5, 7])
        prefix = torch.tensor([2, 4])
        batch = SimpleNamespace(
            batch_size=2,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.clone() if host_mirror else None,
            req_pool_indices=torch.tensor([0, 1]),
            forward_mode=mode,
            extend_seq_lens_cpu=[3, 3],
            extend_seq_lens=torch.tensor([3, 3]),
            extend_prefix_lens_cpu=[2, 4] if host_mirror else None,
            extend_prefix_lens=prefix,
            extend_num_tokens=6,
            seq_lens_sum=12 if host_mirror else 0,
        )
        backend = SimpleNamespace(
            speculative_num_draft_tokens=3,
            req_to_token=table,
            req_to_token_pool=SimpleNamespace(req_to_token=table),
            _kv_shard_pool=None,
            set_dsa_prefill_impl=lambda _: None,
            dsa_decode_impl="triton",
            use_mha=False,
            get_topk_transform_method=lambda _: TopkTransformMethod.PAGED,
            dsa_index_kpool=1,
            dsa_index_topk=4,
            physical_page_size=1,
            get_device_int32_arange=lambda n: torch.arange(n, dtype=torch.int32),
            _cal_indexer_k_start_end=lambda *args: (None, None),
            _transform_table_1_to_real=lambda table: table,
            _build_topk_v2_plan=lambda _: None,
            _init_kpool_metadata=lambda metadata, *args, **kwargs: metadata,
        )

        def cumulative(lengths):
            return torch.cat(
                (torch.zeros(1, dtype=torch.int32), lengths.cumsum(0).to(torch.int32))
            )

        with (
            patch.object(
                seq_lens, "max", side_effect=AssertionError("device max/item sync")
            ),
            patch.object(
                seq_lens, "cpu", side_effect=AssertionError("device seq_lens copy")
            ),
            patch.object(
                prefix, "cpu", side_effect=AssertionError("device prefix copy")
            ),
            patch.object(module, "compute_cu_seqlens", side_effect=cumulative),
            patch.object(
                module,
                "seqlens_expand_triton",
                side_effect=lambda extend, lens, *args: lens.repeat_interleave(extend),
            ),
            patch.object(
                module,
                "compute_dsa_seqlens",
                side_effect=lambda original_seq_lens, **kwargs: original_seq_lens.clamp(
                    max=4
                ),
            ),
            patch.object(
                module, "pad_dsa_cache_seqlens", side_effect=lambda batch, lens: lens
            ),
            patch.object(module, "is_cuda", return_value=False),
        ):
            module.DeepseekSparseAttnBackend.init_forward_metadata(backend, batch)
        width = 7 if host_mirror else 32
        repeats = 3 if mode.is_draft_extend_v2() else 1
        self.assertEqual(backend.forward_metadata.max_seq_len_k, width)
        torch.testing.assert_close(
            backend.forward_metadata.page_table_1,
            table[:, :width].repeat_interleave(repeats, dim=0),
        )
        if not host_mirror:
            self.assertIsNone(batch.seq_lens_cpu)
            self.assertIsNone(batch.extend_prefix_lens_cpu)
            self.assertEqual(batch.seq_lens_sum, 0)

    def test_decode_without_host_mirror(self):
        self.check_metadata(ForwardMode.DECODE, False)

    def test_draft_extend_without_host_mirror(self):
        self.check_metadata(ForwardMode.DRAFT_EXTEND_V2, False)

    def test_existing_host_mirror_keeps_narrow_table(self):
        for mode in (ForwardMode.DECODE, ForwardMode.DRAFT_EXTEND_V2):
            with self.subTest(mode=mode):
                self.check_metadata(mode, True)


if __name__ == "__main__":
    unittest.main()
