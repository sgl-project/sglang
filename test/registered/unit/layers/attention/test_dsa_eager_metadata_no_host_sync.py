import sys
import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.kernels.ops.attention.dsa.fp4_index_hip import (
    Fp4IndexKeys,
    Fp4IndexQuery,
    fp4_ragged_mqa_logits,
)
from sglang.srt.layers.attention import dsa_backend as module
from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
from sglang.srt.model_executor.forward_mode import ForwardMode
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
            dsa_shards_index_k=False,
            dsa_index_topk=4,
            physical_page_size=1,
            get_device_int32_arange=lambda n: torch.arange(n, dtype=torch.int32),
            _cal_indexer_k_start_end=lambda *args: (None, None, None),
            _transform_table_1_to_real=lambda table: table,
            _build_topk_v2_plan=lambda _: None,
            _dcp_topk_length_buffer=lambda *args: None,
            init_kernel_metadata=lambda _: None,
            _init_kpool_metadata=lambda metadata, *args, **kwargs: metadata,
            # A static pool: no id translation, so the gather reads the table.
            runner_ids=None,
        )
        for name in ("_read_v2p", "_gather_token_table"):
            method = getattr(module.DeepseekSparseAttnBackend, name)
            setattr(backend, name, MethodType(method, backend))

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

    def test_fp4_prefill_logits_split_requests_from_host_ks(self):
        # Three requests at k_start 0, 100 and 300: rows [0, 4), [4, 7), [7, 10)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            batch_size=3,
            seq_lens_cpu=torch.tensor([100, 200, 70]),
            extend_seq_lens_cpu=[4, 3, 3],
        )
        backend = SimpleNamespace(device="cpu", speculative_num_draft_tokens=0)
        (ks, ke), _, ks_cpu = module.DeepseekSparseAttnBackend._cal_indexer_k_start_end(
            backend, batch
        )
        self.assertTrue(torch.equal(ks_cpu, ks))

        keys = Fp4IndexKeys(torch.zeros(6, 4096), torch.zeros(6, 256), 370)
        query = Fp4IndexQuery(torch.zeros(10, 16, 64), torch.zeros(10, 1, 4, 16, 4))
        aiter = MagicMock()
        aiter.compute_prefill_schedule.return_value = (None, None, 0)
        aiter_module = "aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill"
        with (
            patch.dict(sys.modules, {aiter_module: aiter}),
            patch.object(ks, "cpu", side_effect=AssertionError("device ks copy")),
        ):
            fp4_ragged_mqa_logits(
                query, torch.ones(10, 16), 1.0, keys, ks, ke, ks_cpu=ks_cpu
            )
        calls = aiter.flydsl_pa_mqa_logits_fp4_prefill.call_args_list
        # (rows, table width) of each kernel launch
        self.assertEqual(
            [(len(c.args[0]), c.args[9]) for c in calls],
            [(4, 370), (3, 306), (3, 114)],
        )


if __name__ == "__main__":
    unittest.main()
