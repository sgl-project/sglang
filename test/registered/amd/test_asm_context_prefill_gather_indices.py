"""Per-batch gather indices for the ROCm ASM context-chunk prefill.

For a plain extend batch the prefill indices updater lays kv_indices out token by
token with kv_indptr = cumsum(seq_lens), so the ASM context prefill can take the
first sum(seq_lens) entries and kv_indptr itself without any device sync. Guards:
the fast path equals the generic (syncing) gather, is computed once per batch,
and batches that break the invariant fall back to the generic gather.
"""

import types
import unittest
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")


@unittest.skipUnless(torch.version.hip, "ROCm attention backend")
class TestAsmContextPrefillGatherIndices(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.attention import aiter_backend as ab

        cls.ab = ab
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _batch(self, seq_lens, pad=256, spec_info=None, kv_indptr=None):
        seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int32)
        seq_lens_dev = seq_lens_cpu.to(self.device)
        bs = len(seq_lens)
        if kv_indptr is None:
            kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=self.device)
            kv_indptr[1:] = torch.cumsum(seq_lens_dev, 0)
        total = int(kv_indptr[bs].item())
        # pool slots in a shuffled order, padded the way the updater pads
        pool = torch.randperm(total + pad, device=self.device)[:total].to(torch.int32)
        kv_indices = torch.cat([pool, pool[:1].expand(pad)])
        fm = self.ab.ForwardMetadata(
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            qo_indptr=None,
            kv_last_page_len=None,
            max_q_len=1,
            max_kv_len=max(seq_lens),
        )
        backend = types.SimpleNamespace(forward_metadata=fm)
        fb = types.SimpleNamespace(
            seq_lens=seq_lens_dev,
            seq_lens_cpu=seq_lens_cpu,
            spec_info=spec_info,
            forward_mode=types.SimpleNamespace(is_extend=lambda: True),
        )
        return backend, fb, bs, total + pad

    def test_fast_path_matches_generic_gather(self):
        torch.manual_seed(0)
        for seq_lens in ([5], [7, 3, 12], [1, 1, 1, 1], [1024, 17, 300]):
            with self.subTest(seq_lens=seq_lens):
                backend, fb, bs, num_slots = self._batch(seq_lens)
                fm = backend.forward_metadata
                tok_idx, cu_k = self.ab.AiterAttnBackend._asm_context_prefill_indices(
                    backend, fb, bs, num_slots
                )
                ref_tok, ref_cu = self.ab._asm_context_prefill_gather_indices(
                    fm.kv_indptr[: bs + 1], fm.kv_indices, fb.seq_lens, num_slots
                )
                self.assertTrue(torch.equal(tok_idx.to(torch.long), ref_tok))
                self.assertTrue(torch.equal(cu_k, ref_cu.to(torch.int32)))
                self.assertEqual(cu_k.dtype, torch.int32)
                self.assertEqual(tok_idx.numel(), sum(seq_lens))
                # cached: the second call for the same batch returns the same tensors
                again = self.ab.AiterAttnBackend._asm_context_prefill_indices(
                    backend, fb, bs, num_slots
                )
                self.assertIs(again[0], tok_idx)
                self.assertIs(again[1], cu_k)

    def test_spec_batch_uses_generic_gather(self):
        backend, fb, bs, num_slots = self._batch([9, 4], spec_info=object())
        fm = backend.forward_metadata
        tok_idx, cu_k = self.ab.AiterAttnBackend._asm_context_prefill_indices(
            backend, fb, bs, num_slots
        )
        ref_tok, ref_cu = self.ab._asm_context_prefill_gather_indices(
            fm.kv_indptr[: bs + 1], fm.kv_indices, fb.seq_lens, num_slots
        )
        self.assertTrue(torch.equal(tok_idx.to(torch.long), ref_tok))
        self.assertTrue(torch.equal(cu_k, ref_cu.to(torch.int32)))

    def test_short_table_falls_back(self):
        # Host lengths exceed the table; the generic helper clamps them.
        backend, fb, bs, num_slots = self._batch([6, 6], pad=0)
        fb.seq_lens_cpu = torch.tensor([6, 100], dtype=torch.int32)
        fb.seq_lens = fb.seq_lens_cpu.to(self.device)
        tok_idx, cu_k = self.ab.AiterAttnBackend._asm_context_prefill_indices(
            backend, fb, bs, num_slots
        )
        # the generic gather clamps the lengths to the table, so it still returns
        self.assertIsNotNone(tok_idx)
        self.assertEqual(tok_idx.numel(), 12)

    def test_missing_host_lengths_uses_and_caches_generic_gather(self):
        for speculative in (False, True):
            with self.subTest(speculative=speculative):
                backend, fb, bs, num_slots = self._batch(
                    [9, 4], spec_info=object() if speculative else None
                )
                fm = backend.forward_metadata
                fb.seq_lens_cpu = None
                ref_tok, ref_cu = self.ab._asm_context_prefill_gather_indices(
                    fm.kv_indptr[: bs + 1], fm.kv_indices, fb.seq_lens, num_slots
                )
                with patch.object(
                    self.ab,
                    "_asm_context_prefill_gather_indices",
                    wraps=self.ab._asm_context_prefill_gather_indices,
                ) as generic:
                    tok_idx, cu_k = (
                        self.ab.AiterAttnBackend._asm_context_prefill_indices(
                            backend, fb, bs, num_slots
                        )
                    )
                    again = self.ab.AiterAttnBackend._asm_context_prefill_indices(
                        backend, fb, bs, num_slots
                    )
                self.assertEqual(generic.call_count, 1)
                self.assertTrue(torch.equal(tok_idx.to(torch.long), ref_tok))
                self.assertTrue(torch.equal(cu_k, ref_cu.to(torch.int32)))
                self.assertIs(again[0], tok_idx)
                self.assertIs(again[1], cu_k)


if __name__ == "__main__":
    unittest.main()
