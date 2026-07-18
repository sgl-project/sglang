import sys
import types
import unittest
from unittest.mock import patch

import torch

from sglang.jit_kernel.dsa.paged_mqa_logits import aiter_paged_mqa_logits
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestAiterPagedMqaLogits(unittest.TestCase):
    @staticmethod
    def _aiter_modules(kernel):
        module = types.ModuleType("aiter.ops.triton.pa_mqa_logits")
        module.deepgemm_fp8_paged_mqa_logits = kernel
        return {
            "aiter": types.ModuleType("aiter"),
            "aiter.ops": types.ModuleType("aiter.ops"),
            "aiter.ops.triton": types.ModuleType("aiter.ops.triton"),
            "aiter.ops.triton.pa_mqa_logits": module,
        }

    def test_ignores_dp_padding_rows(self):
        observed = {}

        def kernel(q, kv, weights, logits, seq_lens, block_tables, max_seq_len, **_):
            observed["q_shape"] = tuple(q.shape)
            observed["weights_shape"] = tuple(weights.shape)
            observed["logits_shape"] = tuple(logits.shape)

        modules = self._aiter_modules(kernel)

        q = torch.empty((24, 2, 132))
        kv = torch.empty((32, 1, 1, 132))
        weights = torch.empty((24, 2))
        seq_lens = torch.ones(18, dtype=torch.int32)
        block_tables = torch.zeros((18, 4), dtype=torch.int32)

        with patch.dict(sys.modules, modules):
            logits = aiter_paged_mqa_logits(
                q,
                kv,
                weights,
                seq_lens,
                block_tables,
                4,
                q_offset=18,
                preshuffle=False,
                kv_block_size=1,
            )

        self.assertEqual(observed["q_shape"], (18, 1, 2, 132))
        self.assertEqual(observed["weights_shape"], (18, 2))
        self.assertEqual(observed["logits_shape"], (18, 4))
        self.assertEqual(tuple(logits.shape), (18, 4))

    def test_rejects_metadata_rows_that_do_not_match_q_offset(self):
        q = torch.empty((24, 2, 132))
        kv = torch.empty((32, 1, 1, 132))
        weights = torch.empty((24, 2))
        seq_lens = torch.ones(17, dtype=torch.int32)
        block_tables = torch.zeros((18, 4), dtype=torch.int32)

        with patch.dict(sys.modules, self._aiter_modules(lambda *args, **kwargs: None)):
            with self.assertRaisesRegex(ValueError, "metadata rows must match"):
                aiter_paged_mqa_logits(
                    q,
                    kv,
                    weights,
                    seq_lens,
                    block_tables,
                    4,
                    q_offset=18,
                    preshuffle=False,
                    kv_block_size=1,
                )


if __name__ == "__main__":
    unittest.main()
