import dataclasses
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.sampling.sampler_params import SamplerParams, SamplerParamsPool
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NUM_ROWS = 8
DEVICE = "cpu"


def _req(slot, temperature=1.0, top_p=1.0, top_k=TOP_K_ALL, min_p=0.0):
    return SimpleNamespace(
        kv=SimpleNamespace(req_pool_idx=slot),
        sampling_params=SimpleNamespace(
            temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p
        ),
    )


def _gather(pool, slots):
    indices = torch.tensor(slots, dtype=torch.int64)
    return pool.gather(indices, indices)


class TestSamplerParamsPool(CustomTestCase):
    def test_gather_reads_each_slot_in_batch_order(self):
        pool = SamplerParamsPool(num_rows=NUM_ROWS, device=DEVICE)
        pool.write(
            [
                _req(5, temperature=0.7, top_p=0.9, top_k=20, min_p=0.05),
                _req(2, temperature=1.3, top_p=0.5, top_k=1, min_p=0.0),
            ]
        )

        params = _gather(pool, [2, 5])

        self.assertEqual(params.temperatures.shape, (2, 1))
        self.assertEqual(params.top_ks.dtype, torch.int32)
        torch.testing.assert_close(
            params.temperatures,
            torch.tensor([[1.3], [0.7]], dtype=torch.float32),
            rtol=0,
            atol=0,
        )
        self.assertEqual(params.top_ps.tolist(), torch.tensor([0.5, 0.9]).tolist())
        self.assertEqual(params.top_ks.tolist(), [1, 20])
        self.assertEqual(params.min_ps.tolist(), torch.tensor([0.0, 0.05]).tolist())
        # The samplers read each field as a contiguous per-row tensor.
        for field in (params.temperatures, params.top_ps, params.top_ks, params.min_ps):
            self.assertTrue(field.is_contiguous())

    def test_unchanged_batch_reuses_its_params_until_a_write(self):
        pool = SamplerParamsPool(num_rows=NUM_ROWS, device=DEVICE)
        pool.write([_req(1, temperature=0.7), _req(4, top_k=1)])

        first = _gather(pool, [1, 4])
        self.assertIs(_gather(pool, [1, 4]), first)
        self.assertIsNot(_gather(pool, [4, 1]), first)

        after_slot_change = _gather(pool, [1, 4])
        self.assertIsNot(after_slot_change, first)
        pool.write([_req(4, temperature=1.2)])
        params = _gather(pool, [1, 4])
        self.assertIsNot(params, after_slot_change)
        self.assertAlmostEqual(params.temperatures[1].item(), 1.2, places=6)
        self.assertFalse(params.is_any_greedy)

    def test_rewritten_slot_reads_the_new_request(self):
        """A freed slot that a later admission reuses holds the new request's
        params, host flags included."""
        pool = SamplerParamsPool(num_rows=NUM_ROWS, device=DEVICE)
        pool.write([_req(3, temperature=0.5, top_p=0.8, top_k=1, min_p=0.2)])
        pool.write([_req(3, temperature=1.1)])

        params = _gather(pool, [3])

        self.assertAlmostEqual(params.temperatures.item(), 1.1, places=6)
        self.assertEqual(params.top_ks.tolist(), [TOP_K_ALL])
        self.assertFalse(params.need_top_p_sampling)
        self.assertFalse(params.need_min_p_sampling)
        self.assertFalse(params.is_any_greedy)

    def test_flags_follow_the_gathered_rows(self):
        # (top_ks, is_all_greedy, is_any_greedy, need_top_k, npu eligible, max_top_k)
        cases = (
            ((), True, False, False, True, None),
            ((1,), True, True, True, True, 1),
            ((1, 1024), False, True, True, True, 1024),
            ((1, 1025), False, True, True, False, 1025),
            ((20, 50), False, False, True, True, 50),
            ((20, TOP_K_ALL), False, False, True, False, None),
            ((TOP_K_ALL, TOP_K_ALL), False, False, False, False, None),
        )
        for top_ks, all_greedy, any_greedy, need_top_k, npu, max_top_k in cases:
            with self.subTest(top_ks=top_ks):
                pool = SamplerParamsPool(num_rows=NUM_ROWS, device=DEVICE)
                slots = list(range(1, len(top_ks) + 1))
                pool.write([_req(s, top_k=k) for s, k in zip(slots, top_ks)])

                params = _gather(pool, slots)

                self.assertEqual(
                    (
                        params.is_all_greedy,
                        params.is_any_greedy,
                        params.need_top_k_sampling,
                        params.npu_top_k_top_p_eligible,
                        params.max_top_k,
                    ),
                    (all_greedy, any_greedy, need_top_k, npu, max_top_k),
                )

    def test_flags_drop_when_the_last_row_using_them_leaves(self):
        """Each gather derives need_top_p_sampling and need_min_p_sampling from
        its own rows, so they go off once the last top-p or min-p row leaves."""
        pool = SamplerParamsPool(num_rows=NUM_ROWS, device=DEVICE)
        pool.write([_req(1, top_p=0.9, min_p=0.1, top_k=1), _req(2, top_k=20)])

        both = _gather(pool, [1, 2])
        self.assertTrue(both.need_top_p_sampling)
        self.assertTrue(both.need_min_p_sampling)

        rest = _gather(pool, [2])
        self.assertFalse(rest.need_top_p_sampling)
        self.assertFalse(rest.need_min_p_sampling)
        self.assertFalse(rest.is_any_greedy)
        self.assertEqual(rest.max_top_k, 20)

    def test_sampler_params_fields_are_sampling_batch_info_fields(self):
        """set_sampler_params sets attributes by name and copy_for_forward copies
        only dataclass fields, so a SamplerParams field SamplingBatchInfo does not
        declare would not reach the forward."""
        fields = {f.name for f in dataclasses.fields(SamplingBatchInfo)}
        self.assertLessEqual(set(SamplerParams._fields), fields)


if __name__ == "__main__":
    unittest.main()
