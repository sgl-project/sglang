import dataclasses
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.sampling.penalties import PenaltyStatusPool
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.sampling.sampling_params_pool import SamplingParamsPool, SamplingRows
from sglang.srt.sampling.watermarking.gumbel import adapter as gumbel_adapter
from sglang.srt.sampling.watermarking.synthid import adapter as synthid_adapter
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NUM_ROWS = 8
DEVICE = "cpu"
POOL = "sglang.srt.sampling.sampling_params_pool"


def _req(
    slot,
    temperature=1.0,
    top_p=1.0,
    top_k=TOP_K_ALL,
    min_p=0.0,
    seed=None,
    filler_seed=0,
    penalty_index=None,
    watermark=None,
):
    return SimpleNamespace(
        kv=SimpleNamespace(req_pool_idx=slot),
        sampling_params=SimpleNamespace(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            sampling_seed=seed,
            watermark=watermark,
        ),
        filler_seed=filler_seed,
        penalty_index=penalty_index,
    )


def _gather(pool, slots) -> SamplingRows:
    indices = torch.tensor(slots, dtype=torch.int64)
    return pool.gather(indices, indices)


def _exec(deterministic=False, watermark_algorithm=None):
    return SimpleNamespace(
        deterministic=SimpleNamespace(enable_deterministic_inference=deterministic),
        features=SimpleNamespace(
            enable_watermark=watermark_algorithm is not None,
            watermark_algorithm=watermark_algorithm,
            watermark_key="0xfedcba9876543210",
            watermark_context_window=4,
            watermark_default_enabled=True,
            watermark_enforce_all=False,
        ),
    )


class _PoolTestCase(CustomTestCase):
    """Runs each test with the flags the pool reads patched, so it does not
    depend on a published (or leaked) process context."""

    def setUp(self):
        super().setUp()
        self.exec_ns = _exec()
        self.spec_ns = SimpleNamespace(speculative_algorithm=None)
        for module in (POOL, "sglang.srt.sampling.watermarking.runtime"):
            exec_patch = patch(f"{module}.get_exec", lambda: self.exec_ns)
            exec_patch.start()
            self.addCleanup(exec_patch.stop)
        spec_patch = patch(f"{POOL}.get_spec", lambda: self.spec_ns)
        spec_patch.start()
        self.addCleanup(spec_patch.stop)


class TestSamplerParams(_PoolTestCase):
    def test_gather_reads_each_slot_in_batch_order(self):
        pool = SamplingParamsPool(num_rows=NUM_ROWS, device=DEVICE)
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

    def test_rewritten_slot_reads_the_new_request(self):
        """A freed slot that a later admission reuses holds the new request's
        params, host flags included."""
        pool = SamplingParamsPool(num_rows=NUM_ROWS, device=DEVICE)
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
                pool = SamplingParamsPool(num_rows=NUM_ROWS, device=DEVICE)
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
        pool = SamplingParamsPool(num_rows=NUM_ROWS, device=DEVICE)
        pool.write([_req(1, top_p=0.9, min_p=0.1, top_k=1), _req(2, top_k=20)])

        both = _gather(pool, [1, 2])
        self.assertTrue(both.need_top_p_sampling)
        self.assertTrue(both.need_min_p_sampling)

        rest = _gather(pool, [2])
        self.assertFalse(rest.need_top_p_sampling)
        self.assertFalse(rest.need_min_p_sampling)
        self.assertFalse(rest.is_any_greedy)
        self.assertEqual(rest.max_top_k, 20)

    def test_rows_fields_are_sampling_batch_info_fields(self):
        """set_rows sets the rows by name and copy_for_forward copies only
        dataclass fields, so a SamplingRows field SamplingBatchInfo does not
        declare would not reach the forward."""
        fields = {f.name for f in dataclasses.fields(SamplingBatchInfo)}
        self.assertLessEqual(set(SamplingRows._fields), fields)


class TestSeeds(_PoolTestCase):
    def test_deterministic_mode_seeds_every_row(self):
        """An explicit seed is kept and a missing one defaults to 42."""
        self.exec_ns = _exec(deterministic=True)
        pool = SamplingParamsPool(num_rows=NUM_ROWS, device=DEVICE)
        pool.write([_req(1, seed=123), _req(2)])

        self.assertEqual(_gather(pool, [1, 2]).sampling_seed.tolist(), [123, 42])
        # An idle batch still seeds, with no rows.
        self.assertEqual(_gather(pool, []).sampling_seed.tolist(), [])

    def test_seed_without_deterministic_inference(self):
        """A non-greedy request's seed applies; a greedy seed, or speculative
        decoding, leaves the batch unseeded."""
        pool = SamplingParamsPool(num_rows=NUM_ROWS, device=DEVICE)
        pool.write([_req(1, seed=123, top_k=50), _req(2, seed=123, top_k=1)])

        self.assertEqual(_gather(pool, [1]).sampling_seed.tolist(), [123])
        self.assertIsNone(_gather(pool, [2]).sampling_seed)
        self.assertIsNone(_gather(pool, []).sampling_seed)
        # The algorithm is fixed per server; a new pool, so no cached gather
        # outlives the change.
        self.spec_ns.speculative_algorithm = "EAGLE"
        pool = SamplingParamsPool(num_rows=NUM_ROWS, device=DEVICE)
        pool.write([_req(1, seed=123, top_k=50)])
        self.assertIsNone(_gather(pool, [1]).sampling_seed)

    def test_unseeded_row_samples_with_its_filler_seed(self):
        """An unseeded request samples with its own filler seed in every seeded
        step, wherever its row is, and unseeded once no seeded row is left."""
        pool = SamplingParamsPool(num_rows=NUM_ROWS, device=DEVICE)
        pool.write(
            [
                _req(1, seed=123, top_k=50),
                _req(2, filler_seed=456),
                _req(3, filler_seed=789),
            ]
        )

        self.assertEqual(_gather(pool, [1, 2]).sampling_seed.tolist(), [123, 456])
        self.assertEqual(
            _gather(pool, [3, 2, 1]).sampling_seed.tolist(), [789, 456, 123]
        )
        self.assertIsNone(_gather(pool, [3, 2]).sampling_seed)

    def test_every_int64_seed_reads_back_exactly(self):
        """Seeds are kept as two int32 halves; the gather rebuilds the int64."""
        seeds = [-(2**63), -1, 0, 2**31, 2**63 - 1]
        pool = SamplingParamsPool(num_rows=NUM_ROWS, device=DEVICE)
        pool.write([_req(i, seed=seed, top_k=50) for i, seed in enumerate(seeds)])

        rows = _gather(pool, list(range(len(seeds))))

        self.assertEqual(rows.sampling_seed.dtype, torch.int64)
        self.assertEqual(rows.sampling_seed.tolist(), seeds)


class TestPenaltyIndices(_PoolTestCase):
    def test_step_penalties_take_the_rows_with_an_index(self):
        """The gather gives StepPenalties the penalized rows in batch order and
        their indices, from the gathered field alone."""
        penalty_pool = SimpleNamespace(kinds=[1, 0, 0, 4])
        pool = SamplingParamsPool(
            num_rows=NUM_ROWS, device=DEVICE, penalty_status_pool=penalty_pool
        )
        pool.write(
            [_req(1), _req(2, penalty_index=3), _req(4), _req(5, penalty_index=0)]
        )

        step = _gather(pool, [1, 2, 4, 5]).step_penalties

        self.assertEqual(step.batch_size, 4)
        self.assertEqual(step.batch_indices.tolist(), [1, 3])
        self.assertEqual(step.penalty_indices.tolist(), [3, 0])
        self.assertEqual(step.penalty_indices.dtype, torch.int64)
        self.assertEqual(step.kinds, 5)
        self.assertIsNone(_gather(pool, [1, 4]).step_penalties)

    def test_set_rows_reads_the_step_penalties_from_the_pool(self):
        penalty_pool = PenaltyStatusPool(vocab_size=8, device=DEVICE)
        req = _req(2)
        req.sampling_params.frequency_penalty = 0.5
        req.sampling_params.presence_penalty = 0.0
        req.sampling_params.repetition_penalty = 1.0
        req.sampling_params.min_new_tokens = 0
        penalty_pool.alloc([req])
        pool = SamplingParamsPool(
            num_rows=NUM_ROWS, device=DEVICE, penalty_status_pool=penalty_pool
        )
        pool.write([req, _req(3)])
        info = SamplingBatchInfo(vocab_size=8, device=DEVICE)

        info.set_rows(_gather(pool, [3, 2]))

        self.assertEqual(info.step_penalties.batch_indices.tolist(), [1])
        self.assertEqual(info.step_penalties.penalty_indices.tolist(), [0])

    def test_unchanged_batch_reuses_its_params_until_a_write(self):
        pool = SamplingParamsPool(num_rows=NUM_ROWS, device=DEVICE)
        pool.write([_req(1, temperature=0.7), _req(4, top_k=1)])

        first = _gather(pool, [1, 4])
        again = _gather(pool, [1, 4])
        self.assertIs(again.temperatures, first.temperatures)
        self.assertIs(again.top_ks, first.top_ks)
        self.assertIsNot(_gather(pool, [4, 1]).temperatures, first.temperatures)

        after_slot_change = _gather(pool, [1, 4])
        self.assertIsNot(after_slot_change.temperatures, first.temperatures)
        pool.write([_req(4, temperature=1.2)])
        params = _gather(pool, [1, 4])
        self.assertIsNot(params.temperatures, after_slot_change.temperatures)
        self.assertAlmostEqual(params.temperatures[1].item(), 1.2, places=6)
        self.assertFalse(params.is_any_greedy)

    def test_unchanged_batch_takes_each_steps_penalties(self):
        """A reused gather still builds the step's penalties from the counts the
        previous step committed."""
        penalty_pool = PenaltyStatusPool(vocab_size=8, device=DEVICE)
        req = _req(2)
        req.output_ids = []
        req.sampling_params.frequency_penalty = 0.5
        req.sampling_params.presence_penalty = 0.0
        req.sampling_params.repetition_penalty = 1.0
        req.sampling_params.min_new_tokens = 0
        penalty_pool.alloc([req])
        penalty_pool.reset_history([req])
        pool = SamplingParamsPool(
            num_rows=NUM_ROWS, device=DEVICE, penalty_status_pool=penalty_pool
        )
        pool.write([_req(3), req])

        first = _gather(pool, [3, 2])
        additive, _ = first.step_penalties.logit_penalties()
        self.assertEqual(additive[1].tolist(), [0.0] * 8)
        first.step_penalties.commit(torch.tensor([0, 5]))

        second = _gather(pool, [3, 2])
        self.assertIs(second.temperatures, first.temperatures)
        additive, _ = second.step_penalties.logit_penalties()
        self.assertEqual(additive[1, 5].item(), -0.5)
        self.assertEqual(additive[0].tolist(), [0.0] * 8)


class TestSlotLifecycle(_PoolTestCase):
    def test_readmission_after_retraction_keeps_the_penalty_index(self):
        """A retraction frees the slot and keeps the penalty index; the
        re-admission writes both to the request's new slot."""
        pool = SamplingParamsPool(
            num_rows=NUM_ROWS,
            device=DEVICE,
            penalty_status_pool=SimpleNamespace(kinds=[0, 0, 1]),
        )
        req = _req(1, seed=7, top_k=20, penalty_index=2)
        pool.write([req])
        req.kv.req_pool_idx = 6
        pool.write([req])

        rows = _gather(pool, [6])

        self.assertEqual(rows.step_penalties.penalty_indices.tolist(), [2])
        self.assertEqual(rows.sampling_seed.tolist(), [7])
        self.assertEqual(rows.top_ks.tolist(), [20])

    def test_reused_slot_reads_only_the_new_request(self):
        """A finished request leaves its entry behind; the next admission to the
        slot overwrites every field, host copies included."""
        self.exec_ns = _exec(watermark_algorithm="synthid")
        pool = SamplingParamsPool(
            num_rows=NUM_ROWS,
            device=DEVICE,
            penalty_status_pool=SimpleNamespace(kinds=[0, 1]),
        )
        pool.write([_req(3, seed=5, top_k=50, penalty_index=1)])
        pool.write([_req(3, top_k=1, filler_seed=11)])

        rows = _gather(pool, [3])

        self.assertIsNone(rows.step_penalties)
        self.assertIsNone(rows.sampling_seed)
        self.assertIsNone(rows.watermark)
        self.assertEqual(pool.table[pool.PENALTY_INDEX, 3].item(), -1)
        self.assertEqual(pool.table[pool.WATERMARK, 3].item(), 0)


class TestWatermarkFields(_PoolTestCase):
    def test_gumbel_rows_read_back_each_request_value(self):
        """Keys (negative ones included) are rebuilt exactly from two int32
        halves; a greedy or disabled row is no candidate."""
        self.exec_ns = _exec(watermark_algorithm="gumbel")
        disabled = SimpleNamespace(enabled=False, key=None, context_window=None)
        own_key = SimpleNamespace(enabled=True, key="0x1234", context_window=2)
        pool = SamplingParamsPool(num_rows=NUM_ROWS, device=DEVICE)
        pool.write(
            [
                _req(1, top_k=50),
                _req(2, top_k=50, watermark=own_key),
                _req(3, top_k=1),
                _req(4, top_k=50, watermark=disabled),
            ]
        )

        rows = _gather(pool, [4, 3, 2, 1]).watermark

        self.assertIsInstance(rows, gumbel_adapter.GumbelRows)
        default_key = 0xFEDCBA9876543210 - (1 << 64)
        self.assertEqual(rows.keys.dtype, torch.int64)
        self.assertEqual(rows.keys.tolist(), [0, default_key, 0x1234, default_key])
        self.assertEqual(rows.context_windows.dtype, torch.int32)
        self.assertEqual(rows.context_windows.tolist(), [4, 4, 2, 4])
        self.assertEqual(rows.enabled.dtype, torch.bool)
        self.assertEqual(rows.enabled.tolist(), [False, True, True, True])
        self.assertEqual(rows.candidates_host, [False, False, True, True])

        self.assertIsNone(_gather(pool, [4, 3]).watermark)

    def test_synthid_rows_index_the_candidates(self):
        self.exec_ns = _exec(watermark_algorithm="synthid")
        pool = SamplingParamsPool(num_rows=NUM_ROWS, device=DEVICE)
        pool.write([_req(1, top_k=50), _req(2, top_k=1), _req(3, top_k=50)])

        rows = _gather(pool, [2, 1, 3]).watermark

        self.assertIsInstance(rows, synthid_adapter.SynthIDRows)
        self.assertEqual(rows.indices.tolist(), [1, 2])
        self.assertEqual(rows.candidates_host, [False, True, True])


if __name__ == "__main__":
    unittest.main()
