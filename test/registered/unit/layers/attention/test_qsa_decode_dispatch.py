"""CPU tests for the QSA scorers' kernel and launch selection (no GPU, no TileLang)."""

import importlib
import unittest
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa import mqa, qsa_indexer
from sglang.test.test_utils import CustomTestCase

# SM counts of current datacenter parts; the selection depends only on this.
_SM_COUNTS = (132, 148, 160)
_CUDA = torch.device("cuda", 0)
_VERIFY = dict(
    rows_per_group=4,
    query_heads=4,
    kernel_heads=8,
    head_dim=128,
    page_size=16,
    scoring_dtype=torch.bfloat16,
    device=_CUDA,
)


class _KernelRecorder:
    """Stands in for the TileLang kernel factories and records which one ran."""

    def __init__(self):
        self.calls = []

    def factory(self, name):
        def build(**config):
            self.calls.append((name, config))
            return lambda *args: None

        return build


def _dispatch(rows, rows_per_group, *, recorder, num_query_heads=4, **extra):
    q = torch.zeros(rows, 8, 128, dtype=torch.bfloat16)
    k_cache = torch.zeros(4, 16, 1, 128, dtype=torch.bfloat16)
    page_table = torch.zeros(rows, 4, dtype=torch.int32)
    lens = torch.full((rows,), 64, dtype=torch.int32)
    mqa.tilelang_qsa_mqa_decode(
        q,
        k_cache,
        page_table,
        lens,
        64,
        min_scored_len=512,
        rows_per_group=rows_per_group,
        num_query_heads=num_query_heads,
        **extra,
    )
    return recorder.calls[-1]


class TestDecodeCtasPerRowLevels(CustomTestCase):
    def tearDown(self):
        importlib.reload(mqa)

    def test_unset_env_keeps_the_stock_levels(self):
        with envs.SGLANG_QSA_DECODE_CTAS_PER_ROW_LEVELS.override(""):
            envs.SGLANG_QSA_DECODE_CTAS_PER_ROW_LEVELS.clear()
            importlib.reload(mqa)
        self.assertEqual(mqa._DECODE_CTAS_PER_ROW_LEVELS, (1, 16, 256))

    def test_env_replaces_the_levels(self):
        with envs.SGLANG_QSA_DECODE_CTAS_PER_ROW_LEVELS.override("1,256"):
            importlib.reload(mqa)
            self.assertEqual(mqa._DECODE_CTAS_PER_ROW_LEVELS, (1, 256))

    def test_malformed_levels_fail_at_import(self):
        for value in ("16,1", "0,256", "1,1,256", "1,x"):
            with self.subTest(value=value):
                with envs.SGLANG_QSA_DECODE_CTAS_PER_ROW_LEVELS.override(value):
                    with self.assertRaises(ValueError):
                        importlib.reload(mqa)

    def test_dropping_a_level_moves_only_its_rows_up(self):
        # Removing 16 from (1, 16, 256) must send exactly the rows that took 16 to
        # 256 and leave every other row count on the stock choice.
        stock, coarse = (1, 16, 256), (1, 256)
        for num_sms in _SM_COUNTS:
            moved = 0
            for rows in range(1, 1025):
                before = mqa._select_decode_ctas_per_row(rows, num_sms, stock)
                after = mqa._select_decode_ctas_per_row(rows, num_sms, coarse)
                if before == 16:
                    self.assertEqual(after, 256, (num_sms, rows))
                    moved += 1
                else:
                    self.assertEqual(after, before, (num_sms, rows))
            self.assertGreater(moved, 0, num_sms)



class TestGrouped4VerifyDispatch(CustomTestCase):
    def setUp(self):
        patcher = mock.patch.object(
            torch.cuda, "get_device_capability", return_value=(10, 7)
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _applies(self, rows, **overrides):
        return mqa._grouped4_verify_applies(rows=rows, **{**_VERIFY, **overrides})

    def test_off_by_default_everywhere(self):
        with mock.patch.object(mqa, "_QSA_GROUPED4_VERIFY", False):
            self.assertFalse(any(self._applies(rows) for rows in range(4, 1025, 4)))

    def test_takes_over_exactly_above_the_stock_grouped_scorer(self):
        # The stock grouped scorer owns 1-16 requests; grouped4 owns 17-144.
        with mock.patch.object(mqa, "_QSA_GROUPED4_VERIFY", True):
            taken = [rows for rows in range(1, 1025) if self._applies(rows)]
        self.assertEqual(taken, list(range(68, 577, 4)))

    def test_shape_dtype_and_arch_gates(self):
        with mock.patch.object(mqa, "_QSA_GROUPED4_VERIFY", True):
            self.assertTrue(self._applies(128))
            for overrides in (
                dict(scoring_dtype=torch.float8_e4m3fn),
                dict(query_heads=8),
                dict(rows_per_group=1),
                dict(page_size=64),
                dict(device=torch.device("cpu")),
            ):
                with self.subTest(**{k: str(v) for k, v in overrides.items()}):
                    self.assertFalse(self._applies(128, **overrides))
            with mock.patch.object(
                torch.cuda, "get_device_capability", return_value=(10, 0)
            ):
                self.assertFalse(self._applies(128))

    def test_dispatch_routes_rows_to_grouped4_only_when_on(self):
        recorder = _KernelRecorder()
        real_applies = mqa._grouped4_verify_applies
        names = (
            "_tilelang_qsa_mqa_decode_grouped4_kernel",
            "_tilelang_qsa_mqa_decode_grouped_v7_kernel",
            "_tilelang_qsa_mqa_decode_kernel",
        )
        with mock.patch.multiple(
            mqa,
            HAS_TILELANG=True,
            _num_sms=lambda index: 148,
            # The CPU tensors stand in for CUDA ones in the arch gate.
            _grouped4_verify_applies=lambda **kw: real_applies(**{**kw, "device": _CUDA}),
            **{name: recorder.factory(name) for name in names},
            create=True,
        ):
            with mock.patch.object(mqa, "_QSA_GROUPED4_VERIFY", False):
                self.assertEqual(
                    _dispatch(128, 4, recorder=recorder)[0],
                    "_tilelang_qsa_mqa_decode_grouped_v7_kernel",
                )
            with mock.patch.object(mqa, "_QSA_GROUPED4_VERIFY", True):
                name, config = _dispatch(128, 4, recorder=recorder)
                self.assertEqual(name, "_tilelang_qsa_mqa_decode_grouped4_kernel")
                self.assertEqual(config["ctas_per_row"], mqa._QSA_GROUPED4_VERIFY_CTAS)
                # Without the real head count the padded Q has 8 heads: stock path.
                self.assertEqual(
                    _dispatch(128, 4, recorder=recorder, num_query_heads=None)[0],
                    "_tilelang_qsa_mqa_decode_grouped_v7_kernel",
                )


class TestGrouped4ChainAndDraftExtendDispatch(CustomTestCase):
    def setUp(self):
        self.recorder = _KernelRecorder()
        real_applies = mqa._grouped4_verify_applies
        names = (
            "_tilelang_qsa_mqa_decode_grouped4_chain_kernel",
            "_tilelang_qsa_mqa_decode_grouped4_kernel",
            "_tilelang_qsa_mqa_decode_grouped_v7_kernel",
            "_tilelang_qsa_mqa_decode_kernel",
        )
        patchers = [
            mock.patch.object(torch.cuda, "get_device_capability", return_value=(10, 7)),
            mock.patch.multiple(
                mqa,
                HAS_TILELANG=True,
                _num_sms=lambda index: 148,
                _grouped4_verify_applies=lambda **kw: real_applies(
                    **{**kw, "device": _CUDA}
                ),
                **{name: self.recorder.factory(name) for name in names},
                create=True,
            ),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _kernel(self, rows, *, grouped4, chain, grouped4_only=False):
        with mock.patch.multiple(
            mqa, _QSA_GROUPED4_VERIFY=grouped4, _QSA_GROUPED4_VERIFY_CHAIN=chain
        ):
            return _dispatch(rows, 4, recorder=self.recorder, grouped4_only=grouped4_only)

    def test_chain_replaces_grouped4_only_where_grouped4_applies(self):
        name, config = self._kernel(128, grouped4=True, chain=True)
        self.assertEqual(name, "_tilelang_qsa_mqa_decode_grouped4_chain_kernel")
        self.assertEqual(config["ctas_per_row"], mqa._QSA_GROUPED4_VERIFY_CHAIN_CTAS)
        # Without grouped4, or at <= 16 requests, the chain switch alone changes nothing.
        for rows, grouped4 in ((128, False), (64, True)):
            self.assertEqual(
                self._kernel(rows, grouped4=grouped4, chain=True)[0],
                self._kernel(rows, grouped4=grouped4, chain=False)[0],
            )

    def test_draft_extend_rows_fall_back_to_per_row_scoring(self):
        # Rows grouped only for grouped4 must not reach the stock grouped kernels,
        # which assume target-verify layout; they keep the per-row kernel instead.
        for rows, grouped4 in ((64, True), (128, False), (8, True)):
            self.assertEqual(
                self._kernel(rows, grouped4=grouped4, chain=False, grouped4_only=True)[0],
                "_tilelang_qsa_mqa_decode_kernel",
            )
        self.assertEqual(
            self._kernel(128, grouped4=True, chain=False, grouped4_only=True)[0],
            "_tilelang_qsa_mqa_decode_grouped4_kernel",
        )


class TestPrefillScaleFold(CustomTestCase):
    def test_store_constant_is_cuda_division_by_sqrt128(self):
        import numpy as np

        # torch's CUDA div_ by a Python scalar casts it to FP32 and multiplies by
        # the FP32 reciprocal; the PTX store constant must be exactly that value.
        divisor = np.float32(np.sqrt(np.float64(128)))
        reciprocal = np.float32(1.0) / divisor
        bits = int(np.array(reciprocal, dtype=np.float32).view(np.uint32))
        self.assertEqual(bits, mqa._QSA_INV_SQRT128_BITS)
        self.assertIn(f"0f{bits:08X}", mqa._QSA_INV_SQRT128_PRELUDE)

    def test_fold_only_replaces_the_default_scale_at_head_dim_128(self):
        with mock.patch.object(mqa, "_QSA_PREFILL_SCALE_FOLD", False):
            self.assertFalse(mqa._prefill_scale_fold_applies(None, 128))
        with mock.patch.object(mqa, "_QSA_PREFILL_SCALE_FOLD", True):
            self.assertTrue(mqa._prefill_scale_fold_applies(None, 128))
            self.assertFalse(mqa._prefill_scale_fold_applies(0.5, 128))
            self.assertFalse(mqa._prefill_scale_fold_applies(None, 64))


class TestGroupedScoreRequestCap(CustomTestCase):
    def tearDown(self):
        importlib.reload(qsa_indexer)

    def test_env_sets_the_cap_and_rejects_non_positive(self):
        with envs.SGLANG_QSA_GROUPED_SCORE_MAX_REQUESTS.override(144):
            importlib.reload(qsa_indexer)
            self.assertEqual(qsa_indexer._QSA_GROUPED_SCORE_MAX_REQUESTS, 144)
        with envs.SGLANG_QSA_GROUPED_SCORE_MAX_REQUESTS.override(0):
            with self.assertRaises(ValueError):
                importlib.reload(qsa_indexer)


if __name__ == "__main__":
    unittest.main()
