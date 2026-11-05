"""CPU dispatch checks; factory doubles do not qualify TileLang GPU numerics."""

import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa import mqa
from sglang.srt.layers.attention.qsa.kernel import qsa_fast_topk
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestQsaPrefillTileV2(unittest.TestCase):
    def test_default_off(self):
        with envs.SGLANG_QSA_PREFILL_TILE_V2.override(False):
            self.assertFalse(envs.SGLANG_QSA_PREFILL_TILE_V2.get())
        with mock.patch.object(mqa, "_QSA_PREFILL_TILE_V2", False):
            self.assertEqual(
                mqa._prefill_relative_resources(torch.bfloat16, torch.bfloat16, 4, 128),
                (64, 3, 512),
            )

    def test_exact_bf16_opt_in(self):
        with mock.patch.object(mqa, "_QSA_PREFILL_TILE_V2", True):
            self.assertEqual(
                mqa._prefill_relative_resources(torch.bfloat16, torch.bfloat16, 4, 128),
                (128, 2, 256),
            )

    def test_other_operands_keep_stock_specialization(self):
        with mock.patch.object(mqa, "_QSA_PREFILL_TILE_V2", True):
            for dtype in (torch.float16, torch.float32, torch.float8_e4m3fn):
                for qd, kd in ((dtype, torch.bfloat16), (torch.bfloat16, dtype)):
                    with self.subTest(q=qd, k=kd):
                        self.assertEqual(
                            mqa._prefill_relative_resources(qd, kd, 4, 128),
                            (64, 3, 512),
                        )
            for heads, dim in ((1, 128), (8, 128), (16, 128), (4, 64), (4, 256)):
                with self.subTest(heads=heads, dim=dim):
                    self.assertEqual(
                        mqa._prefill_relative_resources(
                            torch.bfloat16, torch.bfloat16, heads, dim
                        ),
                        (64, 3, 512),
                    )

    def _route(self, enabled, folded):
        q = (
            torch.arange(35 * 4 * 128)
            .reshape(35, 4, 128)
            .remainder(11)
            .sub(5)
            .to(torch.bfloat16)
        )
        k = (
            torch.arange(192 * 128)
            .reshape(192, 1, 128)
            .remainder(7)
            .sub(3)
            .to(torch.bfloat16)
        )
        starts = torch.arange(35, dtype=torch.int32).remainder(3) * 64
        lengths = 1 + torch.arange(35, dtype=torch.int32).mul(13).remainder(64)
        ends = starts + lengths
        calls = []

        def factory(**config):
            calls.append(config)

            def launch(flat_q, flat_k, logits, a, b):
                # Explicit CPU factory double: exercises real wrapper buffers,
                # padding and scaling, not a replacement GPU implementation.
                queries = flat_q.reshape(
                    -1, config["heads"], config["head_dim"]
                ).float()
                for row in range(queries.shape[0]):
                    scores = (
                        (queries[row] @ flat_k[int(a[row]) : int(b[row])].float().T)
                        .clamp_min(0)
                        .sum(0)
                    )
                    if folded:
                        inv = torch.tensor(
                            [mqa._QSA_INV_SQRT128_BITS], dtype=torch.int32
                        ).view(torch.float32)[0]
                        scores = scores * inv
                    logits[row, : scores.numel()].copy_(scores)

            return launch

        before = [x.clone() for x in (q, k, starts, ends)]
        with (
            mock.patch.object(mqa, "HAS_TILELANG", True),
            mock.patch.object(mqa, "_QSA_PREFILL_TILE_V2", enabled),
            mock.patch.object(mqa, "_QSA_PREFILL_SCALE_FOLD", folded),
            mock.patch.object(mqa, "_num_sms", return_value=212),
            mock.patch.object(
                mqa, "_tilelang_qsa_mqa_prefill_relative_kernel", factory, create=True
            ),
            mock.patch.object(
                mqa,
                "_tilelang_qsa_mqa_prefill_relative_scaled_kernel",
                factory,
                create=True,
            ),
        ):
            logits = mqa.tilelang_qsa_mqa_prefill_relative(q, k, starts, ends, 64)
        for original, actual in zip(before, (q, k, starts, ends)):
            self.assertTrue(torch.equal(original, actual))
        self.assertEqual(logits.shape, (35, 64))
        mask = torch.arange(64)[None, :] < lengths[:, None]
        ids = qsa_fast_topk(logits, torch.zeros_like(starts), lengths, topk=512)
        return logits[mask], ids, calls

    def test_real_wrapper_bits_and_ordered_cpu_selection(self):
        for folded in (False, True):
            with self.subTest(folded=folded):
                a, aid, ac = self._route(False, folded)
                b, bid, bc = self._route(True, folded)
                self.assertTrue(torch.equal(a.view(torch.int32), b.view(torch.int32)))
                self.assertTrue(torch.equal(aid, bid))
                self.assertEqual(len(ac), 1)
                self.assertEqual(len(bc), 1)
                self.assertEqual(
                    (ac[0]["block_n"], ac[0]["num_stages"], ac[0]["threads"]),
                    (64, 3, 512),
                )
                self.assertEqual(
                    (bc[0]["block_n"], bc[0]["num_stages"], bc[0]["threads"]),
                    (128, 2, 256),
                )
                self.assertEqual(ac[0]["block_q"], bc[0]["block_q"])

    def test_public_cpu_fallback_unchanged(self):
        q, k = (
            torch.ones(3, 4, 128, dtype=torch.bfloat16),
            torch.ones(19, 1, 128, dtype=torch.bfloat16),
        )
        starts, ends = torch.tensor([0, 2, 5]), torch.tensor([19, 13, 11])
        with mock.patch.object(mqa, "_QSA_PREFILL_TILE_V2", False):
            a = mqa.qsa_mqa_prefill_relative(q, k, starts, ends, 19)
        with mock.patch.object(mqa, "_QSA_PREFILL_TILE_V2", True):
            b = mqa.qsa_mqa_prefill_relative(q, k, starts, ends, 19)
        self.assertTrue(torch.equal(a.view(torch.int32), b.view(torch.int32)))

    def test_selected_failure_is_not_retried(self):
        factory = mock.Mock(side_effect=RuntimeError("selected tile launch failed"))
        q = torch.ones(32, 4, 128, dtype=torch.bfloat16)
        k = torch.ones(64, 1, 128, dtype=torch.bfloat16)
        starts, ends = (
            torch.zeros(32, dtype=torch.int32),
            torch.full((32,), 64, dtype=torch.int32),
        )
        with (
            mock.patch.object(mqa, "HAS_TILELANG", True),
            mock.patch.object(mqa, "_QSA_PREFILL_TILE_V2", True),
            mock.patch.object(mqa, "_QSA_PREFILL_SCALE_FOLD", False),
            mock.patch.object(mqa, "_num_sms", return_value=212),
            mock.patch.object(
                mqa, "_tilelang_qsa_mqa_prefill_relative_kernel", factory, create=True
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "selected tile launch failed"):
                mqa.tilelang_qsa_mqa_prefill_relative(q, k, starts, ends, 64)
        factory.assert_called_once()


if __name__ == "__main__":
    unittest.main()
