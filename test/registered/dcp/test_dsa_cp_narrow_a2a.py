"""CPU unit test: DSA-CP's narrow all-to-all (handoff §8 W1) is exact.

DSA-CP swaps "my heads for every token" for "every head for my tokens" *after*
``q_b_proj`` and the absorb through ``w_kc``, so the wire carries the 512-wide
absorbed latent on the way in and the 512-wide attention output on the way back.
Both are fixed linear functions of narrower tensors beside them: q is 256 wide
coming out of ``q_b_proj`` and the head output is 256 (``v_head_dim``).

``SGLANG_NPU_ENABLE_DSA_CP_NARROW_A2A`` moves the inbound exchange to the
narrow side, and ``..._OUTPUT`` the return leg as well. Measured on A3 tp16, a
served A/B: **92 ms + 2.08 us/token** with both legs, of which the 92 ms is the
third collective going away and nothing else. The price is the full weights on
every rank -- ``w_kc`` alone is 12 MB per layer, both are 28 -- and at tp16 that
came to **4.06 GiB of KV pool**, which is why it is off by default and why the
return leg is a second switch.

**The claim this file pins:** deferring the absorb past the exchange changes
nothing, because ``npu_transpose_batchmatmul`` here is a product per (token,
head) pair and the all-to-all only moves those pairs between ranks. That holds
only if each rank absorbs with the ``w_kc`` of the head it ENDS UP holding, not
the head it was assigned -- which is exactly what the full weights buy and
exactly what a plausible implementation gets wrong.

The second test is the negative: it shows that reusing the local slice after the
swap is wrong, so the full-weight requirement is not decoration.

Usage:
    python -m pytest test_dsa_cp_narrow_a2a.py -v
    python test_dsa_cp_narrow_a2a.py
"""

import unittest
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# Small but ragged on every axis, so a wrong reshape cannot pass by symmetry.
TP_SIZES = [2, 3, 4, 8]
TOKEN_COUNTS = [1, 7, 16, 17, 33]


def _absorb(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """[t, h, d] x [h, d, l] -> [t, h, l]: the per-head product the NPU op does."""
    return torch.einsum("thd,hdl->thl", x, w)


class TestNarrowA2AIsExact(CustomTestCase):
    def _case(self, total, tp_size, heads_per_rank=2, nope=6, lora=5):
        torch.manual_seed(total * 100 + tp_size)
        heads = heads_per_rank * tp_size
        q_nope = torch.randn(total, heads, nope, dtype=torch.float64)
        w_kc = torch.randn(heads, nope, lora, dtype=torch.float64)
        rows = -(-total // tp_size)
        return q_nope, w_kc, rows

    def test_deferring_the_absorb_past_the_exchange_changes_nothing(self):
        """Absorb-then-swap and swap-then-absorb agree, rank by rank.

        The all-to-all is modelled exactly: rank r ends up holding every head
        for token rows [r * rows, (r + 1) * rows), so whatever the wide path
        delivers to rank r is that slice of the fully absorbed tensor.
        """
        for total in TOKEN_COUNTS:
            for tp_size in TP_SIZES:
                q_nope, w_kc, rows = self._case(total, tp_size)
                wide_all = _absorb(q_nope, w_kc)
                for tp_rank in range(tp_size):
                    lo = min(tp_rank * rows, total)
                    hi = min(lo + rows, total)
                    with self.subTest(total=total, tp=tp_size, rank=tp_rank):
                        wide = wide_all[lo:hi]
                        narrow = _absorb(q_nope[lo:hi], w_kc)
                        self.assertTrue(
                            torch.equal(wide, narrow),
                            "absorbing after the swap differs from absorbing before it",
                        )

    def test_absorbing_with_the_local_slice_after_the_swap_is_wrong(self):
        """The negative: this is why the full w_kc has to be gathered.

        After the exchange a rank holds every head, so indexing w_kc by its own
        assigned head block lines the wrong matrix up against each head. Nothing
        about the shapes objects -- the result is simply different, which is the
        failure mode the full-weight gather exists to prevent.
        """
        heads_per_rank, tp_size, total = 2, 4, 17
        q_nope, w_kc, rows = self._case(total, tp_size, heads_per_rank)
        wide_all = _absorb(q_nope, w_kc)
        wrong_ranks = []
        for tp_rank in range(tp_size):
            lo = min(tp_rank * rows, total)
            hi = min(lo + rows, total)
            if lo == hi:
                continue
            local_w = w_kc[
                tp_rank * heads_per_rank : (tp_rank + 1) * heads_per_rank
            ].repeat(tp_size, 1, 1)
            if not torch.equal(_absorb(q_nope[lo:hi], local_w), wide_all[lo:hi]):
                wrong_ranks.append(tp_rank)
        self.assertEqual(
            wrong_ranks,
            [
                r
                for r in range(tp_size)
                if min(r * rows, total) < min(r * rows + rows, total)
            ],
            "using the local w_kc slice after the swap should be wrong on every "
            "rank that holds a token; if it is not, this test has stopped "
            "modelling the exchange",
        )

    def test_w_vc_commutes_with_the_return_leg_too(self):
        """W1 also moves w_vc BEFORE the restore, which is the same argument."""
        for total in TOKEN_COUNTS:
            for tp_size in TP_SIZES:
                torch.manual_seed(total + tp_size)
                heads = 2 * tp_size
                rows = -(-total // tp_size)
                attn_out = torch.randn(total, heads, 5, dtype=torch.float64)
                w_vc = torch.randn(heads, 5, 4, dtype=torch.float64)
                for tp_rank in range(tp_size):
                    lo = min(tp_rank * rows, total)
                    hi = min(lo + rows, total)
                    with self.subTest(total=total, tp=tp_size, rank=tp_rank):
                        self.assertTrue(
                            torch.equal(
                                _absorb(attn_out, w_vc)[lo:hi],
                                _absorb(attn_out[lo:hi], w_vc),
                            )
                        )


class TestGatheredWeightLayout(CustomTestCase):
    """The gather must hand back w_kc in the layout the loader chose, for free.

    ``deepseek_weight_loader`` stores w_kc as
    ``w_kc.transpose(1, 2).contiguous().transpose(1, 2)``: shape
    [h, qk_nope, kv_lora] but laid out as [h, kv_lora, qk_nope], and the batched
    matmul is tuned for that. An all-gather needs a contiguous send buffer.

    The first implementation gathered into a plain buffer and rebuilt the layout
    with ``.transpose().contiguous().transpose()`` -- correct, but it allocated a
    second full-size copy per layer and copied the send buffer too. Measured on
    A3 tp16 the feature cost 4.05 GiB of KV pool against 2.13 GiB of live
    weights.

    ``dsa_cp_attach_full_kv_b`` now gathers in the PHYSICAL layout and takes the
    logical view back, which costs neither. These tests pin the two facts that
    makes possible.
    """

    @staticmethod
    def _loader_w_kc(h, nope, lora):
        """Exactly what the loader stores: transposed-contiguous."""
        return torch.randn(h, nope, lora).transpose(1, 2).contiguous().transpose(1, 2)

    def test_w_kc_is_its_own_transpose_so_the_send_buffer_is_free(self):
        """The load-bearing fact: no copy is needed to send w_kc."""
        local = self._loader_w_kc(2, 6, 5)
        self.assertFalse(local.is_contiguous(), "the loader's layout is not plain")
        send = local.transpose(1, 2)
        self.assertTrue(
            send.is_contiguous(),
            "w_kc's transpose must already be contiguous, or the gather pays a "
            "full send copy per layer",
        )
        self.assertEqual(
            send.contiguous().data_ptr(),
            send.data_ptr(),
            "contiguous() on it must not copy; same storage, not merely equal",
        )

    def test_gathering_transposed_then_viewing_back_matches_the_loader(self):
        h_local, tp, nope, lora = 2, 4, 6, 5
        local = self._loader_w_kc(h_local, nope, lora)
        send = local.transpose(1, 2)

        # What the new code allocates: the physical layout, contiguous.
        gathered = torch.empty(h_local * tp, *send.shape[1:])
        self.assertTrue(gathered.is_contiguous())
        full = gathered.transpose(1, 2)

        self.assertEqual(full.shape, (h_local * tp, nope, lora))
        self.assertEqual(full.stride()[1:], local.stride()[1:])
        self.assertFalse(full.is_contiguous())

    def test_w_vc_takes_the_plain_path_and_its_transpose_would_not(self):
        """w_vc is plain contiguous, so the transposed branch must not fire."""
        local = torch.randn(2, 5, 4).contiguous()
        self.assertTrue(local.is_contiguous())
        self.assertFalse(
            local.transpose(1, 2).is_contiguous(),
            "if this were contiguous the branch could not tell the two apart",
        )
        gathered = torch.empty(8, 5, 4)
        self.assertEqual(gathered.stride()[1:], local.stride()[1:])


class TestOnlyTheEnabledLegsAreGathered(CustomTestCase):
    """Which weights the loader gathers has to follow which legs are narrowed.

    The inbound leg needs the full ``w_kc``; the return leg needs the full
    ``w_vc`` on top, and that second one is 16 of the 28 MB per layer for about
    12% of the saving. Gathering it anyway is invisible -- everything still
    computes the right answer -- and costs 1.2 GiB of KV pool on a 78-layer
    model at tp16. So the selection is worth pinning.
    """

    TP = 4
    HEADS, NOPE, LORA, VDIM = 2, 6, 5, 4

    def _attach(self, narrow_input=True, narrow_output=False):
        from sglang.srt.layers.attention.dsa import dsa_cp as dsa_cp_module

        class _Attn:
            use_dsa = True

        attn = _Attn()
        # The loader's two layouts: w_kc transposed-contiguous, w_vc plain.
        attn.w_kc = torch.randn(
            self.HEADS, self.LORA, self.NOPE, dtype=torch.bfloat16
        ).transpose(1, 2)
        attn.w_vc = torch.randn(
            self.HEADS, self.LORA, self.VDIM, dtype=torch.bfloat16
        ).contiguous()

        calls = []

        def _fake_all_to_all(recv, send):
            # Every rank sends tp copies of its own slice and keeps one block
            # per peer, so on a single fake rank the result is tp copies. Same
            # contract the real collective has, which is what makes this a
            # gather at all.
            calls.append(tuple(send.shape))
            self.assertEqual(send.shape[0], self.TP)
            recv.copy_(send)

        parallel = mock.Mock()
        parallel.attn_tp_size = self.TP
        parallel.attn_tp_group.all_to_all_single = _fake_all_to_all
        with (
            mock.patch.object(
                dsa_cp_module, "dsa_cp_narrow_a2a_enabled", lambda: narrow_input
            ),
            mock.patch.object(
                dsa_cp_module, "dsa_cp_narrow_a2a_output_enabled", lambda: narrow_output
            ),
            mock.patch.object(dsa_cp_module, "get_parallel", lambda: parallel),
        ):
            dsa_cp_module.dsa_cp_attach_full_kv_b(attn)
        return attn, calls

    def test_input_only_leaves_w_vc_alone(self):
        attn, calls = self._attach(narrow_output=False)
        self.assertEqual(len(calls), 1, "input-only must gather exactly one weight")
        self.assertIsNotNone(getattr(attn, "w_kc_full", None))
        self.assertIsNone(
            getattr(attn, "w_vc_full", None),
            "w_vc was gathered although the return leg is wide, which is the "
            "16 MB per layer this mode exists to save",
        )
        self.assertEqual(
            attn.w_kc_full.shape, (self.HEADS * self.TP, self.NOPE, self.LORA)
        )

    def test_both_legs_gather_both_weights(self):
        attn, calls = self._attach(narrow_input=True, narrow_output=True)
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            attn.w_vc_full.shape, (self.HEADS * self.TP, self.LORA, self.VDIM)
        )

    def test_output_only_leaves_w_kc_alone(self):
        """The return leg runs without the inbound one, and pays only for w_vc.

        Measured at tp16: the inbound leg's effect on wall time sits under the
        run-to-run floor at both 6007 and 16007 tokens, while the pair saves
        105-126 ms. So the return leg on its own is the combination worth
        being able to run, and it must not drag w_kc's 12 MB per layer with it.
        """
        attn, calls = self._attach(narrow_input=False, narrow_output=True)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(getattr(attn, "w_kc_full", None))
        self.assertEqual(
            attn.w_vc_full.shape, (self.HEADS * self.TP, self.LORA, self.VDIM)
        )

    def test_neither_leg_gathers_nothing(self):
        attn, calls = self._attach(narrow_input=False, narrow_output=False)
        self.assertEqual(calls, [])
        self.assertIsNone(getattr(attn, "w_kc_full", None))
        self.assertIsNone(getattr(attn, "w_vc_full", None))

    def test_the_gathered_w_kc_keeps_the_loader_stride_pattern(self):
        attn, _ = self._attach(narrow_output=False)
        self.assertFalse(
            attn.w_kc_full.is_contiguous(),
            "the batched matmul is tuned for the transposed layout; a "
            "contiguous full copy means the gather rebuilt it and paid twice",
        )
        self.assertEqual(attn.w_kc_full.stride()[1:], attn.w_kc.stride()[1:])

    def test_the_gather_uses_all_to_all_not_all_gather(self):
        """The op is load-bearing, and only for memory.

        HCCL charges 2 x HCCL_BUFFSIZE to a communicator the first time it runs
        all_gather_into_tensor -- 2.00 GiB of KV pool at the 1000 MiB default on
        A3 tp16, 1.02 GiB at 500. The same group's first all_to_all_single takes
        0.00 GiB. Both ops compute the same gather, so nothing downstream would
        notice a switch back; the only symptom is 2 GiB of pool, which no test
        that checks results can catch.
        """

        attn, calls = self._attach(narrow_input=True, narrow_output=True)
        self.assertEqual(len(calls), 2, "the fake all-to-all must be what ran")
        for shape in calls:
            self.assertEqual(
                shape[0],
                self.TP,
                "an all-to-all gather sends one block per rank; a bare local "
                "slice here means the call was really an all-gather",
            )

    def test_the_return_leg_still_needs_dsa_cp_itself(self):
        """Independent of the inbound leg, but not of DSA-CP.

        Both legs are rearrangements of DSA-CP's own exchange. With DSA-CP off
        there is no exchange to move work across, and gathering the weights
        would be pure cost.
        """
        from sglang.srt.layers.attention.dsa import dsa_cp as dsa_cp_module

        with (
            mock.patch.object(
                dsa_cp_module, "_dsa_cp_narrow_a2a_output_flag", lambda: True
            ),
            mock.patch.object(dsa_cp_module, "dsa_cp_enabled", lambda: False),
        ):
            self.assertFalse(dsa_cp_module.dsa_cp_narrow_a2a_output_enabled())

        with (
            mock.patch.object(
                dsa_cp_module, "_dsa_cp_narrow_a2a_output_flag", lambda: True
            ),
            mock.patch.object(dsa_cp_module, "dsa_cp_enabled", lambda: True),
            mock.patch.object(
                dsa_cp_module, "dsa_cp_narrow_a2a_enabled", lambda: False
            ),
        ):
            self.assertTrue(
                dsa_cp_module.dsa_cp_narrow_a2a_output_enabled(),
                "the return leg must not need the inbound leg",
            )


if __name__ == "__main__":
    unittest.main()
