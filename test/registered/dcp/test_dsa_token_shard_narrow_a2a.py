"""CPU unit test: the DSA token shard's narrow all-to-all is exact.

the DSA token shard swaps "my heads for every token" for "every head for my tokens" after
``q_b_proj`` and the absorb through ``w_kc``, so the wire carries the 512-wide
absorbed latent in and the 512-wide attention output back. Both are linear
functions of narrower tensors beside them: q is 256 wide out of ``q_b_proj``
and the head output is ``v_head_dim`` 256.
``SGLANG_NPU_ENABLE_DSA_TOKEN_SHARD_NARROW_A2A`` moves the inbound exchange to the
narrow side and ``..._OUTPUT`` the return leg; the two are independent.

**The claim this file pins:** deferring the absorb past the exchange changes
nothing, because ``npu_transpose_batchmatmul`` is a product per (token, head)
pair and the all-to-all only moves those pairs between ranks. That holds only
if each rank absorbs with the ``w_kc`` of the head it ENDS UP holding, not the
head it was assigned -- which is what the full weights buy, and what a
plausible implementation gets wrong. One test is the negative case, showing
that reusing the local slice after the swap really is wrong.

The rest pins that the loader gathers both weights when the flag is on and
neither when it is off, which no test of results can see -- both paths compute
the same answer -- and which costs 28 MB per layer.

Usage:
    python -m pytest test_dsa_token_shard_narrow_a2a.py -v
    python test_dsa_token_shard_narrow_a2a.py
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

    ``dsa_token_shard_attach_full_kv_b`` now gathers in the PHYSICAL layout and takes the
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


class TestTheGatherFollowsTheFlag(CustomTestCase):
    """What the loader gathers is invisible to any test of results.

    Both paths compute the same answer, so gathering the weights when the
    feature is off, or failing to when it is on, shows up only as KV pool --
    28 MB per layer, 2.13 GiB over 78 layers at tp16.
    """

    TP = 4
    HEADS, NOPE, LORA, VDIM = 2, 6, 5, 4

    def _attach(self, narrow):
        from sglang.srt.layers.attention.dsa import (
            dsa_token_shard as dsa_token_shard_module,
        )

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

        def _fake_all_gather(out, inp):
            calls.append(tuple(inp.shape))
            out.copy_(inp.repeat(self.TP, 1, 1))

        parallel = mock.Mock()
        parallel.attn_tp_size = self.TP
        with (
            mock.patch.object(
                dsa_token_shard_module,
                "dsa_token_shard_narrow_a2a_enabled",
                lambda: narrow,
            ),
            mock.patch.object(dsa_token_shard_module, "get_parallel", lambda: parallel),
            mock.patch(
                "sglang.srt.layers.dp_attention.attn_tp_all_gather_into_tensor",
                _fake_all_gather,
            ),
        ):
            dsa_token_shard_module.dsa_token_shard_attach_full_kv_b(attn)
        return attn, calls

    def test_on_gathers_both_weights(self):
        attn, calls = self._attach(narrow=True)
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            attn.w_kc_full.shape, (self.HEADS * self.TP, self.NOPE, self.LORA)
        )
        self.assertEqual(
            attn.w_vc_full.shape, (self.HEADS * self.TP, self.LORA, self.VDIM)
        )

    def test_off_gathers_nothing(self):
        attn, calls = self._attach(narrow=False)
        self.assertEqual(calls, [])
        self.assertIsNone(getattr(attn, "w_kc_full", None))
        self.assertIsNone(getattr(attn, "w_vc_full", None))

    def test_the_gathered_w_kc_keeps_the_loader_stride_pattern(self):
        attn, _ = self._attach(narrow=True)
        self.assertFalse(
            attn.w_kc_full.is_contiguous(),
            "the batched matmul is tuned for the transposed layout; a "
            "contiguous full copy means the gather rebuilt it and paid twice",
        )
        self.assertEqual(attn.w_kc_full.stride()[1:], attn.w_kc.stride()[1:])

    def test_it_still_needs_dsa_token_shard_itself(self):
        """The narrow exchange rearranges the DSA token shard's own exchange, so with the DSA token shard
        off there is nothing to rearrange and the weights would be pure cost."""
        from sglang.srt.layers.attention.dsa import (
            dsa_token_shard as dsa_token_shard_module,
        )

        with (
            mock.patch.object(
                dsa_token_shard_module, "_dsa_token_shard_narrow_a2a_flag", lambda: True
            ),
            mock.patch.object(
                dsa_token_shard_module, "dsa_token_shard_enabled", lambda: False
            ),
        ):
            self.assertFalse(
                dsa_token_shard_module.dsa_token_shard_narrow_a2a_enabled()
            )


if __name__ == "__main__":
    unittest.main()
