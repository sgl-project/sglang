"""
FA3 parity test for `prepare_context_parallel_metadata`.

Drives the real function and feeds its `kv_len_prev/next_tensor` into FA3
via `flash_attn_with_kvcache`. Compares per-rank CP output against a
full-sequence FA3 reference computed over the unpadded `(prefix + extend)`
KV. Any discrepancy indicates the metadata function emitted wrong
`cache_seqlens` for at least one rank.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.utils.cp_utils import prepare_context_parallel_metadata
from sglang.srt.utils.common import ceil_align
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, suite="stage-b-test-1-gpu-large")

_NSA_UTILS = "sglang.srt.layers.attention.nsa.utils"
_DEVICE = "cuda"
_DTYPE = torch.bfloat16
_HEAD_NUM = 8
_HEAD_DIM = 128
_SCALE = _HEAD_DIM**-0.5


class TestCPPrefixLenFA3Parity(CustomTestCase):
    """Per-rank FA3 output under CP must match a full-sequence reference."""

    def _run_parity(self, prefix_len: int, extend_len: int, cp_size: int):
        from sgl_kernel.flash_attn import flash_attn_with_kvcache

        torch.manual_seed(extend_len * 1_000_003 + prefix_len * 101 + cp_size)

        padded_extend = ceil_align(extend_len, cp_size)
        pad = padded_extend - extend_len
        self.assertGreaterEqual(
            padded_extend,
            2 * cp_size,
            "runtime `can_cp_split` would skip this case; pick a larger extend",
        )

        # Reference: one full-sequence FA3 call over the unpadded KV.
        q_full = torch.randn(
            extend_len, _HEAD_NUM, _HEAD_DIM, device=_DEVICE, dtype=_DTYPE
        )
        k_full = torch.randn(
            prefix_len + extend_len, _HEAD_NUM, _HEAD_DIM, device=_DEVICE, dtype=_DTYPE
        )
        v_full = torch.randn(
            prefix_len + extend_len, _HEAD_NUM, _HEAD_DIM, device=_DEVICE, dtype=_DTYPE
        )
        ref = flash_attn_with_kvcache(
            q=q_full.unsqueeze(0),
            k_cache=k_full.unsqueeze(0),
            v_cache=v_full.unsqueeze(0),
            cache_seqlens=torch.tensor(
                [k_full.shape[0]], dtype=torch.int32, device=_DEVICE
            ),
            softmax_scale=_SCALE,
            causal=True,
        ).squeeze(0)

        # CP path sees tensors padded to `ceil_align(extend, cp_size)`,
        # matching what `prepare_mlp_sync_batch` does in production.
        zeros = torch.zeros(pad, _HEAD_NUM, _HEAD_DIM, device=_DEVICE, dtype=_DTYPE)
        q_padded = torch.cat([q_full, zeros], dim=0)
        k_padded = torch.cat([k_full, zeros], dim=0)
        v_padded = torch.cat([v_full, zeros], dim=0)

        seqs_len = [prefix_len + extend_len]
        extend_lens = [extend_len]

        def _call_meta(rank: int):
            return prepare_context_parallel_metadata(
                padded_extend, rank, cp_size, seqs_len, extend_lens=extend_lens
            )

        # Exercise the non-NSA branch; the NSA branch uses a separate
        # `prefix_len` pathway re-added by `_get_topk_ragged_with_cp`.
        with (
            patch(f"{_NSA_UTILS}.is_nsa_enable_prefill_cp", return_value=False),
            patch(
                f"{_NSA_UTILS}.is_nsa_prefill_cp_round_robin_split",
                return_value=False,
            ),
        ):
            meta0 = _call_meta(0)
            cp_segment_num = 2 * cp_size
            blocks_q = list(torch.split(q_padded, meta0.split_list, dim=0))
            outs = [None] * cp_segment_num

            for rank in range(cp_size):
                meta = meta0 if rank == 0 else _call_meta(rank)
                for idx, cs_tensor in (
                    (rank, meta.kv_len_prev_tensor),
                    (cp_size * 2 - rank - 1, meta.kv_len_next_tensor),
                ):
                    if meta0.split_list[idx] == 0:
                        outs[idx] = torch.empty(
                            0, _HEAD_NUM, _HEAD_DIM, device=_DEVICE, dtype=_DTYPE
                        )
                        continue
                    outs[idx] = flash_attn_with_kvcache(
                        q=blocks_q[idx].unsqueeze(0),
                        k_cache=k_padded.unsqueeze(0),
                        v_cache=v_padded.unsqueeze(0),
                        cache_seqlens=cs_tensor,
                        softmax_scale=_SCALE,
                        causal=True,
                    ).squeeze(0)

        cp_out = torch.cat(outs, dim=0)
        err = (cp_out[:extend_len].float() - ref.float()).abs().max().item()

        self.assertLess(
            err,
            1e-2,
            f"CP output diverges from full-sequence FA3 reference by "
            f"max_err={err:.5f} "
            f"(prefix_len={prefix_len}, extend_len={extend_len}, "
            f"cp_size={cp_size}, pad={pad})",
        )

    def test_cp2_prefix1_extend3(self):
        """cp_size=2, prefix_len=1, extend_len=3 (pad=1)."""
        self._run_parity(prefix_len=1, extend_len=3, cp_size=2)

    def test_cp4_prefix1_extend7(self):
        """cp_size=4, prefix_len=1, extend_len=7 (pad=1)."""
        self._run_parity(prefix_len=1, extend_len=7, cp_size=4)

    def test_cp8_prefix1_extend17(self):
        """cp_size=8, prefix_len=1, extend_len=17 (pad=7)."""
        self._run_parity(prefix_len=1, extend_len=17, cp_size=8)


if __name__ == "__main__":
    unittest.main()
