"""SGLANG_ROCM_NO_EXTEND_SYNC: host-side chunk indices must match the
device-derived ones, and chunk KDA must give bit-identical outputs and states
with the hint on or off."""

import unittest

import torch

from sglang.kernels.ops.attention.fla.index import (
    cpu_seqlens_hint,
    prepare_chunk_indices,
)
from sglang.kernels.ops.attention.fla.kda import chunk_kda
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, stage="jit-kernel-unit", runner_config="amd")

CHUNK = 64
# heterogeneous lengths around the chunk boundary
LENS = [1, 63, 64, 65, 200, 7, 128]


def _cu(lens):
    # fresh tensor every call: prepare_chunk_indices caches by tensor identity
    out = torch.zeros(len(lens) + 1, dtype=torch.int32)
    out[1:] = torch.tensor(lens).cumsum(0)
    return out.cuda()


class TestKDAExtendSyncHint(CustomTestCase):
    def test_chunk_indices_match(self):
        for lens in (LENS, [1], [CHUNK] * 4, [5, 0, 70], [0, 3]):
            with self.subTest(lens=lens):
                ref = prepare_chunk_indices(_cu(lens), CHUNK)
                cu = _cu(lens)
                with cpu_seqlens_hint(cu, lens):
                    hinted = prepare_chunk_indices(cu, CHUNK)
                torch.cuda.synchronize()
                torch.testing.assert_close(hinted, ref, rtol=0, atol=0)
                self.assertEqual(hinted.dtype, ref.dtype)

    def test_other_tensor_ignores_hint(self):
        # e.g. a two-batch-overlap half: its own cu_seqlens, not the hinted one
        cu = _cu(LENS)
        half = _cu(LENS[:3])
        with cpu_seqlens_hint(cu, [999] * len(LENS)):
            got = prepare_chunk_indices(half, CHUNK)
        torch.testing.assert_close(
            got, prepare_chunk_indices(_cu(LENS[:3]), CHUNK), rtol=0, atol=0
        )

    def _run(self, hint, lens, seed=0):
        torch.manual_seed(seed)
        H, D, T = 12, 128, sum(lens)
        dev = "cuda"
        q = torch.randn(1, T, H, D, device=dev, dtype=torch.bfloat16)
        k = torch.randn(1, T, H, D, device=dev, dtype=torch.bfloat16)
        v = torch.randn(1, T, H, D, device=dev, dtype=torch.bfloat16)
        g = -torch.rand(1, T, H, D, device=dev, dtype=torch.float32) * 0.1
        beta = torch.rand(1, T, H, device=dev, dtype=torch.bfloat16)
        n = len(lens)
        state = torch.randn(n + 3, H, D, D, device=dev, dtype=torch.float32) * 0.1
        # permuted physical slots
        slots = torch.randperm(n + 3, device=dev)[:n].to(torch.int32)
        cu = _cu(lens)
        ctx = cpu_seqlens_hint(cu, lens) if hint else torch.no_grad()
        with ctx:
            o = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=state,
                initial_state_indices=slots,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu,
            )
        torch.cuda.synchronize()
        return o, state

    def test_chunk_kda_parity(self):
        for lens in (LENS, [300], [1, 1, 1]):
            with self.subTest(lens=lens):
                o_ref, s_ref = self._run(False, lens)
                o_hint, s_hint = self._run(True, lens)
                torch.testing.assert_close(o_hint, o_ref, rtol=0, atol=0)
                torch.testing.assert_close(s_hint, s_ref, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
