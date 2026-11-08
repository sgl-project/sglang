"""The Mamba replay state-indices kernel matches its reference chain for every replay batch size and
compiles one variant for all batch sizes up to the 64-row floor."""

import unittest

import torch

from sglang.kernels.ops.mamba import mamba_state_indices_triton as msi
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def compiled_variants(kernel):
    caches = getattr(kernel, "device_caches", None)
    if caches is None:
        return None
    return sum(len(v[0]) if isinstance(v, tuple) else len(v) for v in caches.values())


class TestMambaReplayStateIndices(CustomTestCase):
    def _check(self, total_bs, valid_bs, v2p):
        g = torch.Generator(device="cpu").manual_seed(total_bs * 1000 + valid_bs)
        req = torch.randint(0, 128, (total_bs,), generator=g).cuda().long()
        mapping = torch.randperm(128, generator=g).cuda().int()
        table = torch.randperm(129, generator=g).cuda().long() if v2p else None
        out = torch.full((total_bs,), -9, device="cuda", dtype=torch.int32)
        expected_idx = mapping[req[:valid_bs]].long()
        if v2p:
            expected_idx = table[expected_idx]
        expected = torch.full((total_bs,), -1, device="cuda", dtype=torch.int32)
        expected[:valid_bs] = expected_idx.int()
        expected_req = req.clone()
        expected_req[valid_bs:] = 0
        msi.fused_replay_state_indices(
            req_pool_indices=req,
            mamba_index_mapping=mapping,
            out_state_indices=out,
            valid_bs=valid_bs,
            total_bs=total_bs,
            v2p=table,
        )
        torch.testing.assert_close(out, expected)
        torch.testing.assert_close(req, expected_req)

    def test_matches_reference_and_compiles_once_up_to_floor(self):
        sizes = [1, 2, 3, 5, 8, 13, 16, 17, 31, 32, 33, 48, 63, 64]
        for v2p in (False, True):
            self._check(sizes[0], sizes[0], v2p)
        before = compiled_variants(msi._fused_replay_state_indices_kernel)
        for v2p in (False, True):
            for total_bs in sizes:
                for valid_bs in sorted({1, max(1, total_bs // 2), total_bs}):
                    self._check(total_bs, valid_bs, v2p)
        after = compiled_variants(msi._fused_replay_state_indices_kernel)
        if before is None:
            self.skipTest("Triton JITFunction exposes no device_caches")
        # The batch size no longer specializes up to the 64-row floor (it was value-specialized before).
        self.assertEqual(after, before)

    def test_matches_reference_above_floor(self):
        for total_bs in [65, 100, 128, 129, 200, 256, 300]:
            self._check(total_bs, total_bs - 3, False)


if __name__ == "__main__":
    unittest.main()
