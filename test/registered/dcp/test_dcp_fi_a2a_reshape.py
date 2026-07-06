# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CPU unit test for the fi_a2a (FlashInfer MNNVL) pack/unpack index math.

Simulates the N-rank all-to-all in-process (GPU-free) to prove that the
``_dcp_fi_a2a_lse_reduce`` reshapes (partial_o ``[B,H_pr,cp,D]`` send /
``o_out`` permute-back) place heads on the right rank — the top correctness
risk of the FlashInfer backend. It pins the fi_a2a layout to the NCCL a2a
layout (``dcp_a2a_lse_reduce``) and to the ground-truth per-rank DCP combine,
all three computed from the same random partials.

The FlashInfer MNNVL kernel itself is exercised end-to-end on GB200; here we
only check the surrounding layout, which is pure ``torch.float32`` and runs on
CPU in CI.

Usage:
    python -m pytest test_dcp_fi_a2a_reshape.py -v
    python test_dcp_fi_a2a_reshape.py
"""

import unittest

import torch

from sglang.srt.layers.dcp.kernels import _lse_weighted_combine_cpu
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestFiA2AReshapeEquivalence(unittest.TestCase):
    """fi_a2a pack/unpack index math == NCCL a2a == ground-truth DCP reduction."""

    def _paths(self, N, B, H_per_rank, D, is_base_e):
        torch.manual_seed(0)
        H = N * H_per_rank
        scale = 5.0 if not is_base_e else 1.0
        # full_out[r] / full_lse[r]: rank r's partial attention over its KV shard
        # for ALL heads (one tensor per rank, all held in this process).
        full_out = [torch.randn(B, H, D, dtype=torch.float32) for _ in range(N)]
        full_lse = [torch.randn(B, H, dtype=torch.float32) * scale for _ in range(N)]

        gt, nccl, fi = [], [], []
        for r in range(N):
            sl = slice(r * H_per_rank, (r + 1) * H_per_rank)

            # Ground truth: combine over all ranks s at rank r's local heads.
            po = torch.stack([full_out[s][:, sl, :] for s in range(N)], dim=0)
            pl = torch.stack([full_lse[s][:, sl] for s in range(N)], dim=0)
            gt.append(_lse_weighted_combine_cpu(po, pl, is_lse_base_on_e=is_base_e))

            # NCCL a2a reshape (mirror dcp_a2a_lse_reduce):
            #   send[s] = full_out[s].view(B,N,H_pr,D).permute(1,0,2,3); after
            #   all_to_all, recv[src] = send[src][r].
            recv_o = torch.stack(
                [full_out[s].view(B, N, H_per_rank, D)[:, r] for s in range(N)], dim=0
            )
            recv_l = torch.stack(
                [full_lse[s].view(B, N, H_per_rank)[:, r] for s in range(N)], dim=0
            )
            nccl.append(
                _lse_weighted_combine_cpu(recv_o, recv_l, is_lse_base_on_e=is_base_e)
            )

            # fi_a2a reshape (mirror _dcp_fi_a2a_lse_reduce):
            #   partial_o[s] = full_out[s].view(B,N,H_pr,D).permute(0,2,1,3) ->
            #   [B,H_pr,N,D]; o_out[b,h,src,:] = partial_o[src][b,h,r,:]; then
            #   permute(2,0,1,3) -> [N,B,H_pr,D] for the combine.
            partial_o = [
                full_out[s].view(B, N, H_per_rank, D).permute(0, 2, 1, 3)
                for s in range(N)
            ]
            lse_v = [
                full_lse[s].view(B, N, H_per_rank).permute(0, 2, 1) for s in range(N)
            ]
            o_out = torch.stack([partial_o[s][:, :, r, :] for s in range(N)], dim=2)
            stats = torch.stack([lse_v[s][:, :, r] for s in range(N)], dim=2)
            fi_o = o_out.permute(2, 0, 1, 3)
            fi_l = stats.permute(2, 0, 1)
            fi.append(
                _lse_weighted_combine_cpu(fi_o, fi_l, is_lse_base_on_e=is_base_e)
            )

        return gt, nccl, fi

    def _check(self, N, is_base_e):
        gt, nccl, fi = self._paths(N, B=3, H_per_rank=5, D=64, is_base_e=is_base_e)
        for r in range(N):
            torch.testing.assert_close(nccl[r], gt[r], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(fi[r], gt[r], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(fi[r], nccl[r], atol=1e-5, rtol=1e-5)

    def test_n2_base_2(self):
        self._check(N=2, is_base_e=False)

    def test_n4_base_2(self):
        self._check(N=4, is_base_e=False)

    def test_n8_base_2(self):
        self._check(N=8, is_base_e=False)

    def test_n4_base_e(self):
        self._check(N=4, is_base_e=True)


if __name__ == "__main__":
    unittest.main()
