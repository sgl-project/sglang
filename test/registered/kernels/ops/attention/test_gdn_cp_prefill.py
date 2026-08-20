"""The in-tree SM100 CP GDN prefill must agree with the installed non-CP kernel.

The vendored CP closure is dispatched by a routing layer copied from the pinned
FlashInfer release, so a re-vendor that changes a call shape (dtype, kwarg, or
state layout) shows up here rather than at serving time.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base", runner_config="1-gpu-b200")

HEADS = 32
HEAD_DIM = 128


def _inputs(seqlens, device):
    total = sum(seqlens)
    q = torch.randn(total, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    k = torch.nn.functional.normalize(
        torch.randn(total, HEADS, HEAD_DIM, device=device), p=2.0, dim=-1
    ).to(torch.bfloat16)
    v = torch.randn(total, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    alpha = torch.rand(total, HEADS, dtype=torch.float32, device=device)
    beta = torch.rand(total, HEADS, dtype=torch.float32, device=device)
    cu_seqlens = torch.tensor(
        [0] + torch.tensor(seqlens).cumsum(0).tolist(),
        dtype=torch.int64,
        device=device,
    )
    return q, k, v, alpha, beta, cu_seqlens


class TestGdnCpPrefill(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            raise unittest.SkipTest("SM100/SM103 GPU required")
        from sglang.kernels.ops.attention.gdn_cp_prefill.gdn_prefill import (
            chunk_gated_delta_rule,
            cp_delta_rule_dsl_sm100,
        )

        if cp_delta_rule_dsl_sm100 is None:
            raise unittest.SkipTest("in-tree SM100 CP kernel unavailable")
        cls.chunk_gated_delta_rule = staticmethod(chunk_gated_delta_rule)
        torch.manual_seed(0)

    def _run(self, seqlens, use_cp, args, checkpoints, pooled):
        q, k, v, alpha, beta, cu_seqlens = args
        total, num_seqs = sum(seqlens), len(seqlens)
        device = q.device
        output = torch.zeros(
            total, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        kwargs = {}
        if pooled:
            pool = torch.zeros(
                64, HEADS, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=device
            )
            slots = torch.tensor(
                [7, 3, 11][:num_seqs], dtype=torch.int32, device=device
            )
            kwargs.update(output_state=pool, state_indices=slots)
        else:
            state = torch.zeros(
                num_seqs, HEADS, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device=device
            )
            kwargs.update(output_state=state)
        if checkpoints:
            every = 512
            starts = torch.tensor(
                [0] + torch.tensor([x // every for x in seqlens]).cumsum(0).tolist(),
                dtype=torch.int64,
                device=device,
            )
            kwargs.update(
                state_checkpoints=torch.zeros(
                    int(starts[-1]),
                    HEADS,
                    HEAD_DIM,
                    HEAD_DIM,
                    dtype=torch.float32,
                    device=device,
                ),
                checkpoint_cu_starts=starts,
                checkpoint_every_n_tokens=every,
            )
        self.chunk_gated_delta_rule(
            q,
            k,
            v,
            alpha,
            beta,
            None,
            None,
            True,
            cu_seqlens,
            True,
            output=output,
            use_cp=use_cp,
            **kwargs,
        )
        final_state = (
            kwargs["output_state"][kwargs["state_indices"].long()]
            if pooled
            else kwargs["output_state"]
        )
        return output.float(), final_state.float()

    def test_cp_matches_non_cp(self):
        cases = [
            ("single", [8192], False, False),
            ("varlen", [8192, 2048, 5000], False, False),
            ("checkpointing", [8192, 4096], True, False),
            ("state_indices_pool", [8192, 4096], False, True),
            ("pool_and_checkpointing", [8192, 4096], True, True),
        ]
        for name, seqlens, checkpoints, pooled in cases:
            with self.subTest(name):
                args = _inputs(seqlens, "cuda")
                ref_o, ref_s = self._run(seqlens, False, args, checkpoints, pooled)
                cp_o, cp_s = self._run(seqlens, True, args, checkpoints, pooled)
                self.assertFalse(cp_o.isnan().any())
                self.assertLess(
                    ((cp_o - ref_o).abs().max() / ref_o.abs().max()).item(), 0.02
                )
                self.assertLess(
                    ((cp_s - ref_s).abs().max() / ref_s.abs().max()).item(), 0.02
                )

    def test_auto_routing_runs_both_ways(self):
        for name, seqlens in (("cp_favoured", [8192]), ("non_cp", [1024] * 16)):
            with self.subTest(name):
                args = _inputs(seqlens, "cuda")
                out, _ = self._run(seqlens, "auto", args, False, False)
                self.assertFalse(out.isnan().any())


if __name__ == "__main__":
    unittest.main()
