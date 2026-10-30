from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_amd_ci(est_time=10, stage="stage-b", runner_config="1-gpu-small-amd")

import unittest

import torch

from sglang.kernels.ops.speculative.simulated_accept import simulated_accept
from sglang.test.test_utils import CustomTestCase


def _torch_simulated_accept(
    accept_index, predict, num_correct_drafts, candidates, target, width, n, real
):
    """The torch ops the fused kernel replaces (generate_simulated_accept_index)."""
    bs = accept_index.shape[0]
    out = torch.full((bs, width), -1, dtype=torch.int32, device=accept_index.device)
    out[:, :n] = accept_index[:, 0].view(-1, 1) + torch.arange(
        n, device=accept_index.device
    )
    num_correct_drafts.fill_(n - 1)
    if not real:
        predict.fill_(100)
        return out
    if n > 1:
        predict[out[:, : n - 1].long()] = candidates[:, 1:n].to(predict.dtype)
    predict[out[:, n - 1].long()] = target[:, n - 1].to(predict.dtype)
    return out


@unittest.skipUnless(torch.cuda.is_available(), "a GPU is required for this test.")
class TestSimulatedAccept(CustomTestCase):
    def test_matches_torch(self):
        dev = torch.device("cuda")
        g = torch.Generator(device=dev).manual_seed(0)
        for bs in (1, 2, 5, 16):
            for width in (2, 4):
                for n in range(1, width + 1):
                    for real in (False, True):
                        with self.subTest(bs=bs, width=width, n=n, real=real):
                            accept_index = torch.full(
                                (bs, width), -1, dtype=torch.int32, device=dev
                            )
                            accept_index[:, 0] = torch.arange(
                                0, bs * width, width, dtype=torch.int32, device=dev
                            )
                            candidates = torch.randint(
                                0, 150000, (bs, width), generator=g, device=dev
                            )
                            target = torch.randint(
                                0, 150000, (bs, width), generator=g, device=dev
                            )
                            init = torch.randint(
                                0, 150000, (bs * width,), generator=g, device=dev
                            ).to(torch.int32)
                            p_ref, p_out = init.clone(), init.clone()
                            c_ref = torch.full((bs,), 7, dtype=torch.int32, device=dev)
                            c_out = c_ref.clone()
                            ref = _torch_simulated_accept(
                                accept_index,
                                p_ref,
                                c_ref,
                                candidates,
                                target,
                                width,
                                n,
                                real,
                            )
                            out = simulated_accept(
                                accept_indices=accept_index,
                                predict_tokens=p_out,
                                num_correct_drafts=c_out,
                                candidate_tokens=candidates,
                                target_tokens=target,
                                width=width,
                                num_accept_tokens=n,
                                real_tokens=real,
                            )
                            torch.testing.assert_close(out, ref, rtol=0, atol=0)
                            torch.testing.assert_close(p_out, p_ref, rtol=0, atol=0)
                            torch.testing.assert_close(c_out, c_ref, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
