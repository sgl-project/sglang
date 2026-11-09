"""ar_agg / ag_agg match the separate collective + attn_res_smallm_hip path
(TP8, H=7168): near-cancelling candidate rows (T=1, NVB=1) and random rows."""

import os
import subprocess
import sys
import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, suite="stage-c-test-large-8-gpu-amd")

_H, _TP = 7168, 8


def _pattern(lead: float) -> torch.Tensor:
    row = torch.full((1, _H), 2.0**-9)
    row[:, 0::2] = lead
    return row


def _check(ca, rank, prefix, bank, cw, ow, nvb, eps, atol, rtol):
    from sglang.kernels.ops.attention.attn_res_smallm_hip import attn_res_smallm_hip
    from sglang.kernels.ops.communication import k3_ar_agg_hip

    T = prefix.shape[0]
    assert k3_ar_agg_hip.usable(ca, T, _H, nvb)
    ref = torch.empty_like(prefix)
    attn_res_smallm_hip(prefix, bank, cw, ow, ref, nvb, eps, eps)

    # AR: the sum of 8 exact eighths is exact
    out, prefix_out = torch.empty_like(prefix), torch.empty_like(prefix)
    k3_ar_agg_hip.ar_agg(
        ca,
        (prefix / _TP).contiguous(),
        None,
        bank,
        cw,
        ow,
        nvb,
        eps,
        eps,
        out=out,
        prefix_out=prefix_out,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(prefix_out, prefix, atol=0, rtol=0)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol, msg="ar_agg")

    # AG: this rank's slice, plus zeros
    ld = _H // _TP
    y = torch.empty(T, ld, dtype=prefix.dtype, device=prefix.device)
    y.copy_(prefix[:, rank * ld : (rank + 1) * ld])
    out, prefix_out = torch.empty_like(prefix), torch.empty_like(prefix)
    k3_ar_agg_hip.ag_agg(
        ca,
        y,
        torch.zeros_like(prefix),
        None,
        bank,
        cw,
        ow,
        nvb,
        eps,
        eps,
        out=out,
        prefix_out=prefix_out,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(prefix_out, prefix, atol=0, rtol=0)
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol, msg="ag_agg")
    return ref


def _worker():
    import torch.distributed as dist
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", device_id=dev)
    ca = CustomAllreduce(dist.new_group(backend="gloo"), dev, max_size=64 << 20)
    bf16, eps = torch.bfloat16, 1e-6

    # near-cancelling: mix = (0, 2^-9, 0, 2^-9, ...)
    bank = torch.zeros(1, 2, _H, dtype=bf16, device=dev)
    bank[:, 0] = _pattern(16.0).to(dev, bf16)
    prefix = _pattern(-16.0).to(dev, bf16)
    cw = torch.zeros(_H, dtype=torch.float32, device=dev)
    ow = torch.ones(_H, dtype=bf16, device=dev)
    ref = _check(ca, rank, prefix, bank, cw, ow, 1, eps, 0, 8e-3)
    expected = torch.zeros_like(prefix)
    expected[:, 1::2] = 2.0**-9 / (2.0**-19 + eps) ** 0.5
    torch.testing.assert_close(ref, expected, atol=0, rtol=8e-3)

    # random rows, every rank the same inputs
    g = torch.Generator(device=dev).manual_seed(0)
    T, nvb = 8, 7
    bank = torch.randn(T, 8, _H, generator=g, device=dev).to(bf16)
    prefix = torch.randn(T, _H, generator=g, device=dev).to(bf16)
    cw = torch.randn(_H, generator=g, device=dev) * _H**-0.5
    ow = (1 + 0.1 * torch.randn(_H, generator=g, device=dev)).to(bf16)
    _check(ca, rank, prefix, bank, cw, ow, nvb, eps, 2e-2, 2e-2)

    dist.barrier()
    dist.destroy_process_group()


class TestK3ArAggGramNorm(unittest.TestCase):
    def test_tp8(self):
        if not torch.cuda.is_available() or torch.cuda.device_count() < _TP:
            self.skipTest("requires 8 GPUs")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={_TP}",
                __file__,
                "--worker",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=900,
        )
        errors = [
            line
            for line in result.stdout.splitlines()
            if "Error" in line or "Mismatch" in line or "Greatest" in line
        ]
        self.assertEqual(result.returncode, 0, "\n".join(errors[:40]))


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _worker()
    else:
        unittest.main()
