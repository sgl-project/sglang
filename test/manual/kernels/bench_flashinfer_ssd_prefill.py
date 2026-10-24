"""Benchmark checkpoint-preserving packed SSD adapters, not just the main kernel."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "registered/kernels/ops/mamba")
)
from test_flashinfer_ssd_prefill import SSDCase, check_error


def milliseconds(fn):
    for _ in range(5):
        fn()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    torch.cuda.synchronize()
    begin.record()
    for _ in range(20):
        fn()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) / 20


@torch.inference_mode()
def scalar_uniform_final(case):
    """Independent FP32 recurrence, batched across equal-length requests."""
    length = case.lengths[0]
    batch = len(case.lengths)
    state = case.initial.float()
    x = case.x.view(batch, length, case.heads, 64)
    dt = case.dt.view(batch, length, case.heads)
    B = case.B.reshape(batch, length, case.groups, 128)
    for token in range(length):
        delta = torch.nn.functional.softplus(dt[:, token].float() + case.bias)
        projection = (
            B[:, token].float().repeat_interleave(case.heads // case.groups, dim=1)
        )
        state = (
            state * torch.exp(delta * case.A)[:, :, None, None]
            + x[:, token].float()[:, :, :, None]
            * (delta[:, :, None] * projection)[:, :, None, :]
        )
    return state


@torch.inference_mode()
def main(diagnostic=False):
    import flashinfer

    print(
        json.dumps(
            dict(
                torch=torch.__version__,
                flashinfer=flashinfer.__version__,
                gpu=torch.cuda.get_device_name(),
            )
        ),
        flush=True,
    )
    for total, sequences in [
        (4096, 1),
        (4096, 16),
        (16384, 16),
        (32768, 16),
        (32768, 64),
    ]:
        lengths = [total // sequences] * sequences
        case = SSDCase(lengths, lengths, heads=128, groups=8)

        def triton():
            _, final = case.reference(checkpoints=False)
            case.pool[case.slot_tensor] = final
            return final

        final = case.run().clone()
        checkpoints = case.pool[case.slot_tensor].clone()
        ref_final = triton()
        passed = True
        for name, actual, expected in [
            ("bench_output", case.out, case.ref_out),
            ("bench_final", final, ref_final),
            ("bench_checkpoint", checkpoints, final),
        ]:
            try:
                check_error(name, actual, expected)
            except AssertionError:
                if not diagnostic:
                    raise
                passed = False
        if diagnostic and total == 4096 and sequences == 16:
            fp32_final = scalar_uniform_final(case)
            for name, actual in [
                ("flashinfer_vs_fp32_final", final),
                ("triton_vs_fp32_final", ref_final),
            ]:
                try:
                    check_error(name, actual, fp32_final)
                except AssertionError:
                    print(json.dumps(dict(diagnostic_failure=name)), flush=True)
        reference_ms, candidate_ms = milliseconds(triton), milliseconds(case.run)
        print(
            json.dumps(
                dict(
                    total=total,
                    sequences=sequences,
                    triton_ms=reference_ms,
                    flashinfer_ms=candidate_ms,
                    speedup=reference_ms / candidate_ms,
                    correctness_pass=passed,
                    diagnostic_only=diagnostic,
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Report failed numerical bounds and continue timings; never a qualification pass.",
    )
    main(parser.parse_args().diagnostic)
