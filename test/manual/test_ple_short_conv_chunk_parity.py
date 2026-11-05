"""Fail-closed native BF16 PLE chunk parity and allocation peak check; no model launch."""

import argparse
import hashlib
import importlib.util
import json
import sys
import traceback
import unittest
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument(
        "--widths", type=int, nargs="+", default=[8192, 16384, 24576, 32768]
    )
    parser.add_argument("--channels", type=int, default=10240)
    parser.add_argument("--dilation", type=int, default=3)
    parser.add_argument("--kernel", type=int, default=4)
    args = parser.parse_args()
    if args.chunk_size <= 0 or min(args.widths) <= 0:
        parser.error("chunk size and widths must be positive")
    if min(args.channels, args.dilation, args.kernel) <= 0:
        parser.error("channels, dilation and kernel must be positive")
    root = args.source_root.resolve()
    sys.path.insert(0, str(root / "python"))
    report = {
        "status": "NOT_RUN",
        "cases": [],
        "source_root": str(root),
        "driver_sha256": sha(Path(__file__).resolve()),
    }
    rc = 1
    try:
        import torch
        import torch.nn.functional as F

        from sglang.srt.layers.ple_short_conv import chunked_packed_short_conv

        report["source_leaves"] = {
            p: sha(root / p)
            for p in (
                "python/sglang/srt/models/qwen4_exp.py",
                "python/sglang/srt/layers/ple_short_conv.py",
                "python/sglang/srt/environ.py",
                "test/registered/unit/models/test_ple_short_conv_chunk.py",
            )
        }
        if not torch.cuda.is_available():
            report["status"] = "REFUSED_NO_CUDA_NOT_PARITY_PASS"
            return 2
        report["environment"] = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "device": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "dtype": "torch.bfloat16",
        }
        report["configuration"] = {
            "channels": args.channels,
            "kernel": args.kernel,
            "dilation": args.dilation,
            "chunk_size": args.chunk_size,
            "widths": args.widths,
            "cudnn_flags_changed": False,
        }

        def peak(fn):
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            output = fn()
            torch.cuda.synchronize()
            return output, torch.cuda.max_memory_allocated() - baseline

        with torch.inference_mode():
            generator = torch.Generator(device="cuda").manual_seed(931)
            for width in args.widths:
                for lengths in ([width], [width, 257, 1, 0]):
                    halo = (args.kernel - 1) * args.dilation
                    inputs = torch.randn(
                        len(lengths),
                        args.channels,
                        width + halo,
                        device="cuda",
                        dtype=torch.bfloat16,
                        generator=generator,
                    )
                    weights = torch.randn(
                        args.channels,
                        1,
                        args.kernel,
                        device="cuda",
                        dtype=torch.bfloat16,
                        generator=generator,
                    )

                    def stock():
                        dense = F.conv1d(
                            inputs,
                            weights,
                            bias=None,
                            dilation=args.dilation,
                            groups=args.channels,
                        ).transpose(1, 2)
                        return torch.cat([dense[i, :n] for i, n in enumerate(lengths)])

                    def candidate():
                        return chunked_packed_short_conv(
                            inputs,
                            weights,
                            lengths,
                            dilation=args.dilation,
                            chunk_size=args.chunk_size,
                        )

                    # Warm both algorithm/allocator paths without changing cuDNN policy.
                    del_ref = stock()
                    del_got = candidate()
                    del del_ref, del_got
                    reference, ref_peak = peak(stock)
                    got, got_peak = peak(candidate)
                    mismatch = int(
                        (reference.view(torch.int16) != got.view(torch.int16)).sum()
                    )
                    activated_ref, activated_got = F.silu(reference), F.silu(got)
                    silu_mismatch = int(
                        (
                            activated_ref.view(torch.int16)
                            != activated_got.view(torch.int16)
                        ).sum()
                    )
                    finite = bool(
                        torch.isfinite(reference).all() and torch.isfinite(got).all()
                    )
                    record = {
                        "lengths": lengths,
                        "channels": args.channels,
                        "raw_bf16_bit_mismatches": mismatch,
                        "silu_bf16_bit_mismatches": silu_mismatch,
                        "both_finite": finite,
                        "stock_incremental_peak_allocated_bytes": ref_peak,
                        "chunked_incremental_peak_allocated_bytes": got_peak,
                        "peak_reduced": got_peak < ref_peak,
                        "input_shape": list(inputs.shape),
                    }
                    report["cases"].append(record)
                    if mismatch or silu_mismatch or not finite:
                        raise AssertionError("native BF16 bitwise parity failed")
                    if not got_peak < ref_peak:
                        raise AssertionError(
                            "measured temporary allocation peak did not fall"
                        )
                    del reference, got, activated_ref, activated_got
                    inputs = weights = None

        # Reuse the CPU test's actual-model-method assertions with real CUDA tensors.
        fixture_path = root / "test/registered/unit/models/test_ple_short_conv_chunk.py"
        spec = importlib.util.spec_from_file_location(
            "ple_chunk_cuda_fixture", fixture_path
        )
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        fixture.TestPleShortConvIntegration.device = "cuda"
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(
            fixture.TestPleShortConvIntegration
        )
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        report["actual_method"] = {
            "tests": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped),
        }
        if not result.wasSuccessful() or result.skipped:
            raise AssertionError("actual PLE method CUDA parity/state tests failed")
        report["status"] = "PASS_BF16_BITS_AND_PEAK_NOT_GSM_OR_THROUGHPUT"
        rc = 0
    except Exception:
        report["status"] = "FAIL"
        report["traceback"] = traceback.format_exc()
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
