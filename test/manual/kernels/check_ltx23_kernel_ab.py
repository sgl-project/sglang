"""Cold-L2 timings of LTX call sites changed by the cleanup, with byte checks."""

import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time


REVISIONS = {
    "A": "1e490772e512317fab95608cbc9fb127776ae28e",
    "B": "a42c97419b70512264442893ec03b46477af45ef",
}


def child(expected_root):
    import torch
    import triton
    import sglang
    from sglang.kernels.ops.diffusion import residual_gate_add
    from sglang.kernels.ops.diffusion.sites.ltx2_rmsnorm_modulate_site import (
        mark_ltx2_rms_norm_modulate_site,
        mount_ltx2_rms_norm_modulate,
    )
    from sglang.multimodal_gen.runtime.layers.layernorm import RMSNormNoWeight
    from sglang.multimodal_gen.runtime.models.dits import ltx_2

    assert Path(sglang.__file__).resolve().is_relative_to(Path(expected_root).resolve())
    torch.manual_seed(20261004)
    props = torch.cuda.get_device_properties(0)
    eviction = torch.zeros(props.L2_cache_size * 5, device="cuda", dtype=torch.uint8)
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    norm = RMSNormNoWeight()
    lossless = torch.nn.Module()
    fused = torch.nn.Module()
    mark_ltx2_rms_norm_modulate_site(fused)
    assert mount_ltx2_rms_norm_modulate(fused)

    def fingerprint(output):
        return {
            "shape": list(output.shape), "stride": list(output.stride()),
            "dtype": str(output.dtype),
            "sha256": hashlib.sha256(output.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest(),
        }

    def bench(name, fn):
        for _ in range(10):
            output = fn()
        torch.cuda.synchronize()
        expected = fingerprint(output)
        samples = {"gpu_us": [], "eager_us": []}
        for mode in samples:
            for _ in range(100):
                if mode == "gpu_us":
                    # Enqueue the whole timed interval while the GPU is busy.
                    torch.cuda._sleep(10_000_000)
                    eviction.add_(1)
                    start.record()
                    output = fn()
                    end.record()
                    end.synchronize()
                    elapsed = start.elapsed_time(end) * 1000
                else:
                    eviction.add_(1)
                    torch.cuda.synchronize()
                    begin = time.perf_counter_ns()
                    output = fn()
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter_ns() - begin) / 1000
                samples[mode].append(elapsed)
            assert fingerprint(output) == expected
        row = {"name": name, "output": expected, "samples": samples,
               "medians": {k: statistics.median(v) for k, v in samples.items()},
               "modulate_verified": ltx_2._LTX2_MODULATE.verified,
               "modulate_disabled": ltx_2._LTX2_MODULATE.disabled}
        print("KERNEL_SAMPLE", json.dumps(row), flush=True)
        return row

    result = {"torch": torch.__version__, "triton": triton.__version__,
              "source": sglang.__file__, "gpu": props.name,
              "l2_bytes": props.L2_cache_size, "eviction_bytes": eviction.numel(),
              "method": "Each invocation follows read/write of 5x L2. Eviction is outside GPU and eager timing. 10 warmups, 100 samples per timing method. No control subtraction.",
              "cases": []}
    with torch.inference_mode():
        empty = torch.zeros(1, device="cuda")
        result["empty_control"] = bench("empty-control", lambda: empty)
        for batch, seq in [(1, 128), (1, 2048), (1, 8192), (2, 512)]:
            for hidden in (2048, 4096):
                def rand(shape):
                    return torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                x = rand((batch, seq, hidden))
                update = rand(x.shape)
                scale, shift, gate = (rand((batch, 1, hidden)) for _ in range(3))
                full_gate = rand(x.shape)
                calls = {
                    "model_modulate": lambda: ltx_2._ltx2_modulate(x, scale, shift),
                    "model_rms_lossless": lambda: ltx_2._ltx2_rms_norm_modulate(lossless, norm, x, scale, shift, 1e-6),
                    "model_rms_quality_high": lambda: ltx_2._ltx2_rms_norm_modulate(fused, norm, x, scale, shift, 1e-6),
                    "residual_row_gate": lambda: residual_gate_add(x, update, gate),
                    "residual_full_gate": lambda: residual_gate_add(x, update, full_gate),
                }
                for name, fn in calls.items():
                    result["cases"].append(bench(f"{name}/{batch}/{seq}/{hidden}", fn))
    print("KERNEL_RESULT", json.dumps(result), flush=True)


def main():
    if "--child" in sys.argv:
        child(sys.argv[sys.argv.index("--child") + 1])
        return
    root = Path(__file__).resolve().parents[3]
    task = Path(tempfile.mkdtemp(prefix="ltx23-kernel-ab-"))
    checkouts = {}
    results = []
    try:
        for variant, revision in REVISIONS.items():
            subprocess.run(["git", "fetch", "--depth=1", "origin", revision], cwd=root, check=True)
            checkout = task / variant
            subprocess.run(["git", "worktree", "add", "--detach", str(checkout), revision], cwd=root, check=True)
            checkouts[variant] = checkout
        for index, variant in enumerate("ABBAABBA"):
            checkout = checkouts[variant]
            env = dict(os.environ, PYTHONPATH=str(checkout / "python") + os.pathsep + os.environ.get("PYTHONPATH", ""))
            proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", str(checkout)],
                                  cwd=task, env=env, text=True, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, timeout=900)
            (task / f"{index}-{variant}.log").write_text(proc.stdout)
            print(proc.stdout, flush=True)
            proc.check_returncode()
            result = json.loads(next(line.split("KERNEL_RESULT ", 1)[1] for line in proc.stdout.splitlines() if line.startswith("KERNEL_RESULT ")))
            results.append(dict(index=index, variant=variant, revision=REVISIONS[variant], **result))
        reference = {case["name"]: case["output"] for case in results[0]["cases"]}
        for result in results:
            assert {case["name"]: case["output"] for case in result["cases"]} == reference
        print("KERNEL_AB_REPORT", json.dumps(results), flush=True)
        print("Byte equality passed. Timings are measurements, not a replacement for the model CI performance gate.", flush=True)
    finally:
        for checkout in checkouts.values():
            subprocess.run(["git", "worktree", "remove", str(checkout)], cwd=root, check=False)


if __name__ == "__main__":
    main()
