"""Benchmark fused FP8 kernels vs sequential baselines.

Subcommands:
  kernel      In-process kernel microbenchmark (triton.testing)
  one-batch   E2E latency via bench_one_batch (subprocess)
  serving     E2E throughput via bench_serving (subprocess)
  compare     Load JSONLs, print comparison table
  all         kernel -> one-batch -> serving -> compare
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# =============================================================================
# Constants
# =============================================================================

_DEFAULT_HIDDEN_SIZES = [4096, 5120, 8192]
_DEFAULT_INTERMEDIATE_SIZES = [14336, 27648]
_DEFAULT_BATCH_SIZES = [1, 4, 16, 64, 128, 256, 512, 1024]
_H100_HBM_BW_GBS = 3352


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass(frozen=True, slots=True)
class RepoConfig:
    name: str  # "branch" or "baseline"
    repo_path: str  # "/root/sglang" or "/root/sglang-baseline"
    python: str  # venv python path


@dataclass(frozen=True, slots=True)
class BenchConfig:
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    tp: int = 1
    batch_sizes: tuple[int, ...] = (1, 8, 32, 64, 128, 256)
    input_lens: tuple[int, ...] = (256, 512, 1024, 2048)
    output_lens: tuple[int, ...] = (16, 64, 256)
    num_prompts: int = 500
    random_input: int = 512
    random_output: int = 64
    request_rates: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0, 32.0, float("inf"))
    port: int = 30000


# =============================================================================
# GPU clock management
# =============================================================================


_GPU_CLOCKS_LOCKED = False


def _lock_gpu_clocks():
    """Lock GPU and memory clocks to max for reproducible benchmarks."""
    global _GPU_CLOCKS_LOCKED
    if _GPU_CLOCKS_LOCKED:
        return
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        print("[gpu] nvidia-smi not found, skipping clock lock")
        return
    try:
        # Query max clocks
        out = subprocess.check_output(
            [nvidia_smi, "--query-gpu=clocks.max.graphics,clocks.max.memory",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        # Take first GPU line only (multi-GPU returns one line per device)
        first_line = out.split("\n")[0].strip()
        gpu_max, mem_max = [int(x.strip()) for x in first_line.split(",")]
        subprocess.run([nvidia_smi, "-pm", "1"], check=True, capture_output=True)
        subprocess.run(
            [nvidia_smi, "-lgc", f"{gpu_max},{gpu_max}"],
            check=True, capture_output=True,
        )
        subprocess.run(
            [nvidia_smi, "-lmc", f"{mem_max},{mem_max}"],
            check=True, capture_output=True,
        )
        _GPU_CLOCKS_LOCKED = True
        print(f"[gpu] Clocks locked: GPU={gpu_max} MHz, MEM={mem_max} MHz")
    except Exception as e:
        print(f"[gpu] WARNING: Failed to lock clocks: {e}")


def _unlock_gpu_clocks():
    """Reset GPU clocks to default after benchmarking."""
    global _GPU_CLOCKS_LOCKED
    if not _GPU_CLOCKS_LOCKED:
        return
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return
    try:
        subprocess.run([nvidia_smi, "-rgc"], check=True, capture_output=True)
        subprocess.run([nvidia_smi, "-rmc"], check=True, capture_output=True)
        _GPU_CLOCKS_LOCKED = False
        print("[gpu] Clocks unlocked (reset to default)")
    except Exception as e:
        print(f"[gpu] WARNING: Failed to unlock clocks: {e}")


# =============================================================================
# Subprocess environment
# =============================================================================


def _bench_env() -> dict[str, str]:
    """Build environment for benchmark subprocesses."""
    env = os.environ.copy()
    # Set both old and new names for PyTorch memory allocator config
    # (PYTORCH_CUDA_ALLOC_CONF deprecated in PyTorch 2.9+, renamed to PYTORCH_ALLOC_CONF)
    alloc_conf = "expandable_segments:True"
    env["PYTORCH_CUDA_ALLOC_CONF"] = alloc_conf
    env["PYTORCH_ALLOC_CONF"] = alloc_conf
    return env


def _run_bench_subprocess(
    cmd: list[str], cwd: str, timeout: int = 300,
) -> tuple[int, str]:
    """Run a benchmark subprocess in its own process group.

    Ensures ALL child processes (including NCCL workers) are killed on exit,
    and waits for GPU memory to be fully released before returning.
    Returns (returncode, combined_output).
    """
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=_bench_env(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    try:
        stdout, _ = proc.communicate(timeout=timeout)
        output = stdout.decode("utf-8", errors="replace") if stdout else ""
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        # Kill the entire process group
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        try:
            stdout, _ = proc.communicate(timeout=10)
            output = stdout.decode("utf-8", errors="replace") if stdout else ""
        except subprocess.TimeoutExpired:
            # Process stuck in uninterruptible sleep (D-state) — give up on output
            output = ""
        returncode = -1
    finally:
        # Kill entire process group if still alive
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
    # Wait for GPU memory to be fully released
    _wait_gpu_free()
    return returncode, output


def _wait_gpu_free(timeout: int = 60) -> bool:
    """Wait until no compute processes are using the GPU.

    Polls nvidia-smi for compute apps. If any remain after timeout,
    force-kills them and waits for memory to be freed.
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return True
    lines: list[str] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        out = subprocess.run(
            [nvidia_smi, "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        lines = [l.strip() for l in out.stdout.strip().split("\n") if l.strip()]
        if not lines:
            return True
        time.sleep(2)
    # Force-kill remaining GPU processes
    for line in lines:
        pid_str = line.split(",")[0].strip()
        if pid_str and pid_str.isdigit():
            print(f"  [cleanup] Killing orphan GPU process {pid_str}")
            try:
                os.kill(int(pid_str), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
    # Wait for memory to actually free after kill
    time.sleep(5)
    return True


# =============================================================================
# Helpers
# =============================================================================


def _str2int_list(arg: str) -> list[int]:
    if not arg:
        return []
    if re.fullmatch(r"\d+(,\d+)*", arg.strip()) is None:
        raise argparse.ArgumentTypeError(f"Bad int list: {arg}")
    return [int(x) for x in arg.split(",")]


def _str2float_list(arg: str) -> list[float]:
    """Parse comma-separated float list (supports 'inf')."""
    if not arg:
        return []
    parts = [x.strip() for x in arg.split(",")]
    result = []
    for p in parts:
        try:
            result.append(float(p))
        except ValueError:
            raise argparse.ArgumentTypeError(f"Bad float in list: {p!r}")
    return result


def _load_jsonl(path: str) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _color_delta(delta: float) -> str:
    """ANSI-colored delta string. Always 10 visible chars for table alignment."""
    # +9.1f = 9 chars (includes sign), + "%" = 10 visible chars total
    text = f"{delta:+9.1f}%"
    if delta > 0:
        return f"\033[32m{text}\033[0m"
    elif delta < 0:
        return f"\033[31m{text}\033[0m"
    return text


def _compute_delta(
    branch_val: float, baseline_val: float, higher_is_better: bool,
) -> float | None:
    """Compute percentage delta. Returns None if baseline is zero."""
    if baseline_val == 0:
        return None
    delta = (branch_val - baseline_val) / baseline_val * 100
    if not higher_is_better:
        delta = -delta  # invert so positive = improvement
    return delta


def _format_delta(delta: float | None) -> str:
    """Format a delta value (from _compute_delta) as a 10-char ANSI string."""
    if delta is None:
        return "       N/A"  # 10 chars to match _color_delta width
    return _color_delta(delta)


def _is_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def _ensure_logs_dir(output_dir: Path) -> Path:
    """Create and return the logs subdirectory."""
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def _save_log(logs_dir: Path, name: str, content: str) -> Path:
    """Write subprocess output to a log file. Returns the log path."""
    log_path = logs_dir / f"{name}.log"
    with open(log_path, "w") as f:
        f.write(content)
    return log_path


# =============================================================================
# Kernel subcommand — in-process
# =============================================================================


def _check_fused_available():
    try:
        import sgl_kernel

        return hasattr(sgl_kernel, "fused_add_rmsnorm_quant_fp8")
    except ImportError:
        return False


_FUSED_AVAILABLE = None  # lazy init


def _get_fused_available():
    global _FUSED_AVAILABLE
    if _FUSED_AVAILABLE is None:
        _FUSED_AVAILABLE = _check_fused_available()
    return _FUSED_AVAILABLE


# --- Sequential wrappers ---


def _sequential_add_rmsnorm_quant_fp8(input, residual, weight, eps=1e-6):
    """fused_add_rmsnorm -> sgl_per_token_quant_fp8 (2 kernel launches)."""
    import sgl_kernel
    import torch

    sgl_kernel.fused_add_rmsnorm(input, residual, weight, eps=eps)
    output_q = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    output_s = torch.empty((input.size(0), 1), device=input.device, dtype=torch.float32)
    sgl_kernel.sgl_per_token_quant_fp8(input, output_q, output_s)
    return output_q, output_s


def _sequential_silu_mul_quant_fp8(input):
    """silu_and_mul -> sgl_per_token_quant_fp8 (2 kernel launches)."""
    import sgl_kernel
    import torch

    act = sgl_kernel.silu_and_mul(input)
    output_q = torch.empty_like(act, dtype=torch.float8_e4m3fn)
    output_s = torch.empty((act.size(0), 1), device=act.device, dtype=torch.float32)
    sgl_kernel.sgl_per_token_quant_fp8(act, output_q, output_s)
    return output_q, output_s


# --- Bandwidth calculators ---


def _bytes_add_rmsnorm_quant(M: int, H: int) -> int:
    """Reads: input(BF16) + residual(BF16) + weight(BF16).
    Writes: residual(BF16) + output_q(FP8) + output_s(FP32)."""
    return M * H * 2 + M * H * 2 + H * 2 + M * H * 2 + M * H * 1 + M * 4


def _bytes_silu_mul_quant(M: int, d: int) -> int:
    """Reads: input(BF16, shape [M, 2d]).
    Writes: output_q(FP8, [M, d]) + output_s(FP32, [M, 1])."""
    return M * 2 * d * 2 + M * d * 1 + M * 4


# --- Correctness verification ---


def _verify_add_rmsnorm_quant(batch_sizes, hidden_sizes):
    import sgl_kernel
    import torch
    import torch.nn.functional as F

    print("\n=== Correctness: fused_add_rmsnorm_quant_fp8 ===")
    all_ok = True
    for bs in batch_sizes:
        for hs in hidden_sizes:
            torch.manual_seed(42)
            device = "cuda"
            inp = torch.randn(bs, hs, dtype=torch.bfloat16, device=device)
            res = torch.randn(bs, hs, dtype=torch.bfloat16, device=device)
            w = torch.randn(hs, dtype=torch.bfloat16, device=device)

            # Sequential
            inp_seq, res_seq = inp.clone(), res.clone()
            seq_q, seq_s = _sequential_add_rmsnorm_quant_fp8(inp_seq, res_seq, w)

            # Fused
            res_fused = res.clone()
            fused_q, fused_s = sgl_kernel.fused_add_rmsnorm_quant_fp8(
                inp, res_fused, w, eps=1e-6
            )

            deq_seq = seq_q.float() * seq_s
            deq_fused = fused_q.float() * fused_s
            cos = F.cosine_similarity(deq_seq.reshape(-1), deq_fused.reshape(-1), dim=0)
            ok = cos.item() > 0.99
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] bs={bs:4d} hs={hs:5d}  cos_sim={cos.item():.6f}")
            if not ok:
                all_ok = False
    return all_ok


def _verify_silu_mul_quant(batch_sizes, intermediate_sizes):
    import sgl_kernel
    import torch
    import torch.nn.functional as F

    print("\n=== Correctness: fused_silu_mul_quant_fp8 ===")
    all_ok = True
    for bs in batch_sizes:
        for d in intermediate_sizes:
            torch.manual_seed(42)
            device = "cuda"
            inp = torch.randn(bs, 2 * d, dtype=torch.bfloat16, device=device)

            # Sequential
            seq_q, seq_s = _sequential_silu_mul_quant_fp8(inp.clone())

            # Fused
            fused_q, fused_s = sgl_kernel.fused_silu_mul_quant_fp8(inp)

            deq_seq = seq_q.float() * seq_s
            deq_fused = fused_q.float() * fused_s
            cos = F.cosine_similarity(deq_seq.reshape(-1), deq_fused.reshape(-1), dim=0)
            ok = cos.item() > 0.99
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] bs={bs:4d}  d={d:5d}  cos_sim={cos.item():.6f}")
            if not ok:
                all_ok = False
    return all_ok


# --- Triton benchmarks ---


def _make_configs(batch_sizes, sizes):
    """Generate (batch_size, size) grid for triton.testing.perf_report."""
    return list(itertools.product(batch_sizes, sizes))


def _timed(fn):
    """Timing helper — pattern from bench_rmsnorm.py."""
    import torch
    import triton.testing

    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ms, qmin, qmax = triton.testing.do_bench_cudagraph(fn, quantiles=[0.5, 0.2, 0.8])
    return 1000 * ms, 1000 * qmax, 1000 * qmin  # us


def _run_rmsnorm_bench(batch_sizes, hidden_sizes):
    """Benchmark fused_add_rmsnorm_quant_fp8 vs sequential."""
    import sgl_kernel
    import torch
    import triton
    import triton.testing

    fused_avail = _get_fused_available()
    providers = ["sequential"]
    names = ["sequential (2 launches)"]
    styles = [("blue", "-")]
    if fused_avail:
        providers += ["fused", "speedup"]
        names += ["fused (1 launch)", "speedup"]
        styles += [("green", "-"), ("red", ":")]

    configs = _make_configs(batch_sizes, hidden_sizes)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["batch_size", "hidden_size"],
            x_vals=configs,
            line_arg="provider",
            line_vals=providers,
            line_names=names,
            styles=styles,
            ylabel="us (median)  or  x (speed-up)",
            plot_name="fused-add-rmsnorm-quant-fp8",
            args={},
        )
    )
    def bench_fn(batch_size, hidden_size, provider):
        device = torch.device("cuda")
        dtype = torch.bfloat16

        inp = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
        res = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
        w = torch.randn(hidden_size, dtype=dtype, device=device)

        if provider == "sequential":
            return _timed(
                lambda: _sequential_add_rmsnorm_quant_fp8(
                    inp.clone(), res.clone(), w
                )
            )
        elif provider == "fused":
            return _timed(
                lambda: sgl_kernel.fused_add_rmsnorm_quant_fp8(
                    inp, res.clone(), w, eps=1e-6
                )
            )
        else:  # speedup
            t_seq, _, _ = _timed(
                lambda: _sequential_add_rmsnorm_quant_fp8(
                    inp.clone(), res.clone(), w
                )
            )
            t_fused, _, _ = _timed(
                lambda: sgl_kernel.fused_add_rmsnorm_quant_fp8(
                    inp, res.clone(), w, eps=1e-6
                )
            )
            spd = t_seq / t_fused if t_fused > 0 else 1.0
            return (spd, spd, spd)

    bench_fn.run(print_data=True)

    # Bandwidth summary
    if fused_avail:

        def _make_rmsnorm_bench(bs, hs):
            inp = torch.randn(bs, hs, dtype=torch.bfloat16, device="cuda")
            res = torch.randn(bs, hs, dtype=torch.bfloat16, device="cuda")
            w = torch.randn(hs, dtype=torch.bfloat16, device="cuda")
            return lambda: sgl_kernel.fused_add_rmsnorm_quant_fp8(
                inp, res.clone(), w, eps=1e-6
            )

        _print_bandwidth_summary(
            "fused_add_rmsnorm_quant_fp8",
            batch_sizes,
            hidden_sizes,
            _bytes_add_rmsnorm_quant,
            _make_rmsnorm_bench,
        )


def _run_silu_bench(batch_sizes, intermediate_sizes):
    """Benchmark fused_silu_mul_quant_fp8 vs sequential."""
    import sgl_kernel
    import torch
    import triton
    import triton.testing

    fused_avail = _get_fused_available()
    providers = ["sequential"]
    names = ["sequential (2 launches)"]
    styles = [("blue", "-")]
    if fused_avail:
        providers += ["fused", "speedup"]
        names += ["fused (1 launch)", "speedup"]
        styles += [("green", "-"), ("red", ":")]

    configs = _make_configs(batch_sizes, intermediate_sizes)

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["batch_size", "d"],
            x_vals=configs,
            line_arg="provider",
            line_vals=providers,
            line_names=names,
            styles=styles,
            ylabel="us (median)  or  x (speed-up)",
            plot_name="fused-silu-mul-quant-fp8",
            args={},
        )
    )
    def bench_fn(batch_size, d, provider):
        device = torch.device("cuda")
        dtype = torch.bfloat16

        inp = torch.randn(batch_size, 2 * d, dtype=dtype, device=device)

        if provider == "sequential":
            return _timed(lambda: _sequential_silu_mul_quant_fp8(inp.clone()))
        elif provider == "fused":
            return _timed(lambda: sgl_kernel.fused_silu_mul_quant_fp8(inp.clone()))
        else:  # speedup
            t_seq, _, _ = _timed(lambda: _sequential_silu_mul_quant_fp8(inp.clone()))
            t_fused, _, _ = _timed(
                lambda: sgl_kernel.fused_silu_mul_quant_fp8(inp.clone())
            )
            spd = t_seq / t_fused if t_fused > 0 else 1.0
            return (spd, spd, spd)

    bench_fn.run(print_data=True)

    # Bandwidth summary
    if fused_avail:

        def _make_silu_bench(bs, d):
            inp = torch.randn(bs, 2 * d, dtype=torch.bfloat16, device="cuda")
            # No clone needed: fused_silu_mul_quant_fp8 is read-only on input
            return lambda: sgl_kernel.fused_silu_mul_quant_fp8(inp)

        _print_bandwidth_summary(
            "fused_silu_mul_quant_fp8",
            batch_sizes,
            intermediate_sizes,
            _bytes_silu_mul_quant,
            _make_silu_bench,
        )


def _print_bandwidth_summary(name, batch_sizes, sizes, bytes_fn, make_args_fn):
    """Print effective bandwidth table for the fused kernel.

    make_args_fn(bs, sz) -> (callable, ) — returns (bench_fn,) where bench_fn()
    runs the kernel with pre-allocated tensors.
    """
    import torch
    import triton.testing

    print(f"\nEffective Bandwidth: {name}")
    print(f"  {'batch':>6s}  {'size':>6s}  {'us':>8s}  {'GB/s':>8s}  {'%peak':>6s}")
    for bs in batch_sizes:
        for sz in sizes:
            fn = make_args_fn(bs, sz)
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            ms, _, _ = triton.testing.do_bench_cudagraph(
                fn, quantiles=[0.5, 0.2, 0.8]
            )
            us = ms * 1000
            nbytes = bytes_fn(bs, sz)
            gbps = nbytes / (us * 1e-6) / 1e9 if us > 0 else 0
            pct = gbps / _H100_HBM_BW_GBS * 100
            print(f"  {bs:6d}  {sz:6d}  {us:8.2f}  {gbps:8.1f}  {pct:5.1f}%")


def cmd_kernel(args):
    """Kernel microbenchmark subcommand."""
    _lock_gpu_clocks()
    try:
        _cmd_kernel_impl(args)
    finally:
        _unlock_gpu_clocks()


def _cmd_kernel_impl(args):
    batch_sizes = args.batch_sizes
    hidden_sizes = args.hidden_sizes
    intermediate_sizes = args.intermediate_sizes
    kernel_name = args.kernel_name

    verify_bs = [1, 4, 16, 128]

    if _get_fused_available():
        print("Fused kernels detected.")
    else:
        print("Fused kernels NOT available — sequential-only mode.")

    if kernel_name in ("rmsnorm", "all"):
        if _get_fused_available():
            ok = _verify_add_rmsnorm_quant(verify_bs, hidden_sizes)
            if not ok:
                print("WARN: correctness check failed for fused_add_rmsnorm_quant_fp8")
        if not args.verify_only:
            _run_rmsnorm_bench(batch_sizes, hidden_sizes)

    if kernel_name in ("silu", "all"):
        if _get_fused_available():
            ok = _verify_silu_mul_quant(verify_bs, intermediate_sizes)
            if not ok:
                print("WARN: correctness check failed for fused_silu_mul_quant_fp8")
        if not args.verify_only:
            _run_silu_bench(batch_sizes, intermediate_sizes)


# =============================================================================
# Server lifecycle helpers
# =============================================================================


def _launch_server(
    repo: RepoConfig, cfg: BenchConfig, port: int, log_path: Path | None = None,
) -> tuple[subprocess.Popen, object]:
    """Launch the inference server. Returns (process, log_file_handle).

    If log_path is provided, server stdout/stderr is written to that file.
    Caller must close the file handle after killing the server.
    """
    cmd = [
        repo.python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        cfg.model,
        "--quantization",
        "fp8",
        "--tp",
        str(cfg.tp),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    print(f"[{repo.name}] Launching server: {' '.join(cmd)}")
    log_fh = None
    stdout_target = subprocess.DEVNULL
    if log_path is not None:
        log_fh = open(log_path, "w")
        stdout_target = log_fh
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=repo.repo_path,
            env=_bench_env(),
            stdout=stdout_target,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
    except Exception:
        if log_fh is not None:
            log_fh.close()
        raise
    return proc, log_fh


def _wait_for_health(port: int, timeout: int = 300) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def _kill_server(proc: subprocess.Popen, port: int):
    if proc.poll() is not None:
        return
    # Try graceful termination of the entire process group first
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, OSError):
        proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        # Force-kill entire process group (catches NCCL workers for tp>1)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            proc.kill()
        proc.wait(timeout=5)
    # Wait until port is free
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if _is_port_free(port):
            return
        time.sleep(1)
    print(f"WARN: port {port} still in use after server kill")


# =============================================================================
# one-batch subcommand
# =============================================================================


def cmd_one_batch(args):
    """Run bench_one_batch for branch and baseline repos."""
    _lock_gpu_clocks()
    try:
        _cmd_one_batch_impl(args)
    finally:
        _unlock_gpu_clocks()


def _cmd_one_batch_impl(args):
    output_dir = Path(args.output_dir).resolve()  # absolute path
    output_dir.mkdir(parents=True, exist_ok=True)

    repos = _build_repos(args)
    cfg = _build_bench_config(args)

    configs = list(itertools.product(cfg.batch_sizes, cfg.input_lens, cfg.output_lens))

    logs_dir = _ensure_logs_dir(output_dir)

    # Clear previous results for all repos
    for repo in repos:
        result_file = output_dir / f"one_batch_{repo.name}.jsonl"
        if result_file.exists():
            result_file.unlink()

    counters = {repo.name: {"ok": 0, "fail": 0} for repo in repos}

    # ABAB interleaving: for each config, run branch then baseline back-to-back.
    # This eliminates temporal bias (thermal state, driver behavior) that would
    # arise from running all branch configs first, then all baseline configs.
    repo_names = ", ".join(r.name for r in repos)
    print(f"\nRunning bench_one_batch ({len(configs)} configs × {len(repos)} repos [{repo_names}], interleaved)")

    for i, (bs, il, ol) in enumerate(configs, 1):
        tag = f"bs={bs},il={il},ol={ol}"
        for repo in repos:
            result_file = output_dir / f"one_batch_{repo.name}.jsonl"
            log_name = f"one_batch_{repo.name}_bs{bs}_il{il}_ol{ol}"
            # Temp file for this single config — bench_one_batch overwrites it each run
            tmp_file = output_dir / f"_tmp_{repo.name}.jsonl"
            cmd = [
                repo.python,
                "-m",
                "sglang.bench_one_batch",
                "--model-path",
                cfg.model,
                "--quantization",
                "fp8",
                "--tp",
                str(cfg.tp),
                "--batch-size",
                str(bs),
                "--input-len",
                str(il),
                "--output-len",
                str(ol),
                "--run-name",
                repo.name,
                "--result-filename",
                str(tmp_file),
            ]
            print(f"  [{i}/{len(configs)}] [{repo.name}] {tag} ... ", end="", flush=True)
            returncode, output = _run_bench_subprocess(cmd, repo.repo_path)
            # Save subprocess log regardless of success/failure
            _save_log(logs_dir, log_name, output)
            if returncode != 0:
                counters[repo.name]["fail"] += 1
                # Show last 3 lines of output for diagnosis
                err_lines = [l for l in output.splitlines() if l.strip()][-3:]
                err_hint = err_lines[-1] if err_lines else "unknown error"
                print(f"FAIL ({err_hint[:80]})")
                continue
            # Append results from tmp file to main result file
            if tmp_file.exists():
                with open(tmp_file) as f_in, open(result_file, "a") as f_out:
                    for line in f_in:
                        line = line.strip()
                        if line:
                            f_out.write(line + "\n")
                tmp_file.unlink()
                counters[repo.name]["ok"] += 1
                print("OK")
            else:
                counters[repo.name]["fail"] += 1
                print("FAIL (no output)")

    for repo in repos:
        c = counters[repo.name]
        result_file = output_dir / f"one_batch_{repo.name}.jsonl"
        print(f"[{repo.name}] Done: {c['ok']} OK, {c['fail']} failed out of {len(configs)}")
        if result_file.exists():
            print(f"[{repo.name}] Results: {result_file}")
        print(f"[{repo.name}] Logs: {logs_dir}/one_batch_{repo.name}_*.log")


# =============================================================================
# serving subcommand
# =============================================================================


def cmd_serving(args):
    """Run bench_serving for branch and baseline repos."""
    _lock_gpu_clocks()
    try:
        _cmd_serving_impl(args)
    finally:
        _unlock_gpu_clocks()


def _cmd_serving_impl(args):
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = _ensure_logs_dir(output_dir)

    repos = _build_repos(args)
    cfg = _build_bench_config(args)

    for repo in repos:
        port = cfg.port
        server_log = logs_dir / f"server_{repo.name}.log"
        proc, log_fh = _launch_server(repo, cfg, port, log_path=server_log)
        try:
            print(f"[{repo.name}] Waiting for server health on port {port}...")
            print(f"[{repo.name}] Server log: {server_log}")
            if not _wait_for_health(port, timeout=300):
                print(f"[{repo.name}] ERROR: server did not become healthy")
                continue

            for rate in cfg.request_rates:
                rate_str = "inf" if rate == float("inf") else f"{rate:g}"
                result_file = output_dir / f"serving_{repo.name}_rate{rate_str}.jsonl"
                cmd = [
                    repo.python,
                    "-m",
                    "sglang.bench_serving",
                    "--backend",
                    "sglang",
                    "--port",
                    str(port),
                    "--dataset-name",
                    "random",
                    "--random-input-len",
                    str(cfg.random_input),
                    "--random-output-len",
                    str(cfg.random_output),
                    "--num-prompts",
                    str(cfg.num_prompts),
                    "--request-rate",
                    str(rate),
                    "--output-file",
                    str(result_file),
                ]
                print(f"\n[{repo.name}] bench_serving rate={rate_str}")
                try:
                    result = subprocess.run(
                        cmd, cwd=repo.repo_path, env=_bench_env(),
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        timeout=600,  # 10min max per rate
                    )
                except subprocess.TimeoutExpired as e:
                    bench_output = e.stdout.decode("utf-8", errors="replace") if e.stdout else ""
                    _save_log(logs_dir, f"serving_{repo.name}_rate{rate_str}", bench_output)
                    print(f"  TIMEOUT (10min)")
                    continue
                bench_output = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
                _save_log(logs_dir, f"serving_{repo.name}_rate{rate_str}", bench_output)
                if result.returncode != 0:
                    print(f"  FAIL (exit code {result.returncode})")
                else:
                    print(f"  OK → {result_file.name}")
        finally:
            print(f"[{repo.name}] Killing server...")
            try:
                _kill_server(proc, port)
            except Exception as e:
                print(f"[{repo.name}] WARN: _kill_server failed: {e}")
            finally:
                if log_fh is not None:
                    log_fh.close()
            print(f"[{repo.name}] Logs: {logs_dir}/serving_{repo.name}_*.log")


# =============================================================================
# compare subcommand
# =============================================================================


def _compare_one_batch(output_dir: Path, model: str, tp: int):
    """Compare bench_one_batch results between branch and baseline."""
    branch_file = output_dir / "one_batch_branch.jsonl"
    baseline_file = output_dir / "one_batch_baseline.jsonl"

    if not branch_file.exists() or not baseline_file.exists():
        print("Missing one_batch result files, skipping comparison.")
        return

    branch_data = _load_jsonl(str(branch_file))
    baseline_data = _load_jsonl(str(baseline_file))

    # Index by (batch_size, input_len, output_len)
    def _index(data):
        idx = {}
        for entry in data:
            key = (entry.get("batch_size"), entry.get("input_len"), entry.get("output_len"))
            idx[key] = entry
        return idx

    b_idx = _index(branch_data)
    bl_idx = _index(baseline_data)

    all_keys = sorted(set(b_idx.keys()) | set(bl_idx.keys()))
    if not all_keys:
        print("No one_batch data to compare.")
        return

    header = f"bench_one_batch: branch vs baseline ({model}, tp={tp})"
    sep = "\u2501" * 70
    print(f"\n{header}")
    print(sep)
    print(
        f"  {'bs':>4s}  {'il':>5s}  {'ol':>5s}  "
        f"\u2502 {'prefill':>10s}  \u2502 {'decode':>10s}  \u2502 {'overall':>10s}"
    )
    print(sep)

    pf_deltas: list[float] = []
    dc_deltas: list[float] = []
    ov_deltas: list[float] = []

    for key in all_keys:
        bs, il, ol = key
        b = b_idx.get(key)
        bl = bl_idx.get(key)
        if b is None or bl is None:
            na = "       N/A"  # 10 chars
            print(f"  {bs:4d}  {il:5d}  {ol:5d}  \u2502 {na}  \u2502 {na}  \u2502 {na}")
            continue

        pf_d = _compute_delta(b.get("prefill_throughput", 0), bl.get("prefill_throughput", 0), True)
        dc_d = _compute_delta(b.get("median_decode_throughput", 0), bl.get("median_decode_throughput", 0), True)
        ov_d = _compute_delta(b.get("overall_throughput", 0), bl.get("overall_throughput", 0), True)

        if pf_d is not None:
            pf_deltas.append(pf_d)
        if dc_d is not None:
            dc_deltas.append(dc_d)
        if ov_d is not None:
            ov_deltas.append(ov_d)

        print(
            f"  {bs:4d}  {il:5d}  {ol:5d}  "
            f"\u2502 {_format_delta(pf_d)}  "
            f"\u2502 {_format_delta(dc_d)}  "
            f"\u2502 {_format_delta(ov_d)}"
        )

    print(sep)

    # Aggregate statistics
    if pf_deltas or dc_deltas or ov_deltas:
        import statistics

        def _agg_line(label: str, vals: list[float]) -> str:
            if not vals:
                return f"    {label:>10s}:  (no data)"
            mn = statistics.mean(vals)
            md = statistics.median(vals)
            lo, hi = min(vals), max(vals)
            return f"    {label:>10s}:  mean={mn:+6.1f}%  median={md:+6.1f}%  range=[{lo:+.1f}%, {hi:+.1f}%]"

        n = max(len(pf_deltas), len(dc_deltas), len(ov_deltas))
        print(f"\n  Summary ({n} configs):")
        print(_agg_line("prefill", pf_deltas))
        print(_agg_line("decode", dc_deltas))
        print(_agg_line("overall", ov_deltas))


def _compare_serving(output_dir: Path):
    """Compare bench_serving results between branch and baseline."""
    # Discover rate files
    branch_files = sorted(output_dir.glob("serving_branch_rate*.jsonl"))
    baseline_files = sorted(output_dir.glob("serving_baseline_rate*.jsonl"))

    if not branch_files or not baseline_files:
        print("Missing serving result files, skipping comparison.")
        return

    rate_re = re.compile(r"serving_\w+_rate(.+)\.jsonl")

    def _extract_rate(p: Path) -> str:
        m = rate_re.match(p.name)
        return m.group(1) if m else "?"

    def _load_by_rate(files):
        idx = {}
        for f in files:
            rate = _extract_rate(f)
            data = _load_jsonl(str(f))
            if data:
                idx[rate] = data[-1]  # last entry = final result
        return idx

    b_idx = _load_by_rate(branch_files)
    bl_idx = _load_by_rate(baseline_files)
    all_rates = sorted(set(b_idx.keys()) | set(bl_idx.keys()), key=lambda r: float(r) if r != "inf" else float("inf"))

    if not all_rates:
        print("No serving data to compare.")
        return

    sep = "\u2501" * 80
    print(f"\nbench_serving: branch vs baseline")
    print(sep)
    print(
        f"  {'rate':>6s}  "
        f"\u2502 {'tput':>10s}  "
        f"\u2502 {'TTFT_med':>10s}  "
        f"\u2502 {'TTFT_p99':>10s}  "
        f"\u2502 {'ITL_med':>10s}  "
        f"\u2502 {'ITL_p99':>10s}"
    )
    print(sep)

    tput_deltas: list[float] = []
    ttft_med_deltas: list[float] = []
    ttft_p99_deltas: list[float] = []
    itl_med_deltas: list[float] = []
    itl_p99_deltas: list[float] = []

    for rate in all_rates:
        b = b_idx.get(rate)
        bl = bl_idx.get(rate)
        if b is None or bl is None:
            na = "       N/A"  # 10 chars
            print(f"  {rate:>6s}  \u2502 {na}  \u2502 {na}  \u2502 {na}  \u2502 {na}  \u2502 {na}")
            continue

        tput_d = _compute_delta(b.get("output_throughput", 0), bl.get("output_throughput", 0), True)
        ttft_med_d = _compute_delta(b.get("median_ttft_ms", 0), bl.get("median_ttft_ms", 0), False)
        ttft_p99_d = _compute_delta(b.get("p99_ttft_ms", 0), bl.get("p99_ttft_ms", 0), False)
        itl_med_d = _compute_delta(b.get("median_itl_ms", 0), bl.get("median_itl_ms", 0), False)
        itl_p99_d = _compute_delta(b.get("p99_itl_ms", 0), bl.get("p99_itl_ms", 0), False)

        for d, lst in [
            (tput_d, tput_deltas), (ttft_med_d, ttft_med_deltas),
            (ttft_p99_d, ttft_p99_deltas), (itl_med_d, itl_med_deltas),
            (itl_p99_d, itl_p99_deltas),
        ]:
            if d is not None:
                lst.append(d)

        print(
            f"  {rate:>6s}  "
            f"\u2502 {_format_delta(tput_d)}  "
            f"\u2502 {_format_delta(ttft_med_d)}  "
            f"\u2502 {_format_delta(ttft_p99_d)}  "
            f"\u2502 {_format_delta(itl_med_d)}  "
            f"\u2502 {_format_delta(itl_p99_d)}"
        )
    print(sep)

    # Aggregate statistics
    if tput_deltas:
        import statistics

        def _agg_line(label: str, vals: list[float]) -> str:
            if not vals:
                return f"    {label:>10s}:  (no data)"
            mn = statistics.mean(vals)
            md = statistics.median(vals)
            lo, hi = min(vals), max(vals)
            return f"    {label:>10s}:  mean={mn:+6.1f}%  median={md:+6.1f}%  range=[{lo:+.1f}%, {hi:+.1f}%]"

        n = len(tput_deltas)
        print(f"\n  Summary ({n} rates):")
        print(_agg_line("throughput", tput_deltas))
        print(_agg_line("TTFT_med", ttft_med_deltas))
        print(_agg_line("TTFT_p99", ttft_p99_deltas))
        print(_agg_line("ITL_med", itl_med_deltas))
        print(_agg_line("ITL_p99", itl_p99_deltas))


def cmd_compare(args):
    """Load JSONLs and print comparison tables."""
    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        print(f"ERROR: output directory does not exist: {output_dir}")
        print("Use --output-dir to specify the directory containing benchmark results.")
        return
    _compare_one_batch(output_dir, args.model, args.tp)
    _compare_serving(output_dir)


# =============================================================================
# all subcommand
# =============================================================================


def cmd_all(args):
    """Run kernel -> one-batch -> serving -> compare."""
    print("=" * 60)
    print("Phase 1: Kernel microbenchmark")
    print("=" * 60)
    cmd_kernel(args)

    # For E2E phases, use --e2e-batch-sizes if provided, otherwise let
    # BenchConfig defaults apply (not kernel --batch-sizes which differ).
    saved_batch_sizes = args.batch_sizes
    e2e_bs = getattr(args, "e2e_batch_sizes", None)
    args.batch_sizes = e2e_bs  # None → BenchConfig defaults kick in

    print("\n" + "=" * 60)
    print("Phase 2: bench_one_batch")
    print("=" * 60)
    cmd_one_batch(args)

    if not args.skip_serving:
        print("\n" + "=" * 60)
        print("Phase 3: bench_serving")
        print("=" * 60)
        cmd_serving(args)

    # Restore for any future use
    args.batch_sizes = saved_batch_sizes

    print("\n" + "=" * 60)
    print("Phase 4: Comparison")
    print("=" * 60)
    cmd_compare(args)


# =============================================================================
# Repo/config builders
# =============================================================================


def _build_repos(args) -> list[RepoConfig]:
    only = getattr(args, "only", None)
    repos = []
    if only is None or only == "branch":
        repos.append(RepoConfig("branch", args.branch_repo, args.branch_python))
    if only is None or only == "baseline":
        repos.append(RepoConfig("baseline", args.baseline_repo, args.baseline_python))
    return repos


def _build_bench_config(args) -> BenchConfig:
    kwargs: dict = dict(model=args.model, tp=args.tp, port=args.port)
    if getattr(args, "batch_sizes", None):
        kwargs["batch_sizes"] = tuple(args.batch_sizes)
    if getattr(args, "input_lens", None):
        kwargs["input_lens"] = tuple(args.input_lens)
    if getattr(args, "output_lens", None):
        kwargs["output_lens"] = tuple(args.output_lens)
    if getattr(args, "request_rates", None):
        kwargs["request_rates"] = tuple(args.request_rates)
    return BenchConfig(**kwargs)


# =============================================================================
# CLI
# =============================================================================


def _add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model path",
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Results directory (default: ./results/fused_fp8_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--branch-repo",
        type=str,
        default="/root/sglang",
        help="Branch repo path",
    )
    parser.add_argument(
        "--baseline-repo",
        type=str,
        default="/root/sglang-baseline",
        help="Baseline repo path",
    )
    parser.add_argument(
        "--branch-python",
        type=str,
        default="/root/sglang/venv/bin/python3.11",
        help="Python for branch",
    )
    parser.add_argument(
        "--baseline-python",
        type=str,
        default="/root/sglang-baseline/venv/bin/python3.11",
        help="Python for baseline",
    )
    parser.add_argument("--port", type=int, default=30000, help="Server port")
    parser.add_argument(
        "--only",
        type=str,
        choices=["branch", "baseline"],
        default=None,
        help="Run only branch or baseline (default: both)",
    )


def _add_sweep_args(parser: argparse.ArgumentParser):
    """Add --batch-sizes, --input-lens, --output-lens overrides for E2E benchmarks."""
    parser.add_argument(
        "--batch-sizes",
        type=_str2int_list,
        default=None,
        help="Comma-separated batch sizes (default: 1,8,32,64,128,256)",
    )
    parser.add_argument(
        "--input-lens",
        type=_str2int_list,
        default=None,
        help="Comma-separated input lengths (default: 256,512,1024,2048)",
    )
    parser.add_argument(
        "--output-lens",
        type=_str2int_list,
        default=None,
        help="Comma-separated output lengths (default: 16,64,256)",
    )


def _add_kernel_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--batch-sizes",
        type=_str2int_list,
        default=_DEFAULT_BATCH_SIZES,
        help="Comma-separated batch sizes",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=_str2int_list,
        default=_DEFAULT_HIDDEN_SIZES,
        help="Comma-separated hidden sizes",
    )
    parser.add_argument(
        "--intermediate-sizes",
        type=_str2int_list,
        default=_DEFAULT_INTERMEDIATE_SIZES,
        help="Comma-separated intermediate sizes",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Run correctness checks only, no timing",
    )
    parser.add_argument(
        "--kernel-name",
        type=str,
        choices=["rmsnorm", "silu", "all"],
        default="all",
        help="Which kernel(s) to benchmark",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark fused FP8 kernels vs sequential baselines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
subcommands:
  kernel      Kernel microbenchmark (in-process, requires GPU)
  one-batch   E2E latency via bench_one_batch (subprocess)
  serving     E2E throughput via bench_serving (subprocess)
  compare     Load JSONLs, print comparison table
  all         kernel -> one-batch -> serving -> compare
""",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # kernel
    p_kernel = sub.add_parser("kernel", help="Kernel microbenchmark")
    _add_kernel_args(p_kernel)

    # one-batch
    p_one_batch = sub.add_parser("one-batch", help="E2E latency via bench_one_batch")
    _add_common_args(p_one_batch)
    _add_sweep_args(p_one_batch)

    # serving
    p_serving = sub.add_parser("serving", help="E2E throughput via bench_serving")
    _add_common_args(p_serving)
    p_serving.add_argument(
        "--request-rates",
        type=_str2float_list,
        default=None,
        help="Comma-separated request rates (default: 2,4,8,16,32,inf)",
    )

    # compare
    p_compare = sub.add_parser("compare", help="Load JSONLs, print comparison")
    _add_common_args(p_compare)

    # all
    p_all = sub.add_parser("all", help="Run all phases")
    _add_common_args(p_all)
    _add_kernel_args(p_all)
    p_all.add_argument(
        "--e2e-batch-sizes",
        type=_str2int_list,
        default=None,
        help="Comma-separated batch sizes for E2E benchmarks (default: BenchConfig defaults). "
        "Separate from --batch-sizes which controls kernel microbenchmarks.",
    )
    p_all.add_argument(
        "--input-lens",
        type=_str2int_list,
        default=None,
        help="Comma-separated input lengths (default: 256,512,1024,2048)",
    )
    p_all.add_argument(
        "--output-lens",
        type=_str2int_list,
        default=None,
        help="Comma-separated output lengths (default: 16,64,256)",
    )
    p_all.add_argument(
        "--request-rates",
        type=_str2float_list,
        default=None,
        help="Comma-separated request rates (default: 2,4,8,16,32,inf)",
    )
    p_all.add_argument(
        "--skip-serving", action="store_true", help="Skip bench_serving phase"
    )

    args = parser.parse_args()

    # Default output_dir with timestamp
    if hasattr(args, "output_dir") and args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"./results/fused_fp8_{ts}"

    dispatch = {
        "kernel": cmd_kernel,
        "one-batch": cmd_one_batch,
        "serving": cmd_serving,
        "compare": cmd_compare,
        "all": cmd_all,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
