"""Compare VibeCUDA with the optimized LTX-2 QKNorm split-RoPE backend.

The workload and reference definitions come from sgl-project/sglang#29708 and
sgl-project/sglang#46111. The timed KDA backend loads the exact CUDA blob from
PR #46111 at a887ac8fc6eae2bac7b892d108c1f9b939142840; the caller-owned-output
adapter and VibeCUDA candidate use the same public selector and timing boundary.
"""

import hashlib
import math
import random
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import msgspec
import torch

import sglang.kernels.kda_kernels.ltx2_qknorm_split_rope_jit as ltx2_kda
from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.kda_kernels import _cuda_source
from sglang.kernels.ops.diffusion import ltx2_qknorm_split_rope_cuda
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.utils import is_in_ci

register_cuda_ci(
    est_time=120, stage="base-b-kernel-benchmark", runner_config="1-gpu-b200"
)


PR46111_COMMIT = "a887ac8fc6eae2bac7b892d108c1f9b939142840"
PR46111_PATH = (
    "python/sglang/kernels/kda_kernels/csrc/diffusion/ltx2_qknorm_split_rope.cuh"
)
PR46111_BLOB = "3d2079ade3b18a6c3fe97180e5651c996813ea34"
PR46111_SHA256 = "233cf4002745c97a2939db76b42e7fce10fcf8e175275d53e9ea46914a730b3e"


class Workload(msgspec.Struct, frozen=True):
    name: str
    batch: int
    q_seq: int
    k_seq: int
    num_heads: int
    head_dim: int


FULL_WORKLOADS = [
    Workload("stage1_video_self_q1536_k1536_d4096", 2, 1536, 1536, 32, 128),
    Workload("stage1_audio_self_q126_k126_d2048", 2, 126, 126, 32, 64),
    Workload("stage1_audio_to_video_q1536_k126_d2048", 2, 1536, 126, 32, 64),
    Workload("stage1_video_to_audio_q126_k1536_d2048", 2, 126, 1536, 32, 64),
    Workload("stage2_video_self_q6144_k6144_d4096", 1, 6144, 6144, 32, 128),
    Workload("stage2_audio_self_q126_k126_d2048", 1, 126, 126, 32, 64),
    Workload("stage2_audio_to_video_q6144_k126_d2048", 1, 6144, 126, 32, 64),
    Workload("stage2_video_to_audio_q126_k6144_d2048", 1, 126, 6144, 32, 64),
    Workload("hq_stage1_video_self_q8160_k8160_d4096", 1, 8160, 8160, 32, 128),
    Workload("hq_stage1_audio_to_video_q8160_k126_d2048", 1, 8160, 126, 32, 64),
    Workload("hq_stage1_video_to_audio_q126_k8160_d2048", 1, 126, 8160, 32, 64),
    Workload("hq_stage2_video_self_q32640_k32640_d4096", 1, 32640, 32640, 32, 128),
    Workload("hq_stage2_audio_to_video_q32640_k126_d2048", 1, 32640, 126, 32, 64),
    Workload("hq_stage2_video_to_audio_q126_k32640_d2048", 1, 126, 32640, 32, 64),
]
CI_WORKLOADS = [
    Workload("stage1_video_self_q16_k16_d4096", 1, 16, 16, 32, 128),
    Workload("stage1_audio_to_video_q16_k8_d2048", 1, 16, 8, 32, 64),
]


@contextmanager
def _use_pr46111_baseline():
    current_path = Path(_cuda_source("diffusion/ltx2_qknorm_split_rope.cuh"))
    current_source = current_path.read_bytes()
    current_alias = b"device.set_options<kDLGPU>();"
    pinned_alias = b"device.set_options<kDLCUDA>();"
    if current_source.count(current_alias) != 1:
        raise RuntimeError("cannot reconstruct the pinned PR46111 baseline source")
    source = current_source.replace(current_alias, pinned_alias)
    digest = hashlib.sha256(source).hexdigest()
    if digest != PR46111_SHA256:
        raise RuntimeError(f"unexpected PR46111 baseline digest: {digest}")

    with tempfile.TemporaryDirectory(prefix="ltx2-pr46111-") as directory:
        pinned_source = Path(directory) / "ltx2_qknorm_split_rope.cuh"
        pinned_source.write_bytes(source)

        @cache_once
        def load_pinned(round_intermediates: bool = False):
            return load_jit(
                "diffusion_ltx2_qknorm_split_rope_pr46111",
                cuda_files=[str(pinned_source)],
                cuda_wrappers=[
                    (
                        "ltx2_qknorm_split_rope_pair",
                        (
                            "ltx2_qknorm_split_rope::LTX2QKNormSplitRopeKernel::run"
                            f"<{str(round_intermediates).lower()}>"
                        ),
                    )
                ],
            )

        current_loader = ltx2_kda._jit_ltx2_qknorm_split_rope_module
        ltx2_kda._jit_ltx2_qknorm_split_rope_module = load_pinned
        try:
            yield
        finally:
            ltx2_kda._jit_ltx2_qknorm_split_rope_module = current_loader


def _make_cos_sin(
    batch: int, seq_len: int, num_heads: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    half_dim = head_dim // 2
    cos = torch.randn(
        batch, seq_len, num_heads, half_dim, device="cuda", dtype=torch.bfloat16
    ).transpose(1, 2)
    sin = torch.randn(
        batch, seq_len, num_heads, half_dim, device="cuda", dtype=torch.bfloat16
    ).transpose(1, 2)
    return cos, sin


def _apply_split_rotary_ref(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    x_dtype = x.dtype
    batch = x.shape[0]
    _, num_heads, seq_len, _ = cos.shape
    x = x.reshape(batch, seq_len, num_heads, -1).swapaxes(1, 2)
    last = x.shape[-1]
    half = last // 2
    split_x = x.reshape(*x.shape[:-1], 2, half)
    first_x = split_x[..., :1, :]
    second_x = split_x[..., 1:, :]
    cos_u = cos.unsqueeze(-2)
    sin_u = sin.unsqueeze(-2)
    out = split_x * cos_u
    out[..., :1, :].addcmul_(-sin_u, second_x)
    out[..., 1:, :].addcmul_(sin_u, first_x)
    out = out.reshape(*out.shape[:-2], last)
    return out.swapaxes(1, 2).reshape(batch, seq_len, -1).to(dtype=x_dtype)


def _reference_pair(inputs):
    q, k, q_cos, q_sin, k_cos, k_sin, q_norm, k_norm = inputs
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        q_out = _apply_split_rotary_ref(q_norm(q), q_cos, q_sin)
        k_out = _apply_split_rotary_ref(k_norm(k), k_cos, k_sin)
    return q_out.to(dtype=torch.bfloat16), k_out.to(dtype=torch.bfloat16)


def cuda_event_us(fn, warmups: int, repeats: int, rounds: int, l2_flush) -> float:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(rounds):
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        for start, end in zip(starts, ends):
            # The flush is stream-ordered outside the event pair for each call.
            l2_flush.zero_()
            start.record()
            fn()
            end.record()
        ends[-1].synchronize()
        elapsed = sum(start.elapsed_time(end) for start, end in zip(starts, ends))
        samples.append(elapsed * 1000.0 / repeats)
    samples.sort()
    return samples[len(samples) // 2]


def _benchmark() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        raise RuntimeError("This comparison requires SM100 or SM103")

    torch.manual_seed(20260630)
    random.seed(20260630)
    torch.cuda.set_device(0)
    l2_flush = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.uint8)

    workloads = CI_WORKLOADS if is_in_ci() else FULL_WORKLOADS + CI_WORKLOADS
    warmups = 3 if is_in_ci() else 10
    repeats = 3 if is_in_ci() else 10
    rounds = 3 if is_in_ci() else 7

    print("| workload | sglang_pr46111 us | vibecuda us | speedup |")
    print("|---|---:|---:|---:|")
    speedups = []
    for workload in workloads:
        hidden = workload.num_heads * workload.head_dim
        q = torch.randn(
            workload.batch,
            workload.q_seq,
            hidden,
            device="cuda",
            dtype=torch.bfloat16,
        )
        k = torch.randn(
            workload.batch,
            workload.k_seq,
            hidden,
            device="cuda",
            dtype=torch.bfloat16,
        )
        q_cos, q_sin = _make_cos_sin(
            workload.batch, workload.q_seq, workload.num_heads, workload.head_dim
        )
        k_cos, k_sin = _make_cos_sin(
            workload.batch, workload.k_seq, workload.num_heads, workload.head_dim
        )
        q_norm = torch.nn.RMSNorm(hidden, eps=1e-6, device="cuda").to(
            dtype=torch.bfloat16
        )
        k_norm = torch.nn.RMSNorm(hidden, eps=1e-6, device="cuda").to(
            dtype=torch.bfloat16
        )
        inputs = (q, k, q_cos, q_sin, k_cos, k_sin, q_norm, k_norm)
        args = (q, q_cos, q_sin, q_norm.weight, k, k_cos, k_sin, k_norm.weight)
        expected = _reference_pair(inputs)

        calls = {}
        for backend in ("kda", "vibecuda"):
            outputs = (
                torch.full_like(q, float("nan")),
                torch.full_like(k, float("nan")),
            )

            def call(backend=backend, outputs=outputs, args=args, workload=workload):
                return ltx2_qknorm_split_rope_cuda(
                    *args,
                    eps=1e-6,
                    num_heads=workload.num_heads,
                    head_dim=workload.head_dim,
                    backend=backend,
                    out=outputs,
                )

            actual = call()
            torch.cuda.synchronize()
            assert actual[0] is outputs[0] and actual[1] is outputs[1]
            assert torch.equal(actual[0], expected[0]), (workload.name, backend, "q")
            assert torch.equal(actual[1], expected[1]), (workload.name, backend, "k")
            calls[backend] = call

        order = ["kda", "vibecuda"]
        random.shuffle(order)
        times = {
            backend: cuda_event_us(calls[backend], warmups, repeats, rounds, l2_flush)
            for backend in order
        }
        speedup = times["kda"] / times["vibecuda"]
        speedups.append(speedup)
        print(
            f"| {workload.name} | {times['kda']:.4f} | "
            f"{times['vibecuda']:.4f} | {speedup:.4f}x |",
            flush=True,
        )
        torch.cuda.empty_cache()

    arithmetic = sum(speedups) / len(speedups)
    geometric = math.exp(sum(math.log(value) for value in speedups) / len(speedups))
    print(
        f"PASS {len(speedups)}/{len(workloads)} exact workloads; "
        f"arithmetic={arithmetic:.6f}x geometric={geometric:.6f}x "
        f"min={min(speedups):.6f}x max={max(speedups):.6f}x",
        flush=True,
    )


def benchmark() -> None:
    with _use_pr46111_baseline():
        print(
            f"baseline source: {PR46111_COMMIT}:{PR46111_PATH} "
            f"blob={PR46111_BLOB} sha256={PR46111_SHA256}",
            flush=True,
        )
        _benchmark()


if __name__ == "__main__":
    benchmark()
    sys.exit(0)
