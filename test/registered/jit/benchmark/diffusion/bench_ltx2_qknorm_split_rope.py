import random
import sys
from dataclasses import dataclass

import torch

from sglang.jit_kernel.diffusion.ltx2_qknorm_split_rope import (
    ltx2_qknorm_split_rope_cuda,
)
from sglang.utils import is_in_ci


@dataclass(frozen=True)
class Workload:
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


def cuda_event_us(fn, warmups: int, repeats: int, rounds: int) -> float:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / repeats)
    samples.sort()
    return samples[len(samples) // 2]


def benchmark() -> None:
    if not torch.cuda.is_available():
        print("CUDA required")
        return

    torch.manual_seed(20260630)
    random.seed(20260630)
    torch.cuda.set_device(0)

    workloads = CI_WORKLOADS if is_in_ci() else FULL_WORKLOADS
    warmups = 3 if is_in_ci() else 10
    repeats = 3 if is_in_ci() else 10
    rounds = 3 if is_in_ci() else 7

    print("| workload | torch us | cuda us | speedup |")
    print("|---|---:|---:|---:|")

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

        q_ref, k_ref = _reference_pair(inputs)
        q_out, k_out = ltx2_qknorm_split_rope_cuda(
            q,
            q_cos,
            q_sin,
            q_norm.weight,
            k,
            k_cos,
            k_sin,
            k_norm.weight,
            eps=1e-6,
            num_heads=workload.num_heads,
            head_dim=workload.head_dim,
        )
        torch.cuda.synchronize()
        assert torch.equal(q_ref, q_out)
        assert torch.equal(k_ref, k_out)

        fns = {
            "torch": lambda: _reference_pair(inputs),
            "cuda": lambda: ltx2_qknorm_split_rope_cuda(
                q,
                q_cos,
                q_sin,
                q_norm.weight,
                k,
                k_cos,
                k_sin,
                k_norm.weight,
                eps=1e-6,
                num_heads=workload.num_heads,
                head_dim=workload.head_dim,
            ),
        }
        order = ["torch", "cuda"]
        random.shuffle(order)
        times = {
            name: cuda_event_us(fns[name], warmups, repeats, rounds) for name in order
        }
        print(
            f"| {workload.name} | {times['torch']:.2f} | "
            f"{times['cuda']:.2f} | {times['torch'] / times['cuda']:.3f}x |"
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    benchmark()
    sys.exit(0)
