# SPDX-License-Identifier: Apache-2.0
"""AITer attention over the Ulysses exchange must not depend on Q/K/V being
copied apart.

``AttentionImpl.accepts_strided_qkv`` lets the packed exchange hand AITer three
views into one interleaved ``[.., 3 * head_size]`` receive buffer instead of
three contiguous copies. Both carry the same values, so only running the kernel
proves the strides reach it: a backend that ignores a stride reads a
neighbouring projection's rows and returns a plausible, wrong result rather
than failing.

    pytest -v python/sglang/multimodal_gen/test/single_test_file/test_usp_strided_qkv_aiter_2_gpu.py
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import unittest

import torch

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.test.test_utils import CustomTestCase, find_available_port

_WORLD = 2
# Four exchanges per layout: the strided path alternates between two receive
# buffers, so a buffer recycled too early shows up from the second step on.
_STEPS = 4
_B, _S_LOCAL, _H, _D = 1, 256, 8, 128


def _make_shards(layout: str, step: int, *, rank: int, world: int, device):
    """This rank's ``[B, S_local, H, D]`` Q/K/V shards of a shared global tensor.

    ``merged_projection`` is the FLUX.2 layout: three chunks of one packed
    projection, so the shards reaching the pack kernel are themselves strided.
    """
    generator = torch.Generator(device="cpu").manual_seed(1000 + step)
    s_global = _S_LOCAL * world
    rows = slice(rank * _S_LOCAL, (rank + 1) * _S_LOCAL)
    if layout == "contiguous":
        return tuple(
            torch.randn(_B, s_global, _H, _D, generator=generator)
            .to(device, torch.bfloat16)[:, rows]
            .contiguous()
            for _ in range(3)
        )
    hidden = torch.randn(_B, s_global, 3 * _H * _D, generator=generator).to(
        device, torch.bfloat16
    )
    return tuple(t[:, rows].unflatten(-1, (_H, _D)) for t in hidden.chunk(3, dim=-1))


def _build_attention(device):
    """A USPAttention wired to a real AITerImpl, without the model-level config.

    The constructor resolves the backend through the global runtime context; the
    fields below are the ones the dense Ulysses path reads.
    """
    import sglang.multimodal_gen.runtime.layers.attention.layer as L
    from sglang.multimodal_gen.runtime.layers.attention.backends.aiter import AITerImpl
    from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum

    scale = _D**-0.5
    attn = L.USPAttention.__new__(L.USPAttention)
    attn.attn_impl = AITerImpl(
        num_heads=_H,
        head_size=_D,
        softmax_scale=scale,
        causal=False,
        num_kv_heads=_H,
    )
    attn.backend = AttentionBackendEnum.AITER
    attn.softmax_scale = scale
    attn.causal = False
    attn.dropout_p = 0.0
    attn.dtype = torch.bfloat16
    attn.num_heads = _H
    attn.head_size = _D
    attn.num_kv_heads = _H
    attn.skip_sequence_parallel = False
    attn.enable_packed_qkv_input_a2a = False
    attn.is_cross_attention = False
    attn.capture_dense_attention = False
    attn.allow_cudnn_sdp = False
    attn.sp_attention_mode = "ulysses"
    attn.sp_attention_mode_is_auto = False
    attn._pipeline_declined = set()
    return attn


def _run_layout(attn, *, strided: bool, layout: str, rank: int, world: int, device):
    """Returns the per-step outputs and whether each call saw contiguous Q/K/V."""
    attn.attn_impl.accepts_strided_qkv = strided
    kernel_saw_contiguous: list[bool] = []
    impl_forward = type(attn.attn_impl).forward

    def recording_forward(query, key, value, metadata):
        kernel_saw_contiguous.append(
            query.is_contiguous() and key.is_contiguous() and value.is_contiguous()
        )
        return impl_forward(attn.attn_impl, query, key, value, metadata)

    attn.attn_impl.forward = recording_forward
    outputs = []
    with torch.inference_mode():
        for step in range(_STEPS):
            q, k, v = _make_shards(layout, step, rank=rank, world=world, device=device)
            outputs.append(attn.forward(q, k, v).clone())
    del attn.attn_impl.forward
    return outputs, kernel_saw_contiguous


def _worker() -> int:
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    init_distributed_environment(world_size=world, rank=rank, local_rank=rank)
    initialize_model_parallel(
        sequence_parallel_degree=world, ulysses_degree=world, ring_degree=1
    )

    import sglang.multimodal_gen.runtime.layers.attention.layer as L
    from sglang.multimodal_gen.runtime.layers import usp as usp_mod

    L.get_forward_context = lambda: SimpleNamespace(attn_metadata=None)

    device = torch.device(f"cuda:{rank}")
    attn = _build_attention(device)
    failures = []

    for layout in ("contiguous", "merged_projection"):
        probe = _make_shards(layout, 0, rank=rank, world=world, device=device)
        if not usp_mod._can_use_packed_qkv_a2a_4d(*probe, world):
            failures.append(f"{layout}: unpacked fallback taken, nothing to compare")
            continue

        strided_out, strided_contiguous = _run_layout(
            attn, strided=True, layout=layout, rank=rank, world=world, device=device
        )
        copied_out, copied_contiguous = _run_layout(
            attn, strided=False, layout=layout, rank=rank, world=world, device=device
        )

        if any(strided_contiguous):
            failures.append(
                f"{layout}: strided path handed AITer contiguous Q/K/V "
                f"({strided_contiguous}); the comparison proves nothing"
            )
        if not all(copied_contiguous):
            failures.append(
                f"{layout}: copying path handed AITer strided Q/K/V "
                f"({copied_contiguous})"
            )
        for step, (got, want) in enumerate(zip(strided_out, copied_out)):
            if not torch.equal(got, want):
                delta = (got.float() - want.float()).abs()
                failures.append(
                    f"{layout} step{step}: strided != copied, "
                    f"mae={delta.mean():.3e} max={delta.max():.3e}"
                )

    for failure in failures:
        print(f"FAILURE rank{rank}: {failure}", flush=True)
    return 1 if failures else 0


class TestUSPStridedQKVAITer(CustomTestCase):
    def test_strided_and_copied_qkv_agree_two_ranks(self):
        if not current_platform.is_hip():
            self.skipTest("AITer backend is ROCm-only")
        if importlib.util.find_spec("aiter") is None:
            self.skipTest("AITer is a ROCm-only dependency")
        if torch.cuda.device_count() < _WORLD:
            self.skipTest(f"needs {_WORLD} GPUs")
        port = str(find_available_port(29500))
        procs = []
        for rank in range(_WORLD):
            env = os.environ.copy()
            env.update(
                {
                    "RANK": str(rank),
                    "LOCAL_RANK": str(rank),
                    "WORLD_SIZE": str(_WORLD),
                    "MASTER_ADDR": "127.0.0.1",
                    "MASTER_PORT": port,
                }
            )
            procs.append(
                subprocess.Popen(
                    [sys.executable, __file__],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )
        outputs = [p.communicate(timeout=600)[0] for p in procs]
        if any(p.returncode for p in procs):
            self.fail("worker failed:\n" + "\n".join(outputs))


if __name__ == "__main__":
    if "RANK" in os.environ:
        sys.exit(_worker())
    unittest.main()
