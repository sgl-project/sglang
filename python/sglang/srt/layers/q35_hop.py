"""Decode hop kernels (TP sum + residual + RMSNorm, optional static FP8) for hidden 4096.

A drop-in for the FlashInfer MNNVL CuTe DSL low-latency path at small M: one publish launch
(the partial, or the finalized MoE rows) and one consume launch per hop, bit-identical to it.
Kernels: kernels/jit/csrc/distributed/q35_hop_kernels.cu on sgl_kernel/distributed/q35_hop.cuh,
built with nvcc on first use and driven through ctypes. The workspace is one multicast mailbox
and one control block per process, shared by every hop of every graph.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import logging
import os
import pathlib
import subprocess

import torch
import torch.distributed as dist

from sglang.kernels.jit.utils.compile.cache import cache_root

logger = logging.getLogger(__name__)

HIDDEN = 4096
STAGES = 3
ROW_CTAS = 4
_PUBLISH, _PUBLISH_FIN, _CONSUME = 0, 1, 2
_DRAFT_MNNVL, _DRAFT_ADD_NORM = 3, 4
# The same, releasing their dependents at entry.
_DRAFT_MNNVL_EARLY, _DRAFT_ADD_NORM_EARLY = 7, 8

_JIT_ROOT = pathlib.Path(__file__).resolve().parents[2] / "kernels" / "jit"
_SOURCE = _JIT_ROOT / "csrc" / "distributed" / "q35_hop_kernels.cu"
_HEADER = _JIT_ROOT / "include" / "sgl_kernel" / "distributed" / "q35_hop.cuh"


class _Site(ctypes.Structure):
    _fields_ = [
        ("mc", ctypes.c_void_p),
        ("uc", ctypes.c_void_p),
        ("ctrl", ctypes.c_void_p),
        ("rank", ctypes.c_int),
        ("world", ctypes.c_int),
        ("m_cap", ctypes.c_int),
        ("timeout_ns", ctypes.c_ulonglong),
    ]


class _Args(ctypes.Structure):
    _fields_ = (
        [("site", _Site)]
        + [
            (name, ctypes.c_void_p)
            for name in (
                "partial",
                "routed",
                "expert_w",
                "permuted",
                "shared",
                "residual_in",
                "gamma",
                "fp8_scale",
                "residual_out",
                "norm_out",
                "fp8_out",
            )
        ]
        + [
            ("m", ctypes.c_int),
            ("top_k", ctypes.c_int),
            ("eps", ctypes.c_float),
            ("weight_bias", ctypes.c_float),
            ("delay_ns", ctypes.c_int),
        ]
    )


class _OprojArgs(ctypes.Structure):
    _fields_ = (
        [("site", _Site)]
        + [(name, ctypes.c_void_p) for name in ("w", "x", "sx", "sw", "out")]
        + [("m", ctypes.c_int)]
    )


def pack_out_proj(weight_nk: torch.Tensor) -> torch.Tensor:
    """[4096, K] e4m3 row-major -> mma.m16n8k32 A fragments in the out_proj kernel's order:
    [cta 128][warp 8][k-step S][block 2][lane 32][16 B], K = 8 * S * 32."""
    w = weight_nk.contiguous().view(torch.uint8)
    n, k = w.shape
    steps = k // 256
    if n != HIDDEN or k != 8 * steps * 32:
        raise ValueError(
            f"out_proj packing needs [4096, 256 * S], got {tuple(w.shape)}"
        )
    v = w.view(n // 32, 2, 2, 8, 8, steps, 2, 4, 4)
    return v.permute(0, 4, 5, 1, 3, 7, 6, 2, 8).contiguous().view(-1)


def _build() -> ctypes.CDLL:
    text = _SOURCE.read_bytes() + _HEADER.read_bytes()
    digest = hashlib.sha256(text).hexdigest()[:16]
    cache = cache_root() / "decode_hop"
    cache.mkdir(parents=True, exist_ok=True)
    lib = cache / f"libq35hop.{digest}.so"
    # One build per node: the ranks of a node share the cache directory.
    with open(cache / f".lock.{digest}", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not lib.exists():
            tmp = cache / f"libq35hop.{digest}.{os.getpid()}.tmp"
            cmd = [
                "nvcc",
                "-O3",
                "-std=c++17",
                "-arch=sm_103a",
                "-shared",
                "-Xcompiler",
                "-fPIC",
                f"-I{_JIT_ROOT / 'include'}",
                "-o",
                str(tmp),
                str(_SOURCE),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"q35_hop build failed: {res.stderr[-3000:]}")
            os.replace(tmp, lib)
    handle = ctypes.CDLL(str(lib))
    logger.info("q35_hop: loaded %s", lib)
    if handle.q35hop_args_size() != ctypes.sizeof(_Args):
        raise RuntimeError("q35_hop Args layout mismatch")
    if handle.q35hop_oproj_args_size() != ctypes.sizeof(_OprojArgs):
        raise RuntimeError("q35_hop OprojArgs layout mismatch")
    return handle


# The process's hop workspace once built; layers that push into it look it up here.
_active: Q35Hop | None = None


def active() -> Q35Hop | None:
    return _active


def input_fp8_linear(layer: torch.nn.Module) -> torch.nn.Module | None:
    """The static per-tensor FP8 linear that reads a decoder layer's input norm, if any."""
    for owner, name in (("linear_attn", "in_proj_qkvz"), ("self_attn", "qkv_proj")):
        block = layer._modules.get(owner)
        linear = None if block is None else block._modules.get(name)
        if (
            linear is not None
            and type(linear.quant_method).__name__ == "ModelOptFp8LinearMethod"
        ):
            return linear
    return None


class Q35Hop:
    """Process-local workspace and launchers; construct on every TP rank before graph capture."""

    def __init__(
        self,
        *,
        max_m: int,
        rms_epsilon: float,
        process_group: dist.ProcessGroup,
        device: torch.device,
        timeout_s: float = 2.0,
    ) -> None:
        import torch.distributed._symmetric_memory as symm_mem
        from flashinfer.comm.mnnvl_cutedsl.symmetric_buffer import SymmetricBuffer

        self.world = dist.get_world_size(process_group)
        if self.world not in (2, 4, 8):
            raise ValueError(f"q35_hop supports tp 2 / 4 / 8, got {self.world}")
        self.rank = dist.get_rank(process_group)
        self.max_m = int(max_m)
        self.eps = float(rms_epsilon)
        self.device = device
        self._lib = _build()
        if symm_mem.get_backend(device) is None:
            symm_mem.set_backend("NCCL")
        self._mailbox = SymmetricBuffer.allocate(
            (STAGES, self.world, self.max_m, HIDDEN),
            torch.int16,
            device,
            process_group,
            require_multicast=True,
        )
        self._mailbox.tensor.fill_(-32768)  # every slot holds the bf16 -0.0 sentinel
        self._ctrl = torch.zeros(4, dtype=torch.int32, device=device)
        self._site = _Site(
            self._mailbox.multicast_address,
            self._mailbox.tensor.data_ptr(),
            self._ctrl.data_ptr(),
            self.rank,
            self.world,
            self.max_m,
            int(timeout_s * 1e9),
        )
        # bf16 partial of a producer that already pushed it; the next all-reduce call on exactly
        # this tensor only consumes.
        self._pushed: tuple[int, torch.Size] | None = None
        torch.cuda.synchronize(device)
        dist.barrier(group=process_group)
        global _active
        _active = self

    def supports(self, m: int) -> bool:
        return 1 <= m <= self.max_m

    def error_bits(self) -> int:
        """Sticky error word; non-zero means a hop timed out and its output was poisoned."""
        return int(self._ctrl[3].item())

    def _launch(self, which: int, args: _Args) -> None:
        rc = self._lib.q35hop_launch(
            which,
            ctypes.byref(args),
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        )
        if rc != 0:
            raise RuntimeError(f"q35hop_launch({which}) failed with {rc}")

    def _args(self, *, m, residual, gamma, fp8_scale, outs) -> _Args:
        a = _Args()
        a.site = self._site
        a.residual_in, a.gamma = residual.data_ptr(), gamma.data_ptr()
        a.fp8_scale = None if fp8_scale is None else fp8_scale.data_ptr()
        a.residual_out, a.norm_out = outs[1].data_ptr(), outs[0].data_ptr()
        a.fp8_out = None if len(outs) < 3 else outs[2].data_ptr()
        a.m, a.eps = m, self.eps
        return a

    def _outputs(self, m: int, fp8_scale):
        shape = (m, HIDDEN)
        norm = torch.empty(shape, dtype=torch.bfloat16, device=self.device)
        residual_out = torch.empty(shape, dtype=torch.bfloat16, device=self.device)
        if fp8_scale is None:
            return (norm, residual_out)
        return (
            norm,
            residual_out,
            torch.empty(shape, dtype=torch.float8_e4m3fn, device=self.device),
        )

    def out_proj_push(
        self, *, x_fp8, packed_w, input_scale, weight_scale
    ) -> torch.Tensor:
        """out_proj of an [M, K] static-FP8 activation whose result goes straight into the mailbox.

        Returns the bf16 partial (also written to global memory); its all-reduce must come next and
        only consumes.
        """
        m = int(x_fp8.shape[0])
        if self._pushed is not None:
            raise RuntimeError("q35_hop: a pushed partial was never consumed")
        out = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=self.device)
        o = _OprojArgs()
        o.site = self._site
        o.w, o.x = packed_w.data_ptr(), x_fp8.data_ptr()
        o.sx, o.sw = input_scale.data_ptr(), weight_scale.data_ptr()
        o.out, o.m = out.data_ptr(), m
        steps = packed_w.numel() // (HIDDEN * 256)
        rc = self._lib.q35hop_oproj(
            steps,
            ctypes.byref(o),
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        )
        if rc != 0:
            raise RuntimeError(f"q35hop_oproj failed with {rc}")
        self._pushed = (out.data_ptr(), out.shape)
        return out

    def site_fields(self) -> list[int]:
        """[mc, uc, ctrl, rank, world, m_cap, timeout_ns] for producers that push from their own kernel."""
        s = self._site
        return [s.mc, s.uc, s.ctrl, s.rank, s.world, s.m_cap, s.timeout_ns]

    def mark_pushed(self, partial: torch.Tensor) -> None:
        """Record that a producer kernel already pushed this bf16 partial; its all-reduce only consumes."""
        if self._pushed is not None:
            raise RuntimeError("q35_hop: a pushed partial was never consumed")
        self._pushed = (partial.data_ptr(), partial.shape)

    def has_pushed(self, partial: torch.Tensor) -> bool:
        return self._pushed is not None and self._pushed == (
            partial.data_ptr(),
            partial.shape,
        )

    def all_reduce_residual_rms_norm(
        self, *, local_contribution, residual, gamma, fp8_scale=None
    ):
        m = int(local_contribution.shape[0])
        outs = self._outputs(m, fp8_scale)
        a = self._args(
            m=m, residual=residual, gamma=gamma, fp8_scale=fp8_scale, outs=outs
        )
        if self.has_pushed(local_contribution):
            self._pushed = None
        else:
            if self._pushed is not None:
                raise RuntimeError("q35_hop: a pushed partial was never consumed")
            a.partial = local_contribution.data_ptr()
            self._launch(_PUBLISH, a)
        self._launch(_CONSUME, a)
        return outs

    def draft_all_reduce_rms_norm(
        self,
        *,
        local_contribution,
        residual,
        gamma,
        eps,
        weight_bias,
        add_norm,
        early_release=False,
        delay_ns=0,
    ):
        """One draft-step hop with the rounding of the kernels it replaces. add_norm=False: FlashInfer's
        legacy MNNVL fused all-reduce, gamma the applied multiplier (weight_bias 0). add_norm=True: the
        one-shot all-reduce, then FlashInfer's CuTe DSL FusedAddRMSNorm with multiplier weight_bias + gamma.
        early_release: the consume releases its dependents at entry (they must read only static data before
        their PDL wait). delay_ns: diagnostic hold of the output stores."""
        m = int(local_contribution.shape[0])
        outs = self._outputs(m, None)
        a = self._args(m=m, residual=residual, gamma=gamma, fp8_scale=None, outs=outs)
        a.eps, a.weight_bias, a.delay_ns = eps, weight_bias, delay_ns
        if self.has_pushed(local_contribution):
            self._pushed = None
        else:
            if self._pushed is not None:
                raise RuntimeError("q35_hop: a pushed partial was never consumed")
            a.partial = local_contribution.data_ptr()
            self._launch(_PUBLISH, a)
        if early_release:
            self._launch(_DRAFT_ADD_NORM_EARLY if add_norm else _DRAFT_MNNVL_EARLY, a)
        else:
            self._launch(_DRAFT_ADD_NORM if add_norm else _DRAFT_MNNVL, a)
        return outs

    def moe_finalize_all_reduce_rms_norm(
        self,
        *,
        routed_output,
        expert_weights,
        permuted_indices,
        gated_shared_output,
        residual,
        gamma,
        fp8_scale=None,
    ):
        if self._pushed is not None:
            raise RuntimeError("q35_hop: a pushed partial was never consumed")
        m = int(permuted_indices.shape[0])
        outs = self._outputs(m, fp8_scale)
        a = self._args(
            m=m, residual=residual, gamma=gamma, fp8_scale=fp8_scale, outs=outs
        )
        a.routed, a.expert_w = routed_output.data_ptr(), expert_weights.data_ptr()
        a.permuted, a.top_k = (
            permuted_indices.data_ptr(),
            int(permuted_indices.shape[1]),
        )
        a.shared = (
            None if gated_shared_output is None else gated_shared_output.data_ptr()
        )
        self._launch(_PUBLISH_FIN, a)
        self._launch(_CONSUME, a)
        return outs


def prepare_out_proj(model: torch.nn.Module) -> int:
    """Pack the FP8 out_proj weights of every GatedDeltaNet mixer for out_proj_push; returns count."""
    packed = 0
    for module in model.modules():
        if type(module).__name__ != "Qwen3_5GatedDeltaNet":
            continue
        linear = module._modules["out_proj"]
        linear.q35_packed = None
        method = linear.quant_method
        if (
            type(method).__name__ != "ModelOptFp8LinearMethod"
            or linear.weight.dtype != torch.float8_e4m3fn
            or linear.input_scale.numel() != 1
            or linear.weight_scale.numel() != 1
        ):
            continue
        # The method keeps the weight as the [K, N] transpose of a row-major [N, K] tensor.
        weight_nk = linear.weight.t()
        if weight_nk.shape[0] != HIDDEN or weight_nk.shape[1] not in (1024, 2048):
            continue
        linear.q35_packed = pack_out_proj(weight_nk)
        packed += 1
    return packed
