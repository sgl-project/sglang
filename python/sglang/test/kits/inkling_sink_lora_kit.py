"""Inkling sink fixtures and a reference over raw, pre-scaled MoE pool buffers."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang.kernels.ops.moe.inkling_moe import silu_and_mul_triton  # noqa: E402
from sglang.srt.lora.backend.triton_v2_backend import TritonV2LoRABackend  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.models.inkling_common.lora import (  # noqa: E402
    InklingBatchDenseMLPWithLoRAV2,
)
from sglang.srt.models.inkling_common.util import deinterleave_gate_up  # noqa: E402
from sglang.srt.runtime_context import get_context, get_parallel  # noqa: E402

DEVICE = torch.device("cuda")

DTYPE = torch.bfloat16

SLOTS, RMAX = 4, 32

RANKS = [32, 16, 0, 8]  # slot 2 holds no adapter

HIDDEN, N_EXPERTS, F = 256, 2, 128

TOL = dict(rtol=3e-2, atol=3e-2)

V2 = InklingBatchDenseMLPWithLoRAV2


def _fill_slot(bufs, slot: int, rank: int, g: torch.Generator) -> None:
    """One slot as the pool writes it: max-rank spaced, zero tails, pre-scaled B."""
    a_gu, b_gu, a_dn, b_dn = bufs
    outer, f, rmax = a_gu.shape[1], b_gu.shape[2] // 2, b_gu.shape[3]
    for buf in bufs:
        buf[slot].zero_()
    if rank == 0:
        return

    def rnd(*shape):
        return (torch.randn(*shape, generator=g, device=DEVICE) * 0.05).to(DTYPE)

    for half in range(2):
        a_gu[slot, :, half * rmax : half * rmax + rank] = rnd(outer, rank, HIDDEN)
    b_gu[slot, :, :, :rank] = rnd(N_EXPERTS, 2 * f, rank)
    a_dn[slot, :, :rank] = rnd(N_EXPERTS, rank, f)
    b_dn[slot, :, :, :rank] = rnd(outer, HIDDEN, rank)


def _pool(shared_outer: bool, seed: int, f: int = F, ranks=RANKS, rmax: int = RMAX):
    outer = 1 if shared_outer else N_EXPERTS
    bufs = (
        torch.zeros(SLOTS, outer, 2 * rmax, HIDDEN, device=DEVICE, dtype=DTYPE),
        torch.zeros(SLOTS, N_EXPERTS, 2 * f, rmax, device=DEVICE, dtype=DTYPE),
        torch.zeros(SLOTS, N_EXPERTS, rmax, f, device=DEVICE, dtype=DTYPE),
        torch.zeros(SLOTS, outer, HIDDEN, rmax, device=DEVICE, dtype=DTYPE),
    )
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    for slot, rank in enumerate(ranks):
        _fill_slot(bufs, slot, rank, g)
    return bufs


def _swiglu_interleaved(y, gammas):
    """Run the fused gated activation for the served interleaved layout."""
    t, n, two_f = y.shape
    return silu_and_mul_triton(y.reshape(t * n, two_f), gammas.reshape(-1)).view(
        t, n, two_f // 2
    )


def _base_w13(w13_lin):
    """Convert each expert to [gate | up]; reference weights stay interleaved."""
    rows = w13_lin.shape[0] // N_EXPERTS
    return torch.stack(
        [deinterleave_gate_up(e, dim=0) for e in w13_lin.view(N_EXPERTS, rows, -1)]
    ).reshape_as(w13_lin)


def _reference(x, gammas, w13_lin, w2_lin, bufs, token_slots, shared_outer):
    a_gu, b_gu, a_dn, b_dn = (b.float() for b in bufs)
    t, n, f, rmax = x.shape[0], N_EXPERTS, b_gu.shape[2] // 2, b_gu.shape[3]
    y = (x.float() @ w13_lin.float().T).view(t, n, 2 * f)
    for i, s in enumerate(token_slots):
        if s < 0:
            continue
        for e in range(n):
            a = a_gu[s, 0 if shared_outer else e]  # [2R, hidden]
            bridge = (
                (x[i].float() @ a.T).to(DTYPE).float()
            )  # bf16 bridge, as the kernels
            y[i, e, 0::2] += bridge[:rmax] @ b_gu[s, e, :f].T
            y[i, e, 1::2] += bridge[rmax:] @ b_gu[s, e, f:].T
    act = _swiglu_interleaved(y.to(DTYPE), gammas)  # [t, n, f]
    out = act.reshape(t, -1).float() @ w2_lin.float()
    for i, s in enumerate(token_slots):
        if s < 0:
            continue
        if shared_outer:
            bridge = sum(act[i, e].float() @ a_dn[s, e].T for e in range(n))
            out[i] += bridge.to(DTYPE).float() @ b_dn[s, 0].T
        else:
            for e in range(n):
                bridge = (act[i, e].float() @ a_dn[s, e].T).to(DTYPE).float()
                out[i] += bridge @ b_dn[s, e].T
    return out


def _batch(decode: bool, slots: list[int] | None = None, ranks=RANKS):
    seq_lens = [1] * 13 if decode else [5, 17, 1, 9]
    if slots is None:
        slots = [i % SLOTS for i in range(13)] if decode else [1, 0, 2, 3]
    fb = SimpleNamespace(
        forward_mode=ForwardMode.DECODE if decode else ForwardMode.EXTEND,
        batch_size=len(seq_lens),
        extend_seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE),
        extend_seq_lens_cpu=list(seq_lens),
        extend_num_tokens=int(sum(seq_lens)),
        spec_info=None,
        return_logprob=False,
        extend_logprob_start_lens_cpu=None,
    )
    token_slots = [
        slot if ranks[slot] > 0 else -1
        for slot, length in zip(slots, seq_lens)
        for _ in range(length)
    ]
    return fb, slots, token_slots


def _inputs(t: int, f: int = F, seed: int = 11):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(t, HIDDEN, generator=g, device=DEVICE, dtype=DTYPE)
    gammas = torch.rand(t, N_EXPERTS, generator=g, device=DEVICE, dtype=DTYPE) + 0.5
    w13_lin = (
        torch.randn(N_EXPERTS * 2 * f, HIDDEN, generator=g, device=DEVICE) * 0.02
    ).to(DTYPE)
    w2_lin = (torch.randn(N_EXPERTS * f, HIDDEN, generator=g, device=DEVICE) * 0.02).to(
        DTYPE
    )
    return x, gammas, w13_lin, w2_lin


def _layer(cls, backend, bufs, *, f: int = F, tp_size: int = 1, tp_rank: int = 0):
    """The LoRA wrapper on a bare sink (no base weights to build)."""
    layer = cls.__new__(cls)
    torch.nn.Module.__init__(layer)
    layer.n_shared_experts = N_EXPERTS
    layer.tp_group = None
    layer.inference_moe_w13_interleaved = True
    layer._linearized_bf16_enabled = True
    layer._w13_gate_up_contiguous = cls is V2
    layer.moe_tp_size, layer.moe_tp_rank = tp_size, tp_rank
    layer.intermediate_size_per_partition = f
    layer.initialize_lora(backend)
    if bufs is not None:
        layer.set_lora_info(*bufs)
    return layer


def _v2_backend():
    backend = TritonV2LoRABackend(SLOTS, DEVICE)
    backend.is_moe_lora = True
    return backend


SINK_PREFIX = "model.layers.0.mlp.experts.shared_experts"


def _real_sink(
    cls, *, f: int = F, tp_size: int = 1, tp_rank: int = 0, lora_backend=None
):
    """Construct under the serving configuration before any weight is loaded."""
    if lora_backend is None:
        lora_backend = "triton_v2" if cls is V2 else "triton"
    default = torch.get_default_dtype()
    torch.set_default_dtype(DTYPE)
    try:
        # The sink partitions its shared experts over the "tp" group, which
        # resolves to the published tp_size and the overridden tp_rank before
        # any process group exists.
        with (
            get_context().override_server_args(
                enable_lora=True, lora_backend=lora_backend, tp_size=tp_size
            ),
            get_parallel().override(tp_rank=tp_rank),
        ):
            layer = cls(
                n_shared_experts=N_EXPERTS,
                d_model=HIDDEN,
                shared_d_mlp=f * tp_size,
                layer_id=0,
                prefix=SINK_PREFIX,
                inference_moe_w13_interleaved=True,
                parallel_group="tp",
                linearized_bf16=True,
            )
    finally:
        torch.set_default_dtype(default)
    return layer.to(DEVICE)


def _load_w13(layer, w13_lin):
    """Load checkpoint [N, 2F, H] gate/up weights through the TP-sliced loader."""
    full = w13_lin.view(N_EXPERTS, -1, HIDDEN)
    layer.weight_loader_fused(
        layer.w13_weight, full, f"{SINK_PREFIX}.w13_weight", "w13"
    )


def _load_w2(layer, w2_lin):
    """Load the linearized [N*F, H] down rows as the checkpoint's [N, H, F]."""
    full = w2_lin.view(N_EXPERTS, -1, HIDDEN).transpose(1, 2).contiguous()
    layer.weight_loader_fused(layer.w2_weight, full, f"{SINK_PREFIX}.w2_weight", "w2")


def _prepare(backend, fb, slots, ranks=RANKS, *, graph=False, prefill_graph=False):
    backend.prepare_lora_batch(
        fb,
        slots,
        list(ranks),
        [1.0] * SLOTS,
        use_decode_cuda_graph=graph,
        use_prefill_cuda_graph=prefill_graph,
    )
    backend.batch_info.has_active_lora = any(
        slot >= 0 and ranks[slot] > 0 for slot in slots
    )


def _forward(layer, x, gammas, w13_lin, w2_lin):
    return layer._forward_bf16_linearized(
        x, gammas, (w13_lin, w2_lin), use_reduce_scatter=True
    )
