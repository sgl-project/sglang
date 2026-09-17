# SPDX-License-Identifier: Apache-2.0
"""Choose a layer's attention backend by timing the candidates on its own tensors.

Which backend is fastest is not a property of the GPU alone. On sm12x no FA
kernel exists and cuDNN beats torch's flash path at head_dim 128 but loses to it
at head_dim 64 and long sequences; on Hopper the FA backend beats cuDNN. Nor do
synthetic timings settle it: the tensors here are non-contiguous views into a
packed QKV buffer and backends differ in how they take that, so the measurement
uses what the layer was actually handed, on its first forward large enough to be
worth deciding on.
"""

from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionImpl,
    wrap_attention_impl_forward,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Backends only separate on the calls that carry the runtime; tuning on a short
# text or audio stream picks the wrong winner for the long video one.
_MIN_TUNE_NUMEL = 4 << 20
# A candidate has to beat the incumbent by more than the spread of these timings.
_MIN_RELATIVE_GAIN = 0.02
_WARMUP_ITERS = 3
_TIMED_ITERS = 8
# A backend that disagrees this much is not computing the same attention,
# whatever its timing says.
_MAX_OUTPUT_DEVIATION = 0.05

_reported = False


def _timed(impl: AttentionImpl, args, kwargs) -> float:
    for _ in range(_WARMUP_ITERS):
        impl.forward(*args, **kwargs)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(_TIMED_ITERS):
        impl.forward(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / _TIMED_ITERS


def _leading_tensor(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def _agrees(candidate_out, reference_out) -> bool:
    got, want = _leading_tensor(candidate_out), _leading_tensor(reference_out)
    if not (isinstance(got, torch.Tensor) and isinstance(want, torch.Tensor)):
        return False
    if got.shape != want.shape:
        return False
    scale = want.float().abs().max().clamp_min(1e-3)
    return bool(
        ((got.float() - want.float()).abs().max() / scale) <= _MAX_OUTPUT_DEVIATION
    )


def _candidates(layer) -> list[tuple[str, AttentionImpl, AttentionBackendEnum]]:
    from sglang.multimodal_gen.runtime.layers.attention.selector import get_attn_backend

    ctor_kwargs = layer._attn_impl_ctor_kwargs
    built: list[tuple[str, AttentionImpl, AttentionBackendEnum]] = []

    # Whether SDPA is allowed to reach for cuDNN is a backend choice of its own,
    # and on sm12x it is the only one there is.
    if layer.backend is AttentionBackendEnum.TORCH_SDPA:
        flipped = not ctor_kwargs.get("allow_cudnn_sdp", False)
        built.append(
            (
                f"torch_sdpa(cudnn={flipped})",
                type(layer.attn_impl)(**{**ctor_kwargs, "allow_cudnn_sdp": flipped}),
                layer.backend,
            )
        )

    for target in sorted(
        layer._supported_attention_backends or (), key=lambda backend: backend.name
    ):
        if target is layer.backend:
            continue
        try:
            backend_cls = get_attn_backend(
                layer.head_size,
                layer.dtype,
                supported_attention_backends=layer._supported_attention_backends,
                selected_attention_backend=target,
            )
            if backend_cls.get_enum() is not target:
                continue
            built.append(
                (target.name.lower(), backend_cls.get_impl_cls()(**ctor_kwargs), target)
            )
        except Exception as exc:
            logger.debug("attention autotune: %s unavailable (%s)", target, exc)
    return built


def _choose(layer, args, kwargs) -> tuple[AttentionImpl, AttentionBackendEnum] | None:
    """The fastest candidate that agrees with the incumbent, or None to keep it."""
    global _reported

    incumbent = layer.attn_impl
    reference = incumbent.forward(*args, **kwargs)
    incumbent_label = f"{layer.backend.name.lower()} (current)"
    timings: dict[str, tuple[float, AttentionImpl | None, AttentionBackendEnum]] = {
        incumbent_label: (_timed(incumbent, args, kwargs), None, layer.backend)
    }

    for label, candidate, enum in _candidates(layer):
        try:
            output = candidate.forward(*args, **kwargs)
        except Exception as exc:
            logger.debug("attention autotune: %s failed (%s)", label, exc)
            continue
        if not _agrees(output, reference):
            logger.debug(
                "attention autotune: %s disagrees with %s", label, incumbent_label
            )
            continue
        timings[label] = (_timed(candidate, args, kwargs), candidate, enum)

    incumbent_ms = timings[incumbent_label][0]
    best_label = min(timings, key=lambda label: timings[label][0])
    best_ms = timings[best_label][0]

    say = logger.debug if _reported else logger.info
    _reported = True
    report = ", ".join(f"{label} {ms:.3f}ms" for label, (ms, *_) in timings.items())
    if best_label == incumbent_label or best_ms > incumbent_ms * (
        1 - _MIN_RELATIVE_GAIN
    ):
        say("attention autotune: keeping %s (%s)", incumbent_label, report)
        return None
    say(
        "attention autotune: %s -> %s, %.1f%% faster (%s)",
        incumbent_label,
        best_label,
        100 * (1 - best_ms / incumbent_ms),
        report,
    )
    return timings[best_label][1], timings[best_label][2]


def install(layer) -> None:
    """Tune this layer on its first forward worth measuring, then step aside."""
    impl = layer.attn_impl
    default_forward = impl.forward

    def tuning_forward(*args, **kwargs):
        query = args[0] if args else kwargs.get("query")
        if not isinstance(query, torch.Tensor) or query.numel() < _MIN_TUNE_NUMEL:
            return default_forward(*args, **kwargs)
        impl.forward = default_forward
        try:
            winner = _choose(layer, args, kwargs)
        except Exception as exc:
            logger.warning_once(
                f"attention autotune failed, keeping the default: {exc}"
            )
            return default_forward(*args, **kwargs)
        if winner is None:
            return default_forward(*args, **kwargs)
        impl_choice, backend_choice = winner
        layer.attn_impl = wrap_attention_impl_forward(impl_choice)
        layer.backend = backend_choice
        return layer.attn_impl.forward(*args, **kwargs)

    impl.forward = tuning_forward
