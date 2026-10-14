"""Mono (persistent fused decode-layer) kernel runtime; see spec.py for the model contract."""

from sglang.srt.layers.mono.layer import MonoRuntime, attach

__all__ = ["MonoRuntime", "attach"]
