"""Hierarchical wall-clock timer for diffusion server/pipeline startup.

Answers "what is actually taking time when we launch our models?" (see
sgl-project/sglang#19087) without needing an ad-hoc print-timing pass each
time. Disabled by default (SGLANG_DIFFUSION_STARTUP_PROFILE=0): the context
manager is then a plain pass-through with no timing or bookkeeping overhead.
"""

import time
from contextlib import contextmanager
from typing import Iterator

import sglang.multimodal_gen.envs as envs
from sglang.multimodal_gen.runtime.utils.logging_utils import (
    get_is_main_process,
    init_logger,
)

logger = init_logger(__name__)


class _Phase:
    __slots__ = ("name", "duration_ms", "children")

    def __init__(self, name: str):
        self.name = name
        self.duration_ms: float = 0.0
        self.children: list[_Phase] = []


class StartupProfiler:
    """Nests `phase()` calls into a tree and renders a flat, dotted-path summary.

    Percentages are relative to the immediate parent phase, matching the
    breakdown format worked out in #19087 (e.g. `load_modules.text_encoder:
    32960ms (52%)` is 52% of `load_modules`, not of the whole startup).
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._root = _Phase("root")
        self._stack: list[_Phase] = [self._root]

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        node = _Phase(name)
        self._stack[-1].children.append(node)
        self._stack.append(node)
        start = time.perf_counter()
        try:
            yield
        finally:
            node.duration_ms = (time.perf_counter() - start) * 1000
            self._stack.pop()

    def render(self) -> str:
        lines: list[str] = []

        def walk(node: _Phase, prefix: str, parent_ms: float):
            qualified = f"{prefix}.{node.name}" if prefix else node.name
            pct = (node.duration_ms / parent_ms * 100) if parent_ms > 0 else 0.0
            lines.append(f"{qualified}: {node.duration_ms:.2f}ms ({pct:.1f}%)")
            for child in node.children:
                walk(child, qualified, node.duration_ms)

        for top in self._root.children:
            walk(top, "", top.duration_ms)
        return "\n".join(lines)


_profiler: StartupProfiler | None = None


def get_startup_profiler() -> StartupProfiler:
    global _profiler
    if _profiler is None:
        _profiler = StartupProfiler(enabled=envs.SGLANG_DIFFUSION_STARTUP_PROFILE)
    return _profiler


def startup_phase(name: str):
    """`with startup_phase("build_pipeline"): ...` — see `StartupProfiler.phase`."""
    return get_startup_profiler().phase(name)


def log_startup_summary() -> None:
    """Log the breakdown once. Every rank runs the same startup path, so only
    rank 0 reports -- otherwise an 8-GPU launch prints eight identical trees."""
    profiler = get_startup_profiler()
    if not profiler.enabled or not get_is_main_process():
        return
    summary = profiler.render()
    if summary:
        logger.info("[Startup Profile]\n%s", summary)
