# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runtime state for CUDA-graph-compatible dumping.

One process-global object, `cuda_graph_dump`, owns the buffer registry and the
tap/collect dispatch.  It is deliberately a singleton: the taps are reached from
`nn.Module` forward hooks that have no channel to receive an object, and the
collect points are backend methods reached through `__init_subclass__` seams.

The dispatch ladder in `tap()` is the whole design in six lines; see the
docstring there.
"""

from __future__ import annotations

import atexit
import logging
from contextlib import contextmanager
from typing import Any, Hashable, Iterator, Optional

import torch

from sglang.srt.debug_utils.cuda_graph.config import CudaGraphDumpConfig
from sglang.srt.debug_utils.cuda_graph.registry import BufferKey, BufferRegistry

logger = logging.getLogger(__name__)

_TAP_IN_GRAPH = "t1"
_TAP_COMPILED = "t3"
_TAP_EAGER = "eager"


def _stream_is_capturing() -> bool:
    """`torch.cuda.is_current_stream_capturing()` without the CPU-build hazard.

    Only ever reached while a capture session is open, so the import-time cost
    of `torch.cuda` initialization is not on the normal path.
    """
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:  # no CUDA build, or no current device
        return False


def _row_pairs(runner: Any, shape_token: Hashable) -> list[tuple[int, int]]:
    """Candidate `(padded_rows, raw_rows)` pairs for this replay.

    A graph is captured at a padded shape and replayed with the tail rows still
    holding whatever the previous replay left there, so trimming is mandatory.
    But a dump name may be anything -- token-major, request-major, per-expert,
    a scalar -- so the axis cannot be assumed.  Both row bases the runners
    actually pad along are offered here, and `_trim` only fires on an exact
    match of dim 0; anything else is written at full width, with `graph_size` in
    the tags so the padding is visible rather than silently believed.

    The raw counters are read here, at collect time, and not captured when the
    scope opens: `load_batch` (which sets them) runs *inside* `execute`.
    """
    raw_tokens = getattr(runner, "raw_num_token", None) or getattr(
        runner, "raw_num_tokens", None
    )
    raw_bs = getattr(runner, "raw_bs", None)
    size = getattr(shape_token, "size", None)
    padded_bs = getattr(runner, "bs", None) or size
    width = getattr(runner, "captured_req_width", None) or 1
    if getattr(runner, "ragged_verify_mode", False) or hasattr(
        runner, "raw_num_tokens"
    ):
        # ragged decode keys on num_tokens; prefill always does
        padded_tokens = size
    else:
        padded_tokens = padded_bs * width if padded_bs else None
    pairs = []
    for padded, raw in ((padded_tokens, raw_tokens), (padded_bs, raw_bs)):
        if padded and raw and raw < padded and (padded, raw) not in pairs:
            pairs.append((padded, raw))
    return pairs


def _trim(tensor: torch.Tensor, pairs: list[tuple[int, int]]) -> torch.Tensor:
    if tensor.dim() == 0:
        return tensor
    for padded, raw in pairs:
        if tensor.shape[0] == padded:
            return tensor[:raw]
    return tensor


class _CudaGraphDumpState:
    """Process-global owner of the dump buffers and the tap/collect dispatch."""

    def __init__(self) -> None:
        self._config = CudaGraphDumpConfig()
        self._registry: Optional[BufferRegistry] = None
        # occurrence counter per fully-expanded dump name, reset once per
        # `forward_fn()` invocation.  Bumped *before* the channel decision so
        # the eager and graph channels can never disagree about which
        # occurrence a frame belongs to.
        self._occurrences: dict[str, int] = {}
        # non-zero while a backend is being constructed / compiled: frames seen
        # here carry dummy values and must not reach the eager writer.
        self._setup_depth = 0
        self._capture_depth = 0
        self._shape_token: Hashable = None
        self._runner_ctx: dict[str, Any] = {}
        self._channel_of: dict[BufferKey, str] = {}
        self._tap_hits = 0
        self._compiled_tap_hits = 0
        self._collect_calls = 0
        self._reported = False

    # ------------------------------- lifecycle ------------------------------

    def _ensure_configured(self) -> None:
        if self._registry is not None:
            return
        # Imported here, not at module scope: `dumper` imports *this* module to
        # reach `tap()`, and its module-level singleton is built at import time.
        from sglang.srt.debug_utils.dumper import dumper

        self._config = CudaGraphDumpConfig.from_dumper_config(dumper._config)
        self._registry = BufferRegistry(
            budget_bytes=self._config.budget_bytes,
            accepts=self._config.accepts,
            strict=self._config.strict,
        )
        # The only trigger that is guaranteed to run after every capture *and*
        # every replay.  `BaseCudaGraphBackend.cleanup` would be the natural
        # lifecycle hook, but nothing in the tree calls it, so a probe hung
        # there would never fire.  `report()` is one-shot, so an explicit call
        # from a test or from `cleanup` still wins the race harmlessly.
        atexit.register(self._report_at_exit)

    @property
    def enabled(self) -> bool:
        return self._registry is not None and self._config.enable

    @property
    def registry(self) -> Optional[BufferRegistry]:
        return self._registry

    def configure(self) -> None:
        """Idempotent; safe to call from every seam entry point."""
        self._ensure_configured()

    def begin_forward(self) -> None:
        self._occurrences.clear()

    # --------------------------------- taps ---------------------------------

    def tap(self, name: str, value: Any) -> bool:
        """Take `value` through whichever channel is live.  True = handled.

        The ladder, in priority order:

        1. **not enabled** -> False, the caller writes the file eagerly as it
           always has.
        2. **dynamo is tracing** (`tc_piecewise`) -> emit the opaque custom op,
           which becomes a piece boundary and does the `copy_` from inside the
           piece's graph.  A `dumper.dump(...)` here would be a hard trace
           error under `fullgraph=True`.
        3. **a capture session is open and the stream is capturing**
           (`full`, and the captured segments of `breakable`) -> `copy_` into
           the graph-resident buffer, which is recorded.
        4. **setting up or capturing, but neither channel matched** -> swallow.
           This is a warmup / compile-pass / break-body forward whose values are
           dummies; letting it reach the eager writer would publish garbage.
        5. **otherwise** -> False.  Genuine eager execution: `disabled` mode, or
           an `eager_on_graph` break body at replay time, where capture is off
           on both sides and the ordinary eager path is exactly right.
        """
        if self._registry is None:
            if torch.compiler.is_compiling():
                return False
            self._ensure_configured()
        if not self._config.enable or not isinstance(value, torch.Tensor):
            return False

        key = self._next_key(name)
        if torch.compiler.is_compiling():
            return self._emit_compiled_tap(key, value)
        if self._capture_depth and _stream_is_capturing():
            self._record(key, value, _TAP_IN_GRAPH)
            return True
        return bool(self._setup_depth or self._capture_depth)

    def _next_key(self, name: str) -> BufferKey:
        occurrence = self._occurrences.get(name, 0)
        self._occurrences[name] = occurrence + 1
        return BufferKey(name, occurrence)

    def _record(self, key: BufferKey, tensor: torch.Tensor, channel: str) -> None:
        slot = self._registry.record(key, self._shape_token, tensor)
        if slot is None:
            return  # filtered out, or the budget ran out: drop it
        slot.copy_(tensor.reshape(-1))
        self._channel_of[key] = channel
        self._tap_hits += 1

    def _emit_compiled_tap(self, key: BufferKey, tensor: torch.Tensor) -> bool:
        # `split_op` is imported lazily so that a plain `import sglang` does not
        # register a torch custom op as a side effect.
        from sglang.srt.debug_utils.cuda_graph import split_op

        split_op.dumper_tap(tensor, self._registry.tag_for(key))
        self._compiled_tap_hits += 1
        return True

    def tap_by_tag(self, tag: int, tensor: torch.Tensor) -> None:
        """Body of the T3 custom op; runs at execution time, not at trace time.

        First execution lands in the largest-first compile-warmup loop with real
        tensors at the maximum shape, which is what makes allocate-on-first-use
        safe here.
        """
        if not self.enabled:
            return
        self._record(self._registry.key_for_tag(tag), tensor, _TAP_COMPILED)

    def eager_tags(self) -> dict[str, Any]:
        """Tags for a frame `tap()` declined, so the eager channel is labelled.

        Empty when the feature is off, so a default run's frames are exactly
        what they always were.  Enabled, it is what makes the coverage claim
        checkable: the acceptance gate asserts that the `t1`/`t3` frames plus
        the `eager` frames equal the pure-eager module set, and "no tap tag"
        would be indistinguishable from a baseline run's frames.
        """
        return {"tap": _TAP_EAGER} if self.enabled else {}

    # -------------------------------- scopes --------------------------------

    @contextmanager
    def setup_scope(self) -> Iterator[None]:
        """Around backend construction and compilation.  Suppresses dummies."""
        self._ensure_configured()
        self._setup_depth += 1
        try:
            yield
        finally:
            self._setup_depth -= 1

    @contextmanager
    def capture_scope(self) -> Iterator[None]:
        """Around one capture session; arms the T1 stream-capturing check."""
        self._ensure_configured()
        self._capture_depth += 1
        try:
            yield
        finally:
            self._capture_depth -= 1

    @contextmanager
    def shape_scope(self, token: Hashable) -> Iterator[None]:
        previous = self._shape_token
        self._shape_token = token
        try:
            yield
        finally:
            self._shape_token = previous

    @contextmanager
    def runner_scope(self, runner: Any, **tags: Any) -> Iterator[None]:
        """Publish the replay-side runner so `collect` can read it lazily."""
        previous = self._runner_ctx
        self._runner_ctx = {
            "runner": runner,
            "tags": {k: v for k, v in tags.items() if v is not None},
        }
        try:
            yield
        finally:
            self._runner_ctx = previous

    def wrap_forward_fn(self, forward_fn):
        """Reset the occurrence counters once per `forward_fn()` invocation.

        `TcPiecewiseCudaGraphBackend.capture_one` calls `forward_fn()` twice, so
        resetting per capture_one would make the second pass allocate a second
        buffer for every name.
        """

        def wrapped(*args, **kwargs):
            self.begin_forward()
            return forward_fn(*args, **kwargs)

        return wrapped

    # -------------------------------- collect -------------------------------

    def collect(self, shape_token: Hashable, **extra_tags: Any) -> int:
        """Write one frame per buffer recorded under `shape_token`.

        Called from the replay seam, *outside* the graph: the values are this
        replay's, the metadata is this replay's `ForwardBatch`, and
        `dumper.dump` runs on the host as it always has.
        """
        if not self.enabled:
            return 0
        from sglang.srt.debug_utils.dumper import dumper

        ctx = self._runner_ctx
        runner = ctx.get("runner")
        pairs = _row_pairs(runner, shape_token) if runner is not None else []
        tags = {**ctx.get("tags", {}), **extra_tags}
        written = 0
        for key, view in self._registry.views(shape_token):
            frame_tags = {
                "tap": self._channel_of.get(key),
                "occurrence": key.occurrence or None,
                **tags,
            }
            dumper.dump(
                key.name,
                _trim(view, pairs),
                **{k: v for k, v in frame_tags.items() if v is not None},
            )
            written += 1
        self._collect_calls += 1
        return written

    # ------------------------------ diagnostics -----------------------------

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "buffers": 0 if self._registry is None else self._registry.num_buffers,
            "used_mb": (
                0.0 if self._registry is None else self._registry.used_bytes / (1 << 20)
            ),
            "tap_hits": self._tap_hits,
            "compiled_tap_hits": self._compiled_tap_hits,
            "collect_calls": self._collect_calls,
        }

    def report(self) -> None:
        """Fails loudly, never silently.  One-shot per process.

        Zero taps with the feature switched on means the capture path did not
        run at all -- almost always a filter that matches no name.  That is
        exactly the silent hole this package exists to close, so it is reported
        rather than ignored, and raised in strict mode.
        """
        if not self.enabled or self._reported:
            return
        self._reported = True
        summary = self.summary()
        if self._tap_hits == 0:
            message = (
                "cuda-graph dumping was enabled but no tap fired; no module "
                "inside a captured region was dumped. Check "
                "DUMPER_CUDA_GRAPH_FILTER against the dump names, and that the "
                "non-intrusive dumper is attached."
            )
            if self._config.strict:
                raise RuntimeError(message)
            logger.warning(message)
        else:
            logger.info("cuda-graph dump summary: %s", summary)

    def _report_at_exit(self) -> None:
        """`report()` for the `atexit` path: never escapes into shutdown noise."""
        try:
            self.report()
        except Exception as error:  # strict mode, or a teardown race
            logger.error("cuda-graph dump report failed: %s", error)


cuda_graph_dump = _CudaGraphDumpState()
