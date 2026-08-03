"""Auxiliary hidden-state capture for DFlash/DSpark speculative decoding.

``AuxCaptureSink`` collects selected per-layer residual hiddens into one fused
``[T, num_layers * H]`` buffer. ``AuxCaptureMixin`` stashes that buffer on the
inner transformer so the outer ``*ForCausalLM`` / ``*ForConditionalGeneration``
wrapper can pop it after forward.
"""

from __future__ import annotations

import torch

__all__ = ["AuxCaptureMixin", "AuxCaptureSink"]


class AuxCaptureSink:
    """Collects per-layer aux hiddens into one ``[T, num_layers * H]`` buffer.

    ``append`` is signature-compatible with ``list.append``, so call sites that
    thread a Python list (e.g. the layer-communicator capture path) can pass a
    sink unchanged. When ``static_buf`` is set (prefill CUDA-graph capture),
    rows are written into that stable-address buffer for graph replay.
    Consumers must read the buffer through ``finalize()``, which fails loudly
    if any capture layer skipped its write.
    """

    # Every write path copies into the fused buffer, so producers need not
    # clone a tensor they later mutate (the layer communicator keys the
    # defensive residual clone off this flag).
    copies_on_append = True

    def __init__(
        self, num_layers: int, static_buf: torch.Tensor | None = None
    ) -> None:
        self.num_layers = num_layers
        self.static_buf = static_buf
        self.buf: torch.Tensor | None = None
        self._idx = 0

    def _next_column(self, ref: torch.Tensor) -> torch.Tensor:
        """Allocate the fused buffer on first use and return the next column."""
        if self._idx >= self.num_layers:
            raise RuntimeError(
                f"AuxCaptureSink overfilled: configured for {self.num_layers} "
                f"capture layers but received write #{self._idx + 1}. The "
                "model's capture-layer selection disagrees with "
                "enable_aux_capture()."
            )
        if self.buf is None:
            if self.static_buf is not None:
                self.buf = self.static_buf[: ref.shape[0]]
            else:
                self.buf = ref.new_empty(ref.shape[0], self.num_layers * ref.shape[-1])
        width = ref.shape[-1]
        if width * self.num_layers != self.buf.shape[-1]:
            raise RuntimeError(
                f"AuxCaptureSink width mismatch: fused buffer is "
                f"{self.buf.shape[-1]} wide for {self.num_layers} layers, but "
                f"layer {self._idx} wrote a {width}-wide hidden. A mis-sized "
                "static buffer or heterogeneous capture widths would silently "
                "mis-slice the columns."
            )
        col = self.buf[:, self._idx * width : (self._idx + 1) * width]
        self._idx += 1
        return col

    def append(self, hidden: torch.Tensor) -> None:
        """Copy ``hidden`` into the next column of the fused buffer."""
        self._next_column(hidden).copy_(hidden)

    def append_add(
        self, hidden: torch.Tensor, residual: torch.Tensor | None
    ) -> None:
        """Write ``hidden + residual`` into the next column of the fused buffer.

        Falls back to a plain copy when ``residual`` is None (already folded
        into ``hidden`` by the layer).
        """
        if residual is None:
            self.append(hidden)
        elif torch.compiler.is_compiling():
            # Dynamo rejects `out=` into a non-contiguous view (the column is
            # strided by num_layers * H). Inductor fuses this add + copy into
            # one kernel, so the compiled path loses nothing.
            self._next_column(hidden).copy_(hidden + residual)
        else:
            torch.add(hidden, residual, out=self._next_column(hidden))

    def finalize(self) -> torch.Tensor:
        """Return the fused buffer, failing loudly on a partial capture.

        Every capture layer must have written its column: a forward path that
        bypasses capture layers (e.g. a TBO segment that does not thread the
        sink) would otherwise hand uninitialized or stale columns to the draft
        model with no error anywhere downstream.
        """
        if self._idx != self.num_layers:
            raise RuntimeError(
                f"AuxCaptureSink underfilled: {self._idx} of {self.num_layers} "
                "capture layers wrote their aux hidden state, so the fused "
                "buffer is unusable. A forward path that skips capture layers "
                "(e.g. a TBO segment that does not thread the sink) cannot "
                "produce aux hidden states."
            )
        return self.buf


class AuxCaptureMixin:
    """Stash protocol for DFlash/DSpark aux hidden states on inner models.

    Class-scoped defaults mean mixing in needs no ``__init__`` change; call
    ``enable_aux_capture`` after selecting capture layers. The stash is
    one-shot: the inner ``forward`` stashes the fused buffer, the outer wrapper
    pops it. Prefill CUDA-graph replay skips ``forward()`` Python, so the
    runner re-arms the stash from its static buffer before the eager tail pops.
    """

    aux_capture_enabled: bool = False
    num_aux_capture_layers: int = 0
    _aux_hidden_states: torch.Tensor | None = None
    _aux_capture_static_buf: torch.Tensor | None = None

    def enable_aux_capture(self, num_layers: int) -> None:
        """Enable aux capture for ``num_layers`` selected layers."""
        self.num_aux_capture_layers = num_layers
        self.aux_capture_enabled = True

    def set_aux_capture_static_buffer(self, buf: torch.Tensor | None) -> None:
        """Route capture into a caller-owned stable-address buffer.

        The prefill CUDA-graph runner sets this around capture and clears it
        afterward.
        """
        self._aux_capture_static_buf = buf

    def make_aux_sink(self) -> AuxCaptureSink | None:
        """Return a per-forward sink, or None when aux capture is disabled."""
        if not self.aux_capture_enabled:
            return None
        return AuxCaptureSink(
            num_layers=self.num_aux_capture_layers,
            static_buf=self._aux_capture_static_buf,
        )

    def stash_aux_hidden_states(
        self, aux_hidden_states: torch.Tensor | None
    ) -> None:
        """Stash the fused aux buffer for the outer wrapper to pop.

        Also used by the prefill CUDA-graph runner to re-arm after a replay
        that skipped ``forward()`` Python. In eager, stash/pop must stay
        strictly paired: stashing over an unconsumed buffer means an
        interleaved forward on this module (or one aborted between stash and
        pop) and would silently leak one request's aux into another, so it
        fails loudly instead. Runners that drive the inner model without its
        wrapper (which never pops) must discard the stash themselves.

        Inside torch.compile (tc_piecewise prefill), the collision check is
        skipped: the compile trampoline legitimately runs the inner forward
        twice back-to-back on first compile (once to trigger compilation,
        once for the returned value) with no pop in between, and reading the
        slot at trace time would make Dynamo guard on its state and
        re-trace. Overwriting is safe there -- the wrapper pops the buffer
        of the most recent forward.
        """
        if (
            not torch.compiler.is_compiling()
            and self._aux_hidden_states is not None
        ):
            raise RuntimeError(
                "AuxCaptureMixin stash collision: the previously stashed aux "
                "buffer was never popped. The stash is a single slot with "
                "strictly paired stash/pop per forward; an interleaved "
                "forward on the same module, or a caller that runs the inner "
                "model without popping, would silently leak one request's "
                "aux hidden states into another."
            )
        self._aux_hidden_states = aux_hidden_states

    def pop_aux_hidden_states(self) -> torch.Tensor | None:
        """Return and clear the stashed aux buffer."""
        aux_hidden_states = self._aux_hidden_states
        self._aux_hidden_states = None
        return aux_hidden_states
