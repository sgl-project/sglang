"""
Per-layer GPU-side signaling for pipelined LoRA weight loading.
Uses torch.cuda.Event for synchronization between a loading stream
and the compute stream. This only runs during prefill (not CUDA graph
captured).

The protocol:
  Loading stream:  [DMA weights for layer N]  ->  mark_ready() records an event
  Compute stream:  wait_until_ready() -> current_stream.wait_event(event) -> [use weights]
"""

import torch


class LoRAPipelineFlag:
    """
    A per-layer event used to synchronize weight loading with forward compute.
    Protocol:
        - Loading stream calls mark_loading() before / mark_ready() after writing
          this layer's weights.
        - Compute stream calls wait_until_ready() before using the weights; it
          consumes the pending wait exactly once.
    """

    def __init__(self, device: torch.device):
        self._event = torch.cuda.Event()
        # Start as "ready" — record a completed event so an un-loaded flag never
        # forces a wait.
        self._event.record(torch.cuda.current_stream(device))
        # True from the start of a load until the compute stream consumes the wait.
        self._pending = False

    @property
    def needs_wait(self) -> bool:
        """Whether a load is outstanding and the compute stream still owes a wait."""
        return self._pending

    def mark_loading(self) -> None:
        """Signal that weight loading is about to start for this layer.

        Marks a wait as pending; it stays pending until wait_until_ready() consumes
        it, so the compute stream cannot race ahead even though this runs (on the
        CPU) long before the DMA completes on the GPU.
        """
        self._pending = True

    def mark_ready(self, loading_stream: torch.cuda.Stream) -> None:
        """Record this layer's load-completion event on the loading stream.

        Does NOT clear `_pending`: the wait is only satisfied once the compute
        stream has actually waited on this event.
        """
        self._event.record(loading_stream)

    def wait_until_ready(self, compute_stream: torch.cuda.Stream) -> None:
        """Make the compute stream wait for this layer's weights, exactly once.

        No-op if no load is pending (e.g. decode).
        """
        if self._pending:
            compute_stream.wait_event(self._event)
            self._pending = False
