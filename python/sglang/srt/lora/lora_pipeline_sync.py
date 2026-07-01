"""
Per-layer GPU-side signaling for pipelined LoRA weight loading.
Uses torch.cuda.Event for synchronization between a loading stream
and the compute stream. This only runs during prefill (not CUDA graph
captured).

The protocol:
  Loading stream:  [DMA weights for layer N]  ->  event.record(load_stream)
  Compute stream:  current_stream.wait_event(event)  ->  [use weights]
"""

import torch


class LoRAPipelineFlag:
    """
    A per-layer event used to synchronize weight loading with forward compute.
    Protocol:
        - Loading stream calls mark_ready() after writing weights for this layer
        - Compute stream calls wait_until_ready() before using weights
    """

    def __init__(self, device: torch.device):
        self._event = torch.cuda.Event()
        # Start as "ready" — no wait needed if no load is in progress
        self._event.record(torch.cuda.current_stream(device))
        self._loading = False

    def mark_loading(self) -> None:
        """Signal that weight loading is about to start for this layer."""
        self._loading = True

    def mark_ready(self, loading_stream: torch.cuda.Stream) -> None:
        """
        Signal that weight loading is complete for this layer.
        Must be called on the loading stream AFTER all weight DMA.
        """
        self._event.record(loading_stream)
        self._loading = False

    def wait_until_ready(self, compute_stream: torch.cuda.Stream) -> None:
        """
        Block the compute stream until this layer's weights are ready.
        No-op if no load is in progress.

        Only called during prefill forward. Decode skips this entirely.
        """
        if self._loading:
            compute_stream.wait_event(self._event)
