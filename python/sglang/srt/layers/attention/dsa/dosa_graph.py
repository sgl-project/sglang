"""Private CUDA captures with stream and allocator ownership independent of serving graphs."""

from contextlib import contextmanager
import weakref

import torch


def _destroy_stream(runtime, raw, device):
    try:
        with torch.cuda.device(device):
            runtime.cudaStreamDestroy(raw)
    except Exception:
        pass  # CUDA teardown must not replace the original execution failure.


def _stream(device):
    from cuda.bindings import runtime

    with torch.cuda.device(device):
        status, raw = runtime.cudaStreamCreateWithFlags(runtime.cudaStreamNonBlocking)
        if status != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA capture stream creation failed: {status}")
        stream = torch.cuda.ExternalStream(int(raw), device=device)
    cleanup = weakref.finalize(stream, _destroy_stream, runtime, raw, device)
    cleanup.atexit = False
    return stream


class _Graph(torch.cuda.CUDAGraph):
    def reset(self):
        super().reset()
        generation = getattr(self, "generation", None)
        if generation is not None and generation.serial == self.serial:
            generation.retired = True

    def __del__(self):
        # Native reset must precede release of the capture stream and pool.
        stream = getattr(self, "stream", None)
        try:
            super().__del__()
        finally:
            try:
                self.reset()
            except Exception:
                pass
            _ = stream


@contextmanager
def capture(device, *, pool=None, keep_graph=True):
    """Use an unpooled stream: graph reset otherwise clears another graph's BLAS cache."""
    stream = _stream(device) if pool is None else pool.stream
    stream.wait_stream(torch.cuda.current_stream(device))
    graph = _Graph(keep_graph=keep_graph)
    graph.stream = stream
    if pool is not None:
        pool.serial += 1
        pool.retired = False
        graph.generation, graph.serial = pool, pool.serial
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    with torch.cuda.stream(stream):
        options = {} if pool is None else {"pool": pool.pool.id}
        graph.capture_begin(capture_error_mode="global", **options)
        try:
            yield graph
        finally:
            graph.capture_end()


class GraphPool:
    """A retained parent leases its generation; retired parent storage can be reused."""

    def __init__(self, device):
        self.device = device
        self.current = None

    def acquire(self):
        if self.current is None or not self.current.retired:
            self.current = _Generation(self.device)
        return self.current


class _Generation:
    def __init__(self, device):
        self.pool = torch.cuda.MemPool()
        self.stream = _stream(device)
        self.serial = 0
        self.retired = True
        # MemPool leases the device pool; the empty graph also leases the host
        # allocator's same-ID pool until all real parents have released it.
        self.anchor_stream = _stream(device)
        self.anchor = _Graph(keep_graph=True)
        self.anchor.stream = self.anchor_stream
        self.anchor.pool_lease = self.pool
        torch.cuda.synchronize(device)
        with torch.cuda.stream(self.anchor_stream):
            self.anchor.capture_begin(pool=self.pool.id, capture_error_mode="global")
            self.anchor.capture_end()
