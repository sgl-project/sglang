"""Own one invocation's residual stream, including independent TBO children."""

from sglang.srt.layers.communicator.residual.stream import ResidualStream


def start(forward_batch):
    """Begin a fresh embedding-side invocation and return its stream."""
    stream = ResidualStream()
    forward_batch.residual_stream = stream
    return stream


def current(forward_batch):
    """Return the invocation's authoritative stream while its stack is active."""
    stream = forward_batch.residual_stream
    if stream is None:
        raise RuntimeError("start the layer stack before entering a stage")
    return stream
