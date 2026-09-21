"""Experimental exact-FP32 LiteTopK decode kernels for NVIDIA B200."""

SUPPORTED_BATCHES = (1, 2, 4, 8, 16)

# Capture-static choices qualified on B200. Values are
# (histogram bins, selector unroll, active CTAs for <=128K or None).
DISPATCH = {
    1: (2048, 1, 128),
    2: (1024, 1, None),
    4: (1024, 4, None),
    8: (1024, 4, None),
    16: (1024, 4, None),
}


def dispatch_for_batch(batch_size: int) -> tuple[int, int, int | None]:
    try:
        return DISPATCH[batch_size]
    except KeyError as exc:
        raise ValueError(
            f"LiteTopK decode supports batch sizes {SUPPORTED_BATCHES}, "
            f"got {batch_size}"
        ) from exc
