"""Read-only AOT Metal radix attention, composable inside ``mx.compile``."""


def radix_decode(
    q, k, v, kp, vp, table, requests, lengths, scale, *, tails=None, page_size=1
):
    try:
        from sgl_kernel.metal import radix_attention
    except ImportError as error:
        raise ImportError(
            "Compiled MLX radix requires AOT Metal kernels. Build them with "
            "`python python/sglang/kernels/aot/setup_metal.py install`."
        ) from error
    return radix_attention(
        q,
        k,
        v,
        kp,
        vp,
        table,
        requests,
        lengths,
        scale,
        tails=tails,
        page_size=page_size,
    )
