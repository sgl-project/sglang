"""Benchmark the fused ratio-2 decode compressor (``c2_decode_norm_rope_store``).

One CTA per token: pair-pool against the state ring, RMSNorm, RoPE, quantize and
store one compressed KV row. The line axis is the cache page format.

Shape axes:
  - ``bs``: requests per batch.
  - ``draft_len``: 1 is plain decode; > 1 is a target-verify block of that many
    consecutive positions per request.
  - ``parity``: the first position's parity. On decode, ``odd`` rows complete a
    pair and run the full norm/RoPE/store path, ``even`` rows only park in the ring.

Constants follow the DeepSeek-V4 deployment: ``head_dim = 512``, ``rope_dim = 64``,
a 256-token FULL page, hence 128 compressed slots per page.
"""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import (
    DEFAULT_DEVICE,
    create_empty,
    create_random,
)
from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.kernels.ops.attention.dsv4.low_ratio_compress import (
    c2_decode_norm_rope_store,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import get_compress_state_ring_size
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

HEAD_DIM = 512
ROPE_DIM = 64
PAGE_SIZE = 256 // 2
MAX_POS = 65536
EPS = 1e-6


@marker.parametrize("parity", ["odd", "even", "half"], ["half"])
@marker.parametrize("draft_len", [1, 2, 4, 8], [1, 8])
@marker.parametrize("bs", marker.range(14, pattern="pow2"), [1, 1024])
@marker.benchmark("layout", ["V4", "V41", "V41_FP4"])
def benchmark(bs: int, draft_len: int, parity: str, layout: str):
    if parity == "half" and bs % 2 != 0:
        marker.skip("half parity requires even batch size")
    layout = KVLayout[layout]
    num_tokens = bs * draft_len
    ring_size = get_compress_state_ring_size(2, draft_len if draft_len > 1 else 0)
    torch.manual_seed(0)

    kv_input = create_random(num_tokens, 2 * HEAD_DIM, dtype=torch.float32)
    kv_state = create_random(bs * ring_size, 2 * HEAD_DIM, dtype=torch.float32)
    norm_weight = create_random(HEAD_DIM)
    freqs_cis = create_random(MAX_POS, ROPE_DIM, dtype=torch.float32)
    req_ids = torch.randperm(bs, device=DEFAULT_DEVICE)
    start = torch.randint(1, MAX_POS // 2 - draft_len, (bs,), device=DEFAULT_DEVICE)
    if parity == "odd":
        start = start * 2 + 1
    elif parity == "even":
        start = start * 2
    else:
        start = start * 2
        start[::2] += 1  # half odd, half even

    offsets = torch.arange(draft_len, device=DEFAULT_DEVICE)
    positions = (start[:, None] + offsets).flatten()
    req = req_ids.repeat_interleave(draft_len)

    # Distinct scattered slots; slot 0 is left alone, it marks a padded row.
    num_slots = num_tokens + 1
    num_pages = -(-num_slots // PAGE_SIZE)
    slots = torch.randperm(num_pages * PAGE_SIZE - 1, device=DEFAULT_DEVICE)
    slots = slots[:num_tokens]
    raw_out_loc = (slots + 1) * 2 + 1
    k_cache = create_empty(
        num_pages,
        layout.page_bytes(PAGE_SIZE),
        dtype=torch.uint8,
    )
    out = create_empty(num_tokens, HEAD_DIM)

    # Traffic of the rows that complete a pair; parked rows write the state instead.
    num_complete = int((positions % 2 == 1).sum())
    num_parked = num_tokens - num_complete
    row_bytes = 2 * HEAD_DIM * 4
    complete_bytes = (
        2 * row_bytes  # kv_input + partner
        + ROPE_DIM * 4  # freqs_cis row
        + HEAD_DIM * 2  # out
        + layout.bytes_per_token
    )
    footprint = (
        num_complete * complete_bytes
        + num_parked * 3 * row_bytes  # kv_input + partner read, state write
        + num_tokens * 3 * 8  # positions, req, raw_out_loc
        + norm_weight.nbytes
    )
    return marker.do_bench(
        c2_decode_norm_rope_store,
        input_args=(kv_input, kv_state, norm_weight, positions, req, raw_out_loc),
        input_kwargs=dict(
            eps=EPS,
            freqs_cis=freqs_cis,
            k_cache=k_cache,
            page_size=PAGE_SIZE,
            ring_size=ring_size,
            draft_len=draft_len,
            layout=layout,
            out=out,
        ),
        memory_args=None,
        memory_output=None,
        extra_memory_footprint=footprint,
    )


if __name__ == "__main__":
    benchmark.run()
