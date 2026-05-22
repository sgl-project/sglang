"""
Minimal smoke test: verify that the step03_test_utils MockModelRunner
can instantiate TritonAttnBackend and run a decode forward without errors.

Run this FIRST on the cluster to confirm the mock setup is valid before
running the full test suite.

  python smoke_step03_triton.py

Expected output: "PASS: triton eager decode" on the last line.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

# Patch TP size to 1 before any backend imports.  The step-03 branch uses
# the distributed group system; we bypass it in unit tests.
from sglang.srt.layers import dp_attention as _dp_attn

_dp_attn.get_attention_tp_size = lambda: 1

from step03_test_utils import (
    build_mha_runner,
    fill_req_to_token,
    make_decode_batch,
    make_extend_batch,
    make_qkv,
    make_radix_attention,
)

from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    set_forward_context,
)

assert torch.cuda.is_available(), "CUDA required"

NUM_HEADS = 4
HEAD_DIM = 32
MAX_BS = 8
MAX_CTX = 64
SEQ_LEN = 16
DTYPE = torch.float16

print("Building mock model runner...")
mr = build_mha_runner(
    num_heads=NUM_HEADS,
    head_dim=HEAD_DIM,
    max_bs=MAX_BS,
    max_context_len=MAX_CTX,
    dtype=DTYPE,
)

print("Constructing TritonAttnBackend...")
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

backend = TritonAttnBackend(mr)
layer = make_radix_attention(NUM_HEADS, HEAD_DIM)
set_forward_context(ForwardContext(attn_backend=backend))
print(f"  backend.num_head={backend.num_head}, v_head_dim={backend.v_head_dim}")

# ---- eager decode ----
bs = 2
fill_req_to_token(mr, bs, SEQ_LEN)
fb = make_decode_batch(bs, SEQ_LEN)
print("Calling init_forward_metadata (eager decode)...")
backend.init_forward_metadata(fb)
q, k, v = make_qkv(bs, NUM_HEADS, HEAD_DIM, dtype=DTYPE)
print("Calling forward_decode...")
out = backend.forward_decode(q, k, v, layer, fb)
assert not torch.isnan(out).any(), f"NaN in output: {out}"
assert not torch.isinf(out).any(), f"Inf in output: {out}"
assert out.shape == (bs, NUM_HEADS * HEAD_DIM), f"Wrong shape: {out.shape}"
print(f"  output shape: {out.shape}, dtype: {out.dtype}")

# ---- eager extend ----
fill_req_to_token(mr, bs, SEQ_LEN)
fb_ext = make_extend_batch(bs, extend_len=8, prefix_len=4)
print("Calling init_forward_metadata (eager extend)...")
backend.init_forward_metadata(fb_ext)
q_e, k_e, v_e = make_qkv(bs * 8, NUM_HEADS, HEAD_DIM, dtype=DTYPE)
print("Calling forward_extend...")
out_e = backend.forward_extend(q_e, k_e, v_e, layer, fb_ext)
assert not torch.isnan(out_e).any(), "NaN in extend output"
print(f"  extend output shape: {out_e.shape}")

# ---- graph decode ----
print("Calling init_cuda_graph_state...")
backend.init_cuda_graph_state(MAX_BS, MAX_BS)
fill_req_to_token(mr, MAX_BS, SEQ_LEN)
req_pool = torch.arange(MAX_BS, dtype=torch.int32, device="cuda")
seq_lens = torch.full((MAX_BS,), SEQ_LEN, dtype=torch.int32, device="cuda")
seq_lens_cpu = torch.full((MAX_BS,), SEQ_LEN, dtype=torch.int32)
from step03_test_utils import init_graph_capture, init_graph_replay

from sglang.srt.model_executor.forward_batch_info import ForwardMode

fb_g = make_decode_batch(MAX_BS, SEQ_LEN)
print("Calling graph capture init...")
init_graph_capture(
    backend, fb_g, MAX_BS, MAX_BS, req_pool, seq_lens, ForwardMode.DECODE
)
print("Calling graph replay init...")
init_graph_replay(
    backend,
    fb_g,
    MAX_BS,
    req_pool,
    seq_lens,
    MAX_BS * SEQ_LEN,
    ForwardMode.DECODE,
    seq_lens_cpu,
)
q_g, k_g, v_g = make_qkv(MAX_BS, NUM_HEADS, HEAD_DIM, dtype=DTYPE)
print("Calling forward_decode (graph replay)...")
out_g = backend.forward_decode(q_g, k_g, v_g, layer, fb_g)
assert not torch.isnan(out_g).any(), "NaN in graph output"
print(f"  graph output shape: {out_g.shape}")

print("\nPASS: triton eager decode + extend + graph replay")
