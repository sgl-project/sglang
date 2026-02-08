#################################################################################################
#
# Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

import argparse
import torch
import sys
import os
from piped_subprocess import PipedSubprocess, TORCH_DTYPE_NAME
import math


parser = argparse.ArgumentParser()
parser.add_argument("example_exe", type=str, help="Path to the 41_fused_multi_head_attention_backward executable")
args = parser.parse_args()

torch.manual_seed(0)
dtype = torch.float16
B, Mq, Mkv, H, K, Kv = 2, 1024, 1024, 5, 128, 128
causal = True
repeat_count = 100

ATOL = {
    torch.float: 5e-4,
    torch.half: 9.5e-2,
    torch.bfloat16: 7e-1,
}[dtype]

RTOL = {
    torch.float: 1e-4,
    torch.half: 2e-2,
    torch.bfloat16: 1e-1,
}[dtype]


assert not (causal and Mq < Mkv), "causal only supports seqlenK <= seqlenQ"

fmha_bw_binary = args.example_exe
if not os.path.isfile(fmha_bw_binary):
    print(f"""No such file: `{fmha_bw_binary}`\nDid you forget to run "make 41_fused_multi_head_attention"?""")
    sys.exit(1)

def create_lower_triangular_mask():
    return torch.triu(torch.full(  # type: ignore
        [1, Mq, Mkv],
        dtype=dtype,
        fill_value=float("-inf"),
    ), diagonal=1)

def ref_mha_bmk(q, k, v, mask):
    # Multi-head attention with inputs/outputs in BMK format
    q = q.float()
    k = k.float()
    v = v.float()

    q = q * (1 / q.shape[-1] ** 0.5)
    attn = q @ k.transpose(-2, -1)
    if mask is not None:
        attn += mask
    attn_max = attn.max(-1, True).values
    attn_norm = (attn - attn_max).exp().sum(-1, True)
    attn = attn.softmax(-1)
    lse = attn_max + attn_norm.log()
    lse = lse.squeeze(2)
    return attn @ v, lse


def bmhk2bmk(t):
    return t.permute((0, 2, 1, 3)).reshape(
        [t.shape[0] * t.shape[2], t.shape[1], t.shape[3]]
    )

def ref_mha_bmhk(q, k, v, mask):
    # Multi-head attention with inputs/outputs in BMHK format
    assert q.ndim == 4

    out, lse = ref_mha_bmk(bmhk2bmk(q), bmhk2bmk(k), bmhk2bmk(v), mask=mask)
    out = out.reshape([q.shape[0], q.shape[2], q.shape[1], v.shape[3]])
    return out.permute((0, 2, 1, 3)), lse.reshape([q.shape[0], q.shape[2], q.shape[1]])

def ref_mha_bw_bmhk(q, k, v, mask, lse, out, grad_out, delta):
    lse = lse[:, :, :q.shape[1]]  #BMH, unpad Q dimension
    delta = delta.reshape([-1, delta.shape[-1], 1])

    # bmhk -> bmk
    q, k, v, out, grad_out = [bmhk2bmk(x).float() for x in (q, k, v, out, grad_out)]

    attn_T = k @ q.transpose(-2, -1)
    if mask is not None:
        attn_T += mask.transpose(-2, -1)
    attn_T = attn_T * (1 / q.shape[-1] ** 0.5)
    attn_T = attn_T - lse.reshape([-1, 1, lse.shape[-1]])
    attn_T = attn_T.exp()

    grad_v = attn_T @ grad_out

    dov = grad_out @ v.transpose(-2, -1)
    tmp = (dov - delta) * attn_T.transpose(-2, -1)
    tmp = tmp / (q.shape[-1] ** 0.5)

    grad_q = tmp @ k
    grad_k = tmp.transpose(-2, -1) @ q

    return [x.reshape([B, H, x.shape[1], x.shape[-1]]).permute([0, 2, 1, 3]) for x in [grad_q, grad_k, grad_v]]


print("initializing tensors...")
query = torch.randn([B, Mq, H, K], dtype=dtype)
key = 3 * torch.randn([B, Mkv, H, K], dtype=dtype)
value = 3 * torch.randn([B, Mkv, H, Kv], dtype=dtype)
mask = create_lower_triangular_mask() if causal else None

# let PyTorch compute gradients
query.requires_grad_(True)
key.requires_grad_(True)
value.requires_grad_(True)

print("computing fw...")
out, lse = ref_mha_bmhk(query, key, value, mask=mask)
out = out.to(dtype).contiguous()
grad_out = 3 * torch.randn([B, Mq, H, Kv], dtype=dtype)

print("computing bw with autograd...")
out.backward(grad_out)
scale = (1 / query.shape[-1] ** 0.5)


# Additional data needed by the kernel
delta = (grad_out.float() * out.float()).sum(-1).transpose(-2, -1).contiguous()
pad_amount = (32 - (lse.shape[2] % 32)) % 32
lse = torch.nn.functional.pad(lse, [0, pad_amount], value=math.inf)

print("computing bw with reference implem...")
gQr, gKr, gVr = ref_mha_bw_bmhk(query, key, value, mask, lse, out, grad_out, delta)

with PipedSubprocess(fmha_bw_binary) as bw_kernel:
    # Send kernel arguments
    bw_kernel.write(
        TORCH_DTYPE_NAME[query.dtype],
        "scale", scale,
        "head_dim", K,
        "head_dim_value", Kv,
        "num_queries", Mq,
        "num_keys", Mkv,
        "num_heads", H,
        "custom_mask_type", (1 if causal else 0),
        "num_batches", B,
        "repeat_count", repeat_count,
        "num_splits_key", (Mkv // 128),
    )
    bw_kernel.writeTensor(query, "query", ["q_strideB", "q_strideM", "q_strideH"])
    bw_kernel.writeTensor(key, "key", ["k_strideB", "k_strideM", "k_strideH"])
    bw_kernel.writeTensor(value, "value", ["v_strideB", "v_strideM", "v_strideH"])
    bw_kernel.writeTensor(lse, "logsumexp", ["lse_strideB", "lse_strideH"])
    bw_kernel.writeTensor(out, "output", ["o_strideB", "o_strideM", "o_strideH"])
    bw_kernel.writeTensor(grad_out, "grad_output", ["gO_strideB", "gO_strideM", "gO_strideH"])
    bw_kernel.writeTensor(delta, "delta", ["delta_strideB", "delta_strideH"])

    if bw_kernel.read() != "OK":
        print("Got unexpected output")
        print(bw_kernel.subp.communicate()[0])
        sys.exit(0)

    # Read kernel output
    gQ = bw_kernel.readTensor("grad_query", ["gQ_strideB", "gQ_strideM", "gQ_strideH"], query.shape).float()
    gK = bw_kernel.readTensor("grad_key", ["gK_strideB", "gK_strideM", "gK_strideH"], key.shape).float()
    gV = bw_kernel.readTensor("grad_value", ["gV_strideB", "gV_strideM", "gV_strideH"], value.shape).float()
    runtime_ms = float(bw_kernel.readNamed("runtime_ms"))

float_ops = B * H * sum([
    # att = Q @ K.transpose
    Mq * Mkv * K * 2,
    # att @ dO
    Mkv * Mq * Kv * 2,
    # dov = dO @ V
    Mq * Kv * Mkv * 2,
    # dov @ K
    Mq * K * Mkv * 2,
    # dov @ Q
    Mq * K * Mkv * 2,
])
if causal:
    float_ops //= 2

print(f"""
Fused multi-head attention - backward
    batch_size={B}
    num_queries={Mq}
    num_keys={Mkv}
    num_heads={H}
    head_dim={K}
    head_dim_value={Kv}

    Correctness:
        grad_query: {"PASS" if torch.allclose(gQ, gQr, rtol=RTOL, atol=ATOL) else "FAIL"} (delta: {(gQ - gQr).abs().max()})
        grad_key:   {"PASS" if torch.allclose(gK, gKr, rtol=RTOL, atol=ATOL) else "FAIL"} (delta: {(gK - gKr).abs().max()})
        grad_value: {"PASS" if torch.allclose(gV, gVr, rtol=RTOL, atol=ATOL) else "FAIL"} (delta: {(gV - gVr).abs().max()})
        (atol={ATOL} / rtol={RTOL})
    Runtime: {runtime_ms}ms ({(float_ops / (1024 ** 4)) / (runtime_ms / 1000):.4f} TFlops)
""")

assert torch.allclose(query.grad.float(), gQr, rtol=RTOL, atol=ATOL), "Reference implementation does not match PyTorch autograd!"
assert torch.allclose(key.grad.float(), gKr, rtol=RTOL, atol=ATOL), "Reference implementation does not match PyTorch autograd!"
assert torch.allclose(value.grad.float(), gVr, rtol=RTOL, atol=ATOL), "Reference implementation does not match PyTorch autograd!"
