import itertools
import logging
import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange

logger = logging.getLogger(__name__)


# Adapted from https://github.com/OpenNLPLab/lightning-attention/blob/main/lightning_attn/ops/triton/lightning_attn2.py
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Out,
    S,  # log lambda
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    BLOCK_MODEL: tl.constexpr,
):
    ##### get offset
    off_bh = tl.program_id(0)
    off_h = off_bh % h
    off_e = tl.program_id(1)
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    # channel offset
    e_offset = off_e * BLOCK_MODEL

    ##### get block ptr
    Q_block_ptr = Q + qk_offset + tl.arange(0, d)[None, :]
    K_trans_block_ptr = K + qk_offset + tl.arange(0, d)[:, None]
    V_block_ptr = V + v_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]
    O_block_ptr = Out + o_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]
    S_block_ptr = S + off_h

    ##### init diag decay(Lambda); q, k decay; kv
    s = tl.load(S_block_ptr)
    # q, k decay
    off_block = tl.arange(
        0, BLOCK
    )  # Not bug, this is a bit different from algorithm 1, but is mathematically equivalent
    q_decay = tl.exp(-s.to(tl.float32) * off_block[:, None])
    k_trans_decay = tl.exp(-s.to(tl.float32) * (BLOCK - off_block[None, :]))
    block_decay = tl.exp(-s.to(tl.float32) * BLOCK)
    # diag decay
    index = off_block[:, None] - off_block[None, :]
    s_index = s * index
    s_index = tl.where(index >= 0, -s_index, float("-inf"))
    diag_decay = tl.exp(s_index)
    kv = tl.zeros([d, BLOCK_MODEL], dtype=tl.float32)

    ##### compute
    for i in range(NUM_BLOCK):
        # load
        q = tl.load(
            Q_block_ptr + off_block[:, None] * d, mask=off_block[:, None] < n, other=0.0
        ).to(tl.float32)
        k_trans = tl.load(
            K_trans_block_ptr + off_block[None, :] * d,
            mask=off_block[None, :] < n,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            V_block_ptr + off_block[:, None] * e, mask=off_block[:, None] < n, other=0.0
        ).to(tl.float32)

        # compute
        qk = tl.dot(q, k_trans) * diag_decay
        o_intra = tl.dot(qk, v)
        o_inter = tl.dot(q, kv) * q_decay
        o = o_intra + o_inter

        # save and update
        tl.store(
            O_block_ptr + off_block[:, None] * e,
            o.to(O_block_ptr.dtype.element_ty),
            mask=off_block[:, None] < n,
        )
        kv = block_decay * kv + tl.dot(k_trans * k_trans_decay, v)
        off_block += BLOCK


def lightning_attn2(q, k, v, s):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    s = s.contiguous()

    b, h, n, d = q.shape
    e = v.shape[-1]

    # Pad d to next power of 2
    d_padded = next_power_of_2(d)
    if d_padded != d:
        q_padded = F.pad(q, (0, d_padded - d))
        k_padded = F.pad(k, (0, d_padded - d))
    else:
        q_padded = q
        k_padded = k

    # Pad e to next power of 2
    e_padded = next_power_of_2(e)
    if e_padded != e:
        v_padded = F.pad(v, (0, e_padded - e))
    else:
        v_padded = v

    o_padded = torch.empty((b, h, n, e_padded), dtype=q.dtype, device=q.device)

    BLOCK = 64
    NUM_BLOCK = triton.cdiv(q.shape[2], BLOCK)
    # parallel over channel
    BLOCK_MODEL = min(triton.next_power_of_2(e_padded), 32)
    grid = (b * h, triton.cdiv(e_padded, BLOCK_MODEL))

    _fwd_kernel[grid](
        q_padded,
        k_padded,
        v_padded,
        o_padded,
        s,
        b,
        h,
        n,
        d_padded,
        e_padded,
        BLOCK=BLOCK,
        NUM_BLOCK=NUM_BLOCK,
        BLOCK_MODEL=BLOCK_MODEL,
    )

    # Remove padding from output
    if e_padded != e:
        o = o_padded[..., :e]
    else:
        o = o_padded

    return o


def is_support(dim):
    return 16 % dim


def next_power_of_2(n):
    return 2 ** (int(math.ceil(math.log(n, 2))))


def lightning_attn_func(q, k, v, s):
    b, h, n, d = q.shape
    e = v.shape[-1]
    assert is_support(d) and is_support(e)

    # pad v's feature dim to power of 2
    e_pad = next_power_of_2(e)
    need_pad = e_pad != e
    if need_pad:
        v = F.pad(v, (0, e_pad - e))

    if d > 128:
        # split over head
        if 64 % d:
            m = 64
        elif 32 % d:
            m = 32
        elif 16 % d:
            m = 16
        arr = [m * i for i in range(d // m + 1)]
        if arr[-1] != d:
            arr.append(d)
        n = len(arr)
        o = 0
        for i in range(n - 1):
            start = arr[i]
            end = arr[i + 1]
            q1 = q[..., start:end]
            k1 = k[..., start:end]
            o += lightning_attn2(q1, k1, v, s)
    else:
        o = lightning_attn2(q, k, v, s)

    if need_pad:
        o = o[:, :, :, :e]

    return o


debug = eval(os.environ.get("debug", default="False"))

BLOCK = 256


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->MiniMaxText01
class MiniMaxText01RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MiniMaxText01RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# Copied from https://huggingface.co/MiniMaxAI/MiniMax-Text-01/blob/main/modeling_minimax_text_01.py
def get_activation_fn(activation):
    if debug:
        logger.info(f"activation: {activation}")
    if activation == "gelu":
        return F.gelu
    elif activation == "relu":
        return F.relu
    elif activation == "elu":
        return F.elu
    elif activation == "sigmoid":
        return F.sigmoid
    elif activation == "exp":

        def f(x):
            with torch.no_grad():
                x_max = torch.max(x, dim=-1, keepdims=True).values
            y = torch.exp(x - x_max)

            return y

        return f
    elif activation == "leak":
        return F.leaky_relu
    elif activation == "1+elu":

        def f(x):
            return 1 + F.elu(x)

        return f
    elif activation == "2+elu":

        def f(x):
            return 2 + F.elu(x)

        return f
    elif activation == "silu" or activation == "swish":
        return F.silu
    elif activation == "sine":
        return torch.sin
    else:
        logger.info(f"activation: does not support {activation}, use Identity!!!")
        return lambda x: x


# Copied from https://huggingface.co/MiniMaxAI/MiniMax-Text-01/blob/main/modeling_minimax_text_01.py
class MiniMaxText01LightningAttention(nn.Module):
    def __init__(self, config=None, layer_idx: Optional[int] = None, **kwargs):
        super().__init__()
        if config is None:
            config = type("Config", (), kwargs)

        bias = False
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)

        self.out_proj = nn.Linear(
            self.head_dim * self.num_heads, self.hidden_size, bias=bias
        )
        self.act = get_activation_fn(config.hidden_act)
        self.norm = MiniMaxText01RMSNorm(self.head_dim * self.num_heads)

        self.qkv_proj = nn.Linear(
            self.hidden_size, 3 * self.head_dim * self.num_heads, bias=bias
        )
        self.output_gate = nn.Linear(
            self.hidden_size, self.head_dim * self.num_heads, bias=bias
        )

        # for inference only
        self.offset = 0
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states,
        attn_mask: Optional[torch.Tensor] = None,  # (b, h, n, m)
        output_attentions: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        slope_rate: Optional[torch.Tensor] = None,
        do_eval: bool = False,
        **kwargs,
    ):
        if (not self.training) and (not do_eval):
            return self.inference(
                hidden_states,
                attn_mask,
                output_attentions,
                past_key_value,
                use_cache,
                slope_rate,
            )

    def inference(
        self,
        x,
        attn_mask: Optional[torch.Tensor] = None,  # (b, n)
        output_attentions: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        slope_rate: Optional[torch.Tensor] = None,  # (h, 1, 1)
    ):
        # x: b n d
        b, n, d = x.shape
        # linear map
        qkv = self.act(self.qkv_proj(x))
        new_shape = qkv.size()[:-1] + (self.num_heads, -1)
        qkv = qkv.view(*new_shape)
        q, k, v = torch.split(qkv, [self.head_dim] * 3, dim=3)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if past_key_value is None:
            self.offset = q.shape[-2]
        else:
            self.offset += 1

        # for align with metaseq
        ratio = torch.exp(-slope_rate)

        # only use for the first time
        if past_key_value is None:
            slope_rate = slope_rate.to(torch.float32)
            if attn_mask is not None:
                v = v.masked_fill(
                    (1 - attn_mask).unsqueeze(1).unsqueeze(-1).to(torch.bool), 0
                )
            NUM_BLOCK = (n + BLOCK - 1) // BLOCK
            b, h, n, d = q.shape
            e = v.shape[-1]
            # other
            array = torch.arange(BLOCK).to(q) + 1
            q_decay = torch.exp(-slope_rate * array.reshape(-1, 1))
            k_decay = torch.exp(-slope_rate * (BLOCK - array.reshape(-1, 1)))
            index = array[:, None] - array[None, :]
            s_index = (
                slope_rate
                * index[
                    None,
                    None,
                ]
            )
            s_index = torch.where(index >= 0, -s_index, float("-inf"))
            diag_decay = torch.exp(s_index)

            kv = torch.zeros(b, h, d, e).to(torch.float32).to(q.device)
            output = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)
            for i in range(NUM_BLOCK):
                si = i * BLOCK
                ei = min(si + BLOCK, n)
                m = ei - si
                qi = q[:, :, si:ei].contiguous()
                ki = k[:, :, si:ei].contiguous()
                vi = v[:, :, si:ei].contiguous()
                qkv_none_diag = torch.matmul(qi * q_decay[:, :m], kv).to(torch.float32)

                # diag
                qk = (
                    torch.matmul(qi, ki.transpose(-1, -2)).to(torch.float32)
                    * diag_decay[:, :, :m, :m]
                )
                qkv_diag = torch.matmul(qk, vi.to(torch.float32))
                block_decay = torch.exp(-slope_rate * m)
                output[:, :, si:ei] = qkv_none_diag + qkv_diag
                kv = block_decay * kv + torch.matmul(
                    (ki * k_decay[:, -m:]).transpose(-1, -2).to(vi.dtype), vi
                )

        else:
            kv = past_key_value
            output = []
            for i in range(n):
                kv = ratio * kv + torch.einsum(
                    "... n d, ... n e -> ... d e",
                    k[:, :, i : i + 1],
                    v[:, :, i : i + 1],
                )
                qkv = torch.einsum(
                    "... n e, ... e d -> ... n d", q[:, :, i : i + 1], kv.to(q.dtype)
                )
                output.append(qkv)
            output = torch.cat(output, dim=-2)
        # reshape
        output = rearrange(output, "b h n d -> b n (h d)")
        # normalize
        output = self.norm(output)
        # gate
        output = F.sigmoid(self.output_gate(x)) * output
        # outproj
        output = self.out_proj(output)

        attn_weights = None

        return output, attn_weights, kv


def _build_slope_tensor(n_attention_heads: int):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(
                n
            )  # In the paper, we only train models that have 2^a heads for some a. This function has
        else:  # some good properties that only occur when the input is a power of 2. To maintain that even
            closest_power_of_2 = 2 ** math.floor(
                math.log2(n)
            )  # when the number of heads is not a power of 2, we use this workaround.
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    # h, 1, 1
    slopes = torch.tensor(get_slopes(n_attention_heads)).reshape(
        n_attention_heads, 1, 1
    )

    return slopes


def test_lightning_attention_implementations(model_params):
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 1024
    dtype = torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_states = torch.randn(
        batch_size, seq_len, model_params["hidden_size"], dtype=dtype, device=device
    )

    attention_mask = torch.ones(batch_size, seq_len, dtype=dtype, device=device)

    slope_rate = _build_slope_tensor(model_params["num_attention_heads"]).to(device)

    model_attn = MiniMaxText01LightningAttention(**model_params).to(dtype).to(device)
    model_attn.eval()

    with torch.no_grad():
        model_output, _, _ = model_attn.inference(
            hidden_states, attn_mask=attention_mask, slope_rate=slope_rate
        )

    qkv = model_attn.act(model_attn.qkv_proj(hidden_states))
    new_shape = qkv.size()[:-1] + (model_attn.num_heads, -1)
    qkv = qkv.view(*new_shape)
    q, k, v = torch.split(qkv, [model_attn.head_dim] * 3, dim=-1)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    lib_output = lightning_attn_func(q, k, v, slope_rate)
    lib_output = lib_output.transpose(1, 2).contiguous()
    lib_output = lib_output.view(batch_size, seq_len, -1)
    lib_output = model_attn.norm(lib_output)
    lib_output = torch.sigmoid(model_attn.output_gate(hidden_states)) * lib_output
    lib_output = model_attn.out_proj(lib_output)

    torch.testing.assert_close(
        model_output,
        lib_output,
        rtol=1e-3,
        atol=1e-2,
        msg="Lightning attention implementations produce different results",
    )

    print("✅ Two implementations match")


def get_benchmark():
    batch_size_range = [2**i for i in range(0, 7)]  # max 64
    seq_length_range = [256, 512, 1024, 2048, 4096]  # max 4096
    configs = list(itertools.product(batch_size_range, seq_length_range))

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["batch_size", "seq_len"],
            x_vals=[list(_) for _ in configs],
            line_arg="provider",
            line_vals=["MiniMax-Text-01", "OpenNLPLab"],
            line_names=[
                "MiniMax-Text-01 Model Implementation",
                "OpenNLPLab Library Implementation",
            ],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="us",
            plot_name="lightning-attention-prefill-performance",
            args={},
        )
    )
    def benchmark(batch_size, seq_len, provider):
        dtype = torch.bfloat16
        device = torch.device("cuda")

        params = {
            "hidden_size": 6144,
            "num_attention_heads": 64,
            "head_dim": 96,
            "hidden_act": "gelu",
        }

        hidden_states = torch.randn(
            batch_size, seq_len, params["hidden_size"], dtype=dtype, device=device
        )

        attention_mask = torch.ones(batch_size, seq_len, dtype=dtype, device=device)

        slope_rate = _build_slope_tensor(params["num_attention_heads"]).to(device)
        model_attn = MiniMaxText01LightningAttention(**params).to(dtype).to(device)
        model_attn.eval()

        quantiles = [0.5, 0.2, 0.8]
        if provider == "MiniMax-Text-01":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: model_attn.inference(
                    hidden_states, attn_mask=attention_mask, slope_rate=slope_rate
                ),
                quantiles=quantiles,
            )
        else:

            def run_lib():
                qkv = model_attn.act(model_attn.qkv_proj(hidden_states))
                new_shape = qkv.size()[:-1] + (model_attn.num_heads, -1)
                qkv = qkv.view(*new_shape)
                q, k, v = torch.split(qkv, [model_attn.head_dim] * 3, dim=-1)
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

                lib_output = lightning_attn_func(q, k, v, slope_rate)
                lib_output = lib_output.transpose(1, 2).contiguous()
                lib_output = lib_output.view(batch_size, seq_len, -1)
                lib_output = model_attn.norm(lib_output)
                lib_output = (
                    torch.sigmoid(model_attn.output_gate(hidden_states)) * lib_output
                )
                return model_attn.out_proj(lib_output)

            ms, min_ms, max_ms = triton.testing.do_bench(
                run_lib,
                quantiles=quantiles,
            )

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="./configs/benchmark_ops/lightning_attention_prefill/",
        help="Path to save lightning attention prefill benchmark results",
    )
    args = parser.parse_args()

    # Run correctness test first
    # Adapted from https://huggingface.co/MiniMaxAI/MiniMax-Text-01/blob/main/config.json
    params = {
        "hidden_size": 6144,
        "num_attention_heads": 64,
        "head_dim": 96,
        "hidden_act": "silu",
    }
    test_lightning_attention_implementations(params)

    # Run performance benchmark
    benchmark = get_benchmark()
    benchmark.run(print_data=True, save_path=args.save_path)
