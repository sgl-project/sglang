import itertools
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange
from sgl_kernel import lightning_attention_decode as sgl_lightning_attention_decode


@triton.jit
def _decode_kernel(
    Q,
    K,
    V,
    KV,
    Out,
    S,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    d_original: tl.constexpr,
    e: tl.constexpr,
    e_original: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_h = off_bh % h

    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    kv_offset = off_bh * d * e

    s = tl.load(S + off_h)
    ratio = tl.exp(-s)

    d_idx = tl.arange(0, d)
    e_idx = tl.arange(0, e)

    # Create masks for original dimensions
    d_mask = d_idx < d_original
    e_mask = e_idx < e_original

    # Load with masking
    q = tl.load(Q + qk_offset + d_idx, mask=d_mask, other=0.0)
    k = tl.load(K + qk_offset + d_idx, mask=d_mask, other=0.0)
    v = tl.load(V + v_offset + e_idx, mask=e_mask, other=0.0)

    # Load KV with 2D masking
    kv = tl.load(
        KV + kv_offset + d_idx[:, None] * e + e_idx[None, :],
        mask=(d_mask[:, None] & e_mask[None, :]),
        other=0.0,
    )

    # Compute outer product using element-wise operations
    k_v_prod = k[:, None] * v[None, :]
    kv = ratio * kv + k_v_prod

    # Store KV with 2D masking
    tl.store(
        KV + kv_offset + d_idx[:, None] * e + e_idx[None, :],
        kv.to(KV.dtype.element_ty),
        mask=(d_mask[:, None] & e_mask[None, :]),
    )

    # Compute matrix-vector multiplication using element-wise operations and reduction
    o = tl.sum(q[:, None] * kv, axis=0)

    # Store output with masking
    tl.store(Out + o_offset + e_idx, o.to(Out.dtype.element_ty), mask=e_mask)


def lightning_attn_decode(q, k, v, kv, s):
    """Triton implementation of Lightning Attention decode operation"""
    b, h, n, d = q.shape
    e = v.shape[-1]
    assert n == 1, "Sequence length must be 1 in decode mode"

    # Get padded dimensions (power of 2)
    d_padded = next_power_of_2(d)
    e_padded = next_power_of_2(e)

    # Create output tensor (padded)
    o_padded = torch.empty(b, h, n, e_padded, dtype=v.dtype, device=v.device)

    # Create padded tensors without actually padding the data
    q_padded = torch.empty(b, h, n, d_padded, dtype=q.dtype, device=q.device)
    k_padded = torch.empty(b, h, n, d_padded, dtype=k.dtype, device=k.device)
    v_padded = torch.empty(b, h, n, e_padded, dtype=v.dtype, device=v.device)
    kv_padded = torch.empty(
        b, h, d_padded, e_padded, dtype=torch.float32, device=kv.device
    )

    # Copy data to padded tensors
    q_padded[..., :d] = q
    k_padded[..., :d] = k
    v_padded[..., :e] = v
    kv_padded[..., :d, :e] = kv

    # Launch kernel
    grid = (b * h, 1)
    _decode_kernel[grid](
        q_padded,
        k_padded,
        v_padded,
        kv_padded,
        o_padded,
        s,
        b=b,
        h=h,
        n=n,
        d=d_padded,
        d_original=d,
        e=e_padded,
        e_original=e,
    )

    # Get unpadded outputs
    o = o_padded[..., :e]
    kv_out = kv_padded[..., :d, :e]

    return o, kv_out


def next_power_of_2(n):
    return 2 ** (int(math.ceil(math.log(n, 2))))


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
        q = q.transpose(1, 2)  # [b, n, h, d] -> [b, h, n, d]
        k = k.transpose(1, 2)  # [b, n, h, d] -> [b, h, n, d]
        v = v.transpose(1, 2)  # [b, n, h, d] -> [b, h, n, e]

        self.offset += 1
        ratio = torch.exp(-slope_rate)  # [h, 1, 1]

        # decode mode
        kv = past_key_value  # [b, h, d, e]
        output = []
        for i in range(n):
            # kv: [b, h, d, e]
            # ratio: [h, 1, 1]
            # k: [b, h, n, d]
            # v: [b, h, n, e]
            # k[:, :, i : i + 1]: [b, h, 1, d]
            # v[:, :, i : i + 1]: [b, h, 1, e]
            # ratio * kv: [b, h, d, e]
            # torch.einsum(
            #     "... n d, ... n e -> ... d e",
            #     k[:, :, i : i + 1],
            #     v[:, :, i : i + 1],
            # )
            # [b, h, d, e] + [b, h, d, e] -> [b, h, d, e]
            kv = ratio * kv + torch.einsum(
                "... n d, ... n e -> ... d e",
                k[:, :, i : i + 1],
                v[:, :, i : i + 1],
            )
            # q[:, :, i : i + 1]: [b, h, 1, d]
            # kv.to(q.dtype): [b, h, d, e]
            # torch.einsum(
            #     "... n e, ... e d -> ... n d", q[:, :, i : i + 1], kv.to(q.dtype)
            # )
            # [b, h, 1, d] * [b, h, d, e] -> [b, h, 1, e]
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


def get_activation_fn(activation):
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
        return lambda x: x


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


def test_lightning_attention_implementations(model_params):
    torch.manual_seed(42)

    batch_size = 64
    seq_len = 1
    dtype = torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_states = torch.randn(
        batch_size, seq_len, model_params["hidden_size"], dtype=dtype, device=device
    )

    attention_mask = torch.ones(batch_size, seq_len, dtype=dtype, device=device)

    slope_rate = _build_slope_tensor(model_params["num_attention_heads"]).to(device)

    model_attn = MiniMaxText01LightningAttention(**model_params).to(dtype).to(device)
    model_attn.eval()

    d = model_params["head_dim"]
    past_kv = torch.randn(
        batch_size,
        model_params["num_attention_heads"],
        d,
        d,
        device=device,
    )
    with torch.no_grad():
        model_output, _, new_kv = model_attn.inference(
            hidden_states,
            attn_mask=attention_mask,
            slope_rate=slope_rate,
            past_key_value=past_kv,
        )

    qkv = model_attn.act(model_attn.qkv_proj(hidden_states))
    new_shape = qkv.size()[:-1] + (model_attn.num_heads, -1)
    qkv = qkv.view(*new_shape)
    q, k, v = torch.split(qkv, [model_attn.head_dim] * 3, dim=-1)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    past_kv = past_kv.contiguous()
    slope_rate = slope_rate.contiguous()

    # Test Triton implementation
    triton_output, triton_new_kv = lightning_attn_decode(q, k, v, past_kv, slope_rate)
    triton_output = triton_output.transpose(1, 2).contiguous()
    triton_output = triton_output.view(batch_size, seq_len, -1)
    triton_output = model_attn.norm(triton_output)
    triton_output = torch.sigmoid(model_attn.output_gate(hidden_states)) * triton_output
    triton_output = model_attn.out_proj(triton_output)

    # Test SGL implementation
    sgl_output = torch.empty_like(v)
    sgl_new_kv = torch.empty_like(past_kv)
    sgl_lightning_attention_decode(q, k, v, past_kv, slope_rate, sgl_output, sgl_new_kv)

    sgl_output = sgl_output.transpose(1, 2).contiguous()
    sgl_output = sgl_output.view(batch_size, seq_len, -1)
    sgl_output = model_attn.norm(sgl_output)
    sgl_output = torch.sigmoid(model_attn.output_gate(hidden_states)) * sgl_output
    sgl_output = model_attn.out_proj(sgl_output)

    # Verify Triton implementation results
    torch.testing.assert_close(
        model_output,
        triton_output,
        rtol=1e-3,
        atol=1e-2,
        msg="Triton lightning attention implementation produces different output results",
    )
    torch.testing.assert_close(
        new_kv,
        triton_new_kv,
        rtol=1e-3,
        atol=1e-2,
        msg="Triton lightning attention implementation produces different kv results",
    )

    # Verify SGL implementation results
    torch.testing.assert_close(
        model_output,
        sgl_output,
        rtol=1e-3,
        atol=1e-2,
        msg="SGL lightning attention implementation produces different output results",
    )
    torch.testing.assert_close(
        new_kv,
        sgl_new_kv,
        rtol=1e-3,
        atol=1e-2,
        msg="SGL lightning attention implementation produces different kv results",
    )

    print("✅ All implementations match")


def _build_slope_tensor(n_attention_heads: int):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    slopes = torch.tensor(get_slopes(n_attention_heads)).reshape(
        n_attention_heads, 1, 1
    )
    return slopes


def get_benchmark():
    batch_size_range = [i for i in range(1, 33)]  # max 32
    seq_length_range = [1]  # decode mode sequence length is fixed to 1
    configs = list(itertools.product(batch_size_range, seq_length_range))

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["batch_size", "seq_len"],
            x_vals=[list(_) for _ in configs],
            line_arg="provider",
            line_vals=["Original", "Triton", "SGL"],
            line_names=[
                "Original PyTorch Implementation",
                "Triton Implementation",
                "SGL Implementation",
            ],
            styles=[("blue", "-"), ("green", "-"), ("red", "-")],
            ylabel="us",
            plot_name="lightning-attention-decode-performance",
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

        d = params["head_dim"]
        past_kv = torch.randn(
            batch_size,
            params["num_attention_heads"],
            d,
            d,
            device=device,
        )

        quantiles = [0.5, 0.2, 0.8]
        if provider == "Original":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: model_attn.inference(
                    hidden_states,
                    attn_mask=attention_mask,
                    slope_rate=slope_rate,
                    past_key_value=past_kv,
                ),
                quantiles=quantiles,
            )
        elif provider == "Triton":

            def run_triton():
                qkv = model_attn.act(model_attn.qkv_proj(hidden_states))
                new_shape = qkv.size()[:-1] + (model_attn.num_heads, -1)
                qkv = qkv.view(*new_shape)
                q, k, v = torch.split(qkv, [model_attn.head_dim] * 3, dim=-1)
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

                output, new_kv = lightning_attn_decode(q, k, v, past_kv, slope_rate)
                output = output.transpose(1, 2).contiguous()
                output = output.view(batch_size, seq_len, -1)
                output = model_attn.norm(output)
                output = torch.sigmoid(model_attn.output_gate(hidden_states)) * output
                return model_attn.out_proj(output)

            ms, min_ms, max_ms = triton.testing.do_bench(
                run_triton,
                quantiles=quantiles,
            )
        else:  # SGL

            def run_sgl():
                qkv = model_attn.act(model_attn.qkv_proj(hidden_states))
                new_shape = qkv.size()[:-1] + (model_attn.num_heads, -1)
                qkv = qkv.view(*new_shape)
                q, k, v = torch.split(qkv, [model_attn.head_dim] * 3, dim=-1)
                q = q.transpose(1, 2).contiguous()
                k = k.transpose(1, 2).contiguous()
                v = v.transpose(1, 2).contiguous()

                output = torch.empty_like(v)
                new_kv = torch.empty_like(past_kv)
                sgl_lightning_attention_decode(
                    q, k, v, past_kv, slope_rate, output, new_kv
                )

                output = output.transpose(1, 2).contiguous()
                output = output.view(batch_size, seq_len, -1)
                output = model_attn.norm(output)
                output = torch.sigmoid(model_attn.output_gate(hidden_states)) * output
                return model_attn.out_proj(output)

            ms, min_ms, max_ms = triton.testing.do_bench(
                run_sgl,
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
        default="./configs/benchmark_ops/lightning_attention_decode/",
        help="Path to save lightning attention decode benchmark results",
    )
    args = parser.parse_args()

    params = {
        "hidden_size": 6144,
        "num_attention_heads": 64,
        "head_dim": 96,
        "hidden_act": "silu",
    }
    # Run correctness test first
    # Adapted from https://huggingface.co/MiniMaxAI/MiniMax-Text-01/blob/main/config.json
    test_lightning_attention_implementations(params)

    # Run performance benchmark
    benchmark = get_benchmark()
    benchmark.run(print_data=True, save_path=args.save_path)
