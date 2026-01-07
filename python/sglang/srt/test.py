import math

import einops
import pytest
import torch

import flashinfer
from flashinfer.jit.utils import filename_safe_dtype_map

attention_sink_decl = r"""
struct AttentionSink : AttentionVariantBase {
  static constexpr bool use_softmax = true;

  uint32_t window_left, qo_len, kv_len;
  float sm_scale_log2;

  // Create closure
  template <typename Params>
  __device__ __host__ AttentionSink(const Params& params, uint32_t batch_idx,
                                   uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
    sm_scale_log2 = params.sm_scale * math::log2e;
  }

  REGISTER_OUTPUT_TRANSFORM(params, output, batch_idx, qo_idx, qo_head_idx, m, d, {
    float d_rcp = (m != -math::inf) ? math::ptx_rcp(d + params.sink[qo_head_idx]) : 0.f;
    return output * d_rcp;
  });
};
"""


def sink_softmax(logits, sink):
    sink = einops.repeat(sink, "h -> b h m 1", b=logits.shape[0], m=logits.shape[2])
    # (b, h, m, (n + 1))
    logits = torch.cat([logits, torch.log(sink)], dim=-1)
    # (s_1, s_2, ..., s_n)
    # (s_1, s_2, ..., s_n, log(sink))
    # (exp(s_1), exp(s_2), ..., exp(s_n), sink)
    # (exp(s_1) / (exp(s_1) + exp(s_2) + ... + exp(s_n) + sink),
    #  exp(s_2) / (exp(s_1) + exp(s_2) + ... + exp(s_n) + sink),
    #  ...,
    #  exp(s_n) / (exp(s_1) + exp(s_2) + ... + exp(s_n) + sink))
    #  sink / (exp(s_1) + exp(s_2) + ... + exp(s_n) + sink)
    score = torch.softmax(logits, dim=-1)[..., :-1].contiguous()
    return score


def sink_attention_ref(
    batch_size,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor,
    causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    qo_len = q.shape[0] // batch_size
    kv_len = k.shape[0] // batch_size
    num_qo_heads = q.shape[1]
    num_kv_heads = k.shape[1]  # Get actual number of kv heads from k tensor
    head_dim_qk = q.shape[2]
    head_dim_vo = v.shape[2]
    
    # Reshape q, k, v with their actual head counts
    q_reshaped = q.view(batch_size, qo_len, num_qo_heads, head_dim_qk).float()
    k_reshaped = k.view(batch_size, kv_len, num_kv_heads, head_dim_qk).float()
    v_reshaped = v.view(batch_size, kv_len, num_kv_heads, head_dim_vo).float()
    
    # Expand k and v to match q's num_heads if using MQA/GQA
    if num_kv_heads != num_qo_heads:
        k_reshaped = k_reshaped.repeat_interleave(num_qo_heads // num_kv_heads, dim=2)
        v_reshaped = v_reshaped.repeat_interleave(num_qo_heads // num_kv_heads, dim=2)
    
    logits = (
        torch.einsum(
            "bmhd,bnhd->bhmn",
            q_reshaped,
            k_reshaped,
        )
        * sm_scale
    )

    if causal:
        mask = torch.arange(kv_len - qo_len, kv_len, device=q.device).unsqueeze(
            1
        ) >= torch.arange(0, kv_len, device=q.device).unsqueeze(0)
    else:
        mask = torch.ones(qo_len, kv_len, device=q.device)

    logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))

    p = sink_softmax(logits, sink)
    o_ref = (
        torch.einsum(
            "bhmn,bnhd->bmhd",
            p,
            v_reshaped,
        )
        .contiguous()
        .view(batch_size * qo_len, num_qo_heads, head_dim_vo)
        .to(q)
    )

    return o_ref


@pytest.mark.parametrize("dtype", [torch.bfloat16])  # , torch.bfloat16])
@pytest.mark.parametrize("causal", [True])  # [True, False])
def test_attention_sink(dtype, causal):
    jit_args = (
        f"batch_prefill_attention_sink_{filename_safe_dtype_map[dtype]}",  # uri
        dtype,  # dtype_q
        dtype,  # dtype_kv
        dtype,  # dtype_o
        torch.int32,  # idtype
        64,  # hidden_dim_qk
        64,  # hidden_dim_vo
        ["sink"],  # additional_tensor_names
        ["float"],  # additional_tensor_dtypes
        ["sm_scale"],  # additional_scalar_names
        ["double"],  # additional_scalar_dtypes
        "AttentionSink",
        attention_sink_decl,
    )
    sm_scale = 1.0 / math.sqrt(64)
    float_workspace_buffer = torch.empty(
        64 * 1024 * 1024, dtype=torch.uint8, device="cuda"
    )
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        float_workspace_buffer, kv_layout="NHD", backend="fa2", jit_args=jit_args
    )
    batch_size = 1
    seq_len_per_request = 1
    qo_indptr_host = torch.arange(
        0, batch_size * seq_len_per_request + 1, seq_len_per_request, dtype=torch.int32
    )
    kv_indptr_host = torch.arange(
        0, batch_size * seq_len_per_request + 1, seq_len_per_request, dtype=torch.int32
    )

    num_qo_heads = 1
    num_kv_heads = 1
    head_dim = 64

    wrapper.plan(
        qo_indptr_host,
        kv_indptr_host,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    q = torch.randn(
        batch_size * seq_len_per_request,
        num_qo_heads,
        head_dim,
        dtype=dtype,
        device="cuda",
    )
    # Reshape the hardcoded tensor to match expected shape [batch_size * seq_len_per_request, num_qo_heads, head_dim]
    # q_values 1 openai moe
    # q_values = torch.tensor([3.78125, 2.609375, 9.0625, 0.09033203125, -1.53125, 6.25, 4.5625, 7.90625, 2.890625, -8.875, 0.31640625, 16.75, 3.09375, -2.203125, 0.318359375, -3.859375, 0.115234375, 5.625, -1.3515625, -6.09375, -1.9609375, 9.9375, 0.427734375, -3.59375, -2.296875, 3.09375, 11.5, 9.625, -12.75, 2.359375, -16.5, -1.0390625, 1.15625, -12.625, 4.84375, 7.84375, -5.03125, -5.03125, -0.76171875, -14.6875, 6.21875, -1.2890625, 3.984375, 4.1875, 10.8125, -11.25, 0.65234375, -6.84375, 2.296875, 2.875, -10.75, 7.78125, -4.0625, 2.9375, -0.66015625, 0.8515625, 7.3125, 2.140625, 1.515625, -5.0625, 4.625, 4.375, -14.1875, -12.1875], dtype=dtype, device="cuda")
    # q_values 2 openai moe
    # q_values = torch.tensor([0.005706787109375, 0.0299072265625, -0.314453125, 0.427734375, 0.20703125, -1.2734375, -0.025634765625, -1.6484375, -0.388671875, -1.2578125, 0.5078125, -0.138671875, -0.1201171875, -0.0037384033203125, -0.1826171875, -0.890625, 0.201171875, -2.15625, 0.93359375, -0.94921875, 1.171875, -0.359375, -0.6484375, -1.828125, -0.57421875, -0.4609375, 0.45703125, -0.3203125, 1.015625, -1.9609375, -0.8828125, -3.03125, -0.0751953125, -0.1748046875, 0.142578125, 0.21875, -0.427734375, -1.0078125, -0.90234375, -1.1171875, -0.84375, 0.044921875, -1.0625, -2.03125, 0.828125, 1.265625, 1.046875, -0.0341796875, 0.0966796875, -1.4140625, 0.4453125, -0.8984375, -0.197265625, 1.265625, 0.435546875, -1.296875, 0.75, -0.79296875, 0.65234375, -2.34375, -0.41015625, 1.84375, 0.7890625, -0.271484375], dtype=dtype, device="cuda")
    # q_values 3 qwen3
    q_values = torch.tensor([-2.390625, 1.4375, 1.265625, -2.90625, 0.8671875, 0.77734375, 0.6953125, 0.04638671875, -0.609375, 0.84765625, -0.283203125, 0.8828125, -1.5703125, 0.5859375, -0.96484375, 0.64453125, -0.39453125, -0.6640625, 0.29296875, 0.173828125, -0.65234375, -0.5546875, 0.44140625, -0.31640625, -2.265625, 0.478515625, -0.64453125, -0.8046875, 0.08642578125, 0.8125, 0.6328125, -1.6484375, 1.171875, 0.36328125, -0.4921875, -0.2216796875, 0.380859375, 0.58984375, 5.46875, 0.546875, -1.1015625, -1.21875, -0.46875, -0.490234375, -0.97265625, 1.2890625, 1.4765625, 1.75, -3.125, -1.3671875, -1.5, -3.6875, 5.3125, 3.3125, 3.375, 4.78125, 0.66796875, 1.8671875, -0.126953125, -0.68359375, -3.859375, -2.890625, 2.8125, 0.09716796875], dtype=dtype, device="cuda")
    q = q_values.view(batch_size * seq_len_per_request, num_qo_heads, head_dim)

    k = torch.zeros(
        batch_size * seq_len_per_request,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device="cuda",
    )
    # Reshape the hardcoded tensor to match expected shape
    # k_value 1 openai moe
    # k_values = torch.tensor([8.125, 11.6875, -4.375, 2.265625, 3.21875, -8.5, -1.8828125, -3.4375, 4.03125, -5.78125, -2.765625, -0.1318359375, -3.734375, -1.5, -5.09375, -9.875, 3.734375, 2.796875, -25.875, -3.59375, 0.76171875, -1.03125, 3.71875, 6.59375, 1.53125, 11.8125, -11.75, 6.5, 4.78125, -7.46875, -6.3125, 2.0625, -1.140625, 2.40625, -3.921875, 0.404296875, 2.546875, 3.28125, -4.78125, -4.5, -8.25, 13.25, -10.3125, -0.2021484375, -4.6875, -10.375, -4.5625, -0.478515625, -2.578125, 2.546875, 2.625, -7.25, -8.5, -0.08154296875, 2.640625, -5.53125, -0.9296875, 3.625, -9.0625, -2.34375, 14.4375, -7.9375, 2.5625, 2.328125], dtype=dtype, device="cuda")
    # k_values 2 openai moe
    # k_values = torch.tensor([-0.99609375, -3.65625, 2.453125, -2.390625, 3.40625, 5.46875, 3.765625, 1.75, 0.310546875, -1.1953125, -0.29296875, -38.0, -4.4375, 0.326171875, 0.361328125, 1.6796875, -1.4453125, -3.0, 0.69921875, 0.74609375, 0.56640625, 1.4609375, 0.98046875, 0.5390625, -0.6328125, -3.28125, 0.67578125, -2.078125, -0.046142578125, 2.53125, -1.625, -0.7734375, -5.75, -1.03125, -0.46484375, -0.6171875, 4.1875, 1.890625, 3.765625, 5.96875, 0.07470703125, -7.125, 1.8828125, 1.984375, 1.5234375, -0.64453125, 0.8671875, -2.03125, 1.59375, 1.5625, 0.69921875, 0.94921875, -0.66015625, -0.318359375, 0.9609375, -4.125, -1.265625, 1.0, 1.0078125, -0.189453125, -1.4609375, -2.765625, 1.5859375, 2.09375], dtype=dtype, device="cuda")
    # k_values 3 qwen3
    k_values = torch.tensor([-2.53125, 1.4921875, 0.025146484375, 0.228515625, -0.8671875, -1.125, 0.515625, 0.07666015625, 0.51953125, 1.34375, 0.09765625, 1.1875, -0.1123046875, -1.0703125, 0.73046875, 0.2158203125, 0.96484375, -2.84375, -0.08447265625, -0.81640625, 0.181640625, 0.421875, 0.98046875, 4.125, -3.0625, 0.97265625, 0.4609375, -2.578125, -0.23828125, -0.244140625, 1.46875, 0.28125, -2.453125, 2.765625, 0.2236328125, -2.765625, 3.375, 0.09912109375, 1.21875, -1.6796875, 1.4140625, 0.921875, 1.5390625, 2.59375, -0.8671875, -0.90234375, 1.4921875, 2.34375, -3.0, -0.423828125, 1.828125, -0.6484375, 0.58203125, -0.73828125, 1.4765625, 2.78125, -0.265625, -0.1083984375, 3.84375, 2.25, -1.1328125, -4.5, 1.15625, 6.90625], dtype=dtype, device="cuda")
    k = k_values.view(batch_size * seq_len_per_request, num_kv_heads, head_dim)
    
    v = torch.ones(
        batch_size * seq_len_per_request,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device="cuda",
    )
    # Reshape the hardcoded tensor to match expected shape
    # v_value 1 openai moe
    # v_values = torch.tensor([1.109375, -3.890625, -5.9375, 2.4375, -3.125, -1.2578125, 6.03125, -0.5859375, -3.125, -6.5, -2.5, 5.09375, -5.3125, -7.40625, 0.07421875, -1.6640625, 0.68359375, -3.71875, 4.65625, 3.34375, 7.3125, -0.11572265625, 5.53125, 7.46875, 0.90234375, 1.0703125, 3.203125, 1.703125, -4.5, -4.09375, 8.5625, 10.75, 7.09375, -3.125, 7.875, 1.2578125, -1.2734375, 3.15625, 5.78125, -7.375, -5.28125, 4.25, -1.953125, 8.1875, 7.625, -1.9765625, 4.9375, -0.18359375, -1.1015625, 2.78125, -2.640625, -6.8125, 7.28125, 3.265625, 2.296875, -0.2412109375, 1.4765625, 1.40625, 3.859375, 4.28125, -5.96875, 3.765625, 1.8515625, -3.9375], dtype=dtype, device="cuda")
    # v_value 2 openai moe
    # v_values = torch.tensor([-0.81640625, -0.5234375, 1.109375, -1.046875, 0.5703125, 0.064453125, -1.609375, -0.69921875, 0.328125, 0.028564453125, 1.0078125, 1.8125, -1.53125, 0.0927734375, -1.046875, 2.578125, -3.8125, 0.296875, 2.328125, 2.953125, 0.1591796875, 1.671875, 1.5625, -1.7265625, -1.203125, -1.2265625, 0.0262451171875, 1.03125, 0.302734375, 1.2265625, -2.03125, -1.234375, 0.34375, -0.7890625, -1.6796875, -0.6328125, -3.359375, -0.47265625, 0.228515625, -4.8125, -0.66015625, -0.6484375, 0.498046875, 0.2451171875, 2.046875, 0.734375, 0.94921875, 0.7890625, -0.53515625, -3.328125, -3.171875, 1.3671875, -1.2109375, 0.388671875, -1.09375, -1.4296875, -0.00946044921875, 2.25, 1.1171875, -0.298828125, -1.7890625, -0.84375, 2.515625, 2.265625], dtype=dtype, device="cuda")
    # v_values 3 qwen3
    v_values = torch.tensor([-0.00136566162109375, 0.0024566650390625, 0.0169677734375, 0.000484466552734375, 0.003936767578125, 0.0010528564453125, 0.0027313232421875, -0.0004329681396484375, 0.00012159347534179688, 0.00067138671875, -0.00150299072265625, -0.000701904296875, -0.0001354217529296875, 0.003021240234375, 0.0019989013671875, -0.00225830078125, -0.000946044921875, 0.000598907470703125, 0.0023651123046875, -0.0003490447998046875, 0.0034942626953125, -0.0015869140625, -0.0004673004150390625, -0.004791259765625, -0.0032958984375, -0.000743865966796875, 0.0067138671875, -0.000217437744140625, 0.000560760498046875, 3.147125244140625e-05, 0.00131988525390625, 0.00384521484375, 0.0004253387451171875, -0.0023651123046875, -0.003570556640625, -0.00020694732666015625, 0.001068115234375, 0.00183868408203125, -0.00244140625, 0.0026397705078125, -0.001617431640625, 7.927417755126953e-06, 0.004608154296875, -0.00010013580322265625, 0.000270843505859375, 2.944469451904297e-05, 0.005157470703125, -0.00131988525390625, -0.0026092529296875, -0.0023651123046875, 0.001800537109375, -0.002838134765625, -0.0015869140625, -0.00074005126953125, 0.001007080078125, 0.002838134765625, 0.000759124755859375, -0.0014495849609375, -0.000888824462890625, -0.001953125, 0.0025177001953125, -0.0022125244140625, -0.00174713134765625, 0.0016021728515625],dtype=dtype, device="cuda")
    v = v_values.view(batch_size * seq_len_per_request, num_kv_heads, head_dim)
    
    sink = torch.rand(num_qo_heads, device="cuda", dtype=torch.float32) * 100
    # sink = torch.tensor([8.1157], dtype=torch.float32, device="cuda")
    o = wrapper.run(q, k, v, sink, sm_scale)
    o_ref = sink_attention_ref(
        batch_size, q, k, v, sink, causal=causal, sm_scale=sm_scale
    )
    if dtype == torch.float16:
        torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
    else:
        torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)

    wrapper_paged = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        float_workspace_buffer, kv_layout="NHD", backend="fa2", jit_args=jit_args
    )
    kv_indices_host = torch.arange(
        0,
        batch_size * seq_len_per_request,
        dtype=torch.int32,
    )
    paged_kv_last_page_len_host = torch.full((batch_size,), 1, dtype=torch.int32)
    wrapper_paged.plan(
        qo_indptr_host,
        kv_indptr_host,
        kv_indices_host,
        paged_kv_last_page_len_host,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        1,
        causal=causal,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    o_paged = wrapper_paged.run(q, (k, v), sink, sm_scale)
    if dtype == torch.float16:
        torch.testing.assert_close(o_paged, o_ref, rtol=1e-3, atol=1e-3)
    else:
        torch.testing.assert_close(o_paged, o_ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    test_attention_sink(torch.float16, True)
