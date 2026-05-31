import torch

from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
    FlashInferGDNKernel,
)


def test_sm100_bf16_mtp_uses_cache_scoped_intermediate_buffer():
    captured = _run_sm100_bf16_mtp_wrapper()

    assert captured["intermediate_shape"] == (8, 3, 4, 8, 8)
    assert captured["cache_indices"].tolist() == [4, 6]
    assert captured["target_verify_intermediate_state_indexing"] == "cache"


def _run_sm100_bf16_mtp_wrapper():
    batch_size = 2
    draft_token_num = 3
    pool_size = 8
    num_heads = 2
    num_v_heads = 4
    head_dim = 8

    captured = {}

    def fake_bf16_mtp_fn(**kwargs):
        intermediate_states_buffer = kwargs["intermediate_states_buffer"]
        captured["intermediate_shape"] = tuple(intermediate_states_buffer.shape)
        captured["cache_indices"] = kwargs["initial_state_indices"].clone()
        return torch.empty(
            batch_size,
            draft_token_num,
            num_v_heads,
            head_dim,
            dtype=kwargs["v"].dtype,
        )

    kernel = object.__new__(FlashInferGDNKernel)
    kernel.use_state_pool = True
    kernel._bf16_mtp_fn = fake_bf16_mtp_fn
    kernel.target_verify_intermediate_state_indexing = "cache"

    seq_len = batch_size * draft_token_num
    q = torch.randn(1, seq_len, num_heads, head_dim)
    k = torch.randn(1, seq_len, num_heads, head_dim)
    v = torch.randn(1, seq_len, num_v_heads, head_dim)
    a = torch.randn(seq_len, num_v_heads)
    b = torch.randn(seq_len, num_v_heads)
    ssm_states = torch.randn(pool_size, num_v_heads, head_dim, head_dim)
    cache_indices = torch.tensor([4, 6], dtype=torch.int32)
    intermediate_states_buffer = torch.empty(
        pool_size,
        draft_token_num,
        num_v_heads,
        head_dim,
        head_dim,
    )

    output = kernel.target_verify(
        A_log=torch.randn(num_v_heads),
        dt_bias=torch.randn(num_v_heads),
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        ssm_states=ssm_states,
        cache_indices=cache_indices,
        query_start_loc=torch.arange(0, seq_len + 1, draft_token_num),
        intermediate_states_buffer=intermediate_states_buffer,
        intermediate_state_indices=torch.arange(batch_size, dtype=torch.int32),
        cache_steps=draft_token_num,
        retrieve_parent_token=None,
    )

    assert output.shape == (1, seq_len, num_v_heads, head_dim)
    captured["target_verify_intermediate_state_indexing"] = (
        kernel.target_verify_intermediate_state_indexing
    )
    return captured
