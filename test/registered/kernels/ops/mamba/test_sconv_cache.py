import pytest
import torch

from sglang.srt.models.inkling_common.kernels.sconv import (
    HIS_PREFIX,
    HIS_ZEROS,
    PAD_SLOT_ID,
    causal_conv1d,
    fused_decode_sconv_metadata,
    fused_extend_sconv_metadata,
    update_sconv_cache,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


@requires_cuda
def test_update_sconv_cache_matches_reference():
    torch.manual_seed(0)
    dtype = torch.bfloat16
    dim, width = 128, 4
    query_lens = torch.tensor([0, 1, 2, 3, 4, 5, 7], device="cuda")
    query_start_loc = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            query_lens.cumsum(0).to(torch.int32),
        ]
    )
    cache_indices = torch.tensor(
        [0, 1, PAD_SLOT_ID, 3, 4, 5, 6], dtype=torch.int32, device="cuda"
    )
    has_initial_state = torch.tensor(
        [True, False, True, True, False, True, False],
        dtype=torch.bool,
        device="cuda",
    )
    hidden_states = torch.randn(int(query_lens.sum()), dim, dtype=dtype, device="cuda")
    initial_cache = torch.randn(8, width - 1, dim, dtype=dtype, device="cuda")
    cache = initial_cache.clone()

    update_sconv_cache(
        x=hidden_states,
        sconv_cache=cache,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        query_start_loc=query_start_loc,
    )

    expected = initial_cache.clone()
    for batch_idx, slot in enumerate(cache_indices.tolist()):
        start = int(query_start_loc[batch_idx])
        end = int(query_start_loc[batch_idx + 1])
        if slot == PAD_SLOT_ID or start == end:
            continue
        prior = (
            initial_cache[slot]
            if has_initial_state[batch_idx]
            else torch.zeros_like(initial_cache[slot])
        )
        expected[slot] = torch.cat([prior, hidden_states[start:end]])[-(width - 1) :]

    torch.testing.assert_close(cache, expected, rtol=0, atol=0)


def _extend_metadata(
    cache_indices: torch.Tensor,
    length: int,
    *,
    has_prefix: bool,
):
    lens = torch.tensor([length], dtype=torch.int64, device="cuda")
    result = fused_extend_sconv_metadata(
        B=1,
        T=length,
        cache_indices=cache_indices,
        his_mode=HIS_PREFIX if has_prefix else HIS_ZEROS,
        extend_seq_lens=lens,
        his_src=lens if has_prefix else None,
    )
    assert result is not None
    return result


@requires_cuda
def test_cached_continuations_match_full_prefill():
    torch.manual_seed(1)
    dtype = torch.bfloat16
    length, prefix_len, dim, width = 24, 10, 128, 4
    hidden_states = torch.randn(length, dim, dtype=dtype, device="cuda")
    weight = torch.randn(dim, width, dtype=dtype, device="cuda")
    cache_indices = torch.zeros(1, dtype=torch.int32, device="cuda")

    full_cache = torch.zeros(8, width - 1, dim, dtype=dtype, device="cuda")
    _, _, full_meta = _extend_metadata(cache_indices, length, has_prefix=False)
    full_output = causal_conv1d(
        x=hidden_states,
        weight=weight,
        sconv_cache=full_cache,
        activation="silu",
        use_residual=True,
        **full_meta,
    )

    decode_cache = torch.zeros_like(full_cache)
    query_start_loc, has_initial_state, decode_meta = fused_decode_sconv_metadata(
        B=1, cache_indices=cache_indices
    )
    decode_outputs = []
    for token in hidden_states.split(1):
        decode_outputs.append(
            causal_conv1d(
                x=token,
                weight=weight,
                sconv_cache=decode_cache,
                activation="silu",
                use_residual=True,
                is_decode=True,
                **decode_meta,
            )
        )
        update_sconv_cache(
            x=token,
            sconv_cache=decode_cache,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            query_start_loc=query_start_loc,
        )
    decode_output = torch.cat(decode_outputs)

    extend_cache = torch.zeros_like(full_cache)
    prefix = hidden_states[:prefix_len]
    prefix_qsl, prefix_his, prefix_meta = _extend_metadata(
        cache_indices, prefix_len, has_prefix=False
    )
    causal_conv1d(
        x=prefix,
        weight=weight,
        sconv_cache=extend_cache,
        activation="silu",
        use_residual=True,
        **prefix_meta,
    )
    update_sconv_cache(
        x=prefix,
        sconv_cache=extend_cache,
        cache_indices=cache_indices,
        has_initial_state=prefix_his,
        query_start_loc=prefix_qsl,
    )
    suffix = hidden_states[prefix_len:]
    _, _, suffix_meta = _extend_metadata(cache_indices, len(suffix), has_prefix=True)
    suffix_output = causal_conv1d(
        x=suffix,
        weight=weight,
        sconv_cache=extend_cache,
        activation="silu",
        use_residual=True,
        **suffix_meta,
    )

    torch.testing.assert_close(decode_output, full_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        suffix_output, full_output[prefix_len:], rtol=2e-2, atol=2e-2
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-x"]))
