import unittest
from typing import Optional

import sgl_kernel  # noqa: F401
import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.cpu_test_utils import parametrize, precision
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="extra-test-cpu-intel")

causal_conv1d_weight_pack = torch.ops.sgl_kernel.causal_conv1d_weight_pack
causal_conv1d_fwd = torch.ops.sgl_kernel.causal_conv1d_fwd_cpu
causal_conv1d_update = torch.ops.sgl_kernel.causal_conv1d_update_cpu


torch.manual_seed(1234)

PAD_SLOT_ID = -1


def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    initial_states: Optional[torch.Tensor] = None,
    return_final_states: bool = False,
    final_states_out: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1)

    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(
            dtype_in
        )  # (batch, dim, width - 1)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


def causal_conv1d_update_ref(
    x, conv_state, weight, bias=None, activation=None, cache_seqlens=None
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the
        conv_state starting at the index
        @cache_seqlens % state_len before performing the convolution.

    out: (batch, dim) or (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")

    x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]

    x_new = torch.cat([conv_state, x], dim=-1)
    conv_state.copy_(x_new[:, :, -state_len:])
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0, groups=dim)[
        :, :, -seqlen:
    ]

    out = out.squeeze(-1)
    return out if activation is None else F.silu(out)


class TestCausalConv1d(CustomTestCase):
    activation = "silu"

    @parametrize(
        batch=[1, 1024],
        dim=[96, 512],
        seqlen=[2, 36],
        width=[4],
        has_bias=[True, False],
        has_initial_state=[True, False],
    )
    def test_causal_conv1d(
        self,
        batch,
        dim,
        seqlen,
        width,
        has_bias,
        has_initial_state,
        dtype=torch.bfloat16,
        prepack=True,
    ):
        x = torch.randn(batch, seqlen, dim).to(dtype).transpose_(-1, -2)
        weight = torch.randn(dim, width).to(dtype)
        bias = torch.randn(dim).to(dtype) if has_bias else None

        if has_initial_state:
            initial_states = torch.randn(batch, dim, width - 1, dtype=dtype)
            has_initial_state_tensor = torch.ones(batch, dtype=torch.bool)
        else:
            initial_states = None
            has_initial_state_tensor = None

        packed_weight = causal_conv1d_weight_pack(weight) if prepack else weight

        out_ref, final_states_ref = causal_conv1d_ref(
            x,
            weight,
            bias,
            initial_states,
            return_final_states=has_initial_state,
            activation=self.activation,
        )

        out = causal_conv1d_fwd(
            x,
            packed_weight,
            bias,
            initial_states,
            None,
            None,
            has_initial_state_tensor,
            self.activation in ["silu"],
            PAD_SLOT_ID,
            prepack,
        )

        atol = rtol = precision[dtype]
        torch.testing.assert_close(out_ref, out, atol=atol, rtol=rtol)
        torch.testing.assert_close(
            final_states_ref, initial_states, atol=atol, rtol=rtol
        )

    @parametrize(
        batch=[11],
        dim=[96],
        max_seqlen=[66],
        width=[4],
    )
    def test_causal_conv1d_varlen(
        self,
        batch,
        dim,
        max_seqlen,
        width,
        has_bias=False,
        dtype=torch.bfloat16,
        prepack=False,
    ):
        total_entries = batch + 3

        seqlens = torch.randint(1, max_seqlen, (batch + 1,))
        seqlens[0] = 0
        # 1 or 2 must test
        seqlens[-2] = 2

        query_start_loc = torch.cumsum(seqlens, dim=0).to(torch.int32)

        seqlen = query_start_loc[-1].item()
        x = torch.randn(seqlen, dim, dtype=dtype).transpose_(-1, -2)
        weight = torch.randn(dim, width, dtype=dtype)
        bias = torch.randn(dim, dtype=dtype) if has_bias else None

        final_states = torch.randn(total_entries, dim, width - 1, dtype=dtype)
        final_states_ref = final_states.clone()

        has_initial_states = torch.randint(0, 2, (batch,), dtype=torch.bool).fill_(
            False
        )
        state_indices = torch.randperm(total_entries, dtype=torch.int32)[:batch]

        out_ref = []
        out_ref_b = []

        return_final_states = final_states is not None
        splits = torch.split(x, seqlens[1:].tolist(), dim=1)
        for i, x_s in enumerate(splits):
            out_ref_b.append(
                causal_conv1d_ref(
                    x_s.unsqueeze(0),
                    weight,
                    bias,
                    activation=self.activation,
                    return_final_states=return_final_states,
                    final_states_out=(
                        final_states_ref[state_indices[i]].unsqueeze(0)
                        if return_final_states
                        else None
                    ),
                    initial_states=(
                        final_states_ref[state_indices[i]].unsqueeze(0)
                        if has_initial_states[i]
                        else None
                    ),
                )
            )
        out_ref.append(torch.cat([t[0] for t in out_ref_b], dim=2))
        out_ref_tensor = torch.cat(out_ref, dim=0).squeeze(0)

        out = causal_conv1d_fwd(
            x,
            weight,
            bias,
            final_states,
            query_start_loc,
            state_indices,
            has_initial_states,
            self.activation in ["silu"],
            PAD_SLOT_ID,
            prepack,
        )

        atol = rtol = precision[dtype]
        torch.testing.assert_close(out_ref_tensor, out, atol=atol, rtol=rtol)
        torch.testing.assert_close(final_states_ref, final_states, atol=atol, rtol=rtol)

    @parametrize(
        dim=[96],
        width=[4],
        num_tokens=[1, 7],
    )
    def test_causal_conv1d_prefill_no_bias(
        self, dim, width, num_tokens, dtype=torch.bfloat16
    ):
        # Regression: MambaMixer2 CPU prefill must accept use_conv_bias=False
        # (conv1d.bias is None).
        from sgl_kernel.mamba import causal_conv1d_fn_cpu

        # Slice out of a wider projection (as torch.split does in
        # MambaMixer2.forward), then repack contiguously like the CPU branch.
        proj = torch.randn(num_tokens, dim + 32, dtype=dtype)
        x = proj[:, :dim].contiguous().transpose(0, 1)

        weight = torch.randn(dim, width, dtype=dtype)
        conv_states = torch.zeros(2, dim, width - 1, dtype=dtype)
        conv_states_ref = conv_states.clone()
        query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32)
        cache_indices = torch.tensor([1], dtype=torch.int32)
        has_initial_state = torch.tensor([False])

        out_ref, _ = causal_conv1d_ref(
            x.unsqueeze(0),
            weight,
            None,
            return_final_states=True,
            final_states_out=conv_states_ref[1].unsqueeze(0),
            activation=self.activation,
        )
        out = causal_conv1d_fn_cpu(
            x,
            causal_conv1d_weight_pack(weight),
            None,
            self.activation,
            conv_states,
            has_initial_state,
            cache_indices,
            query_start_loc,
            None,
        )

        atol = rtol = precision[dtype]
        torch.testing.assert_close(out_ref[0], out, atol=atol, rtol=rtol)
        torch.testing.assert_close(conv_states_ref, conv_states, atol=atol, rtol=rtol)

    @parametrize(
        dim=[96],
        width=[4],
        has_bias=[True, False],
        prepack=[False, True],
    )
    def test_causal_conv1d_varlen_mixed_initial_states(
        self, dim, width, has_bias, prepack, dtype=torch.bfloat16
    ):
        # Mixed cache hits (has_initial_state=True) and misses (False), including
        # one-token sequences on both sides, to exercise the cached-state branch
        # of causal_conv1d_fwd_varlen_kernel_impl. Checks output and cache.
        seqlens_list = [1, 5, 1, 2, 17, 3, 1, 40]
        has_initial_list = [True, False, False, True, True, False, True, False]
        batch = len(seqlens_list)
        total_entries = batch + 3

        seqlens = torch.tensor([0] + seqlens_list)
        query_start_loc = torch.cumsum(seqlens, dim=0).to(torch.int32)
        seqlen = query_start_loc[-1].item()

        x = torch.randn(seqlen, dim, dtype=dtype).transpose_(-1, -2)
        weight = torch.randn(dim, width, dtype=dtype)
        bias = torch.randn(dim, dtype=dtype) if has_bias else None
        packed_weight = causal_conv1d_weight_pack(weight) if prepack else weight

        conv_states = torch.randn(total_entries, dim, width - 1, dtype=dtype)
        conv_states_ref = conv_states.clone()
        has_initial_states = torch.tensor(has_initial_list, dtype=torch.bool)
        state_indices = torch.randperm(total_entries, dtype=torch.int32)[:batch]

        out_ref_list = []
        for i, x_s in enumerate(torch.split(x, seqlens_list, dim=1)):
            idx = state_indices[i]
            init = (
                conv_states_ref[idx].unsqueeze(0).clone()
                if has_initial_states[i]
                else None
            )
            out_s, _ = causal_conv1d_ref(
                x_s.unsqueeze(0),
                weight,
                bias,
                initial_states=init,
                return_final_states=True,
                final_states_out=conv_states_ref[idx].unsqueeze(0),
                activation=self.activation,
            )
            out_ref_list.append(out_s[0])
        out_ref = torch.cat(out_ref_list, dim=-1)

        out = causal_conv1d_fwd(
            x,
            packed_weight,
            bias,
            conv_states,
            query_start_loc,
            state_indices,
            has_initial_states,
            self.activation in ["silu"],
            PAD_SLOT_ID,
            prepack,
        )

        atol = rtol = precision[dtype]
        torch.testing.assert_close(out_ref, out, atol=atol, rtol=rtol)
        torch.testing.assert_close(conv_states_ref, conv_states, atol=atol, rtol=rtol)

    @parametrize(
        batch=[11],
        dim=[32, 64, 96],
        width=[4],
    )
    def test_causal_conv1d_update(
        self, batch, dim, width, has_bias=False, dtype=torch.bfloat16, prepack=True
    ):
        x = torch.randn(batch, dim).to(dtype)
        conv_state = torch.randn(batch, dim, width - 1, dtype=dtype)
        weight = torch.randn(dim, width).to(dtype)
        bias = torch.randn(dim).to(dtype) if has_bias else None

        packed_weight = causal_conv1d_weight_pack(weight) if prepack else weight

        conv_state_ref = conv_state.clone()
        out_ref = causal_conv1d_update_ref(
            x, conv_state_ref, weight, bias, activation=self.activation
        )

        cache_seqlens = None
        conv_state_indices = None
        out = causal_conv1d_update(
            x,
            conv_state,
            packed_weight,
            bias,
            self.activation in ["silu"],
            cache_seqlens,
            conv_state_indices,
            PAD_SLOT_ID,
            prepack,
        )

        atol = rtol = precision[dtype]
        torch.testing.assert_close(out_ref, out, atol=atol, rtol=rtol)
        torch.testing.assert_close(conv_state_ref, conv_state, atol=atol, rtol=rtol)

    @parametrize(
        batch=[7],
        dim=[96],
        width=[4],
    )
    def test_causal_conv1d_update_with_batch_gather(
        self, batch, dim, width, has_bias=False, dtype=torch.bfloat16, prepack=True
    ):
        total_entries = batch + 3

        x = torch.randn(batch, dim).to(dtype=dtype)

        conv_state_indices = torch.randperm(total_entries)[:batch].to(dtype=torch.int32)
        conv_state = torch.randn(total_entries, dim, width - 1, dtype=dtype)

        weight = torch.randn(dim, width).to(dtype=dtype)
        bias = torch.randn(dim).to(dtype=dtype) if has_bias else None
        conv_state_ref = conv_state[conv_state_indices, :]

        packed_weight = causal_conv1d_weight_pack(weight) if prepack else weight

        out_ref = causal_conv1d_update_ref(
            x, conv_state_ref, weight, bias, activation=self.activation
        )

        cache_seqlens = None
        out = causal_conv1d_update(
            x,
            conv_state,
            packed_weight,
            bias,
            self.activation in ["silu"],
            cache_seqlens,
            conv_state_indices,
            PAD_SLOT_ID,
            prepack,
        )

        atol = rtol = precision[dtype]
        torch.testing.assert_close(out_ref, out, atol=atol, rtol=rtol)
        torch.testing.assert_close(
            conv_state_ref, conv_state[conv_state_indices, :], atol=atol, rtol=rtol
        )


if __name__ == "__main__":
    unittest.main()
