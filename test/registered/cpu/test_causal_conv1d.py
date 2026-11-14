import unittest
from typing import Optional

import sgl_kernel  # noqa: F401
import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.cpu_test_utils import parametrize, precision
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="stage-a-test-cpu-intel")

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

    @parametrize(width=[3, 4])
    def test_prefix_and_pad_slots(self, width):
        """Short continuations retain old state and padded requests never access a cache slot."""
        dim = 64
        dtype = torch.bfloat16
        weight = torch.randn(dim, width, dtype=dtype)
        bias = torch.randn(dim, dtype=dtype)
        inputs = torch.randn(40, dim, dtype=dtype)
        states = torch.randn(4, width - 1, dim, dtype=dtype).transpose(1, 2)
        previous = states.clone()
        slots = torch.tensor([2, -1, 0], dtype=torch.int32)
        starts = torch.tensor([0, 1, 4, 40], dtype=torch.int32)
        output = causal_conv1d_fwd(
            inputs.t(),
            causal_conv1d_weight_pack(weight),
            bias,
            states,
            starts,
            slots,
            torch.tensor([True, True, False]),
            False,
            -1,
            True,
        )
        for start, end, slot, initial in [(0, 1, 2, True), (4, 40, 0, False)]:
            reference, final_state = causal_conv1d_ref(
                inputs[start:end].t().unsqueeze(0),
                weight,
                bias,
                initial_states=previous[slot].unsqueeze(0) if initial else None,
                return_final_states=True,
                activation=None,
            )
            torch.testing.assert_close(
                output[:, start:end],
                reference[0],
                atol=precision[dtype],
                rtol=precision[dtype],
            )
            torch.testing.assert_close(states[slot], final_state[0])
        torch.testing.assert_close(states[1], previous[1])
        torch.testing.assert_close(states[3], previous[3])
        previous = states.clone()
        decoded = causal_conv1d_update(
            inputs[:3].contiguous(),
            states,
            causal_conv1d_weight_pack(weight),
            bias,
            False,
            None,
            slots,
            -1,
            True,
        )
        for batch_index, slot in [(0, 2), (2, 0)]:
            state_reference = previous[slot].clone().unsqueeze(0)
            reference = causal_conv1d_update_ref(
                inputs[batch_index : batch_index + 1],
                state_reference,
                weight,
                bias,
            )
            torch.testing.assert_close(
                decoded[batch_index],
                reference[0],
                atol=precision[dtype],
                rtol=precision[dtype],
            )
            torch.testing.assert_close(states[slot], state_reference[0])
        self.assertEqual(decoded[1].abs().max().item(), 0)

    def test_width3_prefill_decode(self):
        """Width-three convolution must preserve its two-token state across decode."""
        dim, width = 64, 3
        weight = torch.randn(dim, width, dtype=torch.bfloat16)
        inputs = torch.randn(6, dim, dtype=torch.bfloat16)
        states = torch.zeros(1, width - 1, dim, dtype=torch.bfloat16).transpose(1, 2)
        indices = torch.tensor([0], dtype=torch.int32)
        prefill = causal_conv1d_fwd(
            inputs[:5].t(),
            weight,
            None,
            states,
            torch.tensor([0, 5], dtype=torch.int32),
            indices,
            torch.tensor([False]),
            False,
            PAD_SLOT_ID,
            False,
        )
        decode = causal_conv1d_update(
            inputs[5:],
            states,
            weight,
            None,
            False,
            None,
            indices,
            PAD_SLOT_ID,
            False,
        )
        reference, _ = causal_conv1d_ref(
            inputs.t().unsqueeze(0),
            weight,
            activation=None,
        )
        torch.testing.assert_close(
            torch.cat([prefill.t(), decode]),
            reference.squeeze(0).t(),
            atol=precision[torch.bfloat16],
            rtol=precision[torch.bfloat16],
        )
        torch.testing.assert_close(states[0], inputs[-2:].t())

    @parametrize(
        batch=[1, 1024],
        dim=[96, 512],
        seqlen=[2, 36],
        width=[3, 4],
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
        width=[3, 4],
        prepack=[False, True],
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

        has_initial_states = torch.arange(batch) % 2 == 0
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
            causal_conv1d_weight_pack(weight) if prepack else weight,
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
        batch=[11],
        dim=[32, 64, 96],
        width=[3, 4],
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
        width=[3, 4],
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
