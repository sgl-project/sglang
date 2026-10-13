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
causal_conv1d_fwd_unpacked = torch.ops.sgl_kernel.causal_conv1d_fwd_unpacked_cpu
causal_conv1d_update_unpacked = torch.ops.sgl_kernel.causal_conv1d_update_unpacked_cpu


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


def make_conv_states(num_slots, dim, width, dtype, dim_contiguous):
    if dim_contiguous:
        # the layout the CPU AMX mamba pool allocates
        return torch.randn(num_slots, width - 1, dim, dtype=dtype).transpose(1, 2)
    return torch.randn(num_slots, dim, width - 1, dtype=dtype)


class TestCausalConv1dUnpacked(CustomTestCase):
    @parametrize(
        width=[2, 3, 4],
        dim=[64, 80],
        dtype=[torch.bfloat16, torch.float32],
        has_bias=[True, False],
        silu=[True, False],
        dim_contiguous=[True, False],
    )
    def test_fwd_varlen(self, width, dim, dtype, has_bias, silu, dim_contiguous):
        """Chunk boundaries, initial states (incl. seqlen < width - 1), pad slots and dim tails."""
        seqlens = [1, 70, 2, 130, 5]
        slots = [3, 0, PAD_SLOT_ID, 5, 1]
        use_init = [True, True, True, False, True]
        query_start_loc = torch.tensor(
            [0] + torch.tensor(seqlens).cumsum(0).tolist(), dtype=torch.int32
        )
        x = torch.randn(sum(seqlens), dim, dtype=dtype).t()
        weight = torch.randn(dim, width, dtype=dtype)
        bias = torch.randn(dim, dtype=dtype) if has_bias else None
        states = make_conv_states(7, dim, width, dtype, dim_contiguous)
        states_before = states.clone()

        out = causal_conv1d_fwd_unpacked(
            x,
            weight,
            bias,
            states,
            query_start_loc,
            torch.tensor(slots, dtype=torch.int32),
            torch.tensor(use_init),
            silu,
            PAD_SLOT_ID,
        )

        # fast_silu is accurate to ~3e-5 relative, looser than the fp32 default
        atol = rtol = max(precision[dtype], 1e-4)
        activation = "silu" if silu else None
        for i, slot in enumerate(slots):
            if slot == PAD_SLOT_ID:
                continue
            start, end = query_start_loc[i].item(), query_start_loc[i + 1].item()
            out_ref, state_ref = causal_conv1d_ref(
                x[:, start:end].unsqueeze(0),
                weight,
                bias,
                initial_states=(
                    states_before[slot].unsqueeze(0) if use_init[i] else None
                ),
                return_final_states=True,
                activation=activation,
            )
            torch.testing.assert_close(
                out[:, start:end], out_ref.squeeze(0), atol=atol, rtol=rtol
            )
            torch.testing.assert_close(states[slot], state_ref.squeeze(0))

        for slot in (2, 4, 6):
            torch.testing.assert_close(states[slot], states_before[slot])

    @parametrize(
        width=[2, 3, 4],
        dim=[64, 80],
        has_bias=[True, False],
        dim_contiguous=[True, False],
    )
    def test_update(self, width, dim, has_bias, dim_contiguous, dtype=torch.bfloat16):
        slots = [4, PAD_SLOT_ID, 0, 2]
        x = torch.randn(len(slots), dim, dtype=dtype)
        weight = torch.randn(dim, width, dtype=dtype)
        bias = torch.randn(dim, dtype=dtype) if has_bias else None
        states = make_conv_states(6, dim, width, dtype, dim_contiguous)
        states_before = states.clone()

        out = causal_conv1d_update_unpacked(
            x,
            states,
            weight,
            bias,
            True,
            torch.tensor(slots, dtype=torch.int32),
            PAD_SLOT_ID,
        )

        atol = rtol = precision[dtype]
        for i, slot in enumerate(slots):
            if slot == PAD_SLOT_ID:
                self.assertEqual(out[i].abs().max().item(), 0.0)
                continue
            state_ref = states_before[slot].clone().unsqueeze(0)
            out_ref = causal_conv1d_update_ref(
                x[i : i + 1], state_ref, weight, bias, activation="silu"
            )
            torch.testing.assert_close(out[i], out_ref[0], atol=atol, rtol=rtol)
            torch.testing.assert_close(states[slot], state_ref[0])

        for slot in (1, 3, 5):
            torch.testing.assert_close(states[slot], states_before[slot])

    def test_prefill_then_decode_matches_full_sequence(
        self, width=3, dim=64, seqlen=37, n_decode=4, dtype=torch.bfloat16
    ):
        """Prefill then token-by-token decode must equal one full-sequence conv."""
        x_full = torch.randn(seqlen + n_decode, dim, dtype=dtype)
        weight = torch.randn(dim, width, dtype=dtype)
        states = make_conv_states(2, dim, width, dtype, dim_contiguous=True)
        slot = torch.tensor([1], dtype=torch.int32)

        outs = [
            causal_conv1d_fwd_unpacked(
                x_full[:seqlen].t(),
                weight,
                None,
                states,
                torch.tensor([0, seqlen], dtype=torch.int32),
                slot,
                torch.tensor([False]),
                False,
                PAD_SLOT_ID,
            ).t()
        ]
        for t in range(seqlen, seqlen + n_decode):
            outs.append(
                causal_conv1d_update_unpacked(
                    x_full[t : t + 1].contiguous(),
                    states,
                    weight,
                    None,
                    False,
                    slot,
                    PAD_SLOT_ID,
                )
            )

        out_ref, _ = causal_conv1d_ref(x_full.t().unsqueeze(0), weight, activation=None)
        atol = rtol = precision[dtype]
        torch.testing.assert_close(
            torch.cat(outs), out_ref.squeeze(0).t(), atol=atol, rtol=rtol
        )


if __name__ == "__main__":
    unittest.main()
