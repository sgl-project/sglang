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


def chain_verify_window_ref(x, history, width):
    """Per-token ancestor walk, transcribed from the Triton
    ``_causal_conv1d_update_kernel`` tree branch.

    The kernel stores the j-th ancestor of each step into window slot
    ``W - j - 2``; once the walk runs off the front of the draft block it reads
    the prior conv-state columns. Spelled out with Python loops so the CPU
    path's window layout is checked against a different formulation. The
    arithmetic itself is not modelled here -- the C++ kernel consumes a
    VNNI-prepacked weight, so a plain-weight reference cannot reproduce it;
    ``test_verify_matches_cpp_decode_kernel`` covers that instead.
    """
    batch, dim, seqlen = x.shape
    window = torch.zeros((batch, seqlen, dim, width - 1), dtype=x.dtype)
    for b in range(batch):
        for t in range(seqlen):
            for j in range(width):
                idx = t - j
                # idx < 0 walks off the block into the conv-state history.
                val = x[b, :, idx] if idx >= 0 else history[b, :, idx + width - 1]
                if width - j - 2 >= 0:
                    window[b, t, :, width - j - 2] = val
    final_state = torch.cat([history, x], dim=-1)[:, :, seqlen:]
    return window, final_state


class TestCausalConv1dChainVerify(CustomTestCase):
    """Covers the DFlash target-verify path of ``causal_conv1d_update_cpu``.

    DFlash drafts a linear chain, so the kernel's parent walk collapses into a
    rolling window and the block becomes the decode kernel replayed token by
    token. These cases pin the index math (window layout, state roll, parent
    ids, cache-slot targeting) and the arithmetic against that kernel.
    """

    @parametrize(
        batch=[1, 3],
        dim=[96],
        draft_token_num=[4, 16],
        width=[4],
        has_bias=[True, False],
    )
    def test_chain_verify_matches_ancestor_walk(
        self, batch, dim, draft_token_num, width, has_bias
    ):
        from sgl_kernel.mamba import causal_conv1d_update_cpu

        dtype = torch.bfloat16
        state_len = width - 1
        num_lines = batch + 5

        x = torch.randn(batch, dim, draft_token_num, dtype=dtype)
        weight = torch.randn(dim, width, dtype=dtype)
        bias = torch.randn(dim, dtype=dtype) if has_bias else None

        conv_states = torch.randn(num_lines, dim, state_len, dtype=dtype)
        # Non-contiguous, non-identity slots to catch index-vs-position bugs.
        conv_state_indices = torch.arange(batch, dtype=torch.int32) + 2
        history = conv_states[conv_state_indices.long()].clone()

        window_buf = torch.zeros(
            num_lines, draft_token_num, dim, width - 1, dtype=dtype
        )
        intermediate_state_indices = torch.arange(batch, dtype=torch.int32) + 1
        retrieve_next_token = (
            torch.arange(1, draft_token_num + 1).unsqueeze(0).repeat(batch, 1)
        )
        retrieve_next_token[:, -1] = -1
        retrieve_next_sibling = torch.full((batch, draft_token_num), -1)
        retrieve_parent_token = torch.empty_like(retrieve_next_token)

        causal_conv1d_update_cpu(
            x,
            conv_states,
            weight,
            bias,
            "silu",
            conv_state_indices,
            intermediate_conv_window=window_buf,
            intermediate_state_indices=intermediate_state_indices,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_parent_token=retrieve_parent_token,
        )

        window_ref, state_ref = chain_verify_window_ref(x, history, width)

        torch.testing.assert_close(
            window_buf[intermediate_state_indices.long()], window_ref
        )
        torch.testing.assert_close(conv_states[conv_state_indices.long()], state_ref)

        expected_parent = (
            torch.arange(-1, draft_token_num - 1).unsqueeze(0).repeat(batch, 1)
        )
        expected_parent[:, 0] = 0
        torch.testing.assert_close(retrieve_parent_token, expected_parent)

    def test_untouched_slots_are_not_written(self):
        """The write must land only on the indexed cache lines; a bug that
        ignores the index tensors would still pass an out-value-only check."""
        from sgl_kernel.mamba import causal_conv1d_update_cpu

        dtype = torch.bfloat16
        batch, dim, draft_token_num, width = 2, 96, 8, 4

        conv_states = torch.randn(6, dim, width - 1, dtype=dtype)
        window_buf = torch.randn(6, draft_token_num, dim, width - 1, dtype=dtype)
        before_states = conv_states.clone()
        before_window = window_buf.clone()

        conv_state_indices = torch.tensor([4, 1], dtype=torch.int32)
        intermediate_state_indices = torch.tensor([3, 0], dtype=torch.int32)

        causal_conv1d_update_cpu(
            torch.randn(batch, dim, draft_token_num, dtype=dtype),
            conv_states,
            torch.randn(dim, width, dtype=dtype),
            None,
            "silu",
            conv_state_indices,
            intermediate_conv_window=window_buf,
            intermediate_state_indices=intermediate_state_indices,
            retrieve_next_token=None,
            retrieve_next_sibling=None,
            retrieve_parent_token=None,
        )

        untouched_states = [i for i in range(6) if i not in (4, 1)]
        untouched_window = [i for i in range(6) if i not in (3, 0)]
        torch.testing.assert_close(
            conv_states[untouched_states], before_states[untouched_states]
        )
        torch.testing.assert_close(
            window_buf[untouched_window], before_window[untouched_window]
        )

    def test_output_layout_allows_caller_transpose_view(self):
        """The GDN backend does ``out.transpose(1, 2).view(seq_len, -1)``, which
        only works if the output keeps the caller's transposed strides (the
        Triton kernel returns ``torch.empty_like(x)``). A plain [B, D, T]
        contiguous result raises "view size is not compatible"."""
        from sgl_kernel.mamba import causal_conv1d_update_cpu

        dtype = torch.bfloat16
        batch, dim, draft_token_num, width = 2, 96, 8, 4
        seq_len = batch * draft_token_num

        # Exactly how gdn_backend builds the input.
        packed = torch.randn(seq_len, dim, dtype=dtype)
        mixed_qkv = packed.view(batch, draft_token_num, -1).transpose(1, 2)

        out = causal_conv1d_update_cpu(
            mixed_qkv,
            torch.randn(4, dim, width - 1, dtype=dtype),
            torch.randn(dim, width, dtype=dtype),
            None,
            "silu",
            torch.zeros(batch, dtype=torch.int32),
            intermediate_conv_window=torch.zeros(
                4, draft_token_num, dim, width - 1, dtype=dtype
            ),
            intermediate_state_indices=torch.arange(batch, dtype=torch.int32),
            retrieve_next_token=None,
            retrieve_next_sibling=None,
            retrieve_parent_token=None,
        )

        self.assertEqual(out.transpose(1, 2).view(seq_len, -1).shape, (seq_len, dim))

    def test_verify_matches_cpp_decode_kernel(self):
        """Replaying the C++ decode kernel token by token is the only oracle for
        the verify rewrite that shares no code with it."""
        from sgl_kernel.mamba import causal_conv1d_update_cpu

        torch.manual_seed(0)
        dtype = torch.bfloat16
        batch, dim, width, lines = 2, 128, 4, 6
        cache_idx = torch.tensor([0, 3], dtype=torch.int32)

        for draft_token_num in (1, 4):
            with self.subTest(draft_token_num=draft_token_num):
                x = torch.randn(batch, dim, draft_token_num, dtype=dtype)
                base_state = torch.randn(lines, dim, width - 1, dtype=dtype)
                weight = torch.randn(dim, width, dtype=dtype) * 0.5
                bias = torch.randn(dim, dtype=dtype)

                verify_state = base_state.clone()
                window = torch.zeros(
                    lines, draft_token_num, dim, width - 1, dtype=dtype
                )
                out = causal_conv1d_update_cpu(
                    x,
                    verify_state,
                    weight,
                    bias,
                    "silu",
                    cache_idx,
                    intermediate_conv_window=window,
                    intermediate_state_indices=torch.arange(batch, dtype=torch.int32),
                    retrieve_next_token=None,
                    retrieve_next_sibling=None,
                    retrieve_parent_token=None,
                )

                decode_state = base_state.clone()
                for t in range(draft_token_num):
                    ref = causal_conv1d_update_cpu(
                        x[:, :, t].contiguous(),
                        decode_state,
                        weight,
                        bias,
                        "silu",
                        cache_idx,
                    )
                    torch.testing.assert_close(out[:, :, t], ref, rtol=5e-2, atol=5e-2)
                    for req in range(batch):
                        torch.testing.assert_close(
                            window[req, t],
                            decode_state[cache_idx[req]],
                            rtol=5e-2,
                            atol=5e-2,
                        )
                torch.testing.assert_close(
                    verify_state, decode_state, rtol=5e-2, atol=5e-2
                )

    def test_tree_draft_is_rejected(self):
        """The chain collapse is only valid without siblings; a real EAGLE tree
        must not silently get chain semantics."""
        from sgl_kernel.mamba import causal_conv1d_update_cpu

        dtype = torch.bfloat16
        batch, dim, draft_token_num, width = 1, 96, 8, 4

        sibling = torch.full((batch, draft_token_num), -1)
        sibling[0, 2] = 3

        with self.assertRaises(NotImplementedError):
            causal_conv1d_update_cpu(
                torch.randn(batch, dim, draft_token_num, dtype=dtype),
                torch.randn(4, dim, width - 1, dtype=dtype),
                torch.randn(dim, width, dtype=dtype),
                None,
                "silu",
                torch.zeros(batch, dtype=torch.int32),
                intermediate_conv_window=torch.zeros(
                    4, draft_token_num, dim, width - 1, dtype=dtype
                ),
                intermediate_state_indices=torch.zeros(batch, dtype=torch.int32),
                retrieve_next_token=torch.zeros(batch, draft_token_num),
                retrieve_next_sibling=sibling,
                retrieve_parent_token=None,
            )


if __name__ == "__main__":
    unittest.main()
