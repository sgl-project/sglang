"""Unit tests for causal_conv1d dispatch when conv cache dtype != activation dtype."""

import unittest
from typing import Optional
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.mamba import causal_conv1d as mamba_causal_conv1d
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# fp32-exact value whose low mantissa bit (2^-23) is lost if rounded through bf16/fp16.
FP32_EXACT_PROBE = 1.0 + 2.0**-20 + 2.0**-23


def _varlen_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_states: Optional[torch.Tensor],
    query_start_loc: torch.Tensor,
    cache_indices: Optional[torch.Tensor],
    has_initial_state: Optional[torch.Tensor],
    activation: Optional[str],
) -> torch.Tensor:
    dim, width = weight.shape
    state_len = width - 1
    starts = query_start_loc.tolist()
    outs = []
    for i in range(len(starts) - 1):
        seqlen = starts[i + 1] - starts[i]
        x_i = x[:, starts[i] : starts[i + 1]]
        slot = int(cache_indices[i]) if cache_indices is not None else i
        use_init = (
            bool(has_initial_state[i]) if has_initial_state is not None else False
        )
        if conv_states is not None and use_init:
            init = conv_states[slot].to(x.dtype)
            cat_conv = torch.cat([init, x_i], dim=-1)
            out_i = F.conv1d(
                cat_conv.unsqueeze(0).float(),
                weight.unsqueeze(1).float(),
                bias.float() if bias is not None else None,
                padding=0,
                groups=dim,
            ).squeeze(0)[:, :seqlen]
        else:
            out_i = F.conv1d(
                x_i.unsqueeze(0).float(),
                weight.unsqueeze(1).float(),
                bias.float() if bias is not None else None,
                padding=state_len,
                groups=dim,
            ).squeeze(0)[:, :seqlen]

        if conv_states is not None:
            if seqlen >= state_len:
                conv_states[slot].copy_(x_i[:, -state_len:].to(conv_states.dtype))
            elif use_init:
                shifted = torch.cat(
                    [
                        conv_states[slot, :, seqlen:],
                        x_i.to(conv_states.dtype),
                    ],
                    dim=-1,
                )
                conv_states[slot].copy_(shifted)
            else:
                padded = F.pad(x_i, (state_len - seqlen, 0)).to(conv_states.dtype)
                conv_states[slot].copy_(padded)

        if activation in ("silu", "swish"):
            out_i = F.silu(out_i)
        outs.append(out_i.to(x.dtype))
    return torch.cat(outs, dim=-1)


def _update_conv1d_ref(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    activation: Optional[str],
    conv_state_indices: Optional[torch.Tensor],
) -> torch.Tensor:
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    state_len = conv_state.shape[-1]
    outs = []
    for i in range(batch):
        slot = int(conv_state_indices[i]) if conv_state_indices is not None else i
        prior = conv_state[slot]
        cat_state = torch.cat([prior, x[i].to(conv_state.dtype)], dim=-1)
        out_i = F.conv1d(
            cat_state.unsqueeze(0).float(),
            weight.unsqueeze(1).float(),
            bias.float() if bias is not None else None,
            padding=0,
            groups=dim,
        ).squeeze(0)[:, -seqlen:]
        conv_state[slot].copy_(cat_state[:, -state_len:])
        if activation in ("silu", "swish"):
            out_i = F.silu(out_i)
        outs.append(out_i.to(x.dtype))
    out = torch.stack(outs, dim=0)
    return out.squeeze(-1) if unsqueeze else out


class TestCausalConv1dMixedDtypeDispatch(CustomTestCase):
    def test_fn_routes_mixed_cache_dtype_to_triton(self):
        torch.manual_seed(0)
        dim, width = 16, 4
        seqlens = [2, 5]
        query_start_loc = torch.tensor([0, 2, 7], dtype=torch.int32)
        cache_indices = torch.tensor([1, 0], dtype=torch.int32)
        has_initial_state = torch.tensor([True, False], dtype=torch.bool)

        calls = {"compiled": 0, "triton": 0, "seq_lens_cpu": None}

        def fake_compiled_fwd(
            x,
            weight,
            bias,
            conv_states,
            qsl,
            c_idx,
            has_init,
            silu_act,
            pad_slot_id,
        ):
            if conv_states is not None and conv_states.dtype != x.dtype:
                raise RuntimeError(
                    f"Dtype value [{conv_states.dtype}] not in the allowed options: [{x.dtype}]"
                )
            calls["compiled"] += 1
            out = _varlen_conv1d_ref(
                x,
                weight,
                bias,
                conv_states,
                qsl,
                c_idx,
                has_init,
                "silu" if silu_act else None,
            )
            x.copy_(out)

        def fake_triton_fn(
            x,
            weight,
            bias,
            *,
            conv_states,
            query_start_loc,
            cache_indices,
            has_initial_state,
            activation,
            pad_slot_id,
            **kwargs,
        ):
            calls["triton"] += 1
            calls["seq_lens_cpu"] = kwargs.get("seq_lens_cpu")
            return _varlen_conv1d_ref(
                x,
                weight,
                bias,
                conv_states,
                query_start_loc,
                cache_indices,
                has_initial_state,
                activation,
            )

        with (
            patch.object(mamba_causal_conv1d, "_HAS_CONV1D_KERNEL", True),
            patch.object(
                mamba_causal_conv1d, "causal_conv1d_fwd", fake_compiled_fwd, create=True
            ),
            patch.object(
                mamba_causal_conv1d, "_causal_conv1d_fn_triton", fake_triton_fn
            ),
        ):
            for x_dtype in (torch.bfloat16, torch.float16):
                x = torch.randn(dim, sum(seqlens), dtype=x_dtype).contiguous()
                self.assertEqual(x.stride(-1), 1)
                weight = torch.randn(dim, width, dtype=x_dtype)
                bias = torch.randn(dim, dtype=x_dtype)

                conv_states = torch.randn(3, dim, width - 1, dtype=torch.float32)
                conv_states[1, 0, 2] = FP32_EXACT_PROBE
                conv_states_ref = conv_states.clone()
                expected = _varlen_conv1d_ref(
                    x.clone(),
                    weight,
                    bias,
                    conv_states_ref,
                    query_start_loc,
                    cache_indices,
                    has_initial_state,
                    "silu",
                )

                calls["triton"] = 0
                out = mamba_causal_conv1d.causal_conv1d_fn(
                    x,
                    weight,
                    bias=bias,
                    query_start_loc=query_start_loc,
                    cache_indices=cache_indices,
                    has_initial_state=has_initial_state,
                    conv_states=conv_states,
                    activation="silu",
                )
                self.assertEqual(calls["compiled"], 0)
                self.assertEqual(calls["triton"], 1)
                self.assertEqual(calls["seq_lens_cpu"], seqlens)
                self.assertEqual(out.dtype, x_dtype)
                torch.testing.assert_close(out, expected)
                self.assertTrue(torch.equal(conv_states, conv_states_ref))
                # seqlen=2 shifts col 2 -> col 0 in slot 1 without rounding fp32.
                self.assertEqual(conv_states[1, 0, 0].item(), FP32_EXACT_PROBE)

            # Matching dtypes on contiguous input stay on the compiled path.
            x_same = torch.randn(dim, sum(seqlens), dtype=torch.bfloat16).contiguous()
            w_same = torch.randn(dim, width, dtype=torch.bfloat16)
            states_same = torch.randn(3, dim, width - 1, dtype=torch.bfloat16)
            mamba_causal_conv1d.causal_conv1d_fn(
                x_same,
                w_same,
                bias=None,
                query_start_loc=query_start_loc,
                cache_indices=cache_indices,
                has_initial_state=has_initial_state,
                conv_states=states_same,
                activation="silu",
            )
            self.assertEqual(calls["compiled"], 1)

    def test_update_routes_mixed_cache_dtype_to_triton(self):
        torch.manual_seed(1)
        batch, dim, width = 2, 16, 4
        indices = torch.tensor([2, 0], dtype=torch.int32)
        calls = {"compiled": 0, "triton": 0}

        def fake_compiled_update(
            x, conv_state, weight, bias, silu_act, cache_seqlens, state_indices, pad_id
        ):
            if conv_state is not None and conv_state.dtype != x.dtype:
                raise RuntimeError(
                    f"Dtype value [{conv_state.dtype}] not in the allowed options: [{x.dtype}]"
                )
            calls["compiled"] += 1
            out = _update_conv1d_ref(
                x,
                conv_state,
                weight,
                bias,
                "silu" if silu_act else None,
                state_indices,
            )
            x.copy_(out)

        def fake_triton_update(
            x,
            conv_state,
            weight,
            bias=None,
            activation=None,
            cache_seqlens=None,
            conv_state_indices=None,
            pad_slot_id=-1,
        ):
            calls["triton"] += 1
            return _update_conv1d_ref(
                x, conv_state, weight, bias, activation, conv_state_indices
            )

        with (
            patch.object(mamba_causal_conv1d, "_HAS_CONV1D_KERNEL", True),
            patch.object(
                mamba_causal_conv1d,
                "causal_conv1d_update_kernel",
                fake_compiled_update,
                create=True,
            ),
            patch.object(
                mamba_causal_conv1d, "_causal_conv1d_update_triton", fake_triton_update
            ),
        ):
            x = torch.randn(batch, dim, dtype=torch.bfloat16)
            weight = torch.randn(dim, width, dtype=torch.bfloat16)
            bias = torch.randn(dim, dtype=torch.bfloat16)
            conv_state = torch.randn(4, dim, width - 1, dtype=torch.float32)
            conv_state[2, 0, 1] = FP32_EXACT_PROBE
            conv_state_ref = conv_state.clone()
            expected = _update_conv1d_ref(
                x.clone(), conv_state_ref, weight, bias, "silu", indices
            )

            out = mamba_causal_conv1d.causal_conv1d_update(
                x,
                conv_state,
                weight,
                bias=bias,
                activation="silu",
                conv_state_indices=indices,
            )
            self.assertEqual(calls["compiled"], 0)
            self.assertEqual(calls["triton"], 1)
            self.assertEqual(out.dtype, torch.bfloat16)
            torch.testing.assert_close(out, expected)
            self.assertTrue(torch.equal(conv_state, conv_state_ref))
            self.assertEqual(conv_state[2, 0, 0].item(), FP32_EXACT_PROBE)

            # Matching dtypes stay on the compiled update kernel.
            conv_state_same = torch.randn(4, dim, width - 1, dtype=torch.bfloat16)
            mamba_causal_conv1d.causal_conv1d_update(
                x.clone(),
                conv_state_same,
                weight,
                bias=bias,
                activation="silu",
                conv_state_indices=indices,
            )
            self.assertEqual(calls["compiled"], 1)


if __name__ == "__main__":
    unittest.main()
