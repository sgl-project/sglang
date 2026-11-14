"""GDN MIS mapped stores, split/norm rounding, and normalization ownership."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
from sglang.kernels.ops.attention.fla.index import prepare_chunk_indices
from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
from sglang.kernels.ops.attention.linear.fused_gdn_gating import fused_gdn_gating
from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkv_gating_gdn_prefill,
    fused_qkv_split_gdn_prefill,
)
from sglang.srt.layers.attention.linear import gdn_backend
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestGDNMISOutput(CustomTestCase):
    @torch.inference_mode()
    def test_mis_reset_preserves_other_slots_without_host_synchronization(self):
        """Reset request slots in place without synchronizing the host,
        including strided state pools and repeated padding slots."""
        for device in ("cpu", "cuda"):
            for index_dtype in (torch.int32, torch.int64):
                for selected in ([0, 3, 3, 5], []):
                    for strided in (False, True):
                        with self.subTest(
                            device=device,
                            index_dtype=index_dtype,
                            selected=selected,
                            strided=strided,
                        ):
                            conv_storage = torch.full(
                                (2, 6, 8, 6), 7, device=device, dtype=torch.bfloat16
                            )
                            ssm_storage = torch.full((2, 6, 2, 4, 8), 11, device=device)
                            conv = conv_storage[1]
                            ssm = ssm_storage[1]
                            if strided:
                                conv = conv[:, :, ::2]
                                ssm = ssm[:, :, :, ::2]
                            expected_conv = conv_storage.clone()
                            expected_ssm = ssm_storage.clone()
                            for slot in set(selected):
                                expected_conv[1, slot, :, :: 2 if strided else 1] = 0
                                expected_ssm[1, slot, :, :, :: 2 if strided else 1] = 0
                            slots = torch.tensor(
                                selected, device=device, dtype=index_dtype
                            )
                            backend = object.__new__(gdn_backend.GDNAttnBackend)
                            backend.mis_metadata = gdn_backend.build_gdn_mis_metadata(
                                SimpleNamespace(
                                    is_prefill_only=True,
                                    extend_prefix_lens_cpu=[0] * len(selected),
                                    extend_seq_lens_cpu=[2] * len(selected),
                                    multi_item_delimiter_indices=[
                                        torch.tensor([0, 1]) for _ in selected
                                    ],
                                    input_ids=torch.zeros(
                                        len(selected) * 2, device=device
                                    ),
                                ),
                                cache_indices=slots,
                            )
                            packed = torch.zeros(len(selected) * 2, 8, device=device)
                            layer = SimpleNamespace(num_v_heads=2, head_v_dim=4)
                            previous_debug_mode = torch.cuda.get_sync_debug_mode()
                            try:
                                torch.cuda.set_sync_debug_mode("error")
                                with patch.object(
                                    backend, "_forward_mis_segments", return_value=None
                                ):
                                    backend._forward_extend_mis(
                                        layer=layer,
                                        mixed_qkv=packed,
                                        a=packed,
                                        b=packed,
                                        conv_states=conv,
                                        ssm_states=ssm,
                                        cache_indices=slots,
                                    )
                            finally:
                                torch.cuda.set_sync_debug_mode(previous_debug_mode)
                            torch.testing.assert_close(
                                conv_storage, expected_conv, rtol=0, atol=0
                            )
                            torch.testing.assert_close(
                                ssm_storage, expected_ssm, rtol=0, atol=0
                            )

    @torch.inference_mode()
    def test_mapped_chunk_output_preserves_unmapped_positions_and_state(self):
        for dtype in (torch.bfloat16, torch.float16):
            for dimension, index_dtype in ((64, torch.int32), (128, torch.int64)):
                for update in (False, True):
                    with self.subTest(dtype=dtype, dimension=dimension, update=update):
                        torch.manual_seed(45)
                        total = 90
                        q = torch.randn(
                            1, total, 2, dimension * 2, device="cuda", dtype=dtype
                        )[..., ::2]
                        k = torch.randn_like(q)
                        v = torch.randn(
                            1, total, 4, dimension, device="cuda", dtype=dtype
                        )
                        state = (
                            torch.randn(7, 4, dimension, dimension, device="cuda") * 0.1
                        )
                        expected_state, before = state.clone(), state.clone()
                        args = dict(
                            q=q,
                            k=k,
                            v=v,
                            g=-torch.rand(1, total, 4, device="cuda"),
                            beta=torch.rand(1, total, 4, device="cuda"),
                            initial_state_indices=torch.tensor(
                                [2, 3, 5, 6] if update else [2, 2, 5, 2],
                                device="cuda",
                                dtype=torch.int32,
                            ),
                            cu_seqlens=torch.tensor(
                                [0, 1, 8, 25, 90], device="cuda", dtype=torch.int32
                            ),
                            inplace_update=update,
                            use_qk_l2norm_in_kernel=True,
                        )
                        expected, _, _ = chunk_gated_delta_rule(
                            **args, initial_state=expected_state
                        )
                        destination = torch.full(
                            (1, total + 31, 4, dimension),
                            123,
                            device="cuda",
                            dtype=dtype,
                        )
                        mapping = (
                            torch.randperm(total + 31, device="cuda")[:total]
                            .to(index_dtype)
                            .contiguous()
                        )
                        actual, _, _ = chunk_gated_delta_rule(
                            **args,
                            initial_state=state,
                            out=destination,
                            output_indices=mapping,
                        )
                        self.assertEqual(actual.data_ptr(), destination.data_ptr())
                        torch.testing.assert_close(
                            actual[:, mapping.long()], expected, rtol=0, atol=0
                        )
                        untouched = torch.ones(
                            total + 31, device="cuda", dtype=torch.bool
                        )
                        untouched[mapping.long()] = False
                        self.assertTrue(bool(torch.all(actual[:, untouched] == 123)))
                        torch.testing.assert_close(
                            state, expected_state, rtol=0, atol=0
                        )
                        if not update:
                            torch.testing.assert_close(state, before, rtol=0, atol=0)

    @torch.inference_mode()
    def test_graph_tables_and_mapped_output_forward_through_dispatcher(self):
        torch.manual_seed(11)
        q = torch.randn(1, 128, 2, 64, device="cuda", dtype=torch.bfloat16)
        state = torch.randn(5, 2, 64, 64, device="cuda")
        cu_seqlens = torch.tensor([0, 65, 128], device="cuda", dtype=torch.int32)
        args = dict(
            q=q,
            k=torch.randn_like(q),
            v=torch.randn_like(q),
            g=-torch.rand(1, 128, 2, device="cuda"),
            beta=torch.rand(1, 128, 2, device="cuda"),
            cache_indices=torch.tensor([1, 3], device="cuda", dtype=torch.int32),
            query_start_loc=cu_seqlens,
        )
        expected_state, actual_state = state.clone(), state.clone()
        expected, _, expected_h = chunk_gated_delta_rule(
            q=args["q"],
            k=args["k"],
            v=args["v"],
            g=args["g"],
            beta=args["beta"],
            initial_state=expected_state,
            initial_state_indices=args["cache_indices"],
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )
        chunk_indices = prepare_chunk_indices(cu_seqlens, 64)
        offsets = torch.tensor([0, 2, 3], device="cuda", dtype=torch.int32)
        mapped = torch.full_like(q, 123)
        indices = torch.arange(127, -1, -1, device="cuda")
        actual, _, actual_h = TritonGDNKernel().extend(
            **args,
            ssm_states=actual_state,
            chunk_indices=chunk_indices,
            chunk_offsets=offsets,
            out=mapped,
            output_indices=indices,
        )
        torch.testing.assert_close(actual[:, indices], expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_h, expected_h, rtol=0, atol=0)
        torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)

    @torch.inference_mode()
    def test_fused_split_qk_norm_matches_existing_rounding(self):
        self._assert_fused_split_norm(num_tokens=19)

    @torch.inference_mode()
    def test_fused_split_qk_norm_within_one_ulp_across_chunk_boundary(self):
        self._assert_fused_split_norm(num_tokens=198, allow_norm_ulp=True)

    @torch.inference_mode()
    def test_mapped_prepare_preserves_gate_rounding_and_strides(self):
        for dtype, index_dtype in (
            (torch.bfloat16, torch.int64),
            (torch.float16, torch.int32),
            (torch.float32, torch.int64),
        ):
            with self.subTest(dtype=dtype, index_dtype=index_dtype):
                torch.manual_seed(55)
                tokens, source_tokens, heads, dim = 19, 29, 5, 64
                packed = torch.randn(
                    tokens, (2 * 3 + heads) * dim, device="cuda", dtype=dtype
                )
                a = torch.randn(source_tokens, heads * 3, device="cuda", dtype=dtype)[
                    :, ::3
                ]
                b = torch.randn(source_tokens, heads * 4, device="cuda", dtype=dtype)[
                    :, 1::4
                ]
                a[0] = 40
                a[1] = -40
                A_log = torch.randn(heads * 2, device="cuda")[::2]
                dt_bias = torch.randn(heads * 2, device="cuda", dtype=dtype)[::2]
                indices = (
                    torch.arange(tokens - 1, -1, -1, device="cuda")
                    .to(index_dtype)
                    .contiguous()
                )
                expected = fused_gdn_gating(
                    A_log.contiguous(),
                    a[indices],
                    b[indices],
                    dt_bias.contiguous(),
                )
                actual = fused_qkv_gating_gdn_prefill(
                    packed,
                    3,
                    3,
                    heads,
                    dim,
                    dim,
                    dim,
                    A_log=A_log,
                    a=a,
                    b=b,
                    dt_bias=dt_bias,
                    token_indices=indices,
                )
                for result, reference in zip(actual[3:], expected):
                    torch.testing.assert_close(result, reference, rtol=0, atol=0)
                    self.assertEqual(result.dtype, torch.float32)

    @torch.inference_mode()
    def test_mapped_prepare_preserves_large_gate_offsets(self):
        """Mapped reads preserve valid gate offsets beyond signed int32."""
        heads, row_stride = 2, 1 << 30
        storage = torch.empty(
            2 * row_stride + heads, device="cuda", dtype=torch.bfloat16
        )
        strided = torch.as_strided(storage, (3, heads), (row_stride, 1))
        strided[2].fill_(0.5)
        contiguous = torch.full((3, heads), -0.5, device="cuda", dtype=storage.dtype)
        packed = torch.zeros(1, 384, device="cuda", dtype=storage.dtype)
        A_log = torch.zeros(heads, device="cuda")
        dt_bias = torch.zeros(heads, device="cuda", dtype=storage.dtype)
        for index_dtype in (torch.int64, torch.int32):
            for a, b in ((strided, contiguous), (contiguous, strided)):
                with self.subTest(index_dtype=index_dtype, a_stride=a.stride(0)):
                    indices = torch.tensor([2], device="cuda", dtype=index_dtype)
                    expected = fused_gdn_gating(A_log, a[indices], b[indices], dt_bias)
                    actual = fused_qkv_gating_gdn_prefill(
                        packed,
                        2,
                        2,
                        2,
                        64,
                        64,
                        64,
                        A_log=A_log,
                        a=a,
                        b=b,
                        dt_bias=dt_bias,
                        token_indices=indices,
                    )
                    for result, reference in zip(actual[3:], expected):
                        torch.testing.assert_close(result, reference, rtol=0, atol=0)

    @torch.inference_mode()
    def test_mapped_prepare_preserves_large_qkv_offsets(self):
        """Compact token reads preserve valid QKV offsets beyond signed int32."""
        row_stride = 1 << 30
        storage = torch.empty(2 * row_stride + 384, device="cuda", dtype=torch.bfloat16)
        packed = torch.as_strided(storage, (3, 384), (row_stride, 1))
        for row in range(3):
            packed[row].fill_(0.5)
        gates = torch.full((3, 2), 0.5, device="cuda", dtype=storage.dtype)
        parameters = dict(
            A_log=torch.zeros(2, device="cuda"),
            a=gates,
            b=gates,
            dt_bias=torch.zeros(2, device="cuda", dtype=storage.dtype),
            token_indices=torch.arange(3, device="cuda"),
        )
        actual = fused_qkv_gating_gdn_prefill(packed, 2, 2, 2, 64, 64, 64, **parameters)
        expected = fused_qkv_gating_gdn_prefill(
            packed.contiguous(), 2, 2, 2, 64, 64, 64, **parameters
        )
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)

    @torch.inference_mode()
    def test_mapped_prepare_preserves_large_head_offsets(self):
        """Strided gate heads and parameters preserve large element offsets."""
        heads, stride = 3, 1 << 30
        storage = torch.empty(2 * stride + 1, device="cuda")
        strided = torch.as_strided(storage, (heads,), (stride,))
        strided.fill_(0.5)
        packed = torch.zeros(1, heads * 3 * 64, device="cuda", dtype=torch.bfloat16)
        indices = torch.zeros(1, device="cuda", dtype=torch.int64)
        for field in ("a", "b", "A_log", "dt_bias"):
            with self.subTest(field=field):
                values = dict(
                    A_log=torch.zeros(heads, device="cuda"),
                    dt_bias=torch.zeros(heads, device="cuda"),
                    a=torch.zeros(1, heads, device="cuda"),
                    b=torch.zeros(1, heads, device="cuda"),
                )
                values[field] = strided.unsqueeze(0) if field in ("a", "b") else strided
                expected = fused_gdn_gating(
                    values["A_log"].contiguous(),
                    values["a"][indices],
                    values["b"][indices],
                    values["dt_bias"].contiguous(),
                )
                actual = fused_qkv_gating_gdn_prefill(
                    packed,
                    heads,
                    heads,
                    heads,
                    64,
                    64,
                    64,
                    **values,
                    token_indices=indices,
                )
                for result, reference in zip(actual[3:], expected):
                    torch.testing.assert_close(result, reference, rtol=0, atol=0)

    @torch.inference_mode()
    def test_mapped_prepare_preserves_large_qkv_column_offsets(self):
        """Strided QKV columns preserve offsets beyond signed int32."""
        column_stride = 1 << 23
        storage = torch.empty(
            383 * column_stride + 1, device="cuda", dtype=torch.bfloat16
        )
        packed = torch.as_strided(storage, (1, 384), (1, column_stride))
        packed.fill_(0.5)
        gates = torch.zeros(1, 2, device="cuda", dtype=storage.dtype)
        parameters = dict(
            A_log=torch.zeros(2, device="cuda"),
            a=gates,
            b=gates,
            dt_bias=torch.zeros(2, device="cuda", dtype=storage.dtype),
            token_indices=torch.zeros(1, device="cuda", dtype=torch.int64),
        )
        for prepare in (
            lambda tensor: fused_qkv_gating_gdn_prefill(
                tensor, 2, 2, 2, 64, 64, 64, **parameters
            ),
            lambda tensor: fused_qkv_split_gdn_prefill(tensor, 2, 2, 2, 64, 64, 64),
        ):
            actual, expected = prepare(packed), prepare(packed.contiguous())
            for result, reference in zip(actual, expected):
                torch.testing.assert_close(result, reference, rtol=0, atol=0)

    @torch.inference_mode()
    def test_prepare_rejects_malformed_gate_mapping(self):
        packed = torch.empty(3, 384, device="cuda", dtype=torch.bfloat16)
        a = torch.empty(5, 2, device="cuda", dtype=packed.dtype)
        args = dict(
            A_log=torch.zeros(2, device="cuda"),
            a=a,
            b=a,
            dt_bias=torch.zeros(2, device="cuda"),
            token_indices=torch.arange(3, device="cuda"),
        )
        for override in (
            {"token_indices": torch.arange(2, device="cuda")},
            {"token_indices": torch.arange(3, device="cuda").float()},
            {"token_indices": torch.arange(6, device="cuda")[::2]},
            {"token_indices": torch.arange(3)},
        ):
            with self.subTest(fields=tuple(override)):
                with self.assertRaisesRegex(ValueError, "GDN preparation requires"):
                    fused_qkv_gating_gdn_prefill(
                        packed, 2, 2, 2, 64, 64, 64, **(args | override)
                    )

    def _assert_within_ulp(self, actual, expected, max_ulp=1):
        self.assertTrue(bool(torch.isfinite(actual).all()))
        self.assertTrue(bool(torch.isfinite(expected).all()))
        lower = expected
        upper = expected
        negative_direction = torch.full_like(expected, -float("inf"))
        positive_direction = torch.full_like(expected, float("inf"))
        for _ in range(max_ulp):
            lower = torch.nextafter(lower, negative_direction)
            upper = torch.nextafter(upper, positive_direction)
        outside = (actual < lower) | (actual > upper)
        self.assertFalse(
            bool(outside.any()),
            f"{int(outside.sum())} values differ by more than {max_ulp} output-dtype ULPs",
        )

    def _assert_fused_split_norm(self, *, num_tokens, allow_norm_ulp=False):
        for dtype in (torch.bfloat16, torch.float16):
            for strided in (False, True):
                for heads, dims in (
                    ((2, 2, 4), (128, 128, 128)),
                    ((3, 3, 5), (96, 96, 64)),
                    ((16, 16, 16), (128, 128, 128)),
                    ((1, 1, 2), (768, 640, 64)),
                ):
                    with self.subTest(
                        dtype=dtype, strided=strided, heads=heads, dims=dims
                    ):
                        torch.manual_seed(53)
                        channels = sum(head * dim for head, dim in zip(heads, dims))
                        storage = torch.randn(
                            num_tokens, channels * 2, device="cuda", dtype=dtype
                        )
                        packed = (
                            storage[:, ::2]
                            if strided
                            else storage[:, :channels].contiguous()
                        )
                        packed[0] = 0
                        packed[1] *= 1e-4
                        before = packed.clone()
                        q, k, v = fused_qkv_split_gdn_prefill(packed, *heads, *dims)
                        raw = tuple(
                            part.contiguous().view(1, num_tokens, head, dim)
                            for part, head, dim in zip(
                                packed.split(
                                    [head * dim for head, dim in zip(heads, dims)], -1
                                ),
                                heads,
                                dims,
                            )
                        )
                        for materialized, reference in zip((q, k, v), raw):
                            torch.testing.assert_close(
                                materialized, reference, rtol=0, atol=0
                            )
                        expected = l2norm_fwd(raw[0]), l2norm_fwd(raw[1]), raw[2]
                        actual = fused_qkv_split_gdn_prefill(
                            packed, *heads, *dims, normalize_qk=True
                        )
                        for index, (left, right) in enumerate(zip(actual, expected)):
                            if allow_norm_ulp and index < 2:
                                self._assert_within_ulp(left, right)
                                reference = raw[index].double()
                                reference = (
                                    reference
                                    / (
                                        reference.square().sum(-1, keepdim=True) + 1e-6
                                    ).sqrt()
                                )
                                self._assert_within_ulp(left, reference.to(dtype))
                                torch.testing.assert_close(
                                    left[:, 0], right[:, 0], rtol=0, atol=0
                                )
                            else:
                                torch.testing.assert_close(left, right, rtol=0, atol=0)
                            self.assertTrue(left.is_contiguous())
                            self.assertFalse(bool(torch.signbit(left[:, 0]).any()))
                        torch.testing.assert_close(packed, before, rtol=0, atol=0)

    @torch.inference_mode()
    def test_mis_norm_composition_preserves_output_and_state(self):
        self._assert_mis_composition(mapped_gating=False)

    @torch.inference_mode()
    def test_mis_mapped_gating_preserves_output_and_state(self):
        self._assert_mis_composition(mapped_gating=True)

    def _assert_mis_composition(self, *, mapped_gating):
        split = gdn_backend.fused_qkv_split_gdn_prefill

        def separate_norm(*args, normalize_qk=False):
            q, k, v = split(*args)
            return (l2norm_fwd(q), l2norm_fwd(k), v) if normalize_qk else (q, k, v)

        def separate_prepare(*args, A_log, a, b, dt_bias, token_indices):
            qkv = (
                split(*args, normalize_qk=True)
                if mapped_gating
                else separate_norm(*args, normalize_qk=True)
            )
            return (
                *qkv,
                *fused_gdn_gating(A_log, a[token_indices], b[token_indices], dt_bias),
            )

        for dtype in (torch.bfloat16, torch.float16):
            for query_len in (0, 5):
                with self.subTest(dtype=dtype, query_len=query_len):
                    torch.manual_seed(46)
                    tokens, heads, key_heads, dim = 198, 4, 2, 128
                    channels = (key_heads * 2 + heads) * dim
                    slots = torch.tensor([3], device="cuda")
                    backend = object.__new__(gdn_backend.GDNAttnBackend)
                    backend.mis_metadata = gdn_backend.build_gdn_mis_metadata(
                        SimpleNamespace(
                            is_prefill_only=True,
                            extend_prefix_lens_cpu=[0],
                            extend_seq_lens_cpu=[tokens],
                            multi_item_delimiter_indices=[
                                torch.tensor([query_len, 68, 132, 197])
                            ],
                            input_ids=torch.zeros(
                                tokens + 3, device="cuda", dtype=torch.int64
                            ),
                        ),
                        cache_indices=slots,
                    )
                    backend.kernel_dispatcher = gdn_backend.GDNKernelDispatcher(
                        LinearAttnKernelBackend.TRITON, LinearAttnKernelBackend.TRITON
                    )
                    extend = backend.kernel_dispatcher.extend

                    def scatter_reference(**kwargs):
                        out = kwargs.pop("out")
                        indices = kwargs.pop("output_indices")
                        compact, final_state, h = extend(**kwargs)
                        out[:, indices] = compact
                        return out, final_state, h

                    layer = SimpleNamespace(
                        num_q_heads=key_heads,
                        num_k_heads=key_heads,
                        num_v_heads=heads,
                        head_q_dim=dim,
                        head_k_dim=dim,
                        head_v_dim=dim,
                        q_dim=key_heads * dim,
                        k_dim=key_heads * dim,
                        v_dim=heads * dim,
                        conv_weights=torch.randn(
                            channels, 4, device="cuda", dtype=dtype
                        )
                        * 0.1,
                        bias=None,
                        activation="silu",
                        A_log=torch.randn(heads, device="cuda"),
                        dt_bias=torch.zeros(heads, device="cuda", dtype=dtype),
                    )
                    inputs = torch.randn(
                        tokens + 3, channels, device="cuda", dtype=dtype
                    )
                    if mapped_gating:
                        gates = torch.randn(
                            tokens + 3, heads * 2, device="cuda", dtype=dtype
                        )
                        a, b = gates.split(heads, dim=-1)
                    else:
                        a = torch.randn(tokens + 3, heads, device="cuda", dtype=dtype)
                        b = torch.randn_like(a)
                    conv = torch.randn(6, channels, 3, device="cuda", dtype=dtype)
                    ssm = torch.randn(6, heads, dim, dim, device="cuda")
                    expected_conv, expected_ssm = conv.clone(), ssm.clone()
                    args = dict(
                        layer=layer,
                        mixed_qkv=inputs,
                        a=a,
                        b=b,
                        cache_indices=slots,
                    )
                    with (
                        patch.object(
                            gdn_backend,
                            "fused_qkv_gating_gdn_prefill",
                            separate_prepare,
                        ),
                        patch.object(
                            backend.kernel_dispatcher,
                            "extend",
                            side_effect=scatter_reference,
                        ),
                    ):
                        expected = backend._forward_extend_mis(
                            **args, conv_states=expected_conv, ssm_states=expected_ssm
                        )
                    actual = backend._forward_extend_mis(
                        **args, conv_states=conv, ssm_states=ssm
                    )
                    if mapped_gating or dtype == torch.bfloat16:
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    else:
                        # Fused FP32 reductions can differ by one ULP and the
                        # recurrent GDN computation can amplify that to two.
                        self._assert_within_ulp(actual, expected, max_ulp=2)
                    torch.testing.assert_close(
                        actual[:, tokens:],
                        torch.zeros_like(actual[:, tokens:]),
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(conv, expected_conv, rtol=0, atol=0)
                    torch.testing.assert_close(ssm, expected_ssm, rtol=0, atol=0)

    @torch.inference_mode()
    def test_norm_dispatch_preserves_shape_and_device_fallbacks(self):
        for heads, cuda_split in ((4, True), (32, True), (4, False)):
            with self.subTest(heads=heads, cuda_split=cuda_split):
                torch.manual_seed(54)
                tokens, dimension = 3, 128
                channels = heads * dimension * 3
                packed = torch.randn(
                    tokens, channels, device="cuda", dtype=torch.bfloat16
                )
                backend = object.__new__(gdn_backend.GDNAttnBackend)
                backend.kernel_dispatcher = SimpleNamespace(
                    extend=Mock(return_value=(None, None, None))
                )
                layer = SimpleNamespace(
                    q_dim=heads * dimension,
                    k_dim=heads * dimension,
                    v_dim=heads * dimension,
                    num_q_heads=heads,
                    num_k_heads=heads,
                    num_v_heads=heads,
                    head_q_dim=dimension,
                    head_k_dim=dimension,
                    head_v_dim=dimension,
                    conv_weights=None,
                    bias=None,
                    activation="silu",
                    A_log=torch.randn(heads, device="cuda"),
                    dt_bias=torch.randn(heads, device="cuda", dtype=packed.dtype),
                )
                index = torch.tensor([0], device="cuda")
                token_indices = torch.tensor([4, 0, 2], device="cuda")
                a = torch.randn(tokens + 2, heads, device="cuda", dtype=packed.dtype)
                b = torch.randn_like(a)
                with (
                    patch.object(
                        gdn_backend,
                        "causal_conv1d_fn",
                        side_effect=lambda x, *args, **kwargs: x,
                    ),
                    patch.object(gdn_backend, "is_cuda", return_value=cuda_split),
                    patch.object(gdn_backend, "is_hip", return_value=False),
                ):
                    backend._forward_mis_segments(
                        layer=layer,
                        mixed_qkv=packed,
                        a=a,
                        b=b,
                        conv_states=torch.empty(
                            1, channels, 3, device="cuda", dtype=packed.dtype
                        ),
                        conv_cache_indices=index,
                        has_initial_states=torch.zeros(
                            1, device="cuda", dtype=torch.bool
                        ),
                        query_start_loc=torch.tensor(
                            [0, tokens], device="cuda", dtype=torch.int32
                        ),
                        seq_lens_cpu=[tokens],
                        ssm_states=torch.empty(
                            1, heads, dimension, dimension, device="cuda"
                        ),
                        ssm_cache_indices=index,
                        inplace_update=True,
                        output_indices=token_indices,
                    )
                captured = backend.kernel_dispatcher.extend.call_args.kwargs
                supported = (
                    cuda_split and channels <= gdn_backend.MAX_FUSED_QKV_SPLIT_DIM
                )
                self.assertIs(captured["use_qk_l2norm_in_kernel"], not supported)
                expected_g, expected_beta = fused_gdn_gating(
                    layer.A_log, a[token_indices], b[token_indices], layer.dt_bias
                )
                torch.testing.assert_close(captured["g"], expected_g, rtol=0, atol=0)
                torch.testing.assert_close(
                    captured["beta"], expected_beta, rtol=0, atol=0
                )
                for field, unnormalized in zip(
                    ("q", "k"), packed.split(heads * dimension, -1)[:2]
                ):
                    expected = unnormalized.contiguous().view(
                        1, tokens, heads, dimension
                    )
                    if supported:
                        expected = l2norm_fwd(expected)
                    torch.testing.assert_close(
                        captured[field], expected, rtol=0, atol=0
                    )

    @torch.inference_mode()
    def test_invalid_mapped_output_rejected_before_launch(self):
        q = torch.empty(1, 3, 2, 64, device="cuda", dtype=torch.bfloat16)
        out = torch.empty_like(q)
        indices = torch.arange(3, device="cuda")
        base = dict(
            q=q,
            k=q,
            v=q,
            g=torch.zeros(1, 3, 2, device="cuda"),
            beta=torch.zeros(1, 3, 2, device="cuda"),
            cu_seqlens=torch.tensor([0, 3], device="cuda", dtype=torch.int32),
            out=out,
            output_indices=indices,
        )
        for override in (
            {"out": None},
            {"output_indices": None},
            {"cu_seqlens": None},
            {"out": out.float()},
            {"out": out.transpose(1, 2)},
            {"out": torch.empty(1, 3, 2, 128, device="cuda", dtype=q.dtype)[..., ::2]},
            {"out": out.cpu()},
            {"output_indices": indices.float()},
            {"output_indices": indices[:2]},
            {"output_indices": indices.cpu()},
            {"output_indices": torch.arange(6, device="cuda")[::2]},
            {"q": q.detach().requires_grad_()},
            {
                "initial_state": torch.empty(
                    1, 2, 64, 64, device="cuda", requires_grad=True
                )
            },
        ):
            with self.subTest(fields=tuple(override)):
                with self.assertRaisesRegex(
                    ValueError, "mapped output|supplied together"
                ):
                    chunk_gated_delta_rule(**(base | override))


if __name__ == "__main__":
    unittest.main()
