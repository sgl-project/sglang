"""CPU unit tests for the graph-safe ``RadixAttention`` interface."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.layers.radix_attention as radix_attention_module
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _RecordingAttentionBackend:
    def __init__(self, *, return_lse=True):
        self.calls = []
        self.return_lse = return_lse

    def forward(
        self,
        query,
        key,
        value,
        attention_layer,
        forward_batch,
        save_kv_cache,
        **kwargs,
    ):
        self.calls.append(
            SimpleNamespace(
                query=query,
                key=key,
                value=value,
                attention_layer=attention_layer,
                output=forward_batch._attn_output,
                out_cache_loc=forward_batch.out_cache_loc.clone(),
                save_kv_cache=save_kv_cache,
                kwargs=kwargs,
            )
        )
        output = torch.full_like(query, 3)
        lse = torch.full((query.shape[0], query.shape[1]), 7, dtype=torch.float32)
        return (output, lse) if self.return_lse else output


class TestRadixAttentionGraphInterface(CustomTestCase):
    @staticmethod
    def _new_layer() -> RadixAttention:
        layer = RadixAttention(
            num_heads=2,
            head_dim=3,
            scaling=1.0,
            num_kv_heads=2,
            layer_id=0,
        )
        return layer

    @staticmethod
    def _new_impl_context(
        attention_layers,
        *,
        mha_companion_layers=None,
        num_tokens=4,
        real_num_tokens=2,
    ):
        forward_batch = SimpleNamespace(
            num_token_non_padded_cpu=real_num_tokens,
            out_cache_loc=torch.arange(num_tokens, dtype=torch.int64),
            positions=torch.arange(num_tokens, dtype=torch.int64),
            _attn_output=None,
            mha_return_lse=False,
        )
        return SimpleNamespace(
            forward_batch=forward_batch,
            attention_layers=attention_layers,
            mha_companion_layers=mha_companion_layers,
            num_tokens=None,
            raw_num_tokens=None,
        )

    def test_forward_dispatches_all_graph_and_lse_variants(self):
        layer = self._new_layer()
        query = torch.zeros((4, 2, 3))
        key = torch.zeros_like(query)
        value = torch.zeros_like(query)

        op_names = {
            (False, False): "unified_attention_with_output",
            (False, True): "unified_attention_with_output_and_lse",
            (True, False): "breakable_unified_attention_with_output",
            (True, True): "breakable_unified_attention_with_output_and_lse",
        }

        for breakable in (False, True):
            for return_lse in (False, True):
                with self.subTest(breakable=breakable, return_lse=return_lse):
                    forward_batch = SimpleNamespace(
                        forward_mode=ForwardMode.EXTEND,
                        mha_return_lse=return_lse,
                    )
                    calls = []

                    def output_only(*args, **kwargs):
                        args[3].fill_(5)
                        calls.append(kwargs)

                    def output_and_lse(*args, **kwargs):
                        args[3].fill_(5)
                        calls.append(kwargs)
                        return torch.full((4, 2), 11, dtype=torch.float32)

                    with ExitStack() as stack:
                        stack.enter_context(
                            patch.object(
                                radix_attention_module,
                                "get_tc_piecewise_forward_context",
                                return_value=SimpleNamespace(
                                    mha_companion_layers=[layer]
                                ),
                            )
                        )
                        stack.enter_context(
                            patch.object(
                                radix_attention_module,
                                "is_in_breakable_cuda_graph",
                                return_value=breakable,
                            )
                        )
                        mocks = {
                            name: stack.enter_context(
                                patch.object(
                                    radix_attention_module,
                                    name,
                                    side_effect=(
                                        output_and_lse
                                        if name.endswith("and_lse")
                                        else output_only
                                    ),
                                )
                            )
                            for name in op_names.values()
                        }
                        result = layer(
                            query,
                            key,
                            value,
                            forward_batch,
                            key_value_num_tokens=3,
                        )

                    selected_name = op_names[(breakable, return_lse)]
                    for name, mock in mocks.items():
                        self.assertEqual(mock.call_count, int(name == selected_name))

                    self.assertEqual(
                        calls,
                        [
                            {
                                "use_mha_companion": True,
                                "key_value_num_tokens": 3,
                            }
                        ],
                    )
                    if return_lse:
                        output, lse = result
                        self.assertEqual(lse.shape, (4, 2))
                        self.assertTrue(torch.all(lse == 11))
                    else:
                        output = result
                    self.assertEqual(output.shape, query.shape)
                    self.assertTrue(torch.all(output == 5))

    def test_impl_preserves_attention_identity_and_lse(self):
        mqa = SimpleNamespace()
        mha = SimpleNamespace()
        context = self._new_impl_context([mqa], mha_companion_layers=[mha])
        forward_batch = context.forward_batch
        original_out_cache_loc = forward_batch.out_cache_loc
        backend = _RecordingAttentionBackend()
        query = torch.zeros((4, 2, 3))

        with (
            patch.object(
                radix_attention_module,
                "get_tc_piecewise_forward_context",
                return_value=context,
            ),
            patch.object(
                radix_attention_module, "get_attn_backend", return_value=backend
            ),
        ):
            for use_mha_companion, expected_layer in ((False, mqa), (True, mha)):
                with self.subTest(use_mha_companion=use_mha_companion):
                    output = torch.empty_like(query)
                    lse = radix_attention_module._unified_attention_with_output_impl(
                        query,
                        query,
                        query,
                        output,
                        False,
                        0,
                        use_mha_companion,
                        True,
                    )

                    call_record = backend.calls[-1]
                    self.assertIs(call_record.attention_layer, expected_layer)
                    self.assertEqual(call_record.query.shape, (2, 2, 3))
                    self.assertEqual(call_record.key.shape, (2, 2, 3))
                    self.assertEqual(call_record.value.shape, (2, 2, 3))
                    self.assertEqual(call_record.output.shape, (2, 2, 3))
                    self.assertEqual(call_record.out_cache_loc.tolist(), [0, 1])
                    self.assertFalse(call_record.save_kv_cache)
                    self.assertTrue(torch.all(output[:2] == 3))
                    self.assertEqual(lse.shape, (4, 2))
                    self.assertTrue(torch.all(lse[:2] == 7))
                    self.assertTrue(torch.all(lse[2:] == 0))
                    self.assertIs(forward_batch.out_cache_loc, original_out_cache_loc)

    def test_extra_kwargs_path_returns_bucket_shaped_lse(self):
        attention_layer = SimpleNamespace()
        context = self._new_impl_context([attention_layer])
        backend = _RecordingAttentionBackend()
        query = torch.zeros((4, 2, 3))

        with (
            patch.object(
                radix_attention_module,
                "get_tc_piecewise_forward_context",
                return_value=context,
            ),
            patch.object(
                radix_attention_module, "get_attn_backend", return_value=backend
            ),
        ):
            lse = radix_attention_module.attention_with_output_extra_kwargs(
                query,
                query,
                query,
                torch.empty_like(query),
                False,
                0,
                {"return_lse": True},
            )

        self.assertEqual(lse.tolist(), [[7, 7], [7, 7], [0, 0], [0, 0]])

    def test_impl_uses_independent_query_and_key_value_extents(self):
        attention_layer = SimpleNamespace()
        context = self._new_impl_context([attention_layer])
        forward_batch = context.forward_batch
        original_out_cache_loc = forward_batch.out_cache_loc
        backend = _RecordingAttentionBackend()
        query = torch.zeros((4, 2, 3))
        key = torch.zeros((6, 2, 3))
        value = torch.zeros((6, 2, 3))
        k_rope = torch.zeros((6, 2, 1))
        output = torch.empty_like(query)

        with (
            patch.object(
                radix_attention_module,
                "get_tc_piecewise_forward_context",
                return_value=context,
            ),
            patch.object(
                radix_attention_module, "get_attn_backend", return_value=backend
            ),
        ):
            lse = radix_attention_module._unified_attention_with_output_impl(
                query,
                key,
                value,
                output,
                False,
                0,
                False,
                True,
                key_value_num_tokens=5,
                k_rope=k_rope,
            )

        call_record = backend.calls[-1]
        self.assertEqual(call_record.query.shape, (2, 2, 3))
        self.assertEqual(call_record.key.shape, (5, 2, 3))
        self.assertEqual(call_record.value.shape, (5, 2, 3))
        self.assertEqual(call_record.kwargs["k_rope"].shape, (5, 2, 1))
        self.assertEqual(call_record.output.shape, (2, 2, 3))
        self.assertEqual(lse.shape, (4, 2))
        self.assertIs(forward_batch.out_cache_loc, original_out_cache_loc)

    def test_impl_preserves_output_only_contract(self):
        attention_layer = SimpleNamespace()
        context = self._new_impl_context([attention_layer])
        forward_batch = context.forward_batch
        original_out_cache_loc = forward_batch.out_cache_loc
        backend = _RecordingAttentionBackend(return_lse=False)
        query = torch.zeros((4, 2, 3))
        output = torch.empty_like(query)

        with (
            patch.object(
                radix_attention_module,
                "get_tc_piecewise_forward_context",
                return_value=context,
            ),
            patch.object(
                radix_attention_module, "get_attn_backend", return_value=backend
            ),
        ):
            lse = radix_attention_module._unified_attention_with_output_impl(
                query,
                query,
                query,
                output,
                False,
                0,
                False,
                False,
            )

        self.assertIsNone(lse)
        self.assertIs(backend.calls[-1].attention_layer, attention_layer)
        self.assertTrue(torch.all(output[:2] == 3))
        self.assertIs(forward_batch.out_cache_loc, original_out_cache_loc)

    def test_impl_zero_real_tokens_returns_zeroed_lse(self):
        # Regression: an idle DP rank whose fabricated EXTEND batch is masked to
        # 0 real tokens skips attention entirely. The skip must still honor the
        # LSE return mode -- unified_attention_with_output_and_lse asserts a
        # tensor comes back, so returning a bare None raised AssertionError as
        # soon as any 0-real-token call needed LSE (chunked-prefix MHA merge).
        attention_layer = SimpleNamespace()
        context = self._new_impl_context([attention_layer], real_num_tokens=0)
        backend = _RecordingAttentionBackend()
        query = torch.zeros((4, 2, 3))
        output = torch.full_like(query, float("nan"))

        with (
            patch.object(
                radix_attention_module,
                "get_tc_piecewise_forward_context",
                return_value=context,
            ),
            patch.object(
                radix_attention_module, "get_attn_backend", return_value=backend
            ),
        ):
            lse = radix_attention_module._unified_attention_with_output_impl(
                query,
                query,
                query,
                output,
                False,
                0,
                False,
                True,
            )

        self.assertEqual(backend.calls, [])
        self.assertTrue(torch.all(output == 0))
        # Same shape/dtype the registered fake impl declares, so
        # unified_attention_with_output_and_lse's `assert lse is not None` holds.
        self.assertEqual(lse.shape, (4, 2))
        self.assertEqual(lse.dtype, torch.float32)
        self.assertTrue(torch.all(lse == 0))

    def test_impl_zero_real_tokens_output_only_returns_none(self):
        # The 0-token skip must not start returning a tensor on the non-LSE
        # path: unified_attention_with_output is registered with an inplace
        # (None-returning) schema.
        attention_layer = SimpleNamespace()
        context = self._new_impl_context([attention_layer], real_num_tokens=0)
        backend = _RecordingAttentionBackend(return_lse=False)
        query = torch.zeros((4, 2, 3))
        output = torch.full_like(query, float("nan"))

        with (
            patch.object(
                radix_attention_module,
                "get_tc_piecewise_forward_context",
                return_value=context,
            ),
            patch.object(
                radix_attention_module, "get_attn_backend", return_value=backend
            ),
        ):
            lse = radix_attention_module._unified_attention_with_output_impl(
                query,
                query,
                query,
                output,
                False,
                0,
                False,
                False,
            )

        self.assertIsNone(lse)
        self.assertEqual(backend.calls, [])
        self.assertTrue(torch.all(output == 0))

    def test_extra_kwargs_zero_real_tokens_zeroes_output(self):
        # Regression: attention_with_output_extra_kwargs (Inkling score_mod /
        # aux_tensors) narrowed to query[:0] and copied output[:0], so with 0
        # real tokens the preallocated torch.empty output was never written and
        # its garbage (NaN/Inf) flowed into residuals and MoE routing. Only ROCm
        # zeroed the padded tail, so every other platform leaked it.
        attention_layer = SimpleNamespace()
        context = self._new_impl_context([attention_layer], real_num_tokens=0)
        backend = _RecordingAttentionBackend(return_lse=False)
        query = torch.zeros((4, 2, 3))
        output = torch.full_like(query, float("nan"))

        with (
            patch.object(
                radix_attention_module,
                "get_tc_piecewise_forward_context",
                return_value=context,
            ),
            patch.object(
                radix_attention_module, "get_attn_backend", return_value=backend
            ),
        ):
            radix_attention_module.attention_with_output_extra_kwargs(
                query,
                query,
                query,
                output,
                False,
                0,
                {},
            )

        self.assertEqual(backend.calls, [])
        self.assertTrue(torch.all(output == 0))

    def test_lse_fake_impl_declares_shape_and_dtype(self):
        query = torch.empty((5, 3, 7), dtype=torch.float16)
        output = torch.empty_like(query)

        lse = radix_attention_module._unified_attention_with_output_and_lse_fake(
            query,
            None,
            None,
            output,
            False,
            0,
        )

        self.assertEqual(lse.shape, (5, 3))
        self.assertEqual(lse.dtype, torch.float32)
        self.assertEqual(lse.device, query.device)


if __name__ == "__main__":
    unittest.main()
