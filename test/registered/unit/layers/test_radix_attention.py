"""CPU unit tests for the graph-safe ``RadixAttention`` interface."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.layers.radix_attention as radix_attention_module
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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
            layer_id=7,
        )
        layer._pcg_is_mha_companion = True
        return layer

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

                    patches = {
                        name: patch.object(
                            radix_attention_module,
                            name,
                            side_effect=(
                                output_and_lse
                                if name.endswith("and_lse")
                                else output_only
                            ),
                        )
                        for name in op_names.values()
                    }
                    with (
                        patch.object(
                            radix_attention_module,
                            "get_tc_piecewise_forward_context",
                            return_value=object(),
                        ),
                        patch.object(
                            radix_attention_module,
                            "is_in_breakable_cuda_graph",
                            return_value=breakable,
                        ),
                        patches["unified_attention_with_output"] as regular,
                        patches["unified_attention_with_output_and_lse"] as regular_lse,
                        patches[
                            "breakable_unified_attention_with_output"
                        ] as breakable_op,
                        patches[
                            "breakable_unified_attention_with_output_and_lse"
                        ] as breakable_lse,
                    ):
                        result = layer(query, key, value, forward_batch)

                    mocks = {
                        "unified_attention_with_output": regular,
                        "unified_attention_with_output_and_lse": regular_lse,
                        "breakable_unified_attention_with_output": breakable_op,
                        "breakable_unified_attention_with_output_and_lse": breakable_lse,
                    }
                    selected_name = op_names[(breakable, return_lse)]
                    for name, mock in mocks.items():
                        self.assertEqual(mock.call_count, int(name == selected_name))

                    self.assertEqual(calls, [{"use_mha_companion": True}])
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
        mqa._pcg_mha_companion = mha
        original_out_cache_loc = torch.arange(4, dtype=torch.int64)
        forward_batch = SimpleNamespace(
            num_token_non_padded_cpu=2,
            out_cache_loc=original_out_cache_loc,
            _attn_output=None,
        )
        context = SimpleNamespace(
            forward_batch=forward_batch,
            attention_layers=[mqa],
            num_tokens=None,
            raw_num_tokens=None,
        )
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

    def test_impl_preserves_output_only_contract(self):
        attention_layer = SimpleNamespace()
        original_out_cache_loc = torch.arange(4, dtype=torch.int64)
        forward_batch = SimpleNamespace(
            num_token_non_padded_cpu=2,
            out_cache_loc=original_out_cache_loc,
            _attn_output=None,
        )
        context = SimpleNamespace(
            forward_batch=forward_batch,
            attention_layers=[attention_layer],
            num_tokens=None,
            raw_num_tokens=None,
        )
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
