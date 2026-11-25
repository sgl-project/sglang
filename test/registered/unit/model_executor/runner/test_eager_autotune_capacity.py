"""Synthetic BF16 tuning must not exceed dispatch capacity or shrink live buffers."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.batch_overlap.two_batch_overlap import MaybeTboDeepEPDispatcher
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher
from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPDispatcher
from sglang.srt.layers.moe.utils import DeepEPMode
from sglang.srt.model_executor.runner import eager_runner as eager
from sglang.srt.model_executor.runner import flashinfer_autotune as autotune
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def dispatcher(capacity, mode=DeepEPMode.AUTO):
    instance = DeepEPDispatcher.__new__(DeepEPDispatcher)
    instance.deepep_mode = mode
    instance._low_latency_dispatcher = SimpleNamespace(
        num_max_dispatch_tokens_per_rank=capacity
    )
    return instance


def model_with_dispatchers(*dispatchers, sharded=None):
    model = torch.nn.Module()
    for index, item in enumerate(dispatchers):
        layer = FusedMoE.__new__(FusedMoE)
        torch.nn.Module.__init__(layer)
        if sharded is not None:
            layer.dispatch_input_boundary = SimpleNamespace(
                input_on_attn_tp_slices=sharded
            )
        layer.dispatcher = item
        model.add_module(str(index), layer)
    return model


class TestEagerAutotuneCapacity(CustomTestCase):
    def test_original_overflow_is_bounded_without_changing_live_registry(self):
        """The synthetic whole-model forward must fit while live capacity stays large."""
        runner = eager.EagerRunner.__new__(eager.EagerRunner)
        runner._eager_max_bs = 4096
        registry = runner._eager_registry = object()
        runner.model_runner = SimpleNamespace(
            is_generation=True,
            is_draft_worker=False,
            spec_algorithm=SimpleNamespace(
                is_none=lambda: True, is_speculative=lambda: False
            ),
            model=model_with_dispatchers(
                dispatcher(256), dispatcher(128), sharded=True
            ),
            server_args=object(),
        )
        runner._alloc_dummy_decode_buffers = lambda bs, **kw: SimpleNamespace(
            input_rows=bs * kw["num_tokens_per_req"]
        )
        parallel = SimpleNamespace(
            dp_size=1,
            attn_dp_size=1,
            attn_cp_size=1,
            moe_dp_size=1,
            pp_size=1,
            tp_size=4,
            attn_tp_size=4,
            moe_ep_size=4,
        )
        with (
            patch.object(eager, "get_parallel", return_value=parallel),
            patch.object(
                eager,
                "get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(moe_a2a_backend="deepep")
                ),
            ),
            patch.object(eager.DllmConfig, "from_server_args", return_value=None),
            patch("sglang.srt.layers.moe.utils.is_tbo_enabled", return_value=False),
            patch.object(autotune, "should_run_flashinfer_autotune", return_value=True),
        ):
            buffers, batch = runner._autotune_buffers()
            # The original 4096/TP4 rows exceed the instantiated 128-row bound.
            self.assertGreater(runner._eager_max_bs // 4, 128)
            self.assertEqual(buffers.input_rows // 4, 128)
            self.assertEqual(batch, 512)
            self.assertGreaterEqual(batch // 4, 16)
            self.assertEqual(runner._eager_max_bs, 4096)
            self.assertIs(runner._eager_registry, registry)
            # Each case removes an eligibility condition; none may clamp.
            for field, value in (
                ("dp_size", 2),
                ("attn_dp_size", 2),
                ("attn_cp_size", 2),
                ("moe_dp_size", 2),
                ("pp_size", 2),
                ("moe_ep_size", 2),
            ):
                previous = vars(parallel)[field]
                vars(parallel)[field] = value
                try:
                    self.assertEqual(runner._autotune_buffers()[1], 4096)
                finally:
                    vars(parallel)[field] = previous
            runner.model_runner.is_draft_worker = True
            self.assertEqual(runner._autotune_buffers()[1], 4096)
            runner.model_runner.is_draft_worker = False
            with patch.object(
                autotune, "should_run_flashinfer_autotune", return_value=False
            ):
                self.assertEqual(runner._autotune_buffers()[1], 4096)
            with patch("sglang.srt.layers.moe.utils.is_tbo_enabled", return_value=True):
                self.assertEqual(runner._autotune_buffers()[1], 4096)
            runner.model_runner.is_generation = False
            self.assertEqual(runner._autotune_buffers()[1], 4096)
            runner.model_runner.is_generation = True
            with patch.object(
                eager.DllmConfig, "from_server_args", return_value=object()
            ):
                self.assertEqual(runner._autotune_buffers()[1], 4096)
            with patch.object(
                eager,
                "get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(moe_a2a_backend="none")
                ),
            ):
                self.assertEqual(runner._autotune_buffers()[1], 4096)
            runner.model_runner.spec_algorithm = SimpleNamespace(
                is_none=lambda: False, is_speculative=lambda: True
            )
            runner.model_runner.decode_num_tokens_per_req = lambda: 2
            buffers, batch = runner._autotune_buffers()
            self.assertEqual(batch, 4096)
            self.assertEqual(buffers.input_rows, 8192)
            self.assertEqual(runner._eager_max_bs, 4096)
            self.assertIs(runner._eager_registry, registry)

    def test_bf16_only_eligibility_excludes_mixed_tuning(self):
        mr = SimpleNamespace(
            device="cuda",
            model_config=SimpleNamespace(quantization=None),
            spec_algorithm=SimpleNamespace(is_speculative=lambda: False),
        )
        moe = SimpleNamespace(moe_runner_backend="deep_gemm", moe_a2a_backend="deepep")
        config = SimpleNamespace(
            moe=moe,
            kernel=SimpleNamespace(disable_flashinfer_autotune=False),
            deterministic=SimpleNamespace(enable_deterministic_inference=False),
        )
        with (
            patch.object(autotune, "get_exec", return_value=config),
            patch.object(
                autotune,
                "get_parallel",
                return_value=SimpleNamespace(enable_kv_cache_sharding=False),
            ),
            patch.object(
                autotune, "_serves_block_fp8_as_mxfp8_on_flashinfer", return_value=False
            ) as fp8,
            patch(
                "sglang.srt.layers.quantization.unquant.bf16_gemm_needs_autotune",
                return_value=True,
            ) as bf16,
            patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)),
        ):
            eligible = autotune.should_run_flashinfer_autotune
            self.assertTrue(eligible(mr, require_bf16_only=True))
            bf16.return_value = False
            self.assertFalse(eligible(mr, require_bf16_only=True))
            bf16.return_value = True
            fp8.return_value = True
            self.assertTrue(eligible(mr))
            self.assertFalse(eligible(mr, require_bf16_only=True))
            fp8.return_value = False
            moe.moe_runner_backend = "flashinfer_cutlass"
            self.assertTrue(eligible(mr))
            self.assertFalse(eligible(mr, require_bf16_only=True))

    def test_instantiated_capacity_and_wrapper_minimum(self):
        wrapper = MaybeTboDeepEPDispatcher.__new__(MaybeTboDeepEPDispatcher)
        wrapper._inners = [dispatcher(256), dispatcher(128)]
        self.assertEqual(wrapper.get_decode_token_capacity(), 128)
        wrapper._inners.reverse()
        self.assertEqual(wrapper.get_decode_token_capacity(), 128)
        wrapper._inners = [dispatcher(128, DeepEPMode.NORMAL)]
        self.assertIsNone(wrapper.get_decode_token_capacity())
        # Normal-only instances need not construct a low-latency implementation.
        del wrapper._inners[0]._low_latency_dispatcher
        self.assertIsNone(wrapper.get_decode_token_capacity())

    def test_tp16_decode_capacity(self):
        # At TP16 the 2,048-row batch fits exactly. An unaligned input can
        # take the replicated path, so round it down before relying on TP.
        for batch in (2048, 2047, 2049, 4096):
            with self.subTest(batch=batch):
                selected = eager._bf16_autotune_batch_size(
                    model=model_with_dispatchers(dispatcher(128), sharded=True),
                    batch_size=batch,
                    attn_tp_size=16,
                )
                self.assertEqual(selected, min(batch // 16, 128) * 16)

    def test_replicated_and_unknown_inputs_use_per_rank_bound(self):
        for sharded in (False, None):
            with self.subTest(sharded=sharded):
                model = model_with_dispatchers(dispatcher(128), sharded=sharded)
                model.add_module(
                    "sharded", model_with_dispatchers(dispatcher(32), sharded=True)
                )
                self.assertEqual(
                    eager._bf16_autotune_batch_size(
                        model=model, batch_size=2048, attn_tp_size=16
                    ),
                    128,
                )

    def test_small_capacity(self):
        select = eager._bf16_autotune_batch_size
        # A BF16 probe can use fewer than 16 rows when dispatch requires it.
        self.assertEqual(
            select(
                model=model_with_dispatchers(dispatcher(7), sharded=True),
                batch_size=2048,
                attn_tp_size=16,
            ),
            112,
        )
        model = model_with_dispatchers(dispatcher(7))
        model.add_module("sharded", model_with_dispatchers(dispatcher(3), sharded=True))
        with patch(
            "sglang.srt.layers.layer_boundary.layout.batches_are_unpadded",
            return_value=True,
        ):
            self.assertEqual(select(model=model, batch_size=2048, attn_tp_size=16), 3)
        with (
            patch(
                "sglang.srt.layers.layer_boundary.layout.batches_are_unpadded",
                return_value=False,
            ),
            self.assertRaisesRegex(ValueError, "TP-aligned"),
        ):
            select(model=model, batch_size=2048, attn_tp_size=16)
        for bound in (0, -1):
            with (
                self.subTest(bound=bound),
                self.assertRaisesRegex(ValueError, "positive"),
            ):
                select(
                    model=model_with_dispatchers(dispatcher(bound)),
                    batch_size=2048,
                    attn_tp_size=16,
                )

    def test_safe_or_unadvertised_capacity_keeps_original_shape(self):
        for batch, layers in (
            (31, [dispatcher(128)]),
            (128, [dispatcher(128)]),
            (4096, []),
            (4096, [dispatcher(128, DeepEPMode.NORMAL)]),
        ):
            with self.subTest(batch=batch, layers=len(layers)):
                self.assertEqual(
                    eager._bf16_autotune_batch_size(
                        model=model_with_dispatchers(*layers),
                        batch_size=batch,
                        attn_tp_size=4,
                    ),
                    batch,
                )
        self.assertIsNone(BaseDispatcher.get_decode_token_capacity(object()))


if __name__ == "__main__":
    unittest.main()
