"""Unit tests for srt/distributed/device_communicators/quick_all_reduce.py"""

import logging
import os
import sys
import types
import unittest
from unittest.mock import patch

import torch

import sglang.srt.distributed.device_communicators.custom_all_reduce_ops as ops
from sglang.srt.distributed.device_communicators.quick_all_reduce import (
    MB,
    QuickAllReduce,
    QuickReduceRegime,
)
from sglang.srt.environ import envs
from sglang.srt.utils import print_warning_once, temp_set_env
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_QR_MODULE = "sglang.srt.distributed.device_communicators.quick_all_reduce"
_AITER_QR_MODULE = "aiter.dist.device_communicators.quick_all_reduce"
_AITER_ENV = (
    "AITER_QUICK_REDUCE_QUANTIZATION",
    "AITER_QUICK_REDUCE_CAST_BF16_TO_FP16",
    "AITER_QUICK_REDUCE_MAX_SIZE_BYTES_MB",
    "AITER_FLY_AR",
)


class _FakeAiterQuickAllReduce:
    """Stands in for aiter's QuickAllReduce dispatcher."""

    disabled = False

    def __init__(self, group=None, device=None):
        self.env_at_init = {key: os.environ.get(key) for key in _AITER_ENV}
        self.qr_max_size = 1024 * MB
        self.accepts = True
        self.calls = 0
        self.closed = False

    def should_quick_allreduce(self, inp):
        return self.accepts

    def quick_all_reduce(self, inp, *, out=None):
        self.calls += 1
        if out is None:
            out = torch.empty_like(inp)
        out.copy_(inp)
        return out

    def close(self):
        self.closed = True


class _DisabledAiterQuickAllReduce(_FakeAiterQuickAllReduce):
    disabled = True


def _aiter_module(cls=None) -> types.ModuleType:
    module = types.ModuleType(_AITER_QR_MODULE)
    if cls is not None:
        module.QuickAllReduce = cls
    return module


def _make_dispatcher_quick_reduce(
    dispatcher: _FakeAiterQuickAllReduce,
) -> QuickAllReduce:
    quick_reduce = QuickAllReduce.__new__(QuickAllReduce)
    quick_reduce.disabled = False
    quick_reduce._flydsl_int4 = dispatcher
    quick_reduce.world_size = 4
    quick_reduce.qr_quant_level = QuickReduceRegime.INT4
    quick_reduce.use_fp16_kernels = 1
    quick_reduce.qr_max_size = 2048 * MB
    return quick_reduce


def _make_uninitialized_quick_reduce() -> QuickAllReduce:
    quick_reduce = QuickAllReduce.__new__(QuickAllReduce)
    quick_reduce.disabled = True
    quick_reduce._flydsl_int4 = None
    quick_reduce.group = None
    quick_reduce.device = torch.device("cpu")
    quick_reduce.rank = 0
    quick_reduce.world_size = 4
    quick_reduce.qr_quant_level = QuickReduceRegime.INT4
    quick_reduce.use_fp16_kernels = 1
    return quick_reduce


class TestQuickAllReduceAiterDispatcher(CustomTestCase):
    def test_sglang_size_window_applies_before_the_dispatcher(self):
        """SGLang's INT4 size window still gates; aiter decides inside it."""
        numel = MB  # 2 MiB of 16-bit data, the INT4 minimum size at TP4
        dispatcher = _FakeAiterQuickAllReduce()
        quick_reduce = _make_dispatcher_quick_reduce(dispatcher)

        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                self.assertTrue(
                    quick_reduce.should_quick_allreduce(torch.empty(numel, dtype=dtype))
                )
        self.assertFalse(
            quick_reduce.should_quick_allreduce(
                torch.empty(numel // 2, dtype=torch.bfloat16)
            )
        )
        dispatcher.accepts = False
        self.assertFalse(
            quick_reduce.should_quick_allreduce(
                torch.empty(numel, dtype=torch.bfloat16)
            )
        )

    def test_quick_all_reduce_goes_through_the_dispatcher(self):
        dispatcher = _FakeAiterQuickAllReduce()
        quick_reduce = _make_dispatcher_quick_reduce(dispatcher)

        inp = torch.ones(MB, dtype=torch.bfloat16)
        out = quick_reduce.quick_all_reduce(inp)

        self.assertEqual(dispatcher.calls, 1)
        self.assertTrue(torch.equal(out, inp))

    def test_init_sets_aiter_env_only_while_building_the_dispatcher(self):
        """AITER_FLY_AR also switches aiter's custom all-reduce, so it must not leak."""
        quick_reduce = _make_uninitialized_quick_reduce()
        with (
            temp_set_env(**{key: None for key in _AITER_ENV}),
            patch.dict(
                sys.modules, {_AITER_QR_MODULE: _aiter_module(_FakeAiterQuickAllReduce)}
            ),
            self.assertLogs(level="INFO") as logs,
        ):
            self.assertTrue(quick_reduce._init_flydsl_int4(512 * MB))
            leaked = {key: os.environ.get(key) for key in _AITER_ENV}

        self.assertEqual(
            quick_reduce._flydsl_int4.env_at_init,
            {
                "AITER_QUICK_REDUCE_QUANTIZATION": "INT4",
                "AITER_QUICK_REDUCE_CAST_BF16_TO_FP16": "1",
                "AITER_QUICK_REDUCE_MAX_SIZE_BYTES_MB": "512",
                "AITER_FLY_AR": "1",
            },
        )
        self.assertEqual(leaked, {key: None for key in _AITER_ENV})
        self.assertEqual(quick_reduce.qr_max_size, 1024 * MB)
        self.assertFalse(quick_reduce.disabled)
        self.assertIn("QuickAllReduce dispatcher", "\n".join(logs.output))

    def test_missing_dispatcher_falls_back_to_hip_visibly(self):
        """Without aiter's dispatcher, QuickReduce stays on the HIP kernel, warns once and says so."""
        print_warning_once.cache_clear()
        with (
            patch.dict(sys.modules, {_AITER_QR_MODULE: _aiter_module()}),
            envs.SGLANG_USE_AITER_FLYDSL_QUICK_REDUCE.override(True),
            temp_set_env(ROCM_QUICK_REDUCE_QUANTIZATION="INT4"),
            patch(f"{_QR_MODULE}.is_gfx95_supported", return_value=True),
            patch.object(ops, "init_custom_qr", create=True, return_value=1),
            patch.object(ops, "qr_max_size", create=True, return_value=2048 * MB),
            patch.object(ops, "qr_destroy", create=True),
            patch.object(QuickAllReduce, "create_shared_buffer"),
            self.assertLogs(level="INFO") as logs,
        ):
            for _ in range(2):
                quick_reduce = _make_uninitialized_quick_reduce()
                quick_reduce.init_quick_all_reduce()
                self.assertIsNone(quick_reduce._flydsl_int4)
                self.assertFalse(quick_reduce.disabled)
                quick_reduce.close()

        warnings = [r for r in logs.records if r.levelno == logging.WARNING]
        self.assertEqual(len(warnings), 1)
        self.assertIn("uses the HIP kernel", warnings[0].getMessage())
        activation = [
            r
            for r in logs.records
            if r.getMessage().startswith("QuickReduce INT4 uses the HIP kernel")
        ]
        self.assertEqual(len(activation), 2)

    def test_disabled_dispatcher_falls_back_to_hip(self):
        print_warning_once.cache_clear()
        quick_reduce = _make_uninitialized_quick_reduce()
        with (
            patch.dict(
                sys.modules,
                {_AITER_QR_MODULE: _aiter_module(_DisabledAiterQuickAllReduce)},
            ),
            self.assertLogs(level="WARNING") as logs,
        ):
            self.assertFalse(quick_reduce._init_flydsl_int4(0))
        self.assertIsNone(quick_reduce._flydsl_int4)
        self.assertIn("dispatcher is disabled", "\n".join(logs.output))

    def test_close_releases_the_dispatcher(self):
        dispatcher = _FakeAiterQuickAllReduce()
        quick_reduce = _make_dispatcher_quick_reduce(dispatcher)

        quick_reduce.close()

        self.assertTrue(dispatcher.closed)
        self.assertIsNone(quick_reduce._flydsl_int4)
        self.assertTrue(quick_reduce.disabled)


if __name__ == "__main__":
    unittest.main()
