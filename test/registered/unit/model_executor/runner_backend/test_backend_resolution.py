"""CPU-only tests for decode graph backend resolution."""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.runner_backend import utils as backend_utils
from sglang.srt.model_executor.runner_backend.base_breakable_graph_backend import (
    BaseBreakableGraphBackend,
)
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _make_graph_runner(*, device="custom"):
    return SimpleNamespace(
        model_runner=SimpleNamespace(
            device=device,
            lora_manager=SimpleNamespace(
                lora_backend=SimpleNamespace(lora_workspace=object(), is_moe_lora=False)
            ),
            attn_backend=SimpleNamespace(requires_per_shape_decode_graph_warmup=False),
        ),
        enable_profile_cuda_graph=False,
        attn_backend=SimpleNamespace(requires_per_shape_decode_graph_warmup=False),
    )


def _make_exec_config(
    *,
    backend=Backend.FULL,
    enable_memory_saver=False,
    debug_cuda_graph=False,
):
    return SimpleNamespace(
        graph=SimpleNamespace(
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(backend=backend),
                prefill=SimpleNamespace(backend=backend),
            ),
            debug_cuda_graph=debug_cuda_graph,
            enable_profile_cuda_graph=False,
        ),
        features=SimpleNamespace(
            enable_memory_saver=enable_memory_saver,
        ),
    )


def _make_platform(*, is_out_of_tree, full_backend_cls=None, cuda=True):
    return SimpleNamespace(
        is_cuda=mock.Mock(return_value=cuda),
        is_out_of_tree=mock.Mock(return_value=is_out_of_tree),
        get_full_graph_backend_cls=mock.Mock(return_value=full_backend_cls),
    )


class _OutOfTreePlatformWithoutFullGraphBackend(SRTPlatform):
    _enum = PlatformEnum.OOT


class TestGraphBackendTypes(CustomTestCase):
    def test_cuda_breakable_backend_implements_neutral_contract(self):
        self.assertTrue(
            issubclass(BreakableCudaGraphBackend, BaseBreakableGraphBackend)
        )


class TestResolveDecodeBackend(CustomTestCase):
    def test_out_of_tree_platform_provides_full_backend(self):
        runner = _make_graph_runner()
        exec_config = _make_exec_config(enable_memory_saver=True)
        expected_backend = object()
        backend_cls = mock.Mock(return_value=expected_backend)
        platform = _make_platform(
            is_out_of_tree=True,
            full_backend_cls=backend_cls,
        )

        with (
            mock.patch.object(
                backend_utils,
                "get_exec",
                return_value=exec_config,
            ),
            mock.patch.object(
                backend_utils,
                "current_platform",
                platform,
            ),
            mock.patch.object(
                backend_utils,
                "FullCudaGraphBackend",
            ) as default_backend_cls,
        ):
            actual_backend = backend_utils.resolve_decode_backend(runner)

        self.assertIs(actual_backend, expected_backend)
        platform.get_full_graph_backend_cls.assert_called_once_with()
        backend_cls.assert_called_once_with(
            runner,
            enable_memory_saver=True,
        )
        default_backend_cls.assert_not_called()

    def test_out_of_tree_platform_without_full_backend_uses_default(self):
        runner = _make_graph_runner()
        exec_config = _make_exec_config()
        expected_backend = object()
        platform = _OutOfTreePlatformWithoutFullGraphBackend()

        with (
            mock.patch.object(
                backend_utils,
                "get_exec",
                return_value=exec_config,
            ),
            mock.patch.object(
                backend_utils,
                "current_platform",
                platform,
            ),
            mock.patch.object(
                backend_utils,
                "FullCudaGraphBackend",
                return_value=expected_backend,
            ) as default_backend_cls,
        ):
            actual_backend = backend_utils.resolve_decode_backend(runner)

        self.assertIs(actual_backend, expected_backend)
        default_backend_cls.assert_called_once_with(
            runner,
            enable_memory_saver=False,
        )

    def test_retired_backend_is_not_replaced_with_full(self):
        runner = _make_graph_runner(device="cuda")
        exec_config = _make_exec_config(backend=Backend.TC_PIECEWISE)
        with (
            mock.patch.object(backend_utils, "get_exec", return_value=exec_config),
            mock.patch.object(backend_utils, "FullCudaGraphBackend") as full_backend,
        ):
            with self.assertRaisesRegex(ValueError, "Unsupported decode graph backend"):
                backend_utils.resolve_decode_backend(runner)
            full_backend.assert_not_called()

    def test_breakable_backend_does_not_consult_full_backend_factory(self):
        runner = _make_graph_runner()
        exec_config = _make_exec_config(
            backend=Backend.BREAKABLE,
            enable_memory_saver=True,
            debug_cuda_graph=True,
        )
        expected_backend = object()
        platform = _make_platform(is_out_of_tree=True)

        with (
            mock.patch.object(
                backend_utils,
                "get_exec",
                return_value=exec_config,
            ),
            mock.patch.object(
                backend_utils,
                "current_platform",
                platform,
            ),
            mock.patch.object(
                backend_utils,
                "BreakableCudaGraphBackend",
                return_value=expected_backend,
            ) as breakable_backend_cls,
        ):
            actual_backend = backend_utils.resolve_decode_backend(runner)

        self.assertIs(actual_backend, expected_backend)
        platform.is_out_of_tree.assert_not_called()
        platform.get_full_graph_backend_cls.assert_not_called()
        breakable_backend_cls.assert_called_once_with(
            runner,
            enable_memory_saver=True,
            debug_eager=True,
        )


class _CaptureBackend:
    def __init__(self, *args, **kwargs):
        self._group_preparation_enabled = False


class TestGroupedBackendResolution(CustomTestCase):
    def _resolve(
        self,
        phase,
        *,
        manager=None,
        compile=False,
        profile=False,
        debug=False,
        precarve=False,
        borrow=False,
        backend=None,
        custom=False,
        attention_warmup=False,
        cuda=True,
    ):
        runner = _make_graph_runner(device="cuda")
        runner.model_runner.lora_manager = manager
        runner.enable_profile_cuda_graph = profile
        runner.attn_backend.requires_per_shape_decode_graph_warmup = attention_warmup
        config = _make_exec_config(
            backend=backend
            or (Backend.FULL if phase == "decode" else Backend.BREAKABLE),
            debug_cuda_graph=debug,
        )
        config.graph.enable_profile_cuda_graph = profile
        flags = SimpleNamespace(capture=SimpleNamespace(enable_torch_compile=compile))
        envs = SimpleNamespace(
            SGLANG_ENABLE_GRAPH_POOL_PRECARVE=SimpleNamespace(get=lambda: precarve),
            SGLANG_ENABLE_GRAPH_POOL_BORROW=SimpleNamespace(get=lambda: borrow),
        )
        platform = _make_platform(
            cuda=cuda,
            is_out_of_tree=custom,
            full_backend_cls=_OtherCaptureBackend if custom else None,
        )
        with mock.patch.multiple(
            backend_utils,
            get_exec=lambda: config,
            get_memory=lambda: SimpleNamespace(enable_hierarchical_cache=False),
            get_flags=lambda: flags,
            envs=envs,
            current_platform=platform,
            FullCudaGraphBackend=_CaptureBackend,
            BreakableCudaGraphBackend=_CaptureBackend,
        ):
            return getattr(backend_utils, f"resolve_{phase}_backend")(runner)

    def test_supported_paths_and_eligibility(self):
        def dense(workspace, expert):
            return SimpleNamespace(
                lora_backend=SimpleNamespace(
                    lora_workspace=workspace, is_moe_lora=expert
                )
            )

        scenarios = [
            ({}, True),
            ({"cuda": False}, False),
            ({"manager": dense(None, False)}, True),
            ({"manager": dense(object(), False)}, False),
            ({"manager": dense(None, True)}, False),
            ({"manager": dense(object(), True)}, False),
        ]
        scenarios += [
            ({flag: True}, False)
            for flag in ("compile", "profile", "debug", "precarve", "borrow")
        ]
        for phase in ("decode", "prefill"):
            for options, expected in scenarios:
                with self.subTest(phase=phase, options=options):
                    self.assertEqual(
                        self._resolve(phase, **options)._group_preparation_enabled,
                        expected,
                    )
            self.assertEqual(
                self._resolve(phase, attention_warmup=True)._group_preparation_enabled,
                phase == "prefill",
            )

    def test_attention_requirement_propagates_through_wrappers(self):
        from sglang.srt.layers.attention import minimax_sparse_backend as sparse
        from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
        from sglang.srt.layers.attention.tbo_backend import TboAttnBackend

        for sm90, page_size, disabled, hip, npu, expected in (
            (True, 1, {0}, False, False, True),
            (True, 128, {0}, False, False, False),
            (True, 256, {0}, False, False, False),
            (True, 64, {0}, False, False, True),
            (False, 128, {0}, False, False, True),
            (False, 1, set(), False, False, False),
            (False, 1, {0}, True, False, False),
            (False, 1, {0}, False, True, False),
        ):
            with self.subTest(
                sm90=sm90, page_size=page_size, disabled=disabled, hip=hip, npu=npu
            ):
                leaf = sparse.MiniMaxSparseAttnBackend.__new__(
                    sparse.MiniMaxSparseAttnBackend
                )
                leaf.is_hip, leaf.is_npu = hip, npu
                leaf.page_size, leaf.block_size_k = page_size, 128
                leaf.disable_value_layer_ids = disabled
                neutral = SimpleNamespace(requires_per_shape_decode_graph_warmup=False)
                hybrid = sparse.MiniMaxHybridAttnBackend.__new__(
                    sparse.MiniMaxHybridAttnBackend
                )
                hybrid.dense, hybrid.sparse = neutral, leaf
                tbo = TboAttnBackend.__new__(TboAttnBackend)
                tbo.primary, tbo.children = neutral, [hybrid]
                with mock.patch.object(
                    sparse, "get_platform", return_value=SimpleNamespace(is_sm90=sm90)
                ):
                    self.assertEqual(
                        [
                            b.requires_per_shape_decode_graph_warmup
                            for b in (leaf, hybrid, tbo)
                        ],
                        [expected] * 3,
                    )
                    self.assertEqual(
                        AttentionBackend.requires_per_shape_decode_graph_warmup.fget(
                            SimpleNamespace(attn_backend_list=[leaf])
                        ),
                        expected,
                    )
        self.assertFalse(
            AttentionBackend.requires_per_shape_decode_graph_warmup.fget(
                SimpleNamespace(attn_backend_list=None)
            )
        )

    def test_other_backends_remain_unoptimized(self):
        for phase, options in (
            ("decode", {"backend": Backend.BREAKABLE}),
            ("prefill", {"backend": Backend.FULL}),
            ("decode", {"custom": True}),
        ):
            with self.subTest(phase=phase, options=options):
                self.assertFalse(
                    self._resolve(phase, **options)._group_preparation_enabled
                )


class _OtherCaptureBackend(_CaptureBackend):
    pass


if __name__ == "__main__":
    unittest.main()
