"""A model owns its shared-expert layout across construction and execution."""

import unittest
from types import MethodType, SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.moe.utils import (
    draft_model_build_scope,
    get_num_fused_shared_experts,
    resolve_shared_experts_config,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import get_context, get_model
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class _AlwaysDisables:
    """A model class whose checkpoint can never fuse."""

    @staticmethod
    def shared_experts_fusion_disable_reason(hf_config, quant_config):
        return "stand-in: this checkpoint cannot fuse."


class _NoGate:
    """A model family without an auto-disable gate: it follows the intent."""


class _FusesOne:
    """A family that fuses one shared expert as a routed slot unless vetoed."""

    @staticmethod
    def shared_experts_fusion_disable_reason(hf_config, quant_config):
        return None

    @staticmethod
    def num_fusable_shared_experts(hf_config):
        return 1


class _AlwaysDisablesButCountsOne(_FusesOne):
    @staticmethod
    def shared_experts_fusion_disable_reason(hf_config, quant_config):
        return "stand-in: this checkpoint cannot fuse."


class _CountsOneWithoutGate:
    @staticmethod
    def num_fusable_shared_experts(hf_config):
        return 1


class TestSharedExpertsModelScope(CustomTestCase):
    """Target/draft construction and execution must select the same fixed layout."""

    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(
            disable_shared_experts_fusion=False
        )
        override.install()
        self.addCleanup(override.restore)

    def _build(self, count, *, disabled=False, fail=False):
        from sglang.srt.layers.moe.topk import TopK
        from sglang.srt.model_loader import loader

        class Model(torch.nn.Module):
            @staticmethod
            def shared_experts_fusion_disable_reason(hf_config, quant_config):
                return "checkpoint veto" if hf_config.disabled else None

            @staticmethod
            def num_fusable_shared_experts(hf_config):
                return hf_config.count

            def __init__(self, config, quant_config):
                super().__init__()
                self.topk = TopK(top_k=2, layer_id=0)
                self.built_config = get_model().shared_experts
                if config.fail:
                    raise RuntimeError("construction failed")

            def forward_split_prefill(self, input_ids, positions, batch, layer_range):
                # Reads both the current model view and the layout retained at build.
                return (
                    input_ids
                    + get_num_fused_shared_experts()
                    + self.topk.topk_config.num_shared_topk_slots
                )

        with mock.patch.object(
            loader, "get_model_architecture", return_value=(Model, "Model")
        ):
            return loader._initialize_model(
                SimpleNamespace(
                    hf_config=SimpleNamespace(count=count, disabled=disabled, fail=fail)
                ),
                SimpleNamespace(draft_model_idx=None),
            )

    def _runner(self, model):
        runner = ModelRunner.__new__(ModelRunner)
        runner.model = model
        runner.model_config = SimpleNamespace(num_hidden_layers=2)
        runner.device_timer = None
        return runner

    def _run(self, runner):
        batch = SimpleNamespace(
            split_index=1, input_ids=torch.tensor([3]), positions=None
        )
        return runner.forward_split_prefill(batch)

    def test_loader_and_alternating_runners_keep_their_own_layout(self):
        ambient = get_model().shared_experts
        target = self._build(2)
        with draft_model_build_scope():
            draft = self._build(1, disabled=True)
        self.assertIs(get_model().shared_experts, ambient)
        for model in (target, draft):
            self.assertIs(model.shared_experts_config, model.built_config)
            self.assertIs(
                model.shared_experts_config, model.topk.topk_config.shared_experts
            )
        target_runner, draft_runner = self._runner(target), self._runner(draft)
        for runner, expected in (
            (target_runner, 7),
            (draft_runner, 3),
            (target_runner, 7),
        ):
            self.assertEqual(self._run(runner).item(), expected)
            self.assertIs(get_model().shared_experts, ambient)
        # A model replacement must not leave the runner reading its old config.
        target_runner.model = draft
        self.assertEqual(self._run(target_runner).item(), 3)

    def test_nested_build_and_execution_restore_on_failure(self):
        target = self._build(1)
        with get_model().override(shared_experts=target.shared_experts_config):
            with self.assertRaisesRegex(RuntimeError, "construction failed"):
                self._build(2, fail=True)
            self.assertIs(get_model().shared_experts, target.shared_experts_config)
            draft = self._build(1, disabled=True)
            runner = self._runner(draft)
            with mock.patch.object(
                draft,
                "forward_split_prefill",
                side_effect=RuntimeError("forward failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "forward failed"):
                    self._run(runner)
            self.assertIs(get_model().shared_experts, target.shared_experts_config)
            self.assertEqual(self._run(self._runner(target)).item(), 5)

    def test_resolution_preserves_intent_and_requires_both_hooks(self):
        from sglang.srt.runtime_context import get_exec

        ambient = get_model().shared_experts
        for cls, disabled, count in (
            (_NoGate, False, 0),
            (_CountsOneWithoutGate, False, 0),
            (_AlwaysDisablesButCountsOne, True, 0),
            (_FusesOne, False, 1),
        ):
            with self.subTest(cls=cls.__name__):
                config = resolve_shared_experts_config(cls, SimpleNamespace(), None)
                self.assertEqual(
                    (config.fusion_disabled, config.num_fused_shared_experts),
                    (disabled, count),
                )
        with get_exec().moe.override(disable_shared_experts_fusion=True):
            config = resolve_shared_experts_config(_FusesOne, SimpleNamespace(), None)
            self.assertTrue(config.fusion_disabled)
            self.assertEqual(config.num_fused_shared_experts, 0)
        with get_exec().moe.override(enable_waterfill=True):
            fused = resolve_shared_experts_config(_FusesOne, SimpleNamespace(), None)
            unfused = resolve_shared_experts_config(
                _AlwaysDisablesButCountsOne, SimpleNamespace(), None
            )
            self.assertEqual(
                (fused.num_fused_shared_experts, fused.num_shared_topk_slots), (1, 0)
            )
            self.assertTrue(fused.use_waterfill)
            self.assertFalse(unfused.use_waterfill)
        self.assertFalse(get_exec().moe.disable_shared_experts_fusion)
        self.assertIs(get_model().shared_experts, ambient)

    def test_warmup_and_direct_capture_bind_the_model_config(self):
        from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
        from sglang.srt.model_executor.runner.base_runner import BaseRunner
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )
        from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
            PrefillCudaGraphRunner,
        )

        model = self._build(1)
        ambient = get_model().shared_experts
        for entry, body in (
            (BaseRunner.warmup, "_warmup"),
            (DecodeCudaGraphRunner.capture, "_capture"),
            (PrefillCudaGraphRunner.capture, "_capture"),
            (CPUGraphRunner.capture, "_capture"),
        ):
            observed = []

            def capture():
                observed.append(get_model().shared_experts)
                raise RuntimeError("capture failed")

            graph_runner = SimpleNamespace(
                model_runner=self._runner(model), **{body: capture}
            )
            graph_runner.model_context = MethodType(
                BaseRunner.model_context, graph_runner
            )
            with self.subTest(entry=entry.__qualname__):
                with self.assertRaisesRegex(RuntimeError, "capture failed"):
                    entry(graph_runner)
                self.assertEqual(observed, [model.shared_experts_config])
                self.assertIs(get_model().shared_experts, ambient)


class TestDraftWeightUpdateRecord(CustomTestCase):
    def _seed(self, **fields):
        override = get_context().override_server_args(**fields)
        server_args = override.install()
        self.addCleanup(override.restore)
        return server_args

    def _update(self, *, is_draft_worker: bool):
        runner = ModelRunner.__new__(ModelRunner)
        runner.is_draft_worker = is_draft_worker
        runner.update_model_fields(
            object(),
            model_path="/new/checkpoint",
            load_format="auto",
            load_config=object(),
        )

    def test_a_target_update_is_recorded(self):
        self._seed()
        self._update(is_draft_worker=False)
        self.assertEqual(get_model().model_path, "/new/checkpoint")

    def test_a_draft_update_keeps_the_targets_record(self):
        seeded = self._seed()
        self._update(is_draft_worker=True)
        self.assertEqual(get_model().model_path, seeded.model_path)


class TestDraftBoundaryPolicy(unittest.TestCase):
    def test_scope_restores_target_policy_after_nested_failure(self):
        from sglang.srt.runtime_context import (
            get_exec,
            get_spec,
            publish,
            reset_context,
        )
        from sglang.srt.server_args import ServerArgs

        reset_context()
        try:
            publish(
                ServerArgs(
                    model_path="dummy",
                    boundary_reduction="rs+rsv",
                ),
                role="test",
            )
            self.assertEqual(get_exec().comm.boundary_reduction, "rs+rsv")
            with (
                get_spec().override(speculative_boundary_reduction="ar"),
                self.assertRaisesRegex(RuntimeError, "draft failed"),
            ):
                with draft_model_build_scope():
                    self.assertEqual(get_exec().comm.boundary_reduction, "ar")
                    with draft_model_build_scope():
                        self.assertEqual(get_exec().comm.boundary_reduction, "ar")
                    raise RuntimeError("draft failed")
            self.assertEqual(get_exec().comm.boundary_reduction, "rs+rsv")
        finally:
            reset_context()


if __name__ == "__main__":
    unittest.main()
