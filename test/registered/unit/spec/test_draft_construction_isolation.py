"""A draft's construction decides for itself and leaves the process state alone.

The shared-experts-fusion decision is per checkpoint: each MoE model's gate
writes the ACTIVE moe flag (both ways) before its own layers build and read
it, and ``draft_model_build_scope`` — which brackets every draft
construction — records it on the speculative leaf and restores the target's
value on exit. The config bag keeps the
user's intent. A draft's weight update does not rewrite the
process's model_path record.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.layers.moe.utils import (
    draft_model_build_scope,
    install_shared_experts_fusion_decision,
    is_shared_experts_fusion_disabled,
    speculative_moe_backend_context,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import get_context, get_flags, get_model
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _AlwaysDisables:
    """A model class whose checkpoint can never fuse."""

    @staticmethod
    def shared_experts_fusion_disable_reason(hf_config, quant_config):
        return "stand-in: this checkpoint cannot fuse."


class _NoGate:
    """A model family without an auto-disable gate: it follows the intent."""


def _install(model_class):
    install_shared_experts_fusion_decision(model_class, SimpleNamespace(), None)


class TestFusionDecisionFlag(CustomTestCase):
    def setUp(self):
        super().setUp()
        moe = get_flags().moe
        self._saved = (
            moe.disable_shared_experts_fusion,
            moe.speculative_disable_shared_experts_fusion,
        )
        moe.disable_shared_experts_fusion = None
        moe.speculative_disable_shared_experts_fusion = None
        moe.in_speculative_scope = False

    def tearDown(self):
        moe = get_flags().moe
        (
            moe.disable_shared_experts_fusion,
            moe.speculative_disable_shared_experts_fusion,
        ) = self._saved
        super().tearDown()

    def _seed(self, **fields):
        override = get_context().override_server_args(**fields)
        override.install()
        self.addCleanup(override.restore)

    def test_unset_flag_falls_back_to_the_config_intent(self):
        self._seed(disable_shared_experts_fusion=True)
        self.assertTrue(is_shared_experts_fusion_disabled())
        self._seed(disable_shared_experts_fusion=False)
        # A fresh install replaces the published config; the flag is still None.
        self.assertFalse(is_shared_experts_fusion_disabled())

    def test_the_installed_decision_wins_over_the_intent(self):
        self._seed(disable_shared_experts_fusion=False)
        _install(_AlwaysDisables)
        self.assertTrue(is_shared_experts_fusion_disabled())
        _install(_NoGate)
        self.assertFalse(is_shared_experts_fusion_disabled())

    def test_the_intent_short_circuits_the_gate(self):
        # A user who passed --disable-shared-experts-fusion is not overruled,
        # and the gate is not even asked.
        self._seed(disable_shared_experts_fusion=True)
        _install(_NoGate)
        self.assertTrue(is_shared_experts_fusion_disabled())

    def test_the_draft_build_scope_restores_the_targets_decision(self):
        self._seed(disable_shared_experts_fusion=False)
        _install(_NoGate)  # the target's build
        with draft_model_build_scope():
            _install(_AlwaysDisables)  # the draft's build
            self.assertTrue(is_shared_experts_fusion_disabled())
        self.assertFalse(is_shared_experts_fusion_disabled())
        # The draft's decision stays inspectable on the twin leaf.
        self.assertTrue(get_flags().moe.speculative_disable_shared_experts_fusion)

    def test_a_gateless_draft_inherits_the_active_decision(self):
        self._seed(disable_shared_experts_fusion=False)
        _install(_AlwaysDisables)  # the target's build
        with draft_model_build_scope():
            # A draft whose family has no gate follows the intent, which is what
            # the target's own build already resolved to here.
            self.assertTrue(is_shared_experts_fusion_disabled())
        self.assertTrue(is_shared_experts_fusion_disabled())

    def test_post_build_scopes_do_not_clobber_the_draft_leaf(self):
        # init_attention_backends / cuda-graph capture / draft forwards enter
        # scopes after construction; no gate runs there, so the persisted
        # draft decision must survive.
        self._seed(disable_shared_experts_fusion=False)
        _install(_NoGate)  # target's build
        with draft_model_build_scope():
            _install(_AlwaysDisables)  # draft's build
        for _ in range(3):
            with draft_model_build_scope():
                pass
            with speculative_moe_backend_context():
                pass
        self.assertTrue(get_flags().moe.speculative_disable_shared_experts_fusion)
        self.assertFalse(get_flags().moe.disable_shared_experts_fusion)

    def test_the_build_scope_leaves_the_runner_backend_alone(self):
        # Swapping runner_backend is speculative_moe_backend_context's job and
        # must bracket the draft's whole lifecycle; dflash/dspark run their
        # draft outside it, so a construction-only swap would build and
        # execute the draft under different backends.
        self._seed()
        before = get_flags().moe.runner_backend
        with draft_model_build_scope():
            self.assertEqual(get_flags().moe.runner_backend, before)
        self.assertEqual(get_flags().moe.runner_backend, before)

    def test_a_record_outside_any_scope_is_target_only(self):
        self._seed(disable_shared_experts_fusion=False)
        get_flags().moe.speculative_disable_shared_experts_fusion = True
        _install(_NoGate)  # target's build
        self.assertTrue(get_flags().moe.speculative_disable_shared_experts_fusion)

    def test_initialize_moe_config_seeds_both_leaves(self):
        from sglang.srt.layers.moe.utils import initialize_moe_config
        from sglang.srt.server_args import ServerArgs

        self._seed()
        initialize_moe_config(
            ServerArgs(model_path="dummy", disable_shared_experts_fusion=True)
        )
        moe = get_flags().moe
        self.assertTrue(moe.disable_shared_experts_fusion)
        self.assertTrue(moe.speculative_disable_shared_experts_fusion)

    def test_a_forward_time_read_is_refused(self):
        # The invariant behind the whole design: the decision is consumed at
        # construction only. During a draft's build the flag holds the draft's
        # value, so a forward reading it would race the build window.
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        self._seed()
        with forward_context(ForwardContext(attn_backend=SimpleNamespace())):
            with self.assertRaises(AssertionError):
                is_shared_experts_fusion_disabled()

    def test_the_intent_stays_on_the_bag(self):
        self._seed(disable_shared_experts_fusion=False)
        _install(_AlwaysDisables)
        from sglang.srt.runtime_context import get_exec

        self.assertFalse(get_exec().moe.disable_shared_experts_fusion)


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


if __name__ == "__main__":
    unittest.main()
