"""Each runner carries its own linear-attn kernel choice.

A target and its draft coexist in one process and can want different kernels:
only the runner whose model is GDN gets the SM100 FlashInfer prefill default,
and the operator's explicit flag applies to the launch. So the choice is a
per-runner stamp read from the runner, the way the full-attention pair already
works (`prefill_attention_backend_str` / `decode_attention_backend_str`), rather
than one process-wide table.

It used to be that table: `attn_backend_wrapper` rebuilt a module-level dict
once per runner, from the handed record plus a local default. Two consequences,
both pinned below -- a second runner could not hold a different choice, and its
rebuild replaced the first runner's (the record never carries the recorded
default, so a runner with no default of its own resolved back to the base
backend).
"""

import unittest
from types import SimpleNamespace

from sglang.srt.layers.attention.linear.utils import (
    LinearAttnKernelBackend,
    resolve_linear_attn_backends,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Runner:
    """The only thing the readers need from a runner: the stamp."""


class TestLinearAttnBackends(CustomTestCase):
    def _publish(self, **fields):
        from sglang.srt.runtime_context import get_context

        override = get_context().override_server_args(**fields)
        override.install()
        self.addCleanup(override.restore)

    def test_the_default_applies_when_the_flag_is_unset(self):
        self._publish(linear_attn_backend="triton")
        backends = resolve_linear_attn_backends(prefill_default="flashinfer")
        self.assertEqual(backends.prefill, LinearAttnKernelBackend.FLASHINFER)

    def test_the_base_backend_applies_without_a_default(self):
        self._publish(linear_attn_backend="triton")
        backends = resolve_linear_attn_backends()
        self.assertEqual(backends.prefill, LinearAttnKernelBackend.TRITON)
        self.assertEqual(backends.decode, LinearAttnKernelBackend.TRITON)

    def test_the_default_does_not_reach_the_decode_backend(self):
        self._publish(linear_attn_backend="triton")
        backends = resolve_linear_attn_backends(prefill_default="flashinfer")
        self.assertEqual(backends.decode, LinearAttnKernelBackend.TRITON)

    def test_explicit_flag_wins_over_the_default(self):
        """Precedence lives in the gate, which declines once the flag is set.

        The resolver takes the default as an argument, so the flag has to win
        upstream: `flashinfer_gdn_prefill_default` returns None the moment the
        leaf is set, and that condition is checked before anything touches the
        device -- which is what lets this run anywhere.
        """
        from types import SimpleNamespace

        from sglang.srt.layers.attention.linear.gdn_backend import (
            flashinfer_gdn_prefill_default,
        )
        from sglang.srt.runtime_context import get_server_args

        self._publish(
            linear_attn_backend="triton", linear_attn_prefill_backend="cutedsl"
        )
        runner = SimpleNamespace(server_args=get_server_args())
        self.assertIsNone(flashinfer_gdn_prefill_default(runner))
        self.assertEqual(
            resolve_linear_attn_backends().prefill, LinearAttnKernelBackend.CUTEDSL
        )

    def test_two_runners_hold_different_choices(self):
        """The property the process-wide table could not express.

        The GDN target gets the SM100 default; the draft that is not GDN has no
        default of its own. Both stamps stand, and reading one does not disturb
        the other -- under the old table the draft's rebuild replaced the
        target's choice with the base backend.
        """
        self._publish(linear_attn_backend="triton")

        target, draft = _Runner(), _Runner()
        target.linear_attn_backends = resolve_linear_attn_backends(
            prefill_default="flashinfer"
        )
        draft.linear_attn_backends = resolve_linear_attn_backends()

        self.assertEqual(
            target.linear_attn_backends.prefill, LinearAttnKernelBackend.FLASHINFER
        )
        self.assertEqual(
            draft.linear_attn_backends.prefill, LinearAttnKernelBackend.TRITON
        )

    def test_an_unstamped_runner_raises_rather_than_guessing(self):
        """No default for "nobody stamped this", on the production read path.

        `attn_backend_wrapper` stamps before it builds the backends that read
        the stamp, so a missing one means a backend was built outside that
        path. The runner double below satisfies everything `GDNAttnBackend`
        touches *before* the stamp read, so the `AttributeError` this asserts
        comes from `model_runner.linear_attn_backends` itself -- a default
        stamp on the runner or a restored module-level fallback would turn
        this red-to-green, which is the regression it guards. Silent triton
        fallback would hide the wiring mistake behind a working-but-wrong
        kernel.
        """
        import torch

        from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend

        # The draft-token width is a bag leaf read before the stamp.
        self._publish(speculative_eagle_topk=0)
        runner = SimpleNamespace(
            device="cpu",
            server_args=SimpleNamespace(enable_unified_memory=False),
            is_draft_worker=False,
            req_to_token_pool=SimpleNamespace(
                mamba_pool=SimpleNamespace(
                    mamba_cache=SimpleNamespace(conv=[torch.zeros(1, 1, 4, 7)])
                )
            ),
            token_to_kv_pool=None,
        )
        with self.assertRaises(AttributeError) as caught:
            GDNAttnBackend(runner)
        self.assertIn("linear_attn_backends", str(caught.exception))

    def test_the_per_runner_default_stays_out_of_the_process_config(self):
        """The process-wide config cannot represent one runner's choice.

        Recording the auto-default there is how a second runner used to inherit
        it: the leaf then reads as "the operator named a backend", which is the
        one question the gate asks. So the default lives in the runner's stamp
        and the leaf keeps meaning what was asked for at launch.
        """
        from sglang.srt.runtime_context import get_context, get_exec

        self._publish(linear_attn_backend="triton")
        backends = resolve_linear_attn_backends(prefill_default="flashinfer")

        self.assertEqual(backends.prefill, LinearAttnKernelBackend.FLASHINFER)
        self.assertIsNone(get_exec().mamba.linear_attn_prefill_backend)
        self.assertIsNone(
            get_context().resolved_server_args_dict()["linear_attn_prefill_backend"]
        )


if __name__ == "__main__":
    unittest.main()
