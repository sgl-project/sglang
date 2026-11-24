import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.environ import envs
from sglang.srt.sampling.sampling_pipeline import VerifyRoute, VerifyStep
from sglang.srt.speculative import eagle_worker_common
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestNativeEagerVerify(CustomTestCase):
    def test_master_passes_the_runtime_hooks_local_keys(self):
        # EAGLE passes its sparse and simulated-acceptance options; NGRAM the defaults.
        eagle_options = dict(
            uno_target_max_top_k=17,
            simulated_acc_len_rng=object(),
            sparse_bonus_argmax=True,
        )
        for label, options in (("eagle", eagle_options), ("defaults", {})):
            # A custom processor matters only with the switch on.
            for enabled, custom in ((False, False), (True, False), (True, True)):
                with self.subTest(options=label, enabled=enabled, custom=custom):
                    batch = SimpleNamespace(
                        sampling_info=SimpleNamespace(
                            has_custom_logit_processor=custom
                        ),
                        seq_lens=object(),
                        req_pool_indices=object(),
                        device="cpu",
                        return_logprob=False,
                        forward_mode=SimpleNamespace(is_idle=lambda: False),
                    )
                    verify = VerifyStep(
                        batch=batch,
                        route=VerifyRoute(
                            adjusts_logits=True, need_truncated_sampling=False
                        ),
                        watermark=object(),
                        watermark_full_mask=True,
                        grammar_mask=object(),
                    )
                    verify_input = object()
                    logits_output = SimpleNamespace(next_token_logits=None)
                    outputs = (object(), object(), object())
                    hooks = SimpleNamespace(local_keys=object())
                    sample = Mock(return_value=outputs)
                    with (
                        patch.object(eagle_worker_common, "eagle_sample", sample),
                        envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(
                            enabled
                        ),
                    ):
                        accept = eagle_worker_common.accept_draft_tree(
                            verify,
                            verify_input=verify_input,
                            logits_output=logits_output,
                            speculative_runtime_hooks=hooks,
                            **options,
                        )
                    self.assertEqual(
                        (accept.tokens, accept.accept_lens, accept.accept_index),
                        outputs,
                    )
                    sample.assert_called_once_with(
                        verify_input,
                        verify,
                        logits_output,
                        uno_target_max_top_k=options.get("uno_target_max_top_k"),
                        simulated_acc_len=None,
                        simulated_acc_len_rng=options.get("simulated_acc_len_rng"),
                        sparse_bonus_argmax=options.get("sparse_bonus_argmax", False),
                        local_keys=(
                            hooks.local_keys if enabled and not custom else None
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
