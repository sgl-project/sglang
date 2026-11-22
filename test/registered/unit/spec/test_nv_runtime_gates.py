import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.environ import envs
from sglang.srt.speculative import eagle_worker_common
from sglang.srt.speculative.target_verify import TargetVerify
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestNativeEagerVerify(CustomTestCase):
    def test_master_selects_native_sampling_or_runtime_hooks(self):
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
                    verify = TargetVerify(
                        grammar_mask=object(),
                        watermark=object(),
                        watermark_full_mask=True,
                    )
                    verify_input = object()
                    logits_output = SimpleNamespace(next_token_logits=None)
                    outputs = (object(), object(), object())
                    hooks = SimpleNamespace(
                        post_verify=Mock(
                            return_value=SimpleNamespace(
                                token_ids=outputs[0],
                                accepted_lengths=outputs[1],
                                accepted_indices=outputs[2],
                            )
                        )
                    )
                    sample, make_io = (
                        Mock(return_value=outputs),
                        Mock(return_value=object()),
                    )
                    with (
                        patch.object(eagle_worker_common, "eagle_sample", sample),
                        patch.object(
                            eagle_worker_common, "make_post_verify_io", make_io
                        ),
                        envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(
                            enabled
                        ),
                    ):
                        accept = eagle_worker_common.accept_draft_tree(
                            batch,
                            verify=verify,
                            verify_input=verify_input,
                            logits_output=logits_output,
                            speculative_runtime_hooks=hooks,
                            **options,
                        )
                    self.assertEqual(
                        (accept.tokens, accept.accept_lens, accept.accept_index),
                        outputs,
                    )
                    if enabled and not custom:
                        sample.assert_not_called()
                        make_io.assert_called_once()
                        hooks.post_verify.assert_called_once_with(make_io.return_value)
                    else:
                        make_io.assert_not_called()
                        hooks.post_verify.assert_not_called()
                        sample.assert_called_once_with(
                            verify_input,
                            batch,
                            logits_output,
                            verify.grammar_mask,
                            uno_target_max_top_k=options.get("uno_target_max_top_k"),
                            watermark=verify.watermark,
                            simulated_acc_len_rng=options.get("simulated_acc_len_rng"),
                            sparse_bonus_argmax=options.get(
                                "sparse_bonus_argmax", False
                            ),
                            has_custom_logit_processor=custom,
                        )


if __name__ == "__main__":
    unittest.main()
