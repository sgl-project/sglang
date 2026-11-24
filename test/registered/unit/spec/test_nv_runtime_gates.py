import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.sampling.sampling_pipeline import VerifyRoute, VerifyStep
from sglang.srt.speculative import eagle_worker_common, eagle_worker_v2
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
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
                    if enabled and not custom:
                        sample.assert_not_called()
                        make_io.assert_called_once()
                        hooks.post_verify.assert_called_once_with(make_io.return_value)
                    else:
                        make_io.assert_not_called()
                        hooks.post_verify.assert_not_called()
                        sample.assert_called_once_with(
                            verify_input,
                            verify,
                            logits_output,
                            uno_target_max_top_k=options.get("uno_target_max_top_k"),
                            simulated_acc_len_rng=options.get("simulated_acc_len_rng"),
                            sparse_bonus_argmax=options.get(
                                "sparse_bonus_argmax", False
                            ),
                        )


def _logits_processor():
    processor = LogitsProcessor.__new__(LogitsProcessor)
    torch.nn.Module.__init__(processor)
    processor._sharded_verify_eligible = True
    processor.sharded_verify = processor.sharded_draft = False
    processor.defer_verify_gather = False
    processor.final_logit_softcapping = None
    processor.vocab_size = 64
    return processor


class TestShardedDraftGate(CustomTestCase):
    def test_only_the_stock_draft_top1_forwards_keep_vocab_shards(self):
        lm_head = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
        torch.nn.Module.__init__(lm_head)
        lm_head.shard_indices = SimpleNamespace(
            num_added_elements_padded=0, org_vocab_start_index=0, num_org_elements=32
        )

        def keeps_shards(
            mode=ForwardMode.DECODE,
            target="Qwen3_5MoeForCausalLM",
            megakernel=None,
            rejection=False,
            hot_token_id=None,
            tp_group=None,
            draft_vocab=64,
        ):
            target_model = type(target, (torch.nn.Module,), {})()
            target_model.logits_processor = _logits_processor()
            worker = EagleDraftWorker.__new__(EagleDraftWorker)
            worker.hot_token_id = hot_token_id
            worker.parallel_layout = SimpleNamespace(tp_group=tp_group)
            config = SimpleNamespace(vocab_size=draft_vocab)
            worker.draft_runner = SimpleNamespace(
                model=torch.nn.Module(), model_config=config
            )
            worker.draft_runner.model.logits_processor = processor = _logits_processor()
            worker.target_worker = SimpleNamespace(
                model_config=SimpleNamespace(vocab_size=64),
                model_runner=SimpleNamespace(model=target_model),
            )
            spec = SimpleNamespace(speculative_use_rejection_sampling=rejection)
            with patch.multiple(
                eagle_worker_v2,
                get_spec=lambda: spec,
                built_megakernel=lambda model: megakernel,
            ):
                eagle_worker_common.bind_eagle_runtime_hooks(
                    worker.target_worker,
                    supports_sharded_verify=True,
                    sharded_draft_model=worker.sharded_top1_model(),
                )
            shards = processor._batch_vocab_shards(SimpleNamespace(forward_mode=mode))
            metadata = LogitsMetadata(forward_mode=mode, verify_vocab_shards=shards)
            return processor._verify_vocab_shard(lm_head, metadata) is not None

        self.assertTrue(keeps_shards())
        self.assertTrue(keeps_shards(mode=ForwardMode.DRAFT_EXTEND_V2))
        self.assertFalse(keeps_shards(mode=ForwardMode.EXTEND))
        self.assertFalse(keeps_shards(target="LlamaForCausalLM"))
        self.assertFalse(keeps_shards(megakernel=object()))
        self.assertFalse(keeps_shards(rejection=True))
        self.assertFalse(keeps_shards(hot_token_id=torch.arange(4)))
        self.assertFalse(keeps_shards(tp_group=object()))
        self.assertFalse(keeps_shards(draft_vocab=32))


if __name__ == "__main__":
    unittest.main()
