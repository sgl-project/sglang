from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.speculative.dspark.dspark_accept import (
    accept_target_only_sampling,
)
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockResult
from sglang.srt.speculative.dspark_components.dspark_verify import (
    DSparkPPLocalSamplingCache,
    DSparkPPMicroBatchSamplingCache,
    DSparkPPVerifyInputRaw,
    accept_draft_tokens,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSparkTargetOnlySampling(CustomTestCase):
    def test_pp_raw_prepares_local_sampling_metadata_without_serializing_it(self):
        pp_raw = DSparkPPVerifyInputRaw(
            bonus_tokens=[1, 2],
            draft_tokens=[[3], [4]],
            new_seq_lens=[10, 20],
            accept_lens=[1, 1],
        )
        reqs = [
            SimpleNamespace(sampling_params=SimpleNamespace(top_k=4)),
            SimpleNamespace(sampling_params=SimpleNamespace(top_k=8)),
        ]

        pp_raw.prepare_local_sampling_metadata(reqs)

        self.assertEqual(pp_raw.max_top_k, 8)
        self.assertIsNone(pp_raw.uniform_top_k_value)
        serialized = pp_raw.to_tensor_dict()["pp_spec_output"]
        self.assertNotIn("max_top_k", serialized)
        self.assertNotIn("uniform_top_k_value", serialized)

    def test_microbatch_cache_filter_and_merge_new_rows(self):
        manager = DSparkPPMicroBatchSamplingCache()
        cache = DSparkPPLocalSamplingCache(
            corrected_logits=torch.arange(24).view(3, 2, 4),
            request_keys=[("a", 1), ("b", 2), ("c", 3)],
        )
        manager.publish(2, cache)

        logits, ready = manager.consume(
            2,
            [
                SimpleNamespace(rid="c"),
                SimpleNamespace(rid="new"),
                SimpleNamespace(rid="a"),
            ],
            [3, 4, 1],
            device=torch.device("cpu"),
        )

        self.assertIsNotNone(logits)
        torch.testing.assert_close(ready, torch.tensor([True, False, True]))
        torch.testing.assert_close(logits[0], cache.corrected_logits[2])
        torch.testing.assert_close(
            logits[1], torch.zeros((2, 4), dtype=torch.int64)
        )
        torch.testing.assert_close(logits[2], cache.corrected_logits[0])
        self.assertEqual(
            manager.consume(2, [], [], device=torch.device("cpu")), (None, None)
        )

    def test_microbatch_slots_are_independent(self):
        manager = DSparkPPMicroBatchSamplingCache()
        cache0 = DSparkPPLocalSamplingCache(
            corrected_logits=torch.zeros((1, 2, 4)), request_keys=[("a", 1)]
        )
        cache1 = DSparkPPLocalSamplingCache(
            corrected_logits=torch.ones((1, 2, 4)), request_keys=[("b", 2)]
        )
        manager.publish(0, cache0)
        manager.publish(1, cache1)

        logits1, ready1 = manager.consume(
            1, [SimpleNamespace(rid="b")], [2], device=torch.device("cpu")
        )
        self.assertIsNone(ready1)
        torch.testing.assert_close(logits1, cache1.corrected_logits)
        logits0, ready0 = manager.consume(
            0, [SimpleNamespace(rid="a")], [1], device=torch.device("cpu")
        )
        self.assertIsNone(ready0)
        torch.testing.assert_close(logits0, cache0.corrected_logits)

    def test_samples_only_first_target_position_and_commits_one(self):
        bs, verify_tokens, vocab = 2, 4, 8
        logits = torch.full((bs, verify_tokens, vocab), -100.0)
        logits[0, 0, 3] = 100.0
        logits[1, 0, 5] = 100.0
        # Later verify positions must not affect the target-only result.
        logits[:, 1:, 7] = 1000.0
        sampling_info = SimpleNamespace(
            temperatures=torch.ones(bs),
            top_ks=torch.ones(bs, dtype=torch.int32),
            top_ps=torch.ones(bs),
            need_top_k_sampling=True,
            need_top_p_sampling=False,
        )

        correct_len, bonus, cap_trim_lens = accept_target_only_sampling(
            target_logits=logits.view(bs * verify_tokens, vocab),
            sampling_info=sampling_info,
            verify_num_draft_tokens=verify_tokens,
        )

        torch.testing.assert_close(correct_len, torch.zeros(bs, dtype=torch.int32))
        torch.testing.assert_close(bonus, torch.tensor([3, 5], dtype=torch.int64))
        torch.testing.assert_close(cap_trim_lens, torch.zeros(bs, dtype=torch.int32))

    def test_cache_miss_rows_use_target_only_without_affecting_hits(self):
        bs, gamma, verify_tokens, vocab = 2, 3, 4, 8
        draft_block = DraftBlockResult(
            draft_tokens=torch.zeros((bs, gamma), dtype=torch.int64),
            corrected_logits=torch.zeros((bs, gamma, vocab)),
            greedy_mask=torch.zeros(bs, dtype=torch.bool),
            temperatures=torch.ones(bs),
            corrected_logits_ready=torch.tensor([True, False]),
        )
        sampling_info = SimpleNamespace(is_all_greedy=False, is_any_greedy=False)
        normal = (
            torch.tensor([3, 3], dtype=torch.int32),
            torch.tensor([4, 4], dtype=torch.int64),
            torch.zeros(bs, dtype=torch.int32),
        )
        target_only = (
            torch.zeros(bs, dtype=torch.int32),
            torch.tensor([6, 6], dtype=torch.int64),
            torch.zeros(bs, dtype=torch.int32),
        )

        with (
            patch(
                "sglang.srt.speculative.dspark_components.dspark_verify.SoftmaxTemp.execute",
                return_value=torch.ones(bs * gamma, vocab),
            ),
            patch(
                "sglang.srt.speculative.dspark_components.dspark_verify.AcceptSampling.execute",
                return_value=normal,
            ),
            patch(
                "sglang.srt.speculative.dspark_components.dspark_verify.accept_target_only_sampling",
                return_value=target_only,
            ),
        ):
            correct_len, bonus, cap_trim_lens = accept_draft_tokens(
                candidates=torch.zeros((bs, verify_tokens), dtype=torch.int64),
                target_logits=torch.zeros((bs * verify_tokens, vocab)),
                draft_block=draft_block,
                sampling_info=sampling_info,
                draft_input=SimpleNamespace(),
                gamma=gamma,
                verify_num_draft_tokens=verify_tokens,
            )

        torch.testing.assert_close(correct_len, torch.tensor([3, 0], dtype=torch.int32))
        torch.testing.assert_close(bonus, torch.tensor([4, 6], dtype=torch.int64))
        torch.testing.assert_close(cap_trim_lens, torch.zeros(bs, dtype=torch.int32))


if __name__ == "__main__":
    import unittest

    unittest.main()
