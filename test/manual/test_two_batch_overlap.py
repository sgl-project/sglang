import unittest
from types import SimpleNamespace
from unittest.mock import patch

import requests
import torch

from sglang.srt.batch_overlap.two_batch_overlap import (
    TboForwardBatchPreparer,
    compute_split_seq_index,
    compute_split_token_index,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_ENABLE_THINKING_MODEL_NAME_FOR_TEST,
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestTwoBatchOverlap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_ENABLE_JIT_DEEPGEMM.override(False):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--trust-remote-code",
                    "--tp",
                    "2",
                    "--dp",
                    "2",
                    "--enable-dp-attention",
                    "--moe-a2a-backend",
                    "deepep",
                    "--deepep-mode",
                    "normal",
                    "--disable-cuda-graph",  # DeepEP normal does not support CUDA Graph
                    "--enable-two-batch-overlap",
                ],
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_generate_single_prompt(self):
        response = requests.post(
            self.base_url + "/generate",
            # we use an uncommon start to minimise the chance that the cache is hit by chance
            json={
                "text": "_ 1+1=2, 1+2=3, 1+3=4, 1+4=",
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            },
        )
        print(f"{response.json()=}")
        self.assertEqual(response.json()["text"], "5, 1+5=6")

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.5)


class TestTwoBatchOverlapUnitTest(unittest.TestCase):
    def test_compute_split_seq_and_token_index(self):
        for num_tokens, expect in [
            (0, 0),
            (100, 50),
            (99, 49),
        ]:
            actual = compute_split_seq_index(
                forward_mode=ForwardMode.DECODE,
                num_tokens=num_tokens,
                extend_lens=None,
                token_num_per_seq=1,
            )
            self.assertEqual(actual, expect)

        for extend_lens, expect in [
            ([], (0, 0)),
            ([42], (0, 21)),
            ([42, 999], (1, 520)),
            ([999, 42], (0, 520)),
            ([498, 502], (1, 498)),
            ([4096, 4096, 4096, 4096], (2, 8192)),
            ([4095, 4096, 4096, 4096, 1], (2, 8191)),
            ([1, 4095, 4096, 4096, 4096], (3, 8192)),
            ([4097, 4096, 4096, 4095, 1], (2, 8193)),
            ([1, 1, 1, 1, 99999], (4, 50001)),
            ([99999, 1, 1, 1, 1], (0, 50001)),
        ]:
            actual_seq_idx = compute_split_seq_index(
                forward_mode=ForwardMode.EXTEND,
                num_tokens=None,
                extend_lens=extend_lens,
                token_num_per_seq=None,
            )
            actual_token_idx = compute_split_token_index(
                split_seq_index=actual_seq_idx,
                forward_mode=ForwardMode.EXTEND,
                extend_seq_lens=extend_lens,
                token_num_per_seq=None,
            )
            actual = (actual_seq_idx, actual_token_idx)
            print(f"{extend_lens=} {expect=} {actual=}")
            self.assertEqual(actual, expect)

    def test_eager_prepare_preserves_zero_parent_cpu_count(self):
        """An idle MAX_LEN parent must produce two zero-count TBO children."""
        batch = SimpleNamespace(
            tbo_split_seq_index=2,
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
            extend_seq_lens_cpu=None,
            num_token_non_padded_cpu=0,
        )

        with (
            patch.object(
                TboForwardBatchPreparer,
                "compute_tbo_children_num_token_non_padded",
                return_value=(0, 0),
            ),
            patch.object(TboForwardBatchPreparer, "prepare_raw") as prepare_raw,
        ):
            TboForwardBatchPreparer.prepare(batch)

        prepare_raw.assert_called_once()
        child_cpu_counts = prepare_raw.call_args.kwargs[
            "tbo_children_num_token_non_padded_cpu"
        ]
        self.assertEqual(child_cpu_counts, (0, 0))

    def test_capture_count_falls_back_to_physical_rows(self):
        """A decode CUDA-graph capture batch has no CPU count mirror."""
        batch = SimpleNamespace(
            tbo_split_seq_index=2,
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
            extend_seq_lens_cpu=None,
            input_ids=torch.empty(8, dtype=torch.long),
            num_token_non_padded_cpu=None,
        )

        with patch(
            "sglang.srt.batch_overlap.two_batch_overlap.get_device",
            return_value=SimpleNamespace(device="cpu"),
        ):
            children = (
                TboForwardBatchPreparer.compute_tbo_children_num_token_non_padded(
                    batch
                )
            )

        self.assertEqual(children.cpu().tolist(), [2, 6])

    def test_prepare_falls_back_to_physical_rows_for_missing_cpu_count(self):
        """prepare() must propagate the capture fallback to TBO children."""
        batch = SimpleNamespace(
            tbo_split_seq_index=2,
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
            extend_seq_lens_cpu=None,
            input_ids=torch.empty(8, dtype=torch.long),
            num_token_non_padded_cpu=None,
        )

        with (
            patch.object(
                TboForwardBatchPreparer,
                "compute_tbo_children_num_token_non_padded",
                return_value=torch.tensor([2, 6], dtype=torch.int32),
            ),
            patch.object(TboForwardBatchPreparer, "prepare_raw") as prepare_raw,
        ):
            TboForwardBatchPreparer.prepare(batch)

        prepare_raw.assert_called_once()
        child_cpu_counts = prepare_raw.call_args.kwargs[
            "tbo_children_num_token_non_padded_cpu"
        ]
        self.assertEqual(child_cpu_counts, (2, 6))


class TestQwen3TwoBatchOverlap(TestTwoBatchOverlap):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_ENABLE_THINKING_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-1234"
        with envs.SGLANG_ENABLE_JIT_DEEPGEMM.override(False):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--trust-remote-code",
                    "--tp",
                    "2",
                    "--dp",
                    "2",
                    "--enable-dp-attention",
                    "--moe-a2a-backend",
                    "deepep",
                    "--deepep-mode",
                    "normal",
                    "--disable-cuda-graph",  # DeepEP normal does not support CUDA Graph
                    "--enable-two-batch-overlap",
                ],
            )


if __name__ == "__main__":
    unittest.main()
