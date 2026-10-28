import dataclasses
import types
import unittest
from array import array
from itertools import product
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import (  # noqa: E402
    Req,
    ScheduleBatch,
    split_cached_prefix_by_tier,
)
from sglang.srt.managers.scheduler_components.batch_result_processor import (  # noqa: E402
    SchedulerBatchResultProcessor,
)
from sglang.srt.model_executor.forward_batch_info import (  # noqa: E402
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.runtime_context import get_context  # noqa: E402
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

AUTO_FILL_EXCLUDED_FIELDS = ["reqs"]


class TestCachedPrefixTierAttribution(unittest.TestCase):
    def test_split_cached_prefix_by_tier(self):
        cases = [
            (
                {"prefix_len": 100, "host_hit_len": 70, "storage_hit_len": 20},
                (30, 50, 20),
            ),
            (
                {
                    "prefix_len": 80,
                    "host_hit_len": 50,
                    "storage_hit_len": 20,
                    "storage_hit_start": 80,
                },
                (30, 50, 0),
            ),
            (
                {
                    "prefix_len": 90,
                    "host_hit_len": 60,
                    "storage_hit_len": 20,
                    "storage_hit_start": 80,
                },
                (30, 50, 10),
            ),
            (
                {
                    "prefix_len": 100,
                    "host_hit_len": 70,
                    "storage_hit_len": 70,
                    "host_hit_is_storage": True,
                },
                (30, 0, 70),
            ),
            (
                {
                    "prefix_len": 100,
                    "host_hit_len": 0,
                    "storage_hit_len": 20,
                    "storage_hit_start": 80,
                },
                (80, 0, 20),
            ),
            (
                {
                    "prefix_len": 100,
                    "host_hit_len": 0,
                    "storage_hit_len": 20,
                    "storage_hit_start": 80,
                    "host_hit_is_storage": True,
                },
                (80, 0, 20),
            ),
        ]
        for kwargs, expected in cases:
            with self.subTest(**kwargs):
                self.assertEqual(split_cached_prefix_by_tier(**kwargs), expected)


def make_schedule_batch(bs: int, **overrides) -> ScheduleBatch:
    batch = ScheduleBatch(reqs=overrides.pop("reqs"))
    # init_new always sets a SpeculativeAlgorithm enum, never None.
    batch.spec_algorithm = SpeculativeAlgorithm.NONE
    for field in dataclasses.fields(ScheduleBatch):
        name = field.name
        if name in overrides or name in AUTO_FILL_EXCLUDED_FIELDS:
            continue
        annotation = str(field.type)
        if "List" in annotation or "list[" in annotation:
            setattr(batch, name, [f"{name}-{i}" for i in range(bs)])
        elif "Tensor" in annotation:
            setattr(batch, name, torch.arange(bs, dtype=torch.int64))
    for name, value in overrides.items():
        setattr(batch, name, value)
    return batch


def _snapshot_mutable_fields(batch):
    snapshot = []
    for field in dataclasses.fields(batch):
        value = getattr(batch, field.name, None)
        if isinstance(value, MagicMock):
            continue
        if isinstance(value, list):
            snapshot.append((field.name, value, list(value)))
        elif isinstance(value, torch.Tensor):
            snapshot.append((field.name, value, value.clone()))
    return snapshot


def _assert_snapshot_not_mutated(test_case, snapshot):
    for name, obj, value_copy in snapshot:
        if isinstance(obj, list):
            test_case.assertEqual(obj, value_copy, f"field {name} was mutated in place")
        else:
            test_case.assertTrue(
                torch.equal(obj, value_copy), f"field {name} was mutated in place"
            )


class TestForwardBatchPrefillCompletion(unittest.TestCase):
    def test_first_commit_or_zero_token_completion_keeps_gate_open(self):
        runner = types.SimpleNamespace(
            device="cpu",
            is_draft_worker=False,
            model_config=types.SimpleNamespace(
                requires_mm_token_modalities=False, model_is_mrope=False
            ),
            kv_index_translator=MagicMock(),
            prefill_attention_backend_str="torch_native",
            ngram_embedding_manager=types.SimpleNamespace(enabled=False),
            lora_manager=None,
            attn_backend=MagicMock(),
            attn_tp_sequence_sharded=lambda _: False,
        )
        for algorithm, (output_len, end, mixed, zero_tokens, completing) in product(
            (
                SpeculativeAlgorithm.NONE,
                SpeculativeAlgorithm.EAGLE,
                SpeculativeAlgorithm.EAGLE3,
            ),
            (
                (0, 4, False, False, False),
                (0, 8, False, False, True),
                (0, 4, True, False, False),
                (0, 8, True, False, True),
                (2, 8, False, False, False),
                (2, 10, False, False, True),
                (0, 4, False, True, False),
                (0, 8, False, True, True),
                (0, 4, True, True, False),
                (0, 8, True, True, True),
            ),
        ):
            with self.subTest(
                algorithm=algorithm,
                output_len=output_len,
                end=end,
                mixed=mixed,
                zero_tokens=zero_tokens,
            ):
                incoming = Req(
                    "incoming",
                    "",
                    array("q", range(8)),
                    SamplingParams(max_new_tokens=0 if zero_tokens else 16),
                )
                incoming.sampling_params.normalize(None)
                incoming.output_ids.extend(range(output_len))
                incoming.refresh_fill_ids()
                incoming.extend_end = end
                running = Req("running", "", array("q", range(8)), SamplingParams())
                running.output_ids.extend([0, 1])
                running.refresh_fill_ids()
                running.extend_end = 10
                reqs = [incoming, running] if mixed else [incoming]
                prefix = [2, 9][: len(reqs)]
                lengths = [req.extend_end - start for req, start in zip(reqs, prefix)]
                seq_lens = torch.tensor([req.extend_end for req in reqs])
                batch = ScheduleBatch(
                    reqs=reqs,
                    num_decode_rows=1 if mixed else 0,
                    device="cpu",
                    model_config=types.SimpleNamespace(
                        is_encoder_decoder=False, is_matryoshka=False
                    ),
                    spec_algorithm=algorithm,
                    forward_mode=ForwardMode.MIXED if mixed else ForwardMode.EXTEND,
                    input_ids=torch.zeros(sum(lengths), dtype=torch.int64),
                    req_pool_indices=torch.arange(len(reqs)),
                    req_pool_indices_cpu=torch.arange(len(reqs)),
                    seq_lens=seq_lens,
                    seq_lens_cpu=seq_lens,
                    seq_lens_sum=int(seq_lens.sum()),
                    orig_seq_lens=torch.tensor(
                        [len(req.full_untruncated_fill_ids) for req in reqs]
                    ),
                    out_cache_loc=torch.arange(sum(lengths)),
                    prefix_lens=prefix,
                    extend_lens=lengths,
                    extend_num_tokens=sum(lengths),
                    extend_logprob_start_lens=[0] * len(reqs),
                )
                with get_context().override_server_args(device="cpu"):
                    forward = ForwardBatch.init_new(
                        batch,
                        runner,
                        capture_hidden_mode=CaptureHiddenMode.NULL,
                        return_hidden_states_before_norm=False,
                    )
                output_starts = [len(req.output_ids) for req in reqs]
                if completing:
                    incoming.output_ids.append(100)
                    incoming.update_finish_state()
                if mixed:
                    running.output_ids.append(101)
                commits = SchedulerBatchResultProcessor._build_auxiliary_commits(
                    batch, output_starts
                )
                if completing and zero_tokens:
                    self.assertTrue(incoming.finished())
                    self.assertEqual(incoming.finished_len, 0)
                    self.assertIsNone(commits[0])
                required_rows = [
                    i
                    for i, (req, commit) in enumerate(zip(reqs, commits, strict=True))
                    if (commit is not None and commit.output_index == 0)
                    or (req.finished() and req.finished_len == 0)
                ]
                self.assertEqual(bool(required_rows), completing and output_len == 0)
                for i in required_rows:
                    self.assertTrue(forward.contains_last_prefill_chunk)
                    self.assertEqual(
                        int(forward.extend_prefix_lens[i] + forward.extend_seq_lens[i]),
                        int(forward.orig_seq_lens[i]),
                    )
                self.assertEqual(forward.contains_last_prefill_chunk, completing)
                incoming.extend_end = 4 if completing else 8 + output_len
                self.assertEqual(forward.contains_last_prefill_chunk, completing)
                self.assertEqual(
                    dataclasses.replace(forward).contains_last_prefill_chunk, completing
                )


class TestMixDecodeRows(unittest.TestCase):
    def test_decode_rows_follow_the_prefill_rows(self):
        """A mixed step lists its prefill rows, then each decoding request as a
        one-token extend whose prefix is its row length before this token."""
        extend_step = make_schedule_batch(
            2,
            reqs=[types.SimpleNamespace(rid="e1"), types.SimpleNamespace(rid="e2")],
            model_config=types.SimpleNamespace(is_encoder_decoder=False, vocab_size=8),
            sampling_info=MagicMock(),
            return_logprob=False,
            forward_mode=ForwardMode.EXTEND,
            enable_overlap=False,
            is_prefill_only=True,
            multimodal_inputs=None,
            out_cache_loc=torch.arange(6, dtype=torch.int64),
            prefix_lens=[0, 0],
            extend_lens=[3, 3],
            extend_num_tokens=6,
            extend_logprob_start_lens=[0, 0],
        )
        decode_step = make_schedule_batch(
            1,
            reqs=[types.SimpleNamespace(rid="r1")],
            forward_mode=ForwardMode.DECODE,
            out_cache_loc=torch.arange(6, 7, dtype=torch.int64),
            # The decode build counts this step's token in the row length.
            seq_lens_cpu=torch.tensor([6], dtype=torch.int64),
        )

        extend_step.mix_decode_rows(decode_step)

        self.assertEqual(extend_step.forward_mode, ForwardMode.MIXED)
        self.assertIs(extend_step.mix_running_indices, decode_step.req_pool_indices)
        self.assertEqual([r.rid for r in extend_step.reqs], ["e1", "e2", "r1"])
        self.assertEqual(extend_step.num_decode_rows, 1)
        self.assertEqual([r.rid for r in extend_step.prefill_reqs()], ["e1", "e2"])
        self.assertTrue(
            torch.equal(extend_step.out_cache_loc, torch.arange(7, dtype=torch.int64))
        )
        self.assertEqual(extend_step.prefix_lens, [0, 0, 5])
        self.assertEqual(extend_step.extend_lens, [3, 3, 1])
        self.assertEqual(extend_step.extend_num_tokens, 7)
        self.assertEqual(extend_step.extend_logprob_start_lens, [0, 0, 0])
        self.assertFalse(extend_step.is_prefill_only)


class TestPrepareEncoderInfoExtendOutOfPlace(unittest.TestCase):
    def test_prepare_encoder_info_extend_rebinds_lens_without_mutating_old_lists(self):
        """prepare_encoder_info_extend must strip encoder tokens via rebound lists; old list objects stay intact."""
        req_with_image = types.SimpleNamespace(
            rid="img",
            multimodal_inputs=types.SimpleNamespace(num_image_tokens=2),
            prefix_len=0,
            extend_len=5,
            logprob_start_len=0,
        )
        req_text_only = types.SimpleNamespace(
            rid="txt",
            multimodal_inputs=None,
            prefix_len=0,
            extend_len=4,
            logprob_start_len=0,
        )
        batch = make_schedule_batch(
            2,
            reqs=[req_with_image, req_text_only],
            device="cpu",
            forward_mode=ForwardMode.EXTEND,
            out_cache_loc=torch.arange(9, dtype=torch.int64),
            prefix_lens=[0, 0],
            extend_lens=[5, 4],
            extend_num_tokens=9,
            extend_logprob_start_lens=[0, 0],
            extend_input_logprob_token_ids=torch.arange(9, dtype=torch.int64),
        )

        prefix_before = batch.prefix_lens
        extend_before = batch.extend_lens
        logprob_start_before = batch.extend_logprob_start_lens
        snapshot = _snapshot_mutable_fields(batch)

        batch.prepare_encoder_info_extend(
            input_ids=[array("q", range(5)), array("q", range(4))],
            seq_lens=[5, 4],
        )

        self.assertEqual(batch.encoder_lens_cpu, [2, 0])
        self.assertEqual(batch.encoder_cached, [False, True])
        self.assertEqual(batch.extend_lens, [3, 4])
        self.assertEqual(batch.prefix_lens, [0, 0])
        self.assertEqual(batch.extend_num_tokens, 7)
        self.assertTrue(
            torch.equal(batch.out_cache_loc, torch.arange(2, 9, dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(batch.encoder_out_cache_loc, torch.arange(2, dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(batch.seq_lens_cpu, torch.tensor([3, 4], dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(
                batch.extend_input_logprob_token_ids,
                torch.arange(2, 9, dtype=torch.int64),
            )
        )
        self.assertEqual(batch.extend_logprob_start_lens, [0, 0])
        self.assertIsNot(batch.prefix_lens, prefix_before)
        self.assertIsNot(batch.extend_lens, extend_before)
        self.assertIsNot(batch.extend_logprob_start_lens, logprob_start_before)
        _assert_snapshot_not_mutated(self, snapshot)


if __name__ == "__main__":
    unittest.main()
