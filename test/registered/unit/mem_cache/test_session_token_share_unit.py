"""Unit tests for the fill_ids reconstruction-overhead reduction (PR #27965).

Correctness of the streaming-session in-place token-array share protocol
(`Session.create_req` / `finish_req` / `abort_req`):

- token arrays are extended in place and shared across turns (no per-turn copy);
- committed_* lengths recorded at finish_req trim away tokens appended by a
  turn that aborted before finishing (mid-turn and first-turn aborts);
- max_new_tokens overshoot falls back to a fill_ids rebuild instead of
  carrying an inconsistent array.

Performance regression guards for the same feature. These pin the *allocation
behavior* of the fill_ids path, which a correctness test cannot: the optimized
and the naive forms emit byte-identical token sequences, so only object
identity / no-copy assertions tell them apart. Each fails (red) if its
mechanism is reverted to the O(context)-per-step form:

- `Req._refresh_fill_ids` extends one array in place across decode steps; the
  old `full_untruncated_fill_ids = origin_input_ids + output_ids` rebuilt (and
  decref'd) the whole array every step -> a new object each call;
- `RadixKey(token_ids, limit=...)` caps the logical length by reference; the
  old call site sliced `full_untruncated_fill_ids[:max_prefix_len]`, copying
  O(context) tokens for every prefill-batch build.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import unittest
from array import array
from types import SimpleNamespace

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import Session
from sglang.test.test_utils import CustomTestCase

VOCAB = 1 << 20


def _recv(rid, input_ids, max_new_tokens=8):
    return SimpleNamespace(
        rid=rid,
        input_ids=array("q", input_ids),
        mm_inputs=None,
        session_params=SimpleNamespace(
            id="s", rid=None, offset=None, replace=False, drop_previous_output=False
        ),
        sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
        lora_id=None,
        custom_logit_processor=None,
        stream=False,
        return_logprob=False,
        top_logprobs_num=0,
        token_ids_logprob=None,
        require_reasoning=False,
        return_hidden_states=False,
        return_routed_experts=False,
        routed_experts_start_len=0,
        priority=None,
        routing_key=None,
        extra_key=None,
        http_worker_ipc=None,
        time_stats=None,
    )


class TestSessionTokenShare(CustomTestCase):

    def setUp(self):
        self.session = Session(capacity_of_str_len=0, session_id="s", streaming=True)

    def _create(self, rid, input_ids, max_new_tokens=8):
        return self.session.create_req(
            _recv(rid, input_ids, max_new_tokens=max_new_tokens),
            tokenizer=None,
            vocab_size=VOCAB,
        )

    def _decode_and_finish(self, req, output, baked=None):
        """Simulate decode then a successful finish.

        `baked` output tokens are folded into the fill array before the rest
        arrive (mix_with_running refreshes mid-decode, so the bake is often
        partial).
        """
        if baked is None:
            baked = len(output)
        req.output_ids.extend(output[:baked])
        req._refresh_fill_ids()
        req.output_ids.extend(output[baked:])
        self.session.finish_req(req)

    def test_normal_multi_turn_share_and_carry(self):
        in1, out1 = list(range(100, 110)), [1, 2, 3]
        r1 = self._create("r1", in1)
        self.assertEqual(list(r1.origin_input_ids), in1)
        self._decode_and_finish(r1, out1, baked=2)  # partial bake
        self.assertEqual(self.session.committed_origin_len, len(in1))
        self.assertEqual(self.session.committed_fill_len, len(in1) + 2)

        in2, out2 = [7, 8], [4, 5]
        r2 = self._create("r2", in2)
        # In-place share: same objects, extended to the new prompt.
        self.assertIs(r2.origin_input_ids, r1.origin_input_ids)
        self.assertEqual(list(r2.origin_input_ids), in1 + out1 + in2)
        # Carry: the fill array handed over and equal to the new origin.
        self.assertIs(r2.full_untruncated_fill_ids, r1.full_untruncated_fill_ids)
        self.assertEqual(list(r2.full_untruncated_fill_ids), list(r2.origin_input_ids))
        self._decode_and_finish(r2, out2)

        r3 = self._create("r3", [9])
        self.assertEqual(list(r3.origin_input_ids), in1 + out1 + in2 + out2 + [9])
        self.assertEqual(list(r3.full_untruncated_fill_ids), list(r3.origin_input_ids))

    def test_mid_turn_abort_then_continue(self):
        in1, out1 = list(range(200, 210)), [1, 2, 3]
        r1 = self._create("r1", in1)
        self._decode_and_finish(r1, out1)

        # Turn 2 extends the shared arrays, decodes a bit, then aborts:
        # finish_req never runs, req_nodes still points at r1.
        r2 = self._create("r2", [50, 51])
        self.assertEqual(list(r2.origin_input_ids), in1 + out1 + [50, 51])
        r2.output_ids.extend([6, 7])
        r2._refresh_fill_ids()
        self.session.abort_req()
        self.assertEqual(self.session.committed_origin_len, len(in1))

        # Turn 3 must see exactly r1's history — no [50, 51], no doubled out1.
        r3 = self._create("r3", [60])
        self.assertEqual(list(r3.origin_input_ids), in1 + out1 + [60])
        self.assertEqual(list(r3.full_untruncated_fill_ids), list(r3.origin_input_ids))

        # Two aborted attempts in a row heal idempotently.
        self.session.abort_req()
        r4 = self._create("r4", [70])
        self.assertEqual(list(r4.origin_input_ids), in1 + out1 + [70])
        self.assertEqual(list(r4.full_untruncated_fill_ids), list(r4.origin_input_ids))

    def test_first_turn_abort(self):
        self._create("r1", [1, 2, 3])
        self.assertTrue(self.session._inflight)
        self.session.abort_req()
        self.assertFalse(self.session._inflight)
        # No finish_req ran: nothing committed, next turn starts from scratch.
        self.assertIsNone(self.session.committed_origin_len)
        r2 = self._create("r2", [4, 5])
        self.assertEqual(list(r2.origin_input_ids), [4, 5])
        self._decode_and_finish(r2, [9])
        r3 = self._create("r3", [6])
        self.assertEqual(list(r3.origin_input_ids), [4, 5, 9, 6])

    def test_max_new_tokens_overshoot_falls_back(self):
        in1 = list(range(300, 310))
        r1 = self._create("r1", in1, max_new_tokens=4)
        # Spec-decode overshoot: 6 tokens decoded and baked into the fill
        # array, then output trimmed to finished_len (like _trim_overshoot)
        # before finish.
        r1.output_ids.extend([1, 2, 3, 4, 5, 6])
        r1._refresh_fill_ids()
        del r1.output_ids[4:]
        self.session.finish_req(r1)
        self.assertEqual(
            self.session.committed_fill_len, len(in1) + 6
        )  # fill kept the overshoot

        # Next turn: out_tail is output[:max_new]; the carried fill has more
        # baked than out_tail, so the carry is dropped and the fill rebuilds.
        r2 = self._create("r2", [50])
        self.assertEqual(list(r2.origin_input_ids), in1 + [1, 2, 3, 4] + [50])
        self.assertEqual(len(r2.full_untruncated_fill_ids), 0)  # carry skipped
        r2._refresh_fill_ids()
        self.assertEqual(list(r2.full_untruncated_fill_ids), list(r2.origin_input_ids))


def _decode_req(prompt_len):
    return Req(
        rid="r",
        origin_input_text="",
        origin_input_ids=array("q", range(10, 10 + prompt_len)),
        sampling_params=SamplingParams(max_new_tokens=4096),
    )


class TestFillIdsReuseAcrossDecode(CustomTestCase):

    def test_refresh_reuses_one_array_across_decode_steps(self):
        # The headline guard: across a long decode, fill_ids is grown in place,
        # never reallocated. Old code reallocated origin+output every step.
        req = _decode_req(prompt_len=64)
        req._refresh_fill_ids()  # first call materializes the array
        fill = req.full_untruncated_fill_ids

        # Materialized fill must be its own array, not the origin (extending it
        # in place would otherwise corrupt origin_input_ids).
        self.assertIsNot(fill, req.origin_input_ids)
        origin_len = len(req.origin_input_ids)

        for step in range(256):
            req.output_ids.append(50_000 + step)
            req._refresh_fill_ids()
            self.assertIs(
                req.full_untruncated_fill_ids,
                fill,
                msg=f"fill_ids reallocated at decode step {step}",
            )

        # Origin untouched; in-place growth produced the correct sequence.
        self.assertEqual(len(req.origin_input_ids), origin_len)
        self.assertEqual(
            list(req.full_untruncated_fill_ids),
            list(req.origin_input_ids) + list(req.output_ids),
        )

    def test_refresh_appends_only_the_new_tokens(self):
        # A partial bake (mix_with_running refreshes mid-decode) leaves some
        # output already folded in; the next refresh must append only the tail,
        # still in place.
        req = _decode_req(prompt_len=8)
        req.output_ids.extend([1, 2, 3])
        req._refresh_fill_ids()
        fill = req.full_untruncated_fill_ids
        self.assertEqual(list(fill), list(req.origin_input_ids) + [1, 2, 3])

        req.output_ids.extend([4, 5])
        req._refresh_fill_ids()
        self.assertIs(req.full_untruncated_fill_ids, fill)
        self.assertEqual(list(fill), list(req.origin_input_ids) + [1, 2, 3, 4, 5])


class TestRadixKeyLimit(CustomTestCase):

    def test_limit_caps_logical_length_by_reference(self):
        arr = array("q", range(4096))
        key = RadixKey(arr, limit=10)
        # Capping must not copy the (potentially huge) backing array.
        self.assertIs(key.token_ids, arr)
        self.assertEqual(len(key), 10)
        self.assertEqual(list(key), list(range(10)))

    def test_uncapped_raw_token_ids_avoids_copy(self):
        arr = array("q", range(4096))
        # No limit, or a limit at/above the length, returns the array itself.
        self.assertIs(RadixKey(arr).raw_token_ids(), arr)
        self.assertIs(RadixKey(arr, limit=len(arr)).raw_token_ids(), arr)
        self.assertIs(RadixKey(arr, limit=len(arr) + 100).raw_token_ids(), arr)

    def test_capped_raw_token_ids_copies_only_the_cap(self):
        arr = array("q", range(4096))
        raw = RadixKey(arr, limit=10).raw_token_ids()
        self.assertIsNot(raw, arr)
        self.assertEqual(list(raw), list(range(10)))

    def test_match_clamps_to_limit(self):
        # Two keys over identical backing arrays; the capped one must not match
        # past its limit even though the raw tokens agree further.
        arr = array("q", range(4096))
        capped = RadixKey(arr, limit=10)
        full = RadixKey(arr)
        self.assertEqual(capped.match(full), 10)
        self.assertEqual(full.match(capped), 10)


if __name__ == "__main__":
    unittest.main()
