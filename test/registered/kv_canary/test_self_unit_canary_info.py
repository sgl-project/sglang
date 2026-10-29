from __future__ import annotations

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import sglang.srt.kv_canary.perturb.detokenizer_swap as detokenizer_swap
from sglang.kernels.ops.kv_canary.consts import (
    RealKvHashMode,
    canary_chain_extend_hash,
    canary_chain_initial_hash,
    sampling_digest_from_values,
    splitmix64_mix3,
    to_signed_int64,
)
from sglang.kernels.ops.kv_canary.verify import CanaryLaunchTag, VerifyOrWriteContext
from sglang.kernels.ops.kv_canary.verify_ref import compute_slot_hash
from sglang.kernels.ops.kv_canary.write_ref import (
    launch_canary_write_kernel_torch_reference,
)
from sglang.srt.environ import envs
from sglang.srt.kv_canary.buffer_group import PoolKind
from sglang.srt.kv_canary.canary_info.aux_output import (
    CANARY_INFOS_KEY,
    CanaryInfoHostOutput,
)
from sglang.srt.kv_canary.canary_info.checker import (
    CANARY_INFO_MISMATCH_CODE,
    CanaryInfoChecker,
    CanaryInfoMismatch,
)
from sglang.srt.kv_canary.canary_info.digest import (
    sampling_digest,
)
from sglang.srt.kv_canary.canary_info.tokenizer_manager_hook import (
    CanaryInfoTokenizerManagerHook,
    ReqStateCanaryInfo,
    _build_canary_info_checker,
    _check_canary_infos,
)
from sglang.srt.kv_canary.config import CanaryConfig, CanaryMode
from sglang.srt.kv_canary.runner.canary_manager import CanaryManager
from sglang.srt.managers.auxiliary_output import CommittedTokens
from sglang.srt.managers.io_struct import (
    TokenizedGenerateReqInput,
    unwrap_from_pickle,
    wrap_as_pickle,
)
from sglang.srt.managers.tokenizer_manager import ReqState
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import TOP_K_ALL, SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kernels.kv_canary._canary_helpers import (
    FakeViolationLog,
    make_canary_buf,
    make_write_plan,
)
from sglang.test.kv_canary.canary_info_utils import reference_canary_infos
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


_CPU = torch.device("cpu")


class TestCanaryInfoConfiguration(CustomTestCase):
    def test_unsupported_execution_paths_rejected_only_when_info_active(self) -> None:
        """Reject paths without INFO transport while preserving base canary modes."""
        for overrides in (
            {"pp_size": 2},
            {"disaggregation_mode": "prefill"},
            {"disaggregation_mode": "decode"},
            {"allow_auto_truncate": True},
            {"dllm_algorithm": "LowConfidence"},
            {"speculative_algorithm": "UNO"},
            {"speculative_algorithm": "NGRAM"},
            {"speculative_algorithm": "DFLASH"},
            {"speculative_algorithm": "DSPARK"},
            {"speculative_algorithm": "FROZEN_KV_MTP"},
        ):
            for enabled, mode in (
                (True, "raise"),
                (True, "log"),
                (False, "raise"),
                (True, "none"),
            ):
                with self.subTest(overrides=overrides, enabled=enabled, mode=mode):
                    with (
                        envs.SGLANG_KV_CANARY_ENABLE_INFO.override(enabled),
                        get_context().override_server_args(
                            model_path="dummy",
                            kv_canary=mode,
                            speculative_eagle_topk=1,
                            speculative_num_steps=3,
                            speculative_num_draft_tokens=4,
                            **overrides,
                        ),
                    ):
                        if enabled and mode != "none":
                            with self.assertRaisesRegex(AssertionError, "KvCanaryInfo"):
                                CanaryConfig.from_env()
                        else:
                            self.assertEqual(
                                CanaryConfig.from_env().enable_info, enabled
                            )

    def test_info_requires_linear_eagle_only_when_active(self) -> None:
        """Require a single chain with matching widths only when INFO is active."""
        for enabled, mode, topk, num_draft_tokens, rejected in (
            (True, "raise", 2, 4, True),
            (True, "raise", 1, 4, False),
            (True, "raise", 1, 5, True),
            (False, "raise", 2, 5, False),
            (True, "none", 2, 5, False),
        ):
            with self.subTest(
                enabled=enabled, mode=mode, topk=topk, num_draft_tokens=num_draft_tokens
            ):
                with (
                    envs.SGLANG_KV_CANARY_ENABLE_INFO.override(enabled),
                    get_context().override_server_args(
                        model_path="dummy",
                        kv_canary=mode,
                        speculative_algorithm="EAGLE",
                        speculative_eagle_topk=topk,
                        speculative_num_steps=3,
                        speculative_num_draft_tokens=num_draft_tokens,
                    ),
                ):
                    if rejected:
                        with self.assertRaisesRegex(AssertionError, "KvCanaryInfo"):
                            CanaryConfig.from_env()
                    else:
                        assert CanaryConfig.from_env().enable_info == enabled

    def test_info_uses_resolved_speculative_defaults(self) -> None:
        """Accept resolved EAGLE settings while raw speculative fields stay unset."""
        with (
            envs.SGLANG_KV_CANARY_ENABLE_INFO.override(True),
            get_context().override_server_args(
                kv_canary="raise",
                speculative_algorithm="EAGLE",
                speculative_eagle_topk=1,
                speculative_num_steps=3,
                speculative_num_draft_tokens=4,
            ) as args,
        ):
            self.assertIsNone(args.speculative_algorithm)
            self.assertIsNone(args.speculative_eagle_topk)
            self.assertIsNone(args.speculative_num_steps)
            self.assertIsNone(args.speculative_num_draft_tokens)
            self.assertTrue(CanaryConfig.from_env().enable_info)

    def test_detokenizer_swap_preserves_chunk_lengths(self) -> None:
        """Swap one token and its info without changing either chunk length."""
        with (
            patch.object(detokenizer_swap, "_delivered_counts", {"a": 16, "b": 17}),
            patch.object(detokenizer_swap, "_swapped_rids", set()),
        ):
            output = SimpleNamespace(
                rids=["a", "b"],
                output_ids=[array("q", [1, 2, 3]), array("q", [4, 5])],
                customized_info=wrap_as_pickle(
                    {"canary_infos": [[10, 20, 30], [40, 50]]}
                ),
            )

            detokenizer_swap._swap_due_pairs(recv_obj=output, swap_at_output=16)

            assert [list(row) for row in output.output_ids] == [[4, 2, 3], [1, 5]]
            assert unwrap_from_pickle(output.customized_info)["canary_infos"] == [
                [40, 20, 30],
                [10, 50],
            ]

    def test_info_hook_requires_opt_in_and_active_canary(self) -> None:
        """Info checking requires both the opt-in flag and an active canary mode."""
        for enabled in (False, True):
            for mode in CanaryMode:
                with self.subTest(enabled=enabled, mode=mode):
                    with (
                        envs.SGLANG_KV_CANARY_ENABLE_INFO.override(enabled),
                        patch(
                            "sglang.srt.kv_canary.canary_info.tokenizer_manager_hook.kv_canary_mode",
                            return_value=mode,
                        ),
                    ):
                        hook = CanaryInfoTokenizerManagerHook.maybe_create()
                    assert (hook is not None) == (
                        enabled and mode is not CanaryMode.NONE
                    )


_GOLDEN_CASES: tuple[tuple[list[int], int, dict], ...] = (
    (
        [1],
        2,
        dict(temperature=1.0, top_p=1.0, top_k=TOP_K_ALL, min_p=0.0),
    ),
    (
        [101, 102, 103],
        7,
        dict(temperature=0.7, top_p=0.9, top_k=50, min_p=0.05),
    ),
    (
        [5, 6, 7, 8, 9],
        151643,
        dict(temperature=1.0, top_p=1.0, top_k=1, min_p=0.0, sampling_seed=1234),
    ),
    (
        [42] * 16,
        0,
        dict(temperature=1.3, top_p=0.5, top_k=20, min_p=0.0, sampling_seed=0),
    ),
    (
        list(range(1000, 1033)),
        99999,
        dict(temperature=0.2, top_p=1.0, top_k=TOP_K_ALL, min_p=0.1),
    ),
)


def _golden_canary_info(*, prompt: list[int], token: int, params_kwargs: dict) -> int:
    digest = canary_chain_initial_hash()
    for position, prompt_token in enumerate(prompt):
        digest = canary_chain_extend_hash(digest, prompt_token, position)
    sampling = sampling_digest_from_values(
        temperature=params_kwargs["temperature"],
        top_p=params_kwargs["top_p"],
        top_k=params_kwargs["top_k"],
        min_p=params_kwargs["min_p"],
    )
    return splitmix64_mix3(digest, token, sampling)


def _make_checker(prompt: list[int], params: SamplingParams) -> CanaryInfoChecker:
    return CanaryInfoChecker(rid="r0", sampling_params=params, prompt_ids=prompt)


class TestCanaryInfoDigest(CustomTestCase):
    def test_prefix_digest_matches_write_ref_chain(self) -> None:
        tokens = [11, 22, 33, 44, 55, 66, 77]
        slots = [3, 9, 1, 14, 6, 2, 12]
        buf = make_canary_buf(num_slots=16, slot_stride_bytes=32, device=_CPU)
        log = FakeViolationLog.allocate(device=_CPU)
        launch_canary_write_kernel_torch_reference(
            context=VerifyOrWriteContext(
                canary_buf=buf,
                kernel_kind=CanaryLaunchTag.TAIL_K_FULL,
                violation_ring=log.ring,
                violation_write_index=log.write_index,
                slot_run_counter=log.slot_run_counter,
                kernel_run_counter=log.kernel_run_counter,
                enable_chain_position_assert=log.enable_chain_position_assert,
                real_kv_sources=(),
                real_kv_hash_mode=RealKvHashMode.NONE,
            ),
            plan=make_write_plan(
                write_offsets=[0, len(tokens)],
                seed_slot_indices=[-1],
                num_valid_reqs=1,
                device=_CPU,
            ),
            input_ids=torch.tensor(tokens, dtype=torch.int64),
            positions=torch.arange(len(tokens), dtype=torch.int64),
            out_cache_loc=torch.tensor(slots, dtype=torch.int64),
            enable_write_input_assert=False,
            expected_input_tokens=None,
            expected_input_positions=None,
        )

        buf_i64 = buf.view(torch.int64)
        digest = canary_chain_initial_hash()
        self.assertEqual(compute_slot_hash(buf_i64, -1), digest)
        for position, (token, slot) in enumerate(zip(tokens, slots)):
            digest = canary_chain_extend_hash(digest, token, position)
            self.assertEqual(compute_slot_hash(buf_i64, slot), digest)

    def test_golden_vectors(self) -> None:
        for prompt, token, params_kwargs in _GOLDEN_CASES:
            with self.subTest(prompt_len=len(prompt), token=token):
                checker = _make_checker(prompt, SamplingParams(**params_kwargs))
                expected = _golden_canary_info(
                    prompt=prompt,
                    token=token,
                    params_kwargs=params_kwargs,
                )
                self.assertEqual(checker.expected(token), expected)

    def test_sampling_digest_ignores_seed(self) -> None:
        """Keep CanaryInfo unchanged when only the sampling seed differs."""
        unseeded = SamplingParams(temperature=0.8, top_p=0.9)
        seeded = SamplingParams(temperature=0.8, top_p=0.9, sampling_seed=1234)
        self.assertEqual(
            sampling_digest(unseeded),
            sampling_digest(seeded),
        )
        self.assertEqual(
            sampling_digest(seeded),
            sampling_digest_from_values(
                temperature=0.8,
                top_p=0.9,
                top_k=seeded.top_k,
                min_p=seeded.min_p,
            ),
        )

    def test_sampling_digest_matches_values_and_covers_every_field(self) -> None:
        params = SamplingParams(
            temperature=0.6, top_p=0.95, top_k=40, min_p=0.02, sampling_seed=99
        )
        values = dict(
            temperature=0.6,
            top_p=0.95,
            top_k=40,
            min_p=0.02,
        )
        base = sampling_digest_from_values(**values)
        self.assertEqual(sampling_digest(params), base)
        for changed in (
            dict(temperature=0.61),
            dict(top_p=0.94),
            dict(top_k=41),
            dict(top_k=-1),
            dict(min_p=0.03),
        ):
            with self.subTest(changed=changed):
                self.assertNotEqual(
                    sampling_digest_from_values(**(values | changed)), base
                )


class TestCanaryInfoChecker(CustomTestCase):
    def setUp(self) -> None:
        self.prompt = [7, 8, 9, 10]
        self.outputs = [100, 101, 102, 103, 104]
        self.params = SamplingParams(temperature=0.9, top_k=30)
        self.canary_infos = reference_canary_infos(
            prompt=self.prompt,
            outputs=self.outputs,
            params=self.params,
        )

    def test_checker_rejects_empty_prompt(self) -> None:
        with self.assertRaises(ValueError):
            _make_checker([], self.params)

    def test_checker_accepts_sequence_from_reference(self) -> None:
        checker = _make_checker(self.prompt, self.params)
        for token, info in zip(self.outputs, self.canary_infos):
            self.assertIsNone(checker.advance(token, info))
        self.assertEqual(checker.position, len(self.prompt) + len(self.outputs))

    def test_checker_accepts_whole_chunk(self) -> None:
        checker = _make_checker(self.prompt, self.params)
        self.assertIsNone(checker.advance_many(self.outputs, self.canary_infos))
        self.assertEqual(checker.position, len(self.prompt) + len(self.outputs))

    def test_checker_accepts_signed_canary_info(self) -> None:
        checker = _make_checker(self.prompt, self.params)
        signed = [to_signed_int64(info) for info in self.canary_infos]
        self.assertIsNone(checker.advance_many(self.outputs, signed))

    def test_checker_rejects_wrong_token(self) -> None:
        checker = _make_checker(self.prompt, self.params)
        self.assertIsNone(checker.advance(self.outputs[0], self.canary_infos[0]))
        mismatch = checker.advance(self.outputs[1] + 1, self.canary_infos[1])
        self.assertIsNotNone(mismatch)
        self.assertEqual(mismatch.position, len(self.prompt) + 1)
        self.assertEqual(mismatch.token_id, self.outputs[1] + 1)
        self.assertEqual(mismatch.rid, "r0")

    def test_checker_rejects_reordered_or_dropped_token(self) -> None:
        checker = _make_checker(self.prompt, self.params)
        self.assertIsNone(checker.advance(self.outputs[0], self.canary_infos[0]))
        mismatch = checker.advance(self.outputs[2], self.canary_infos[2])
        self.assertEqual(mismatch.position, len(self.prompt) + 1)
        self.assertEqual(mismatch.actual, self.canary_infos[2])

    def test_checker_rejects_replayed_token(self) -> None:
        checker = _make_checker(self.prompt, self.params)
        self.assertIsNone(checker.advance_many(self.outputs[:2], self.canary_infos[:2]))
        mismatch = checker.advance(self.outputs[1], self.canary_infos[1])
        self.assertEqual(mismatch.position, len(self.prompt) + 2)

    def test_checker_rejects_missing_value(self) -> None:
        checker = _make_checker(self.prompt, self.params)
        mismatch = checker.advance(self.outputs[0], None)
        self.assertIsNone(mismatch.actual)
        self.assertEqual(mismatch.position, len(self.prompt))

    def test_checker_rejects_other_sampling_params(self) -> None:
        checker = _make_checker(self.prompt, SamplingParams(temperature=0.5))
        self.assertIsNotNone(checker.advance(self.outputs[0], self.canary_infos[0]))

    def test_checker_keeps_accepted_prefix_after_mismatch(self) -> None:
        checker = _make_checker(self.prompt, self.params)
        bad = list(self.canary_infos)
        bad[2] ^= 1
        mismatch = checker.advance_many(self.outputs, bad)
        self.assertEqual(mismatch.position, len(self.prompt) + 2)
        self.assertEqual(checker.position, len(self.prompt) + 2)
        self.assertIsNone(checker.advance_many(self.outputs[2:], self.canary_infos[2:]))

    def test_mismatch_log_round_trips(self) -> None:
        checker = _make_checker(self.prompt, self.params)
        mismatch = checker.advance(self.outputs[0], 0)
        found = CanaryInfoMismatch.find_all(f"prefix {mismatch.format()} suffix")
        self.assertEqual(found, [mismatch])


def _make_state(rid: str, prompt: list[int], params: SamplingParams) -> ReqState:
    state = ReqState([], False, MagicMock(), MagicMock(), MagicMock())
    state.canary_info = ReqStateCanaryInfo(
        checker=_build_canary_info_checker(
            SimpleNamespace(rid=rid, input_ids=prompt, sampling_params=params),
        ),
    )
    return state


def _make_hook(*, raise_on_mismatch: bool = True) -> CanaryInfoTokenizerManagerHook:
    return CanaryInfoTokenizerManagerHook(raise_on_mismatch=raise_on_mismatch)


def _make_recv(*, output_ids: list[list[int]]) -> SimpleNamespace:
    return SimpleNamespace(
        output_ids=[array("q", ids) for ids in output_ids],
        finished_reasons=[None] * len(output_ids),
    )


class TestCanaryInfoHookCheck(CustomTestCase):
    def setUp(self) -> None:
        self.params = SamplingParams(temperature=0.7, top_k=40)
        self.prompt_a = [11, 12, 13]
        self.prompt_b = [21, 22, 23, 24]
        self.outputs_a = [101, 102]
        self.outputs_b = [201, 202]
        self.infos_a = reference_canary_infos(
            prompt=self.prompt_a,
            outputs=self.outputs_a,
            params=self.params,
        )
        self.infos_b = reference_canary_infos(
            prompt=self.prompt_b,
            outputs=self.outputs_b,
            params=self.params,
        )

    def _run(self, states, recv, customized_info, *, raise_on_mismatch=True) -> None:
        for index, state in enumerate(states.values()):
            _make_hook(raise_on_mismatch=raise_on_mismatch).on_batch_output(
                state=state,
                recv_obj=recv,
                customized_info=customized_info,
                index=index,
            )

    def _two_states(self) -> dict[str, ReqState]:
        return {
            "a": _make_state("a", self.prompt_a, self.params),
            "b": _make_state("b", self.prompt_b, self.params),
        }

    def test_matching_chunk_is_delivered_unchanged(self) -> None:
        """Pass matching chunks without modifying their payload."""
        recv = _make_recv(output_ids=[self.outputs_a, self.outputs_b])
        customized_info = {CANARY_INFOS_KEY: [self.infos_a, self.infos_b]}
        self._run(self._two_states(), recv, customized_info)
        assert list(recv.output_ids[0]) == self.outputs_a
        assert recv.finished_reasons == [None, None]

    def test_mismatch_raises_without_mutating_batch_output(self) -> None:
        """Raise before processing further requests or changing received data."""
        states = self._two_states()
        bad_outputs = [self.outputs_a[0], self.outputs_a[1] + 1]
        recv = _make_recv(output_ids=[bad_outputs, self.outputs_b])
        recv.output_strs = ["bad", "other"]
        recv.output_token_logprobs_val = [[-0.1, -0.2], [-0.3, -0.4]]
        customized_info = {CANARY_INFOS_KEY: [self.infos_a, self.infos_b]}
        with self.assertRaisesRegex(RuntimeError, CANARY_INFO_MISMATCH_CODE) as error:
            self._run(states, recv, customized_info)
        mismatch = CanaryInfoMismatch.find_all(str(error.exception))[0]
        assert mismatch.position == len(self.prompt_a) + 1
        assert list(recv.output_ids[0]) == bad_outputs
        assert list(recv.output_ids[1]) == self.outputs_b
        assert recv.output_strs == ["bad", "other"]
        assert recv.output_token_logprobs_val == [[-0.1, -0.2], [-0.3, -0.4]]
        assert customized_info[CANARY_INFOS_KEY] == [self.infos_a, self.infos_b]
        assert recv.finished_reasons == [None, None]
        assert states["b"].canary_info.checker.position == len(self.prompt_b)

    def test_invalid_canary_info_chunks_raise(self) -> None:
        """Reject missing values and unequal lengths, including empty token chunks."""
        for tokens, infos in (
            (self.outputs_a, None),
            (self.outputs_a, [None, None]),
            (self.outputs_a, self.infos_a[:1]),
            (self.outputs_a[:1], self.infos_a),
            ([], self.infos_a),
        ):
            states = {"a": _make_state("a", self.prompt_a, self.params)}
            recv = _make_recv(output_ids=[tokens])
            customized_info = None if infos is None else {CANARY_INFOS_KEY: [infos]}
            with self.assertRaisesRegex(
                RuntimeError, CANARY_INFO_MISMATCH_CODE
            ) as error:
                self._run(states, recv, customized_info)
            mismatch = CanaryInfoMismatch.find_all(str(error.exception))[0]
            assert states["a"].canary_info.checker.position == len(self.prompt_a)
            if not tokens:
                assert mismatch.token_id is None
                assert mismatch.expected is None

    def test_checker_spans_multiple_chunks(self) -> None:
        """Preserve the validated prefix between streaming chunks."""
        states = {"a": _make_state("a", self.prompt_a, self.params)}
        for token, info in zip(self.outputs_a, self.infos_a, strict=True):
            recv = _make_recv(output_ids=[[token]])
            self._run(states, recv, {CANARY_INFOS_KEY: [[info]]})
            assert list(recv.output_ids[0]) == [token]
            assert recv.finished_reasons[0] is None
        assert states["a"].canary_info.checker.position == len(self.prompt_a) + len(
            self.outputs_a
        )

    def test_log_mode_records_mismatch_but_delivers_unchanged(self) -> None:
        """Check each chunk in log mode while preserving the received payload."""
        states = self._two_states()
        bad_outputs = [self.outputs_a[0], self.outputs_a[1] + 1]
        recv = _make_recv(output_ids=[bad_outputs, self.outputs_b])
        customized_info = {CANARY_INFOS_KEY: [self.infos_a, self.infos_b]}
        self._run(states, recv, customized_info, raise_on_mismatch=False)
        assert list(recv.output_ids[0]) == bad_outputs
        assert recv.finished_reasons == [None, None]
        mismatch = states["a"].canary_info.mismatch
        assert mismatch.position == len(self.prompt_a) + 1
        assert states["b"].canary_info.mismatch is None
        later = _make_recv(output_ids=[[7]])
        self._run({"a": states["a"]}, later, None, raise_on_mismatch=False)
        assert list(later.output_ids[0]) == [7]
        later_mismatch = states["a"].canary_info.mismatch
        assert later_mismatch is not mismatch
        assert later_mismatch.token_id == 7
        assert later_mismatch.position == mismatch.position

    def test_request_without_checker_is_rejected(self) -> None:
        """Reject output for a request that lacks its checker state."""
        state = ReqState([], False, MagicMock(), MagicMock(), MagicMock())
        recv = _make_recv(output_ids=[self.outputs_a])
        with self.assertRaises(AssertionError):
            self._run({"a": state}, recv, None)

    def test_empty_chunk_passes(self) -> None:
        """Allow empty token chunks when no CanaryInfo values are present."""
        checker = _make_state("a", self.prompt_a, self.params).canary_info.checker
        for infos in (None, []):
            assert (
                _check_canary_infos(checker=checker, token_ids=[], canary_infos=infos)
                is None
            )


class TestCanaryInfoHookSampling(CustomTestCase):
    def _dispatch(self, *, params: SamplingParams, overrides: dict | None = None):
        state = ReqState([], False, MagicMock(), MagicMock(), MagicMock())
        tokenized = MagicMock(
            spec=TokenizedGenerateReqInput,
            rid="a",
            input_ids=[11, 12, 13],
            sampling_params=params,
            mm_inputs=None,
            need_wait_for_mm_inputs=False,
            session_params=None,
            input_embeds=None,
        )
        for name, value in (overrides or {}).items():
            setattr(tokenized, name, value)

        _make_hook().on_request_dispatched(
            rid_to_state={"a": state}, tokenized_obj=tokenized
        )
        return state.canary_info.checker

    def test_unreplayable_prompts_rejected_before_checker_creation(self) -> None:
        """Reject rewritten or absent token prompts before scheduler dispatch."""
        for overrides in (
            {"mm_inputs": SimpleNamespace()},
            {"need_wait_for_mm_inputs": True},
            {"session_params": SimpleNamespace(id="session")},
            {"input_embeds": [[0.1]], "input_ids": None},
            {"input_embeds": [[0.1]], "input_ids": [1]},
            {"input_ids": None},
            {"input_ids": []},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, "KvCanaryInfo requires"):
                    self._dispatch(params=SamplingParams(), overrides=overrides)

    def test_hook_ignores_seed(self) -> None:
        """Check a seeded request against the same unseeded CanaryInfo stream."""
        params = SamplingParams(temperature=0.7, sampling_seed=1234)
        checker = self._dispatch(params=params)
        expected = reference_canary_infos(
            prompt=[11, 12, 13],
            outputs=[101],
            params=SamplingParams(temperature=0.7),
        )
        self.assertIsNone(checker.advance(101, expected[0]))

    def test_beam_search_rejected_before_dispatch(self) -> None:
        """Reject beam leaders whose output ids are length placeholders."""
        with self.assertRaisesRegex(ValueError, "does not support beam search"):
            self._dispatch(params=SamplingParams(beam_width=2))

    def test_single_beam_retains_normal_info_checking(self) -> None:
        """A single beam remains an ordinary checked generation request."""
        self.assertIsNotNone(self._dispatch(params=SamplingParams(beam_width=1)))


class TestCanaryInfoManager(CustomTestCase):
    def test_missing_full_pool_fails_before_device_state_allocation(self) -> None:
        """Reject INFO when the target pool cannot supply a full prompt chain."""
        with self.assertRaisesRegex(
            AssertionError, "KvCanaryInfo requires a FULL pool"
        ):
            CanaryManager(
                config=SimpleNamespace(enable_info=True),
                perturb_config=None,
                buffer_groups=(SimpleNamespace(kind=PoolKind.SWA),),
                device=_CPU,
                req_to_token_pool=None,
                launch_capacities=None,
            )

    def test_verify_info_excludes_dp_padding_slots(self) -> None:
        """Only real verify rows are grouped into per-request INFO outputs."""
        manager = CanaryManager.__new__(CanaryManager)
        manager.config = SimpleNamespace(enable_info=True)
        manager._canary_info_buf = torch.zeros(1)
        manager._is_eagle_draft_decode = False
        sampling_info = SimpleNamespace(
            temperatures=torch.ones(2),
            top_ps=torch.ones(2),
            top_ks=torch.ones(2),
            min_ps=torch.zeros(2),
        )
        for padding in (0, 3):
            with self.subTest(padding=padding):
                batch = SimpleNamespace(
                    out_cache_loc=torch.tensor([4, 5, 6, 7] + [0] * padding),
                    sampling_info=sampling_info,
                )
                logits = SimpleNamespace(auxiliary_device_output=None)
                with patch(
                    "sglang.srt.kv_canary.runner.canary_manager.launch_canary_info_kernel",
                    side_effect=lambda **kwargs: kwargs["out"].copy_(
                        kwargs["out_cache_loc"]
                    ),
                ):
                    manager.on_verify_accept(
                        logits_output=logits,
                        verify_forward_batch=batch,
                        predict=torch.tensor([11, 12, 13, 14]),
                        accept_lens=torch.tensor([2, 1]),
                        num_draft_tokens=2,
                    )
                self.assertEqual(
                    logits.auxiliary_device_output.values.tolist(), [[4, 5], [6, 7]]
                )


class TestCanaryInfoHostOutput(CustomTestCase):
    def _req(self) -> SimpleNamespace:
        return SimpleNamespace(customized_info=None)

    def test_consume_records_values_for_committed_tokens(self) -> None:
        """Record committed rows and skip requests without a commit."""
        reqs = [self._req(), self._req()]
        output = CanaryInfoHostOutput(values=torch.tensor([[7], [9]]))
        output.consume(
            SimpleNamespace(reqs=reqs),
            [CommittedTokens(output_index=0, token_ids=(1,)), None],
        )
        self.assertEqual(reqs[0].customized_info[CANARY_INFOS_KEY], [7])
        self.assertIsNone(reqs[1].customized_info)

    def test_consume_truncates_speculative_rows_to_committed_tokens(self) -> None:
        req = self._req()
        req.customized_info = {CANARY_INFOS_KEY: [5]}
        output = CanaryInfoHostOutput(values=torch.tensor([[11, 12, 13]]))
        output.consume(
            SimpleNamespace(reqs=[req]),
            [CommittedTokens(output_index=1, token_ids=(4, 5))],
        )
        self.assertEqual(req.customized_info[CANARY_INFOS_KEY], [5, 11, 12])

    def test_consume_rejects_gap_before_committed_tokens(self) -> None:
        """Reject a gap in the recorded output sequence."""
        req = self._req()
        output = CanaryInfoHostOutput(values=torch.tensor([[3]]))
        with self.assertRaises(AssertionError):
            output.consume(
                SimpleNamespace(reqs=[req]),
                [CommittedTokens(output_index=2, token_ids=(9,))],
            )


if __name__ == "__main__":
    unittest.main()
