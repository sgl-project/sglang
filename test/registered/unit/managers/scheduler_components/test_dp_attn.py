import unittest
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX  # noqa: E402
from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.scheduler_components import dp_attn  # noqa: E402
from sglang.srt.model_executor.cuda_graph_config import Backend  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.observability.metrics_collector import DPBalanceStats  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestDPAttnSchedulerMetadata(CustomTestCase):
    def test_skip_all_gather_policy(self):
        with envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(False):
            self.assertTrue(dp_attn.should_skip_scheduler_all_gather(num_dp_ranks=1))
            self.assertFalse(dp_attn.should_skip_scheduler_all_gather(num_dp_ranks=2))
        with envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(True):
            self.assertTrue(dp_attn.should_skip_scheduler_all_gather(num_dp_ranks=2))

    def test_dp1_skip_preserves_local_tbo_metadata(self):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            batch_size=lambda: 4,
            spec_info=None,
            reqs=[],
            seq_lens_cpu=torch.ones(4, dtype=torch.int64),
        )
        tbo_preparer = Mock()
        tbo_preparer.prepare_all_gather.return_value = (
            True,
            ForwardMode.DECODE.value,
        )
        tbo_preparer.compute_output.return_value = (2, ForwardMode.DECODE)

        with (
            envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(False),
            patch.object(dp_attn, "TboDPAttentionPreparer", return_value=tbo_preparer),
            patch.object(dp_attn, "world_dp_gather_enabled", return_value=False),
            patch.object(
                dp_attn,
                "get_parallel",
                return_value=SimpleNamespace(
                    num_dp_ranks=1,
                    attn_tp_size=4,
                    attn_cp_size=1,
                    tp_group=SimpleNamespace(
                        device_group=object(), device="cpu", cpu_group=object()
                    ),
                ),
            ),
            patch.object(dp_attn, "check_cuda_graph_backend", return_value=False),
            patch.object(dp_attn.MLPSyncBatchInfo, "all_gather") as all_gather,
        ):
            result = dp_attn.prepare_mlp_sync_batch_raw(
                batch,
                model_runner=SimpleNamespace(
                    prefill_cuda_graph_runner=None,
                    spec_algorithm=SpeculativeAlgorithm.NONE,
                    model_config=object(),
                ),
                get_idle_batch=Mock(
                    side_effect=AssertionError("DP1 must not emit idle batch")
                ),
                disable_cuda_graph=False,
                require_mlp_tp_gather=False,
                disable_overlap_schedule=True,
                offload_tags=set(),
            )

        all_gather.assert_not_called()
        self.assertEqual(result.global_num_tokens, [4])
        self.assertEqual(result.tbo_split_seq_index, 2)
        self.assertEqual(result.global_forward_mode, ForwardMode.DECODE)
        self.assertEqual(result.recv_skipper_forward_mode, ForwardMode.DECODE)
        self.assertEqual(
            tbo_preparer.compute_output.call_args.args[0].tolist(),
            [[1, ForwardMode.DECODE.value]],
        )


class TestDPBalanceStats(CustomTestCase):
    def _prepare_dp2_batch(
        self,
        forward_mode,
        local_tokens: int,
        peer_tokens: int,
        *,
        reqs: Optional[list] = None,
        peer_attention_pairs: Optional[int] = None,
        peer_health_check: bool = False,
        get_idle_batch=None,
        sync_wait_carry=None,
        wait_seconds=None,
    ):
        if reqs is None:
            # One-token rows, the shortest a decode row can be.
            reqs = [
                SimpleNamespace(rid=f"req-{i}", seqlen=1) for i in range(local_tokens)
            ]
        batch = SimpleNamespace(
            forward_mode=forward_mode,
            batch_size=lambda: local_tokens,
            spec_info=None,
            reqs=list(reqs),
            # Spec batches carry no CPU seq_lens at schedule time.
            seq_lens_cpu=None,
            dp_balance_stats=None,
        )
        gathered_mode = forward_mode or ForwardMode.IDLE
        tbo_preparer = Mock()
        tbo_preparer.prepare_all_gather.return_value = (True, gathered_mode.value)
        tbo_preparer.compute_output.return_value = (None, gathered_mode)

        def fake_all_gather_single(output, local, group, **_):
            peer = local.clone()
            peer[0] = peer[1] = peer_tokens  # num_tokens, num_tokens_for_logprob
            peer[9] = int(peer_health_check)
            # attention_pairs; one-token rows unless the test says otherwise.
            peer[10] = (
                peer_tokens if peer_attention_pairs is None else peer_attention_pairs
            )
            output.copy_(torch.stack([local, peer]).flatten())

        # all_gather reads the clock once before and once after the collective.
        clock = (
            patch.object(dp_attn.time, "perf_counter", side_effect=[0.0, wait_seconds])
            if wait_seconds is not None
            else patch.object(
                dp_attn.time, "perf_counter", wraps=dp_attn.time.perf_counter
            )
        )
        with (
            clock,
            envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(False),
            patch.object(dp_attn, "_ENABLE_METRICS_DP_ATTENTION", True),
            patch.object(dp_attn, "TboDPAttentionPreparer", return_value=tbo_preparer),
            patch.object(dp_attn, "world_dp_gather_enabled", return_value=False),
            patch.object(
                dp_attn,
                "get_parallel",
                return_value=SimpleNamespace(
                    num_dp_ranks=2,
                    attn_tp_size=1,
                    attn_cp_size=1,
                    tp_group=SimpleNamespace(
                        device_group=object(),
                        device="cpu",
                        cpu_group=object(),
                        active_ranks_cpu=torch.ones(2, dtype=torch.int64),
                    ),
                ),
            ),
            patch.object(dp_attn, "check_cuda_graph_backend", return_value=False),
            patch.object(
                dp_attn, "all_gather_single", side_effect=fake_all_gather_single
            ),
        ):
            return dp_attn.prepare_mlp_sync_batch_raw(
                batch if forward_mode is not None else None,
                model_runner=SimpleNamespace(
                    prefill_cuda_graph_runner=None,
                    spec_algorithm=SpeculativeAlgorithm.NONE,
                    model_config=object(),
                ),
                get_idle_batch=get_idle_batch
                or Mock(side_effect=AssertionError("local batch must not be replaced")),
                disable_cuda_graph=False,
                require_mlp_tp_gather=True,
                disable_overlap_schedule=True,
                offload_tags=set(),
                sync_wait_carry=sync_wait_carry,
            )

    def test_attached_to_batch_after_dp_gather(self):
        result = self._prepare_dp2_batch(
            ForwardMode.DECODE, local_tokens=4, peer_tokens=8
        )

        self.assertEqual(result.global_num_tokens, [4, 8])
        stats = result.dp_balance_stats
        self.assertEqual(
            (stats.local_tokens, stats.max_tokens, stats.sum_tokens, stats.num_ranks),
            (4, 8, 12, 2),
        )
        self.assertEqual(stats.imbalance_tokens, 4)
        self.assertAlmostEqual(stats.max_over_mean, 8 * 2 / 12)
        self.assertGreater(stats.sync_wait_seconds, 0)

    def test_attention_pairs_follow_context_not_rows(self):
        """Fewer rows with longer contexts is the busier rank in pairs.

        Rows and pairs are gathered side by side, so the two series can
        disagree on which rank is the busiest; this is the case the pair
        series exists for.
        """
        reqs = [SimpleNamespace(rid=f"req-{n}", seqlen=n) for n in (1000, 3000)]
        result = self._prepare_dp2_batch(
            ForwardMode.DECODE,
            local_tokens=2,
            peer_tokens=8,
            reqs=reqs,
            peer_attention_pairs=800,
        )

        stats = result.dp_balance_stats
        self.assertEqual((stats.local_tokens, stats.imbalance_tokens), (2, 6))
        self.assertEqual(
            (stats.local_attention_pairs, stats.max_attention_pairs), (4000, 4000)
        )
        self.assertEqual(stats.imbalance_attention_pairs, 0)
        self.assertAlmostEqual(stats.max_over_mean, 8 * 2 / 10)

    def test_attention_pairs_gathered_without_the_metrics_flag(self):
        """The pair column must not depend on a per-process flag: a rank with
        the flag off would gather zeros and trip the pair-count guard on its
        flag-on peers."""
        with patch.object(dp_attn, "_ENABLE_METRICS_DP_ATTENTION", False):
            info = dp_attn.MLPSyncBatchInfo(
                num_dp_ranks=2,
                tp_size=1,
                cp_size=1,
                num_tokens=2,
                num_tokens_for_logprob=2,
                can_run_decode_cuda_graph=True,
                can_run_draft_cuda_graph=True,
                can_run_prefill_cuda_graph=False,
                is_extend_in_batch=False,
                local_can_run_tbo=False,
                local_forward_mode=ForwardMode.DECODE.value,
                attention_pairs=dp_attn._local_attention_pairs(
                    SimpleNamespace(
                        forward_mode=ForwardMode.DECODE,
                        seq_lens_cpu=None,
                        reqs=[SimpleNamespace(seqlen=5), SimpleNamespace(seqlen=7)],
                    )
                ),
            )
        self.assertEqual(info._get_local_tensor(device="cpu")[10].item(), 12)

    def test_prefill_chunk_pairs_are_causal(self):
        """A chunk after prefix p reads p keys per query plus the causal
        triangle inside the chunk; a one-token chunk after prefix p therefore
        costs the same as a decode row of length p + 1."""
        extend = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND, prefix_lens=[10, 6], extend_lens=[4, 1]
        )
        decode = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            seq_lens_cpu=None,
            reqs=[SimpleNamespace(seqlen=7)],
        )
        self.assertEqual(dp_attn._local_attention_pairs(extend), (10 * 4 + 10) + 7)
        self.assertEqual(dp_attn._local_attention_pairs(decode), 7)
        self.assertEqual(dp_attn._local_attention_pairs(None), 0)
        self.assertEqual(
            dp_attn._local_attention_pairs(
                SimpleNamespace(forward_mode=ForwardMode.IDLE)
            ),
            0,
        )

    def test_decode_pairs_from_seq_lens_cpu_match_request_lengths(self):
        """The prepared seq_lens_cpu fast path and the committed-length
        fallback count the same pairs, so the series has no discontinuity when
        a batch switches between them."""
        lens = [3, 5, 9]
        fast = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            seq_lens_cpu=torch.tensor(lens),
            reqs=[],
        )
        slow = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            seq_lens_cpu=None,
            reqs=[SimpleNamespace(seqlen=n) for n in lens],
        )
        self.assertEqual(
            dp_attn._local_attention_pairs(fast), dp_attn._local_attention_pairs(slow)
        )
        self.assertEqual(dp_attn._local_attention_pairs(fast), 17)

    def _adapter(self):
        return dp_attn.SchedulerDPAttnAdapter(
            model_runner=object(),
            req_to_token_pool=object(),
            token_to_kv_pool_allocator=object(),
            tree_cache=object(),
            offload_tags=set(),
            model_config=object(),
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            get_require_mlp_sync=lambda: True,
        )

    def test_batchless_gathers_carry_wait_until_recorded_or_idle(self):
        adapter = self._adapter()
        carry = adapter.sync_wait_carry
        idle_batch = SimpleNamespace(
            forward_mode=ForwardMode.IDLE, reqs=[], dp_balance_stats=None
        )

        def batchless_gather(wait_seconds):
            result = self._prepare_dp2_batch(
                None,
                local_tokens=0,
                peer_tokens=0,
                get_idle_batch=Mock(return_value=idle_batch),
                sync_wait_carry=carry,
                wait_seconds=wait_seconds,
            )
            self.assertIsNone(result)

        batchless_gather(0.001)
        batchless_gather(0.002)
        self.assertAlmostEqual(carry.seconds, 0.003)
        result = self._prepare_dp2_batch(
            ForwardMode.DECODE,
            local_tokens=4,
            peer_tokens=8,
            sync_wait_carry=carry,
            wait_seconds=0.0005,
        )
        self.assertAlmostEqual(result.dp_balance_stats.sync_wait_seconds, 0.0035)
        self.assertEqual(carry.seconds, 0.0)

        # An iteration that ends without a batch drops what it carried.
        batchless_gather(0.004)
        adapter.drop_sync_wait_carry()
        self.assertEqual(carry.seconds, 0.0)

    def test_unrecorded_batch_steps_do_not_carry_wait(self):
        carry = self._adapter().sync_wait_carry
        carry.seconds = 0.003
        # A prebuilt batch with no work anywhere runs a step but records nothing.
        result = self._prepare_dp2_batch(
            ForwardMode.PREBUILT,
            local_tokens=0,
            peer_tokens=0,
            sync_wait_carry=carry,
            wait_seconds=0.001,
        )
        self.assertIsNone(result.dp_balance_stats)
        self.assertEqual(carry.seconds, 0.0)

    def test_health_check_probe_steps_not_recorded(self):
        probe = SimpleNamespace(rid=f"{HEALTH_CHECK_RID_PREFIX}-0", seqlen=1)
        result = self._prepare_dp2_batch(
            ForwardMode.DECODE, local_tokens=1, peer_tokens=0, reqs=[probe]
        )
        self.assertIsNone(result.dp_balance_stats)

        # The gathered flag skips the step on the ranks that only ran idle.
        idle_batch = SimpleNamespace(
            forward_mode=ForwardMode.IDLE, reqs=[], dp_balance_stats=None
        )
        result = self._prepare_dp2_batch(
            None,
            local_tokens=0,
            peer_tokens=1,
            peer_health_check=True,
            get_idle_batch=Mock(return_value=idle_batch),
        )
        self.assertIs(result, idle_batch)
        self.assertIsNone(result.dp_balance_stats)

    def test_create_rejects_steps_without_tokens_or_pairs(self):
        with self.assertRaises(ValueError):
            DPBalanceStats.create(
                local_tokens=0,
                global_num_tokens=[0, 0],
                local_attention_pairs=0,
                global_attention_pairs=[0, 0],
                sync_wait_seconds=0.0,
            )
        # Rows without pairs cannot come from a correct gather.
        with self.assertRaises(ValueError):
            DPBalanceStats.create(
                local_tokens=0,
                global_num_tokens=[0, 8],
                local_attention_pairs=0,
                global_attention_pairs=[0, 0],
                sync_wait_seconds=0.0,
            )


class TestDecodeToExtendConversionVote(CustomTestCase):
    """A decode batch votes for the prefill graph only when its 1-token extend
    view can represent every row. Beam requests cannot: the converted batch
    takes the prefill result path, which commits them per-req instead of
    through the batch decode fold, and member rows carry no req."""

    def _vote(self, *, beam):
        runner = Mock(spec=dp_attn.PrefillCudaGraphRunner)
        runner.enable_lora = False
        runner.max_context_size = None
        runner.can_replay_locally.return_value = True
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            batch_size=lambda: 2,
            return_logprob=False,
            has_grammar=False,
            multimodal_inputs=None,
            reqs=[
                SimpleNamespace(beam_group=Mock() if beam else None),
                SimpleNamespace(beam_group=None),
            ],
        )
        with (
            patch.object(
                dp_attn, "get_moe_a2a_backend", return_value=Mock(is_none=lambda: True)
            ),
            patch.object(dp_attn, "uses_ssm_state", return_value=False),
            patch.object(
                dp_attn,
                "get_memory",
                return_value=SimpleNamespace(enable_hisparse=False),
            ),
            patch.object(
                dp_attn,
                "get_exec",
                return_value=SimpleNamespace(
                    overlap=SimpleNamespace(enable_two_batch_overlap=False)
                ),
            ),
            patch.object(dp_attn, "get_cp_strategy", return_value=None),
        ):
            return dp_attn._local_prefill_cuda_graph_vote(
                local_batch=batch,
                prefill_graph_runner=runner,
                coordinated_prefill=True,
                breakable_prefill=True,
                spec_algorithm=SpeculativeAlgorithm.NONE,
                model_config=object(),
            )

    def test_plain_decode_batch_votes_for_conversion(self):
        self.assertTrue(self._vote(beam=False))

    def test_beam_request_blocks_conversion(self):
        self.assertFalse(self._vote(beam=True))


class TestPrefillCudaGraphVote(CustomTestCase):
    def _vote(self, mode, *, mm=None):
        runner_cls = dp_attn.PrefillCudaGraphRunner
        runner = Mock(spec=runner_cls)
        runner.enable_lora = False
        runner.max_context_size = None
        runner._qwen_bcg_hc_sidechannel = True
        runner.can_replay_locally.side_effect = lambda **kwargs: (
            not kwargs["contains_mm_inputs"]
            or runner_cls.can_replay_locally(self=runner, **kwargs)
        )
        batch = SimpleNamespace(
            forward_mode=mode,
            extend_num_tokens=4,
            multimodal_inputs=mm,
            input_embeds=None,
            replace_embeds=None,
            prefix_lens=[1, 1],
            return_logprob=False,
            batch_size=lambda: 2,
        )
        vote = dp_attn._local_prefill_cuda_graph_vote(
            local_batch=batch,
            prefill_graph_runner=runner,
            coordinated_prefill=True,
            breakable_prefill=True,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            model_config=object(),
        )
        return vote, runner

    def test_extend_batch_votes_for_prefill_graph(self):
        vote, runner = self._vote(ForwardMode.EXTEND)

        self.assertTrue(vote)
        runner.can_replay_locally.assert_called_once()
        self.assertFalse(runner.can_replay_locally.call_args.kwargs["is_mixed"])

    def test_mixed_batch_delegates_to_runner_policy(self):
        vote, runner = self._vote(ForwardMode.MIXED)

        self.assertTrue(vote)
        runner.can_replay_locally.assert_called_once()
        self.assertTrue(runner.can_replay_locally.call_args.kwargs["is_mixed"])

    def test_batch_over_captured_metadata_bound_votes_eager(self):
        """An extend or mixed batch over the captured attention metadata's
        request bound votes eager, so no dp rank replays while another falls
        back at forward time."""
        runner_cls = dp_attn.PrefillCudaGraphRunner
        runner = Mock(spec=runner_cls)
        runner.enable_lora = False
        runner.max_context_size = None
        runner._qwen_bcg_hc_sidechannel = False
        runner._is_full_backend = False
        runner._captured_attn_metadata_max_bs = 2
        runner.prefer_eager_mixed_prefill = False
        runner.prefill_backend_name = Backend.BREAKABLE
        runner.has_mha_companion_layers = False
        runner._has_uncapturable_chunked_prefix.return_value = False
        runner.max_num_tokens = 64
        runner.capture_num_tokens = [64]
        runner._pad_to_bucket = runner_cls._pad_to_bucket
        runner.can_replay_locally.side_effect = lambda **kwargs: (
            runner_cls.can_replay_locally(self=runner, **kwargs)
        )

        def vote(mode, batch_size):
            batch = SimpleNamespace(
                forward_mode=mode,
                extend_num_tokens=48,
                multimodal_inputs=None,
                input_embeds=None,
                replace_embeds=None,
                prefix_lens=[0] * batch_size,
                return_logprob=False,
                batch_size=lambda: batch_size,
            )
            return dp_attn._local_prefill_cuda_graph_vote(
                local_batch=batch,
                prefill_graph_runner=runner,
                coordinated_prefill=True,
                breakable_prefill=True,
                spec_algorithm=SpeculativeAlgorithm.NONE,
                model_config=object(),
            )

        # Mixed batches (--enable-mixed-chunk) count their decode rows too.
        for mode in (ForwardMode.EXTEND, ForwardMode.MIXED):
            with self.subTest(mode=mode):
                self.assertTrue(vote(mode, batch_size=2))
                self.assertFalse(vote(mode, batch_size=3))

    @patch.object(dp_attn, "get_parallel")
    @patch.object(dp_attn, "all_gather_single")
    def test_multimodal_ranks_vote_eager(self, gather, parallel):
        infos = [
            dp_attn.MLPSyncBatchInfo(
                num_dp_ranks=2,
                tp_size=1,
                cp_size=1,
                num_tokens=4,
                num_tokens_for_logprob=1,
                can_run_decode_cuda_graph=False,
                can_run_draft_cuda_graph=False,
                can_run_prefill_cuda_graph=self._vote(ForwardMode.EXTEND, mm=mm)[0],
                is_extend_in_batch=True,
                local_can_run_tbo=False,
                local_forward_mode=ForwardMode.EXTEND.value,
            )
            for mm in ([SimpleNamespace(contains_mm_input=lambda: True)], None)
        ]
        values = torch.cat([i._get_local_tensor(device="cpu") for i in infos])
        gather.side_effect = lambda output, *a, **kw: output.copy_(values)
        parallel.return_value.tp_group.active_ranks_cpu = torch.ones(2)
        for info in infos:
            info.all_gather(device="cpu", group=None)
            self.assertFalse(info.can_run_prefill_cuda_graph)


if __name__ == "__main__":
    unittest.main()
