"""Relay staging fence: scheduler-thread staging vs the in-flight forward's
tail relay writes.

The race (pool-indexed relay, the default): batch-prep staging seams --
PD-disagg decode bootstrap (get_new_prebuilt_batch -> process_prebuilt),
the EAGLE disagg spec bootstrap (build_eagle_disagg_draft_input), and the
hisparse staging->decode rebuild -- publish/stash into freshly reallocated
req_pool rows on the schedule stream, while the rows' previous owner's final
forward -- still executing on the forward stream -- publishes/stashes those
SAME rows at its tail. Unordered, the stale tail write can land second, and
the bootstrapped request's first resolve silently reads the previous owner's
bonus/topk/seq_lens. The values are in-range: no crash, just wrong tokens.
Scheduler._fence_relay_staging closes the seam by ordering the staging
writes behind everything enqueued on the forward stream.

The fence is wired into the FutureMap (set_staging_fence) so the seams can
invoke it AFTER their H2D payload materialization and immediately BEFORE the
publish/stash pair: a pageable H2D copy on a fenced schedule stream would
host-block every bootstrap iteration until the in-flight forward drains,
while the publish/stash scatter writes are enqueue-only.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers.overlap_utils import FutureMap, RelayPayload
from sglang.srt.speculative.eagle_disaggregation import (
    build_eagle_disagg_draft_input,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")

_GPU_DELAY_CYCLES = 100_000_000  # ~tens of ms >> host enqueue latency


def _make_future_map(
    spec_algo: SpeculativeAlgorithm, device: torch.device = torch.device("cpu")
) -> FutureMap:
    return FutureMap(
        device=device,
        spec_algo=spec_algo,
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.zeros((8, 16), dtype=torch.int32, device=device)
        ),
    )


def _make_disagg_req(token: int = 7, device: str = "cpu") -> SimpleNamespace:
    """The subset of a PD-transferred Req that build_eagle_disagg_draft_input
    reads."""
    return SimpleNamespace(
        output_topk_p=[0.9],
        output_topk_index=[token],
        hidden_states_tensor=torch.zeros(2, device=device),
        output_dsa_topk_indices=None,
    )


def _make_disagg_batch(
    rows: torch.Tensor,
    seq_lens: torch.Tensor,
    token: int,
    device: str = "cpu",
    enable_overlap: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        reqs=[_make_disagg_req(token, device) for _ in range(rows.numel())],
        device=device,
        enable_overlap=enable_overlap,
        seq_lens=seq_lens,
        req_pool_indices=rows,
    )


def _server_args() -> SimpleNamespace:
    return SimpleNamespace(
        speculative_eagle_topk=1,
        speculative_num_steps=5,
        enable_multi_layer_eagle=False,
    )


def _scheduler_cls():
    from sglang.test.test_utils import maybe_stub_sgl_kernel

    maybe_stub_sgl_kernel()
    from sglang.srt.managers.scheduler import Scheduler

    return Scheduler


class TestRelayStagingFence(CustomTestCase):
    def test_fence_orders_schedule_stream_behind_forward_stream(self):
        scheduler_cls = _scheduler_cls()
        scheduler = scheduler_cls.__new__(scheduler_cls)
        scheduler.enable_overlap = True
        scheduler.schedule_stream = mock.MagicMock(name="schedule_stream")
        scheduler.forward_stream = mock.MagicMock(name="forward_stream")

        scheduler._fence_relay_staging()

        scheduler.schedule_stream.wait_stream.assert_called_once_with(
            scheduler.forward_stream
        )

    def test_fence_inert_without_overlap_or_streams(self):
        scheduler_cls = _scheduler_cls()
        # Non-overlap: forwards are synchronous on the scheduler's stream; no
        # in-flight tail writes to order against.
        scheduler = scheduler_cls.__new__(scheduler_cls)
        scheduler.enable_overlap = False
        scheduler.schedule_stream = mock.MagicMock(name="schedule_stream")
        scheduler.forward_stream = mock.MagicMock(name="forward_stream")
        scheduler._fence_relay_staging()
        scheduler.schedule_stream.wait_stream.assert_not_called()

        # Pre-loop / MLX: schedule_stream is None until run_event_loop's
        # stream setup (__init__ contract) -- must no-op. Direct attribute
        # access is deliberate (no getattr defaulting): an instance that
        # skipped __init__ entirely should fail loud, not silently no-op.
        scheduler = scheduler_cls.__new__(scheduler_cls)
        scheduler.enable_overlap = True
        scheduler.schedule_stream = None
        scheduler.forward_stream = mock.MagicMock(name="forward_stream")
        scheduler._fence_relay_staging()  # must not raise
        with self.assertRaises(AttributeError):
            bare = scheduler_cls.__new__(scheduler_cls)
            bare.enable_overlap = True
            bare.forward_stream = mock.MagicMock(name="forward_stream")
            bare._fence_relay_staging()  # skipped __init__: loud, not silent

    def test_run_staging_fence_wired_invoke_and_unwired_noop(self):
        # FutureMap-level contract: staging invokes the wired fence; unwired
        # (direct FutureMap construction in tests, MLX) is a no-op.
        future_map = _make_future_map(SpeculativeAlgorithm.EAGLE)
        future_map.run_staging_fence()  # unwired: must not raise
        fence = mock.MagicMock(name="fence")
        future_map.set_staging_fence(fence)
        future_map.run_staging_fence()
        fence.assert_called_once_with()

    def test_run_event_loop_wires_staging_fence(self):
        # The prod wiring: run_event_loop must hand the scheduler's fence to
        # the FutureMap once the schedule stream exists -- otherwise the seam
        # calls below are silently inert in production.
        scheduler_cls = _scheduler_cls()
        import sglang.srt.managers.scheduler as scheduler_mod

        scheduler = scheduler_cls.__new__(scheduler_cls)
        stream = mock.MagicMock(name="schedule_stream")
        stream.cuda_stream = 11
        scheduler.device = "cuda"
        scheduler.device_module = SimpleNamespace(
            Stream=mock.MagicMock(return_value=stream),
            StreamContext=mock.MagicMock(),
        )
        scheduler.forward_stream = SimpleNamespace(cuda_stream=22)
        scheduler.future_map = mock.MagicMock(name="future_map")
        with mock.patch.object(scheduler_mod, "triton_load_watch"), mock.patch.object(
            scheduler_mod, "use_mlx", return_value=False
        ), mock.patch.object(
            scheduler_mod, "is_cuda", return_value=True
        ), mock.patch.object(
            scheduler_mod, "dispatch_event_loop"
        ) as dispatch:
            scheduler.run_event_loop()
        dispatch.assert_called_once_with(scheduler)
        scheduler.future_map.set_staging_fence.assert_called_once()
        (wired,) = scheduler.future_map.set_staging_fence.call_args.args
        self.assertEqual(
            getattr(wired, "__func__", None),
            scheduler_cls._fence_relay_staging,
        )
        self.assertIs(getattr(wired, "__self__", None), scheduler)

    def test_prebuilt_bootstrap_defers_fence_to_staging_seam(self):
        # Drive the REAL get_new_prebuilt_batch seam: it must NOT fence before
        # process_prebuilt -- process_prebuilt's H2D payload materialization
        # (last_tokens/topk/hidden pageable copies) on a fenced schedule
        # stream would host-block every bootstrap iteration until the
        # in-flight forward drains. The fence belongs INSIDE the seam,
        # immediately before the publish/stash pair (tests below).
        scheduler_cls = _scheduler_cls()
        import sglang.srt.disaggregation.decode as decode_mod

        scheduler = scheduler_cls.__new__(scheduler_cls)
        scheduler.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        req = SimpleNamespace(
            init_next_round_input=mock.MagicMock(),
            kv_committed_len=None,
            last_node=None,
        )
        scheduler.waiting_queue = [req]
        scheduler.enable_priority_scheduling = False
        scheduler.req_to_token_pool = SimpleNamespace(size=8)
        scheduler.max_running_requests = 2
        scheduler.server_args = SimpleNamespace(
            disaggregation_decode_enable_radix_cache=False
        )
        scheduler.tree_cache = object()
        scheduler.token_to_kv_pool_allocator = object()
        scheduler.model_config = object()
        scheduler.enable_overlap = True
        scheduler.spec_algorithm = SpeculativeAlgorithm.EAGLE
        scheduler.future_map = object()

        calls = mock.MagicMock()
        scheduler._fence_relay_staging = calls.fence
        new_batch = mock.MagicMock(name="new_batch")
        new_batch.prepare_for_prebuilt = calls.prepare_for_prebuilt
        new_batch.process_prebuilt = calls.process_prebuilt
        running_batch = SimpleNamespace(batch_size=lambda: 0)

        with mock.patch.object(
            decode_mod.ScheduleBatch, "init_new", return_value=new_batch
        ), mock.patch.object(decode_mod, "set_time_batch"), mock.patch.object(
            # get_new_prebuilt_batch reads the radix flag from the published
            # 'disagg' config namespace; this bare unit process publishes
            # nothing, so stub the bag at the module binding decode.py imports.
            decode_mod,
            "get_disagg",
            return_value=SimpleNamespace(
                disaggregation_decode_enable_radix_cache=False
            ),
        ):
            out = scheduler.get_new_prebuilt_batch(running_batch)

        self.assertIs(out, new_batch)
        names = [name for name, _, _ in calls.mock_calls]
        self.assertNotIn("fence", names)  # the caller-level stall hazard
        calls.process_prebuilt.assert_called_once_with(
            scheduler.server_args, scheduler.future_map
        )

    def test_eagle_staging_fence_after_materialize_before_publish_stash(self):
        # The REAL eagle staging seam: run_staging_fence must be invoked after
        # the H2D payload materialization (guaranteed structurally: the mock
        # future_map's first recorded call IS the fence, and every H2D happens
        # before any future_map call) and immediately before publish/stash.
        future_map = mock.MagicMock(name="future_map")
        batch = _make_disagg_batch(
            rows=torch.tensor([3, 5], dtype=torch.int64),
            seq_lens=torch.tensor([9, 17], dtype=torch.int64),
            token=7,
        )
        with mock.patch(
            "sglang.srt.speculative.spec_utils.spec_need_hidden_states",
            return_value=False,
        ):
            build_eagle_disagg_draft_input(
                batch,
                _server_args(),
                torch.tensor([11, 12], dtype=torch.int64),
                future_map,
            )
        names = [name for name, _, _ in future_map.mock_calls]
        self.assertEqual(
            names, ["run_staging_fence", "publish", "stash"]
        )  # fence first among relay ops, nothing between fence and writes

    def test_eagle_no_overlap_skips_staging_and_fence(self):
        future_map = mock.MagicMock(name="future_map")
        batch = _make_disagg_batch(
            rows=torch.tensor([3], dtype=torch.int64),
            seq_lens=torch.tensor([9], dtype=torch.int64),
            token=7,
            enable_overlap=False,
        )
        draft_input = build_eagle_disagg_draft_input(
            batch, _server_args(), torch.tensor([11], dtype=torch.int64), future_map
        )
        self.assertIsNone(draft_input.future_indices)
        self.assertEqual(future_map.mock_calls, [])

    def test_process_prebuilt_nonspec_fences_before_stash(self):
        # Non-spec disagg staging (always pool mode): the mixin must call
        # run_staging_fence after materializing last_tokens_tensor and
        # immediately before the relay stash.
        from sglang.srt.disaggregation.decode_schedule_batch_mixin import (
            ScheduleBatchDisaggregationDecodeMixin,
        )

        future_map = mock.MagicMock(name="future_map")
        req = SimpleNamespace(output_ids=[42], grammar=None, rid="r0")
        fake_self = SimpleNamespace(
            reqs=[req],
            tree_cache=object(),
            device="cpu",
            spec_algorithm=SimpleNamespace(
                build_disagg_draft_input=lambda *a, **k: None
            ),
            req_pool_indices=torch.tensor([3], dtype=torch.int64),
        )
        with mock.patch(
            "sglang.srt.disaggregation.decode_schedule_batch_mixin."
            "maybe_cache_unfinished_req"
        ):
            ScheduleBatchDisaggregationDecodeMixin.process_prebuilt(
                fake_self, SimpleNamespace(), future_map
            )
        names = [name for name, _, _ in future_map.mock_calls]
        self.assertEqual(names, ["run_staging_fence", "stash"])
        self.assertIsNone(fake_self.input_ids)

    def test_hisparse_rebuild_fences_before_stash(self):
        # Same staging class: the hisparse staging->decode batch rebuild
        # stashes into pool rows on the scheduler thread.
        scheduler_cls = _scheduler_cls()
        import sglang.srt.managers.scheduler as scheduler_mod

        scheduler = scheduler_cls.__new__(scheduler_cls)
        scheduler.device = "cpu"
        scheduler.req_to_token_pool = object()
        scheduler.token_to_kv_pool_allocator = object()
        scheduler.tree_cache = object()
        scheduler.model_config = SimpleNamespace(vocab_size=32)
        scheduler.enable_overlap = True
        scheduler.spec_algorithm = SpeculativeAlgorithm.NONE

        calls = mock.MagicMock()
        scheduler._fence_relay_staging = calls.fence
        scheduler.future_map = SimpleNamespace(stash=calls.stash)
        batch = mock.MagicMock(name="batch")
        batch.return_logprob = False
        reqs = [
            SimpleNamespace(req_pool_idx=3, origin_input_ids=[1, 2], output_ids=[5])
        ]

        with mock.patch.object(
            scheduler_mod.ScheduleBatch, "init_new", return_value=batch
        ), mock.patch.object(scheduler_mod, "SamplingBatchInfo"):
            out = scheduler._build_hisparse_decode_batch(reqs)

        self.assertIs(out, batch)
        names = [name for name, _, _ in calls.mock_calls]
        self.assertIn("fence", names)
        self.assertLess(names.index("fence"), names.index("stash"))

    # ---- empirical race closure (real CUDA streams) ------------------------

    def _gpu_race_arm(self, fence: bool):
        """One arm of the staging race on real streams.

        Previous owner A's final forward is emulated by a forward-stream
        sequence [sleep, publish(seq=40), sleep, stash(bonus/topk=111)] still
        in flight when A's row (5) is freed and reassigned to disagg-
        bootstrapped B, whose staging goes through the REAL seam
        (build_eagle_disagg_draft_input: publish(seq=9) + stash(222)) on the
        schedule stream. The fenced arm exercises the SHIPPED wiring: the
        scheduler fence handed to the FutureMap (set_staging_fence), invoked
        by the seam itself after payload materialization, immediately before
        publish/stash. Returns B's first-resolve (bonus, seq_len).
        """
        scheduler_cls = _scheduler_cls()
        dev = torch.device("cuda")
        future_map = _make_future_map(SpeculativeAlgorithm.EAGLE, device=dev)
        forward_stream = torch.cuda.Stream()
        schedule_stream = torch.cuda.Stream()
        rows = torch.tensor([5], dtype=torch.int64, device=dev)

        scheduler = scheduler_cls.__new__(scheduler_cls)
        scheduler.enable_overlap = True
        scheduler.schedule_stream = schedule_stream
        scheduler.forward_stream = forward_stream
        if fence:
            # The shipped run_event_loop wiring; the seam fences itself.
            future_map.set_staging_fence(scheduler._fence_relay_staging)

        server_args = _server_args()
        # Everything the raced region touches is pre-built on the default
        # stream: a pageable H2D (torch.tensor(..., device=...)) host-syncs its
        # stream and a first-touch cudaMalloc syncs the whole device -- either
        # inside the choreography would serialize the streams and mask the
        # race the unfenced arm must exhibit.
        stale_seq = torch.tensor([40], dtype=torch.int64, device=dev)
        stale_payload = RelayPayload(
            bonus_tokens=torch.tensor([111], dtype=torch.int64, device=dev),
            topk_p=torch.full((1, 1), 0.5, device=dev),
            topk_index=torch.tensor([[111]], dtype=torch.int64, device=dev),
        )
        raced_batch = _make_disagg_batch(
            rows, torch.tensor([9], dtype=torch.int64, device=dev), 222, device=dev
        )
        last_tokens = torch.tensor([222], dtype=torch.int64, device=dev)

        with mock.patch(
            "sglang.srt.speculative.spec_utils.spec_need_hidden_states",
            return_value=False,
        ):
            # Warmup on the default stream: triggers the FutureMap lazy buf
            # init and the caching-allocator pools for both write paths (via
            # an untouched row), so the raced region only enqueues async
            # scatter kernels.
            warm_rows = torch.tensor([1], dtype=torch.int64, device=dev)
            future_map.stash(warm_rows, stale_payload)
            build_eagle_disagg_draft_input(
                _make_disagg_batch(
                    warm_rows,
                    torch.tensor([1], dtype=torch.int64, device=dev),
                    7,
                    device=dev,
                ),
                server_args,
                torch.tensor([7], dtype=torch.int64, device=dev),
                future_map,
            )
            torch.cuda.synchronize()

            # Owner A's in-flight forward tail (forward stream): still mid-
            # forward (sleep), publishes seq_lens, then stashes the payload.
            with torch.cuda.stream(forward_stream):
                torch.cuda._sleep(_GPU_DELAY_CYCLES)
                future_map.publish(rows, stale_seq)
                torch.cuda._sleep(_GPU_DELAY_CYCLES)
                future_map.stash(rows, stale_payload)
            # Scheduler thread: row 5 freed + reassigned to B; disagg decode
            # bootstrap stages through the real seam (schedule stream). No
            # explicit fence call here: in the fenced arm the seam invokes the
            # wired fence itself (post-materialization, pre-publish/stash).
            with torch.cuda.stream(schedule_stream):
                spec_info = build_eagle_disagg_draft_input(
                    raced_batch, server_args, last_tokens, future_map
                )
        torch.cuda.synchronize()

        # B's first decode resolve.
        resolve_batch = SimpleNamespace(
            spec_info=spec_info,
            seq_lens=None,
            seq_lens_cpu=None,
            seq_lens_sum=None,
            req_pool_indices=rows,
            req_pool_indices_cpu=torch.tensor([5], dtype=torch.int64),
        )
        future_map.resolve_seq_lens_cpu(resolve_batch)
        future_map._resolve_spec_extras(resolve_batch)
        torch.cuda.synchronize()
        return (
            int(spec_info.bonus_tokens.flatten()[0].item()),
            int(resolve_batch.seq_lens_cpu[0].item()),
        )

    @unittest.skipUnless(
        torch.cuda.is_available(), "staging race needs real CUDA streams"
    )
    def test_fence_closes_disagg_staging_race_on_gpu(self):
        # Unfenced arm: the race reproduces -- B resolves A's stale values.
        # If this arm ever starts seeing the fresh values, the unordered seam
        # has changed shape; re-derive the fence need before touching it.
        self.assertEqual(self._gpu_race_arm(fence=False), (111, 40))
        # Fenced arm: the shipped fence closes it -- B resolves its own
        # staged values.
        self.assertEqual(self._gpu_race_arm(fence=True), (222, 9))


if __name__ == "__main__":
    unittest.main()
