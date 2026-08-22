"""Spec-V2 overlap lifetime: plan-stream draft-extend pin, relay-read pin.

Two allocator-lifetime seams of the overlap scheduler, each demonstrated with
the production ownership graph intact. What is REAL in every arm: the producer
functions (prepare_for_draft_extend / ForwardBatch.init_new / compute_position
/ ScheduleBatch record + ring semantics / resolve_forward_inputs), the stream
discipline (a real side plan stream, wait_stream exactly where the worker has
it), and the reference lifetimes (objects die by natural scope exit or by the
production sequence that drops them -- no reference is manually deleted that
production keeps). What is COMPRESSED for determinism, and only that: kernel
latency (torch.cuda._sleep holds the forward stream the way a deep kernel
queue does) and allocation pressure (an immediate same-stream allocation
stands in for the next scheduler round's allocations). Reachability is
allocator-config-dependent: the default native allocator's free-list
eventing happens to order the reuse write after pending cross-stream
readers in this pattern (measured 0/40 corruptions), so corruption arms run
in a subprocess pinned to PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,
where the same production ordering corrupts ~19/20 (backend:cudaMallocAsync
behaves the same); the contract (torch.Tensor.record_stream docs) is
violated either way.

1. Plan-stream draft-extend lifetime (SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1):
   prepare_for_draft_extend runs inside the plan stream and its ForwardBatch
   owns plan-stream allocations that are NOT ScheduleBatch attributes --
   init_new's positions / extend_start_loc. wait_stream orders execution but
   does not protect deallocation (see torch.Tensor.record_stream docs): when
   _draft_extend_for_decode returns, the ForwardBatch's only reference dies
   while forward-stream kernels still read those tensors. The fix pins the
   draft-extend ForwardBatch in GenerationBatchResult.extra_keep_alive_refs
   (the designed extension point: the scheduler ring slot is "a list (not
   tuple) so that workers can register additional refs"), beside
   verify_forward_batch. The scheduler's batch-field snapshot cannot cover
   this: the tensors are not reachable from the ScheduleBatch (pinned by a
   dedicated test below).

2. mix_running_indices relay-read (non-spec overlap + mixed chunked prefill):
   ScheduleBatch.mix_with_running aliases the running batch's
   req_pool_indices. Usually self.last_batch or the 2-slot record ring keeps
   the running batch alive through the forward-stream gather -- but that
   protection ages out: intervening non-mixed prefills advance last_batch and
   overwrite both ring slots, after which a later mix_with_running leaves the
   staging field as the LAST reference, and resolve_forward_inputs clears it
   right after enqueueing the gather. The ownership test walks that exact
   sequence and proves last-reference death with a weakref; the corruption
   test shows the gather reads recycled memory without record_stream.
"""

import dataclasses
import gc
import unittest
import weakref
from types import SimpleNamespace

import torch

from sglang.srt.managers.overlap_utils import FutureMap, resolve_forward_inputs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")

_GPU_DELAY_CYCLES = 2_000_000_000  # ~1s: must outlast full-process gc.collect()

_DIST_READY = False


def _ensure_single_rank_parallel_state():
    """init_new consults the parallel state (EP group); give it the real
    single-rank one, same as other unit tests that drive real forwards."""
    global _DIST_READY
    if _DIST_READY:
        return
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        backend="nccl",
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="tcp://127.0.0.1:29811",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)
    _DIST_READY = True


def _make_schedule_batch(**attrs):
    """A ScheduleBatch shell with every dataclass field present (None): the
    real record/snapshot paths getattr every field, and the real
    prepare_for_draft_extend setattrs onto it like on any batch."""
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    batch = object.__new__(ScheduleBatch)
    for f in dataclasses.fields(ScheduleBatch):
        setattr(batch, f.name, None)
    for name, value in attrs.items():
        setattr(batch, name, value)
    return batch


def _record_ring_slot(batch):
    """Mirror Scheduler.record_batch_in_overlap exactly (scheduler.py): the
    dataclass-fields snapshot plus the batch object, as a list so workers can
    extend it with extra_keep_alive_refs."""
    attr_snapshot = [getattr(batch, f.name, None) for f in dataclasses.fields(batch)]
    return [batch, attr_snapshot]


def _flatten_tensors(obj, out, seen=None):
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _flatten_tensors(item, out, seen)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            _flatten_tensors(getattr(obj, f.name, None), out, seen)


class _PlanStreamRunnerShell:
    """The model-runner surface ForwardBatch.init_new + the draft-extend
    prepare actually consume; no model, no kernels."""

    def __init__(self, device):
        self.device = device
        self.spec_algorithm = SpeculativeAlgorithm.EAGLE
        self.req_to_token_pool = None
        self.token_to_kv_pool = None
        self.attn_backend = SimpleNamespace(init_forward_metadata=lambda fb: None)
        self.server_args = SimpleNamespace(
            enable_dp_attention=False,
            disable_radix_cache=True,
            speculative_algorithm="EAGLE",
            enable_lora=False,
            attention_backend="torch_native",
            enable_deterministic_inference=False,
            disable_scheduler_metadata_precompute=True,
        )
        self.model_config = SimpleNamespace(vocab_size=32000, model_is_mrope=False, is_encoder_decoder=False)
        self.sliding_window_size = None
        self.prefill_attention_backend_str = "torch_native"
        self.decode_attention_backend_str = "torch_native"
        self.ngram_embedding_manager = SimpleNamespace(enabled=False)
        self.mm_embedding_cache = None
        self.lora_manager = None
        self.ps = SimpleNamespace(attn_dcp_size=1, attn_dcp_rank=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestDraftExtendPlanStreamLifetime(CustomTestCase):
    """Seam 1: ForwardBatch-only plan-stream tensors outlive their reference.

    Arms:
      pin="ring"      -- production BASE behavior (batch snapshot ring only):
                         positions is not a ScheduleBatch attr, so the read
                         returns recycled memory. RED on base.
      pin="keepalive" -- the fix: the draft-extend ForwardBatch rides
                         extra_keep_alive_refs into the same ring slot
                         (mirrors scheduler.py's ring-slot extend). GREEN.
    """

    def _draft_extend_round(self, pin: str):
        device = "cuda"
        _ensure_single_rank_parallel_state()
        torch.cuda.synchronize()
        forward_stream = torch.cuda.current_stream()
        plan_stream = torch.cuda.Stream()

        bs, ndt = 4, 512  # positions [bs*ndt] = 16KB: its own allocator size class
        seq_lens = torch.tensor([17, 33, 9, 25], dtype=torch.int64, device=device)
        batch = _make_schedule_batch(
            seq_lens=seq_lens,
            seq_lens_cpu=None,  # gpu_only path: prefix/extend lens stay on GPU
            reqs=[
                SimpleNamespace(
                    rid=str(i),
                    lora_id=None,
                    token_type_ids=None,
                    multi_item_delimiter_indices=None,
                )
                for i in range(bs)
            ],
            req_pool_indices=torch.arange(1, bs + 1, dtype=torch.int64, device=device),
            out_cache_loc=torch.arange(bs * ndt, dtype=torch.int64, device=device),
            model_config=SimpleNamespace(vocab_size=32000),
            forward_mode=__import__(
                "sglang.srt.model_executor.forward_batch_info", fromlist=["ForwardMode"]
            ).ForwardMode.DECODE,
            return_logprob=False,
            sampling_info=None,
            spec_algorithm=SpeculativeAlgorithm.EAGLE,
        )
        ring_slot = _record_ring_slot(batch)  # the scheduler's pre-forward pin

        result = SimpleNamespace(extra_keep_alive_refs=None)  # GenerationBatchResult shape
        runner = _PlanStreamRunnerShell(device)

        truth_holder = {}

        def _worker_round():
            """Mirrors _draft_extend_for_decode's ownership: the ForwardBatch
            is a local; its reference dies when this function returns."""
            from sglang.srt.speculative.eagle_worker_common import (
                prepare_for_draft_extend,
            )
            from sglang.srt.speculative.eagle_info import EagleDraftExtendInput

            draft_extend_input = EagleDraftExtendInput(
                hidden_states=None,
                num_correct_drafts=torch.ones(bs, dtype=torch.int32, device=device),
                num_accept_tokens=torch.ones(bs, dtype=torch.int32, device=device),
            )
            predict = torch.randint(0, 31999, (bs * ndt,), dtype=torch.int64, device=device)
            with torch.cuda.stream(plan_stream):
                fb = prepare_for_draft_extend(
                    draft_extend_input,
                    batch,
                    predict,
                    ndt,
                    runner,
                    None,  # no CUDA-graph runner: eager metadata path
                    return_hidden_states_before_norm=False,
                )
            # Production ordering: the worker joins the plan stream, then runs
            # the draft forward on the compute stream.
            torch.cuda.current_stream().wait_stream(plan_stream)

            self.assertIsNotNone(fb.positions, "init_new must produce positions")
            truth_holder["positions"] = fb.positions.detach().clone().cpu()
            truth_holder["nbytes"] = fb.positions.numel()

            # The draft forward reading fb.positions; _sleep compresses the
            # real kernel queue depth in front of it.
            torch.cuda._sleep(_GPU_DELAY_CYCLES)
            out = fb.positions + 0

            if pin == "keepalive":
                # The fix (eagle_worker_v2 / multi_layer_eagle_worker_v2):
                if result.extra_keep_alive_refs is None:
                    result.extra_keep_alive_refs = []
                result.extra_keep_alive_refs.append(fb)
            return out

        out = _worker_round()
        # Scheduler side after forward returns (scheduler.py ring extension).
        if pin == "keepalive" and result.extra_keep_alive_refs:
            ring_slot.extend(result.extra_keep_alive_refs)

        gc.collect()
        # Next-round allocation pressure on the plan stream: reuses the freed
        # block if nothing pinned it.
        with torch.cuda.stream(plan_stream):
            decoys = [
                torch.full(
                    (truth_holder["nbytes"],), 7_777_777, dtype=torch.int64, device=device
                )
                for _ in range(4)
            ]
        torch.cuda.synchronize()
        del decoys, ring_slot
        return out.cpu().tolist(), truth_holder["positions"].tolist()

    def test_ring_snapshot_alone_lets_positions_be_recycled(self):
        self.assertEqual(_run_arm_in_subprocess("seam1_ring"), "CORRUPTED")

    def test_keepalive_preserves_positions(self):
        self.assertEqual(_run_arm_in_subprocess("seam1_keepalive"), "CLEAN")


class TestRingSnapshotCannotReachForwardBatch(CustomTestCase):
    """Design pin (CPU): the scheduler's batch-field snapshot -- and even a
    full vars(batch) capture -- cannot retain ForwardBatch-only tensors, which
    is why the fix pins the ForwardBatch object itself."""

    def test_positions_not_reachable_from_batch(self):
        batch = _make_schedule_batch(
            seq_lens=torch.tensor([3, 5]),
            input_ids=torch.tensor([1, 2]),
        )
        fb_positions = torch.arange(8)
        ring_slot = _record_ring_slot(batch)
        ring_slot_vars = list(vars(batch).values())
        reachable = []
        _flatten_tensors([ring_slot, ring_slot_vars], reachable)
        self.assertTrue(all(t is not fb_positions for t in reachable))


class TestMixRunningIndicesLifetime(CustomTestCase):
    """Seam 2: the aging-out sequence that makes the mixed gather's indices
    lose their last owner, walked with the production data structures."""

    @staticmethod
    def _future_map(vocab_rows, device):
        fm = object.__new__(FutureMap)
        fm.output_tokens_buf = vocab_rows
        fm.spec_algo = SpeculativeAlgorithm.NONE
        return fm

    def _age_out_sequence(self, device):
        """scheduler-lifecycle walk: R decode batch -> two non-mixed prefills
        overwrite the 2-slot ring and advance last_batch -> a later prefill
        mixes R -> running_batch replaced -> staging cleared by
        resolve_forward_inputs. Returns (weakref to R.req_pool_indices,
        the mixed batch, the future map)."""
        rpi = torch.tensor([2, 5], dtype=torch.int64, device=device)
        running = _make_schedule_batch(req_pool_indices=rpi, reqs=[])
        wr = weakref.ref(rpi)

        ring = [None, None]
        ring[0] = _record_ring_slot(running)  # R's decode round
        last_batch = running

        for _ in range(2):  # non-mixed prefills age R out of the ring
            prefill = _make_schedule_batch(reqs=[])
            ring[0], ring[1] = ring[1], _record_ring_slot(prefill)
            last_batch = prefill

        mixed = _make_schedule_batch(
            reqs=[],
            device=device,
            # the alias mix_with_running establishes (schedule_batch.py):
            mix_running_indices=running.req_pool_indices,
            prefill_input_ids_cpu=torch.tensor([11, 12], dtype=torch.int64),
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        # scheduler replaces the running batch; R's object dies.
        del running, rpi
        return wr, mixed, last_batch, ring

    def test_aging_out_drops_last_reference(self):
        device = "cpu"
        wr, mixed, last_batch, ring = self._age_out_sequence(device)
        # Before the forward: staging is the last owner.
        gc.collect()
        self.assertIsNotNone(wr(), "staging field should still own the indices")
        fm = self._future_map(torch.arange(10, dtype=torch.int64), device)
        resolve_forward_inputs(mixed, fm)  # real function; clears staging
        gc.collect()
        self.assertIsNone(
            wr(),
            "after resolve_forward_inputs clears the staging field, nothing "
            "owns the indices while the gather may still be queued",
        )
        del last_batch, ring

    def _gather_round_corrupted(self) -> bool:
        """One aging-out round on the GPU: real resolve_forward_inputs on a
        real side stream; returns whether the gathered decode ids were
        corrupted by scheduler-stream block reuse."""
        device = "cuda"
        torch.cuda.synchronize()
        wr, mixed, last_batch, ring = self._age_out_sequence(device)
        pool = torch.arange(100, 110, dtype=torch.int64, device=device)
        fm = self._future_map(pool, device)
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            torch.cuda._sleep(_GPU_DELAY_CYCLES)
            resolve_forward_inputs(mixed, fm)  # real fn (carries the fix)
            out = mixed.input_ids + 0
        gc.collect()  # last host ref (staging already cleared) dies here
        # scheduler-stream reuse pressure (indices were allocated on the
        # default/scheduler stream in _age_out_sequence)
        decoys = [torch.full((2,), 999, dtype=torch.int64, device=device) for _ in range(4)]
        torch.cuda.synchronize()
        del decoys, last_batch, ring
        result = out.cpu().tolist()
        del out
        gc.collect()
        torch.cuda.synchronize()
        return result != [11, 12, 102, 105]

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_gather_survives_aging_out(self):
        self.assertEqual(_run_arm_in_subprocess("seam2_fix"), "CLEAN")

    def test_cpu_device_mixed_gather_still_works(self):
        device = "cpu"
        wr, mixed, last_batch, ring = self._age_out_sequence(device)
        fm = self._future_map(torch.arange(100, 110, dtype=torch.int64), device)
        resolve_forward_inputs(mixed, fm)
        self.assertEqual(mixed.input_ids.tolist(), [11, 12, 102, 105])


def _run_arm_in_subprocess(arm: str) -> str:
    """Run a corruption arm in a subprocess pinned to the allocator config
    under which the lifetime violation is observable (the parent process may
    already have initialized CUDA with a different allocator)."""
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    out = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--arm", arm],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    marker = [l for l in out.stdout.splitlines() if l.startswith("ARM_RESULT:")]
    assert marker, f"arm subprocess produced no result: {out.stdout[-2000:]} {out.stderr[-2000:]}"
    return marker[-1].split(":", 1)[1].strip()


def _arm_main(arm: str) -> None:
    torch.cuda.init()
    if arm.startswith("seam1_"):
        pin = arm.split("_", 1)[1]
        case = TestDraftExtendPlanStreamLifetime("test_keepalive_preserves_positions")
        # The violation fires ~19/20 per attempt under this allocator; bound
        # the retry so a miss is negligible while a FIXED build stays clean
        # across every attempt.
        results = []
        for _ in range(3):
            got, want = case._draft_extend_round(pin)
            results.append(got != want)
            if pin == "ring" and results[-1]:
                break
        corrupted = any(results)
    else:
        case = TestMixRunningIndicesLifetime("test_gather_survives_aging_out")
        results = []
        for _ in range(3):
            corrupted_once = case._gather_round_corrupted()
            results.append(corrupted_once)
            if arm == "seam2_nofix" and corrupted_once:
                break
        corrupted = any(results)
    print(f"ARM_RESULT: {'CORRUPTED' if corrupted else 'CLEAN'}")


if __name__ == "__main__":
    import sys

    if "--arm" in sys.argv:
        _arm_main(sys.argv[sys.argv.index("--arm") + 1])
    else:
        unittest.main()
