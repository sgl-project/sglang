"""Speculative workers take the scheduled step read-only and hand draft state to
the next step through FutureMap, keyed by req_pool_idx.

- A worker never writes the step: not its fields, its draft input, its
  sampling info, or its tensors in place.
- A gather of a slot waits on the publish of the step that last wrote it.
- The draft state the next step's worker sees through FutureMap is the state
  the old path carried on the running batch (filtered and merged by row), over
  random sequences of joins, leaves and reorders.
- Result tensors are not worker buffers a later step writes again.
"""

import copy
import dataclasses
import importlib
import random
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.overlap_utils import FutureMap, RelayPayload
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

POOL = 16
TOPK = 2
HIDDEN = 3
DSA = 4
NGRAM_WIDTH = 3


def _future_map(algo):
    return FutureMap(
        torch.device("cpu"),
        algo,
        SimpleNamespace(req_to_token=torch.zeros((POOL, 8), dtype=torch.int32)),
    )


def _step(slots, spec_info=None, seq_lens=None):
    step = ScheduleBatch(reqs=[SimpleNamespace(rid=s) for s in slots])
    step.forward_mode = ForwardMode.DECODE
    step.device = "cpu"
    step.enable_overlap = False
    step.spec_algorithm = SpeculativeAlgorithm.EAGLE
    step.spec_info = spec_info
    step.sampling_info = None
    step.req_pool_indices = torch.tensor(slots, dtype=torch.int64)
    step.req_pool_indices_cpu = torch.tensor(slots, dtype=torch.int64)
    lens = [10] * len(slots) if seq_lens is None else seq_lens
    step.seq_lens = torch.tensor(lens, dtype=torch.int64)
    step.seq_lens_cpu = step.seq_lens.clone()
    step.seq_lens_sum = int(step.seq_lens.sum())
    return step


# ---------------------------------------------------------------------------
# A frozen step: any write raises, in-place tensor writes are caught by version.


_FROZEN = set()


def _guarded(cls):
    # By identity: a copy of a frozen object (same class) is writable.
    class Frozen(cls):
        def __setattr__(self, name, value):
            if id(self) in _FROZEN:
                raise AssertionError(f"wrote {cls.__name__}.{name}")
            super().__setattr__(name, value)

    Frozen.__name__ = f"Frozen{cls.__name__}"
    return Frozen


_FrozenStep = _guarded(ScheduleBatch)


@dataclasses.dataclass
class _Sampling:
    penalizer_orchestrator: object = None
    grammars: object = None


_FrozenSampling = _guarded(_Sampling)


def _tensors(obj):
    return {
        name: (value._version, value.data_ptr())
        for name, value in vars(obj).items()
        if isinstance(value, torch.Tensor)
    }


def _freeze(step):
    state = (dict(vars(step)), _tensors(step))
    object.__setattr__(step, "__class__", _FrozenStep)
    _FROZEN.add(id(step))
    if step.sampling_info is not None:
        _FROZEN.add(id(step.sampling_info))
    if step.spec_info is not None:
        object.__setattr__(step.spec_info, "__class__", _guarded(type(step.spec_info)))
        _FROZEN.add(id(step.spec_info))
        state += (dict(vars(step.spec_info)), _tensors(step.spec_info))
    return state


def _assert_unchanged(test, step, state):
    _FROZEN.clear()
    test.assertEqual(dict(vars(step)), state[0])
    test.assertEqual(_tensors(step), state[1])
    if len(state) > 2:
        test.assertEqual(dict(vars(step.spec_info)), state[2])
        test.assertEqual(_tensors(step.spec_info), state[3])


class _SpecNeedsHiddenStates:
    def setUp(self):
        hidden = patch(
            "sglang.srt.speculative.spec_utils.spec_need_hidden_states",
            return_value=True,
        )
        hidden.start()
        self.addCleanup(hidden.stop)


# The v2 workers, by module and class; a module whose optional kernels are not
# installed is skipped.
_WORKERS = (
    ("eagle_worker_v2", "EAGLEWorkerV2"),
    ("standalone_worker_v2", "StandaloneWorkerV2"),
    ("multi_layer_eagle_worker_v2", "MultiLayerEagleWorkerV2"),
    ("frozen_kv_mtp_worker_v2", "FrozenKVMTPWorkerV2"),
    ("dflash_worker_v2", "DFlashWorkerV2"),
    ("dspark_components.dspark_worker_v2", "DSparkWorkerV2"),
    ("uno_worker_v2", "UnoWorkerV2"),
    ("ngram_worker", "NGRAMWorker"),
)


def _fill(batch):
    """The writes a forward's phases make on the batch they run on."""
    batch.forward_mode = ForwardMode.TARGET_VERIFY
    batch.input_ids = torch.zeros(6, dtype=torch.int64)
    batch.out_cache_loc = torch.zeros(6, dtype=torch.int64)
    batch.spec_info.positions = torch.zeros(6, dtype=torch.int64)
    batch.sampling_info.grammars = []


class TestEveryWorkerTakesTheStepReadOnly(_SpecNeedsHiddenStates, CustomTestCase):
    """Each worker's forward runs on a batch of its own: what it rebinds or
    fills there (fields, draft input, sampling info) never reaches the step."""

    def _for_each_worker(self, check):
        for module, name in _WORKERS:
            with self.subTest(worker=name):
                try:
                    mod = importlib.import_module(f"sglang.srt.speculative.{module}")
                except ModuleNotFoundError as e:
                    self.skipTest(f"{name}: {e}")
                check(object.__new__(getattr(mod, name)))

    def test_step_keyed_into_future_map(self):
        def check(worker):
            worker.future_map = _future_map(SpeculativeAlgorithm.EAGLE)
            # The prefill of rows 3 and 5 committed their state.
            worker.future_map.seed(
                torch.tensor([3, 5]),
                torch.tensor([9, 6]),
                RelayPayload(bonus_tokens=torch.tensor([7, 8])),
                indices_cpu=torch.tensor([3, 5]),
            )
            seen = []

            def forward(batch, *args):
                seen.append((batch.seq_lens.tolist(), batch.spec_info.bonus_tokens))
                _fill(batch)
                worker.publish_new_seq_lens(batch, batch.seq_lens + 2)
                return GenerationBatchResult(
                    next_draft_input=EagleDraftInput(
                        bonus_tokens=torch.tensor([17, 18])
                    )
                )

            worker._forward = forward
            # What a worker's draft input factory reads.
            worker.draft_token_num = 3
            worker.forward_width = 3
            slots = torch.tensor([3, 5])
            step = _step(
                [3, 5],
                spec_info=worker.keyed_draft_input(slots, slots),
                seq_lens=[0, 0],
            )
            step.sampling_info = _FrozenSampling()
            state = _freeze(step)
            result = worker.forward_batch_generation(step)
            _assert_unchanged(self, step, state)
            self.assertEqual(seen[0][0], [9, 6])
            self.assertEqual(seen[0][1].tolist(), [7, 8])

            # The result is complete: its draft input keys the rows, its
            # publish is sealed, and the next step (rows reordered) gathers it.
            self.assertEqual(result.next_draft_input.future_indices.tolist(), [3, 5])
            self.assertIsNotNone(result.step_publish.ready)
            seen.clear()
            slots = torch.tensor([5, 3])
            nxt = _step([5, 3], spec_info=worker.keyed_draft_input(slots, slots))
            nxt.sampling_info = _Sampling()
            worker.forward_batch_generation(nxt)
            self.assertEqual(seen[0][0], [8, 11])
            self.assertEqual(seen[0][1].tolist(), [18, 17])

        self._for_each_worker(check)

    def test_step_carrying_its_draft_state(self):
        # Under the pipeline-parallel relay the step carries the draft state
        # itself; the forward fills a copy of it.
        def check(worker):
            worker.future_map = None
            worker._forward = lambda batch, *args: (
                _fill(batch) or GenerationBatchResult()
            )
            step = _step(
                [3, 5], spec_info=EagleDraftInput(bonus_tokens=torch.tensor([7, 8]))
            )
            step.sampling_info = _FrozenSampling()
            state = _freeze(step)
            worker.forward_batch_generation(step)
            _assert_unchanged(self, step, state)

        self._for_each_worker(check)


# ---------------------------------------------------------------------------
# Per-step publishes order a write before the next gather.


class _Event:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def record(self, stream=None):
        self.log.append(("record", self.name, stream))

    def synchronize(self):
        self.log.append(("synchronize", self.name))


class _Stream:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def wait_event(self, event):
        self.log.append(("wait", self.name, event.name))

    def __repr__(self):
        return self.name


class _DeviceModule:
    """Streams and events that log; `stream` picks the current stream."""

    def __init__(self):
        self.log = []
        self.stream = "forward"
        self._streams = {}
        self._events = 0

    def current_stream(self):
        return self._streams.setdefault(self.stream, _Stream(self.log, self.stream))

    def Event(self):
        self._events += 1
        return _Event(self.log, f"e{self._events}")


class TestPerStepPublish(CustomTestCase):
    def setUp(self):
        self.future_map = _future_map(SpeculativeAlgorithm.EAGLE)
        self.device = _DeviceModule()
        self.future_map.device_module = self.device

    def publish(self, slots, lens):
        idx = torch.tensor(slots)
        publish = self.future_map.publish(idx, torch.tensor(lens), indices_cpu=idx)
        self.future_map.seal(publish)
        return publish

    def waits(self):
        return [entry[2] for entry in self.device.log if entry[0] == "wait"]

    def test_gather_waits_on_the_last_writer_of_each_slot(self):
        a = self.publish([1, 2], [11, 12])
        b = self.publish([3], [13])
        self.publish([5], [15])
        c = self.publish([1], [21])

        # The next step's gather runs on another stream.
        self.device.stream = "schedule"
        self.device.log.clear()
        seq_lens, seq_lens_cpu, seq_lens_sum = self.future_map.gather_seq_lens(
            torch.tensor([1, 2, 3])
        )
        self.assertEqual(seq_lens.tolist(), [21, 12, 13])
        self.assertEqual(seq_lens_cpu.tolist(), [21, 12, 13])
        self.assertEqual(seq_lens_sum, 46)
        # Slot 1's writer is c (it overwrote a's write), slot 2's is a, slot
        # 3's is b; the step that wrote only slot 5 is not waited on.
        self.assertEqual(
            sorted(self.waits()),
            sorted(p.seq_lens_ready.name for p in (a, b, c)),
        )

    def test_draft_state_gather_waits_for_the_sealed_writes(self):
        a = self.publish([1], [11])
        self.device.stream = "schedule"
        self.device.log.clear()
        self.future_map.gather_draft_input(
            EagleDraftInput(future_indices=torch.tensor([1])), torch.tensor([1])
        )
        self.assertEqual(self.waits(), [a.ready.name])

    def test_same_stream_gather_needs_no_wait(self):
        self.publish([1], [11])
        self.device.log.clear()
        self.future_map.gather_seq_lens(torch.tensor([1]))
        self.assertEqual(self.waits(), [])

    def test_released_slot_has_no_writer(self):
        self.publish([1], [11])
        self.future_map.release_slots(torch.tensor([1]))
        self.device.stream = "schedule"
        self.device.log.clear()
        self.future_map.gather_seq_lens(torch.tensor([1]))
        self.assertEqual(self.waits(), [])
        self.assertEqual(
            self.future_map.writer_publishes(torch.tensor([1]).numpy()), []
        )

    def test_non_speculative_publish_records_no_step(self):
        future_map = _future_map(SpeculativeAlgorithm.NONE)
        self.assertIsNone(future_map.publish(torch.tensor([1]), torch.tensor([3])))


# ---------------------------------------------------------------------------
# The draft state the next step sees is what each row's last forward produced.


class _FakeWorker(BaseSpecWorker):
    """Commits a per-row draft state that tells rows and steps apart, and
    records the draft input and lengths each forward saw."""

    def __init__(self, future_map, make_draft_input, new_draft_input):
        self.future_map = future_map
        self.make_draft_input = make_draft_input
        self.new_draft_input = new_draft_input
        self.step_no = 0
        self.seen = None
        self.produced = None

    @property
    def target_worker(self):
        return None

    @property
    def draft_worker(self):
        return None

    def _new_draft_input(self, req_pool_indices):
        return self.new_draft_input(req_pool_indices)

    def _forward(self, batch, on_publish, pp_proxy_tensors, pending_grammar):
        self.seen = batch
        self.step_no += 1
        slots = batch.req_pool_indices.tolist()
        new_seq_lens = batch.seq_lens + torch.tensor(
            [1 + (s + self.step_no) % 3 for s in slots]
        )
        on_publish(new_seq_lens)
        produced = self.make_draft_input(slots, self.step_no)
        # What the forward produced, kept before the commit keys it.
        self.produced = copy.copy(produced)
        return GenerationBatchResult(
            next_draft_input=produced, new_seq_lens=new_seq_lens
        )


def _eagle_draft_input(slots, step_no):
    rows = torch.tensor(slots, dtype=torch.float32)[:, None] + step_no * 100
    with_dsa = step_no % 3 != 0
    return EagleDraftInput(
        topk_p=rows.repeat(1, TOPK) / 1000,
        topk_index=rows.long().repeat(1, TOPK) + 1,
        hidden_states=rows.repeat(1, HIDDEN) + 0.5,
        bonus_tokens=torch.tensor(slots) + step_no * 100,
        dsa_topk_indices=(
            rows.int().repeat(1, DSA) + torch.arange(DSA, dtype=torch.int32)
            if with_dsa
            else None
        ),
        cuda_graph_compatible=step_no % 4 != 1,
    )


def _ngram_draft_input(slots, step_no):
    bs = len(slots)
    return NgramVerifyInput(
        draft_token_num=NGRAM_WIDTH,
        new_seq_lens=torch.zeros(bs, dtype=torch.int64),
        accept_tokens=(
            torch.tensor(slots)[:, None] * 10
            + step_no * 1000
            + torch.arange(NGRAM_WIDTH)
        ).flatten(),
        accept_lens=torch.tensor([1 + (s + step_no) % NGRAM_WIDTH for s in slots]),
    )


def _eagle_rows(produced, num_rows):
    """Per row of a forward, the draft state it produced."""
    return [
        SimpleNamespace(
            bonus_tokens=produced.bonus_tokens[j],
            topk_p=produced.topk_p[j],
            topk_index=produced.topk_index[j],
            hidden_states=produced.hidden_states[j],
            dsa_topk_indices=(
                None
                if produced.dsa_topk_indices is None
                else produced.dsa_topk_indices[j]
            ),
            cuda_graph_compatible=produced.cuda_graph_compatible,
        )
        for j in range(num_rows)
    ]


def _ngram_rows(produced, num_rows):
    accept_tokens = produced.accept_tokens.view(num_rows, -1)
    return [
        SimpleNamespace(
            accept_tokens=accept_tokens[j],
            accept_lens=produced.accept_lens[j],
            cuda_graph_compatible=True,
        )
        for j in range(num_rows)
    ]


class TestDraftStateFollowsRows(_SpecNeedsHiddenStates, CustomTestCase):
    def _check_eagle(self, seen, rows):
        self.assertTrue(
            torch.equal(seen.bonus_tokens, torch.stack([r.bonus_tokens for r in rows]))
        )
        self.assertTrue(torch.equal(seen.topk_p, torch.stack([r.topk_p for r in rows])))
        self.assertTrue(
            torch.equal(seen.topk_index, torch.stack([r.topk_index for r in rows]))
        )
        self.assertTrue(
            torch.equal(
                seen.hidden_states, torch.stack([r.hidden_states for r in rows])
            )
        )
        if all(r.dsa_topk_indices is not None for r in rows):
            # FutureMap drops the seed only when one of the step's own rows
            # lacks it.
            self.assertTrue(
                torch.equal(
                    seen.dsa_topk_indices,
                    torch.stack([r.dsa_topk_indices for r in rows]),
                )
            )

    def _check_ngram(self, seen, rows):
        self.assertTrue(
            torch.equal(seen.accept_tokens, torch.cat([r.accept_tokens for r in rows]))
        )
        self.assertTrue(
            torch.equal(seen.accept_lens, torch.stack([r.accept_lens for r in rows]))
        )

    def _run(self, algo, make_draft_input, new_draft_input, split_rows, check, seed):
        rng = random.Random(seed)
        future_map = _future_map(algo)
        worker = _FakeWorker(future_map, make_draft_input, new_draft_input)
        free = list(range(1, POOL))
        rng.shuffle(free)
        # The running rows' slots, and per slot the draft state and length its
        # last forward produced.
        slots, produced, lens = [], {}, {}

        def record(step_slots, result):
            rows = split_rows(worker.produced, len(step_slots))
            for slot, row, length in zip(
                step_slots, rows, result.new_seq_lens.tolist(), strict=True
            ):
                produced[slot], lens[slot] = row, length

        for _ in range(40):
            # Some rows leave, the rest are reordered.
            if slots:
                keep = [i for i in range(len(slots)) if rng.random() < 0.8]
                rng.shuffle(keep)
                free += [s for i, s in enumerate(slots) if i not in keep]
                slots = [slots[i] for i in keep]

            # New rows prefill and join.
            joining = [free.pop() for _ in range(min(len(free), rng.randint(0, 3)))]
            if joining:
                prefill = _step(joining)
                prefill.forward_mode = ForwardMode.EXTEND
                record(joining, worker.forward_batch_generation(prefill))
                slots = slots + joining

            if not slots:
                continue
            # A decode step, its draft input built from the rows' slots: its
            # worker sees what each row's last forward produced.
            indices = torch.tensor(slots)
            built = worker.keyed_draft_input(indices, indices)
            self.assertEqual(built.future_indices.tolist(), slots)
            self.assertEqual(
                built.cuda_graph_compatible,
                all(produced[s].cuda_graph_compatible for s in slots),
            )
            step = _step(slots, spec_info=built, seq_lens=[0] * len(slots))
            result = worker.forward_batch_generation(step)
            check(worker.seen.spec_info, [produced[s] for s in slots])
            expected_lens = [lens[s] for s in slots]
            self.assertEqual(worker.seen.seq_lens.tolist(), expected_lens)
            self.assertEqual(worker.seen.seq_lens_cpu.tolist(), expected_lens)
            record(slots, result)

    def test_eagle(self):
        for seed in range(20):
            with self.subTest(seed=seed):
                self._run(
                    SpeculativeAlgorithm.EAGLE3,
                    _eagle_draft_input,
                    lambda indices: EagleDraftInput(),
                    _eagle_rows,
                    self._check_eagle,
                    seed,
                )

    def test_ngram(self):
        for seed in range(20):
            with self.subTest(seed=seed):
                self._run(
                    SpeculativeAlgorithm.NGRAM,
                    _ngram_draft_input,
                    lambda indices: NgramVerifyInput(
                        draft_token_num=NGRAM_WIDTH, new_seq_lens=indices
                    ),
                    _ngram_rows,
                    self._check_ngram,
                    seed,
                )


# ---------------------------------------------------------------------------
# Result tensors are owned by the result.


class TestResultsDoNotAliasWorkerBuffers(CustomTestCase):
    def test_dflash_accept_outputs_are_fresh_per_step(self):
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        worker = object.__new__(DFlashWorkerV2)
        worker.device = "cpu"
        worker.block_size = 4
        first = worker._alloc_accept_bonus_outputs(3)
        held = [t.data_ptr() for t in first]
        # Steps later, the held result's tensors are still its own.
        for _ in range(3):
            later = worker._alloc_accept_bonus_outputs(3)
            for t in later:
                self.assertNotIn(t.data_ptr(), held)

    def test_dspark_folded_accept_outputs_are_copies(self):
        from sglang.srt.speculative.dspark_components.dspark_verify import (
            AcceptOuts,
            TargetVerifyExecutor,
        )

        buffers = AcceptOuts(
            correct_len=torch.zeros(4, dtype=torch.int32),
            bonus=torch.zeros(4, dtype=torch.int64),
            cap_trim_lens=torch.zeros(4, dtype=torch.int32),
            commit_lens=torch.ones(4, dtype=torch.int32),
            new_seq_lens=torch.full((4,), 9),
            out_tokens=torch.zeros((4, 3), dtype=torch.int64),
        )
        executor = object.__new__(TargetVerifyExecutor)
        executor.verify_epilogue = MagicMock(read_accept=lambda bs: buffers)
        outs = executor.accept_and_finalize(
            folded_accept=True,
            bs=4,
            verify_ids_2d=None,
            target_logits=None,
            draft_block=None,
            sampling_info=None,
            draft_input=None,
            layout=None,
            prefix_lens=None,
            draft_tokens=None,
        )
        for name in AcceptOuts.__struct_fields__:
            self.assertTrue(torch.equal(getattr(outs, name), getattr(buffers, name)))
            self.assertNotEqual(
                getattr(outs, name).data_ptr(), getattr(buffers, name).data_ptr()
            )


if __name__ == "__main__":
    unittest.main()
