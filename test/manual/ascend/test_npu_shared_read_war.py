"""Single-NPU WAR/read-done tests, without model weights or a server.

Run in an installed SGLang NPU environment from the repository root:
    export PYTHONPATH=python
    timeout 180s python -m pytest test/manual/ascend/test_npu_shared_read_war.py -v -s

Real Ascend metadata, scheduler barriers and graph event hooks are exercised
with small runner fixtures. The delayed tail consumes only the metadata
snapshot, like DSA attention; this is not a model-accuracy or throughput test.
Negative controls MUST observe overwritten page IDs. If they do not, the
device delay did not expose the race and the test must not report success.
Graph-marker cases require the external-event API; missing-marker and eager
cases do not. The MTP fixture exercises plan-stream hand-offs and last-reader
selection, not the full EAGLE worker or an attention kernel.
"""

from types import SimpleNamespace

import pytest
import torch


class _SharedReadCase:
    PAGE_SIZE = 64
    REPLAYS = 8
    SLEEP_CYCLES = 100_000_000

    def __init__(self, *, graph, mode="DECODE", marker=True, early=False, plan=False):
        # Keep imports behind the NPU fixture so CPU-only collection can skip.
        from sglang.srt.environ import envs
        from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
            AscendAttnBackend,
            ForwardMetadata,
        )
        from sglang.srt.hardware_backend.npu.graph_runner.npu_cudagraph_backend import (
            NPUCudaGraphBackend,
        )
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.model_executor.forward_mode import ForwardMode
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )
        from sglang.srt.model_executor.runner_utils.shared_read_event import (
            make_external_event,
        )
        from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
        from sglang.srt.speculative.spec_utils import get_plan_stream

        self.envs = envs
        self.graph_enabled = graph
        self.marker = marker
        self.early = early
        self.mtp = mode == "DRAFT_EXTEND_V2"
        self.iteration = 0
        self.forward = torch.npu.Stream()
        self.schedule = torch.npu.Stream()
        with envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.override(plan):
            self.verify_plan, self.verify_plan_ctx = get_plan_stream("npu")
            self.extend_plan, self.extend_plan_ctx = get_plan_stream("npu")

        self.old_table = (
            torch.arange(256, dtype=torch.int32, device="npu") + self.PAGE_SIZE
        ).view(2, 128)
        self.new_table = self.old_table + 4 * self.PAGE_SIZE
        self.request_orders = torch.tensor([[0, 1], [1, 0]], device="npu")
        self.input_requests = self.request_orders[0].clone()
        self.verify_requests = self.input_requests.clone()
        self.batch = SimpleNamespace(
            forward_mode=ForwardMode[mode],
            req_pool_indices=self.input_requests.clone(),
            seq_lens=torch.full((2,), 128, dtype=torch.int32, device="npu"),
        )
        self.attn = AscendAttnBackend.__new__(AscendAttnBackend)
        self.attn.prepare_in_graph = True
        self.attn.shared_read_enabled = True
        self.attn.is_draft_worker = self.mtp
        self.attn.dsa_dcp_target = False
        self.attn.speculative_num_draft_tokens = 3
        self.attn.page_size = self.PAGE_SIZE
        self.attn.req_to_token = self.old_table.clone()
        self.attn.graph_metadata = {
            "page_offsets": torch.arange(
                0,
                self.old_table.shape[1],
                self.PAGE_SIZE,
                dtype=torch.int64,
                device="npu",
            ),
        }
        self.attn.forward_metadata = ForwardMetadata(
            block_tables=torch.empty((2, 2), dtype=torch.int32, device="npu"),
            seq_lens=torch.empty_like(self.batch.seq_lens),
        )
        self.observed = torch.empty_like(self.attn.forward_metadata.block_tables)

        self.runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        self.runner.device_module = torch.npu
        self.runner.model_runner = SimpleNamespace(shared_read_done_event=None)
        self.runner.attn_backend = self.attn
        self.runner.capture_forward_mode = self.batch.forward_mode
        self.runner.in_graph_metadata_prep_done = None
        self.runner.backend = NPUCudaGraphBackend.__new__(NPUCudaGraphBackend)
        self.runner.backend._decode_runner = self.runner
        if graph and marker:
            event = make_external_event(torch.npu)
            if event is None:
                pytest.skip("NPU graph read-done needs the CANN external-event API")
            self.runner.in_graph_metadata_prep_done = event

        self.target = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        self.target.device_module = torch.npu
        self.target.model_runner = SimpleNamespace(shared_read_done_event=None)
        if self.mtp:
            worker = EAGLEWorkerV2.__new__(EAGLEWorkerV2)
            worker._target_worker = SimpleNamespace(
                model_runner=self.target.model_runner
            )
            worker._draft_worker = SimpleNamespace(
                draft_runner=self.runner.model_runner
            )
            assert worker.last_shared_read_runner is self.runner.model_runner
        else:
            worker = SimpleNamespace(last_shared_read_runner=self.runner.model_runner)
        self.scheduler = Scheduler.__new__(Scheduler)
        self.scheduler.model_worker = worker
        self.scheduler.schedule_stream = self.schedule
        self.scheduler.forward_stream = self.forward

        # Older torch_npu packages lack _sleep. A finite native GEMM chain also
        # widens the window; no custom kernel or CPU sleep is needed.
        self.sleep = getattr(torch.npu, "_sleep", None)
        if self.sleep is None:
            self.delay_input = torch.ones(
                (8192, 8192), dtype=torch.bfloat16, device="npu"
            )
            self.delay_output = torch.empty_like(self.delay_input)
        self.forward.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(self.forward):
            for _ in range(2):
                self._forward_body()
            self.forward.synchronize()
        self.graph = None
        if graph:
            self.graph = torch.npu.NPUGraph()
            with torch.npu.graph(
                self.graph, stream=self.forward, auto_dispatch_capture=True
            ):
                self._forward_body()
            self.forward.synchronize()

    def _delay(self):
        if self.sleep is not None:
            self.sleep(self.SLEEP_CYCLES)
        else:
            for _ in range(128):
                torch.mm(self.delay_input, self.delay_input, out=self.delay_output)

    def _record_read_done(self):
        if self.graph_enabled:
            self.runner._record_in_graph_metadata_prep_done()
        else:
            self.runner._publish_read_done(in_graph=False)

    def _forward_body(self):
        if self.marker and self.early:
            self._record_read_done()
        self._delay()
        self.attn.init_forward_metadata_in_graph(self.batch)
        if self.marker and not self.early:
            self._record_read_done()
        self._delay()
        # No live req_to_token reads after the marker: only the private snapshot.
        self.observed.copy_(self.attn.forward_metadata.block_tables)

    def _plan_copy(self, stream, context, destination, source):
        if stream is not None:
            stream.wait_stream(self.forward)
        with context:
            destination.copy_(source)
        if stream is not None:
            self.forward.wait_stream(stream)

    def run(self, protection="fine", *, wrong_phase=False):
        self.runner.model_runner.shared_read_done_event = None
        self.scheduler._war_barrier_enabled = protection != "off"
        order = self.iteration % 2
        with torch.npu.stream(self.schedule):
            self.old_table.add_(8 * self.PAGE_SIZE)
            self.new_table.add_(8 * self.PAGE_SIZE)
            self.attn.req_to_token.copy_(self.old_table)
            self.input_requests.copy_(self.request_orders[order])

        forward_done = torch.npu.Event()
        write_done = torch.npu.Event()
        with torch.npu.stream(self.forward):
            # RAW is always protected. Only the later overwrite's WAR varies.
            self.forward.wait_stream(self.schedule)
            if self.mtp:
                self._plan_copy(
                    self.verify_plan,
                    self.verify_plan_ctx,
                    self.verify_requests,
                    self.input_requests,
                )
                self.target._publish_read_done(in_graph=False)
                self._plan_copy(
                    self.extend_plan,
                    self.extend_plan_ctx,
                    self.batch.req_pool_indices,
                    self.verify_requests,
                )
            else:
                self.batch.req_pool_indices.copy_(self.input_requests)
            if self.graph is not None:
                self.graph.replay()
                if self.marker:
                    self.runner._publish_read_done(in_graph=True)
            else:
                self._forward_body()
            forward_done.record()
            if wrong_phase:
                # Negative control: a verify event cannot cover draft-extend reads.
                self.runner.model_runner.shared_read_done_event = (
                    self.target.model_runner.shared_read_done_event
                )

        with (
            self.envs.SGLANG_FORCE_COARSE_WAR_BARRIER.override(protection == "coarse"),
            torch.npu.stream(self.schedule),
        ):
            self.scheduler._apply_war_barrier()
            if protection != "off":
                assert self.runner.model_runner.shared_read_done_event is None
            self.attn.req_to_token.copy_(self.new_table)
            write_done.record()

        # All competing work is now enqueued. Host waits/readbacks only observe
        # the outcome; they must not be inserted before the competing write.
        write_done.synchronize()
        released_early = not forward_done.query()
        forward_done.synchronize()
        self.iteration += 1
        expected = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        expected = (expected + 8 * self.iteration)[[order, 1 - order]]
        return self.observed.cpu(), expected, released_early


@pytest.fixture
def make_case():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires one NPU")
    cases = []

    def create(**kwargs):
        case = _SharedReadCase(**kwargs)
        cases.append(case)
        return case

    yield create
    # Keep captured buffers/events alive until all submitted work has completed.
    torch.npu.synchronize()


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_without_war_exposes_overwrite(make_case, graph):
    case = make_case(graph=graph, marker=False)
    observed, expected, _ = case.run("off")
    assert torch.equal(observed, expected + 4), "failed to expose the overwrite window"


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_missing_read_done_falls_back_to_whole_forward(make_case, graph):
    case = make_case(graph=graph, marker=False)
    observed, expected, early = case.run()
    assert torch.equal(observed, expected)
    assert not early


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_force_coarse_ignores_an_early_marker(make_case, graph):
    case = make_case(graph=graph, early=True)
    observed, expected, early = case.run("coarse")
    assert torch.equal(observed, expected)
    assert not early


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("mode", ["DECODE", "TARGET_VERIFY", "DRAFT_EXTEND_V2"])
def test_read_done_preserves_snapshot_and_releases_tail(make_case, graph, mode):
    case = make_case(graph=graph, mode=mode)
    early_releases = 0
    for _ in range(case.REPLAYS):
        observed, expected, early = case.run()
        assert torch.equal(
            observed, expected
        ), "read-done released a shared read too early"
        early_releases += early
    # Host descheduling may miss a particular overlap window. Correctness must
    # hold on every replay, but one observed early completion proves release.
    assert early_releases, "writer never finished before the independent forward tail"


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_marker_before_shared_read_exposes_overwrite(make_case, graph):
    case = make_case(graph=graph, early=True)
    observed, expected, _ = case.run()
    assert torch.equal(observed, expected + 4), "failed to expose premature read-done"


@pytest.mark.parametrize("plan", [False, True], ids=["same-stream", "plan-streams"])
@pytest.mark.parametrize("wrong_phase", [False, True], ids=["draft-extend", "verify"])
def test_mtp_waits_for_the_last_reader(make_case, plan, wrong_phase):
    case = make_case(graph=True, mode="DRAFT_EXTEND_V2", plan=plan)
    early_releases = 0
    for _ in range(1 if wrong_phase else case.REPLAYS):
        observed, expected, early = case.run(wrong_phase=wrong_phase)
        assert torch.equal(observed, expected + (4 if wrong_phase else 0))
        early_releases += early
    if not wrong_phase:
        assert early_releases, "read-done unexpectedly waited for the whole forward"
