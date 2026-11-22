"""Single-NPU WAR/read-done tests, without model weights or a server.

Run in an installed SGLang NPU environment from the repository root:
    export PYTHONPATH=python
    timeout 180s python -m pytest test/manual/ascend/test_npu_shared_read_war.py -v -s

Real Ascend metadata, scheduler barriers and ordinary read-done events are exercised
with small runner fixtures. The delayed tail consumes only the metadata
snapshot, like DSA attention; this is not a model-accuracy or throughput test.
Negative controls MUST observe overwritten page IDs. If they do not, the
device delay did not expose the race and the test must not report success.
Metadata stays outside the graph; only the snapshot-consuming tail is captured.
The MTP fixture exercises plan-stream hand-offs and last-reader selection,
not the full EAGLE worker or an attention kernel.
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
        from sglang.srt.layers.attention.base_attn_backend import SharedReadEnds
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )
        from sglang.srt.model_executor.forward_mode import ForwardMode
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )
        from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
        from sglang.srt.speculative.spec_info import SpecInputType
        from sglang.srt.speculative.spec_utils import get_plan_stream

        self.envs = envs
        self.forward_context = forward_context
        self.ForwardContext = ForwardContext
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
            batch_size=2,
            forward_mode=ForwardMode[mode],
            req_pool_indices=self.input_requests.clone(),
            seq_lens=torch.full(
                (2,),
                125 if mode == "TARGET_VERIFY" else 128,
                dtype=torch.int32,
                device="npu",
            ),
            seq_lens_cpu=None,
            extend_seq_lens=None,
            extend_seq_lens_cpu=[],
            out_cache_loc=None,
            spec_algorithm=None,
            spec_info=(
                None
                if mode == "DECODE"
                else SimpleNamespace(
                    draft_token_num=3,
                    spec_input_type=(
                        SpecInputType.EAGLE_VERIFY
                        if mode == "TARGET_VERIFY"
                        else SpecInputType.EAGLE_DRAFT_EXTEND
                    ),
                )
            ),
        )
        self.attn = AscendAttnBackend.__new__(AscendAttnBackend)
        self.attn.shared_read_enabled = True
        self.attn.device = torch.device("npu")
        self.attn.is_draft_worker = self.mtp
        self.attn.use_mla = self.attn.use_dsa = True
        self.attn.use_flash_mla = self.attn.use_paged_mtp = False
        self.attn.use_quant_flash_attn = False
        self.attn.is_hybrid_swa = self.attn.use_sliding_window_kv_pool = False
        self.attn.dense_mla_dcp_target = False
        self.attn.dsa_dcp_target = False
        self.attn.needs_cpu_seq_lens = False
        self.attn.speculative_num_draft_tokens = 3
        self.attn.page_size = self.PAGE_SIZE
        self.attn.req_to_token = self.old_table.clone()
        self.attn.req_to_token_pool = SimpleNamespace(
            req_to_token=self.attn.req_to_token
        )
        self.attn.forward_metadata = ForwardMetadata(
            block_tables=torch.empty((2, 2), dtype=torch.int32, device="npu"),
            seq_lens=torch.empty_like(self.batch.seq_lens),
        )
        self.attn.graph_metadata = {2: self.attn.forward_metadata}
        self.observed = torch.empty_like(self.attn.forward_metadata.block_tables)

        self.runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        self.runner.device_module = torch.npu
        self.runner.model_runner = SimpleNamespace(shared_read_done_event=None)
        self.attn._shared_read_runner = self.runner.model_runner
        self.runner.attn_backend = self.attn
        self.runner.capture_forward_mode = self.batch.forward_mode
        self.runner.backend = NPUCudaGraphBackend.__new__(NPUCudaGraphBackend)
        self.runner.backend._decode_runner = self.runner
        assert self.runner.backend.supports_shared_read_events
        assert (
            self.runner._resolve_shared_read_ends(self.attn, self.batch.forward_mode)
            is SharedReadEnds.PRE_REPLAY
        )

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
                self._prepare_metadata()
                self._compute_tail()
            self.forward.synchronize()
        self.graph = None
        if graph:
            self.graph = torch.npu.NPUGraph()
            with torch.npu.graph(
                self.graph, stream=self.forward, auto_dispatch_capture=True
            ):
                self._compute_tail()
            self.forward.synchronize()

    def _delay(self):
        if self.sleep is not None:
            self.sleep(self.SLEEP_CYCLES)
        else:
            for _ in range(128):
                torch.mm(self.delay_input, self.delay_input, out=self.delay_output)

    def _prepare_metadata(self):
        early_event = None
        if self.marker and self.early:
            self.runner._publish_read_done(in_graph=False)
            early_event = self.runner.model_runner.shared_read_done_event
        self._delay()
        if self.graph_enabled:
            self.attn.init_forward_metadata_out_graph(self.batch)
            self.runner._publish_read_done(in_graph=False)
        else:
            with self.forward_context(self.ForwardContext(attn_backend=self.attn)):
                self.attn.init_forward_metadata(self.batch)
        assert self.runner.model_runner.shared_read_done_event is not None
        if not self.marker or self.early:
            # Negative controls suppress or replace the real publication.
            self.runner.model_runner.shared_read_done_event = early_event

    def _compute_tail(self):
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
                with self.extend_plan_ctx:
                    self.attn._publish_eager_shared_read_done(self.batch)
                assert self.runner.model_runner.shared_read_done_event is None
            else:
                self.batch.req_pool_indices.copy_(self.input_requests)
            self._prepare_metadata()
            if self.graph is not None:
                self.graph.replay()
            else:
                self._compute_tail()
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
    from sglang.srt.environ import envs

    cases = []

    def create(**kwargs):
        case = _SharedReadCase(**kwargs)
        cases.append(case)
        return case

    with envs.SGLANG_FORCE_COARSE_WAR_BARRIER.override(False):
        yield create
    # Keep captured buffers/events alive until all submitted work has completed.
    torch.npu.synchronize()


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_without_war_exposes_overwrite(make_case, graph):
    # Verify uses device-only length preparation, keeping the race observable.
    case = make_case(graph=graph, mode="TARGET_VERIFY", marker=False)
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
    case = make_case(graph=graph, mode="TARGET_VERIFY", early=True)
    observed, expected, _ = case.run()
    assert torch.equal(observed, expected + 4), "failed to expose premature read-done"


@pytest.mark.parametrize("plan", [False, True], ids=["same-stream", "plan-streams"])
@pytest.mark.parametrize("wrong_phase", [False, True], ids=["draft-extend", "verify"])
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_mtp_waits_for_the_last_reader(make_case, plan, wrong_phase, graph):
    case = make_case(graph=graph, mode="DRAFT_EXTEND_V2", plan=plan)
    early_releases = 0
    for _ in range(1 if wrong_phase else case.REPLAYS):
        observed, expected, early = case.run(wrong_phase=wrong_phase)
        assert torch.equal(observed, expected + (4 if wrong_phase else 0))
        early_releases += early
    if not wrong_phase:
        assert early_releases, "read-done unexpectedly waited for the whole forward"
