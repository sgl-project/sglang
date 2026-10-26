"""Native CPU types/allocation checks, not Simulator performance acceptance."""

import importlib.util

import pytest
from test_cpu_bootstrap import run_python

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("sglang") is None,
    reason="requires a native SGLang/PyTorch CPU environment",
)


def test_native_result_next_input_and_cpu_reserve_use_native_sglang_types():
    run_python(
        """
from types import SimpleNamespace as NS
from dataclasses import replace
from unittest.mock import patch
import torch
from sglang_simulator.simulation.sglang.hook_bootstrap import install_simulator_hooks
install_simulator_hooks()
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang_simulator.simulation.sglang.dspark import *
from sglang_simulator.simulation.sglang.dspark_runtime import NativeDSparkResults, prepare_cpu_decode
r = DSparkRequest('a', 512, 6, 384, 512, 1, 20, '{"max_new_tokens":20}')
g = DSparkGroup(0, 'TARGET_VERIFY', (r,), '{"path":"CPU fixture"}')
s = DSparkStep(0, DSparkTopology(2,1), (g,), '{"source":"CPU fixture"}')
t = DSparkStepTiming(0, tuple(DSparkPhaseCost(p, .01, 'CPU fixture') for p in s.required_phases))
p = DSparkProgress(0, (('a', 2),), 'CPU fixture')
prediction = DSparkStepResult(s,t,p)
result = NativeDSparkResults(1).build(batch=NS(seq_lens=torch.tensor([512])), prediction=prediction)
assert isinstance(result, GenerationBatchResult)
assert isinstance(result.next_draft_input, DFlashDraftInputV2)
assert result.next_token_ids.tolist() == [1] * 6
assert result.accept_lens.tolist() == [2]
assert result.block_accept_lens is None and result.cap_lens is None
assert result.new_seq_lens.tolist() == [514]
assert result.next_draft_input.new_seq_lens.tolist() == [514]
assert result.next_draft_input.bonus_tokens.tolist() == [1]
# Native Q6 commits can cross the visible output limit. Keep the full result
# and KV progression; the actual scheduler owns finished_len/output exposure.
tail_request = replace(r, output_tokens_before=19)
tail_step = replace(s, groups=(replace(g, requests=(tail_request,)),))
tail_prediction = DSparkStepResult(tail_step, t, DSparkProgress(0, (('a', 6),), 'CPU fixture'))
tail_result = NativeDSparkResults(1).build(batch=NS(seq_lens=torch.tensor([512])), prediction=tail_prediction)
assert tail_request.remaining_output_tokens == 1
assert tail_result.accept_lens.tolist() == [6]
assert tail_result.new_seq_lens.tolist() == [518]
assert tail_result.next_draft_input.new_seq_lens.tolist() == [518]
assert tail_result.next_token_ids.tolist() == [1] * 6
pool = ReqToTokenPool(1, 32, 'cpu', False)
pool.write((1, slice(0,4)), torch.tensor([1,2,3,4], dtype=torch.int32))
req = NS(kv=NS(kv_committed_len=4, kv_allocated_len=4), decode_batch_idx=0, sampling_params=NS(top_k=100))
batch = NS(device='cpu', reqs=[req], req_pool_indices=torch.tensor([1]),
           req_to_token_pool=pool, token_to_kv_pool_allocator=NS(page_size=1),
           tree_cache=object(), maybe_evict_swa=lambda:None)
with patch('sglang.srt.mem_cache.allocation.alloc_token_slots', return_value=torch.arange(5,17,dtype=torch.int64)) as allocate:
    result.next_draft_input.prepare_for_decode(batch)
    allocate.assert_called_once_with(batch.tree_cache, 12)
assert pool.req_to_token[1,:16].tolist() == list(range(1,17))
assert req.kv.kv_allocated_len == 16 and req.kv.kv_committed_len == 4
assert batch.seq_lens_cpu.tolist() == [4]
assert result.next_draft_input.nxt_kv_lens_cpu.tolist() == [16]
assert req.decode_batch_idx == 1
""",
        bootstrap=True,
        cpu=True,
    )


def test_idle_local_result_matches_native_helper_in_both_joint_phases():
    run_python(
        """
from dataclasses import replace
from types import SimpleNamespace as NS
import torch
from sglang_simulator.simulation.sglang.hook_bootstrap import install_simulator_hooks
install_simulator_hooks()
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang_simulator.simulation.sglang.dspark import (
    DSparkGroup, DSparkPhaseCost, DSparkProgress, DSparkRequest,
    DSparkStep, DSparkStepResult, DSparkStepTiming, DSparkTopology,
)
from sglang_simulator.simulation.sglang.dspark_runtime import NativeDSparkResults

# Execute the installed native helper without a model/worker constructor.
# Compare its actual CPU tensor/default contract, not a copied expected result.
native = DSparkWorkerV2._decode_idle_result(
    NS(device='cpu', verify_num_draft_tokens=6), on_publish=None)
for active_mode in ('EXTEND', 'TARGET_VERIFY'):
    for idle_rank in (0, 1):
        req = DSparkRequest('a', 512, 6, 384, 512, 1, 20, '{"max_new_tokens":20}')
        if active_mode == 'EXTEND':
            req = replace(req, target_prefix_len=0, extend_len=17,
                          draft_valid_from=0, draft_valid_to=0, output_tokens_before=0)
        groups = tuple(
            DSparkGroup(rank, 'IDLE', (), '{"source":"CPU fixture"}')
            if rank == idle_rank else
            DSparkGroup(rank, active_mode, (req,), '{"source":"CPU fixture"}')
            for rank in (0, 1))
        step = DSparkStep(0, DSparkTopology(2, 2), groups, '{"source":"CPU fixture"}')
        timing = DSparkStepTiming(0, (DSparkPhaseCost('whole_worker', .01, 'CPU fixture'),))
        progress = DSparkProgress(0, (('a', 1 if active_mode == 'EXTEND' else 2),), 'CPU fixture')
        prediction = DSparkStepResult(step, timing, progress, idle_rank)
        result = NativeDSparkResults(1).build(
            batch=NS(seq_lens=torch.empty((0,), dtype=torch.int64)), prediction=prediction)
        assert type(result) is type(native)
        assert result.logits_output is native.logits_output is None
        for field in ('next_token_ids', 'accept_lens', 'block_accept_lens', 'new_seq_lens'):
            actual, expected = getattr(result, field), getattr(native, field)
            assert actual.dtype == expected.dtype and actual.shape == expected.shape == (0,)
            assert torch.equal(actual, expected)
        assert type(result.next_draft_input) is type(native.next_draft_input)
        for field in ('bonus_tokens', 'new_seq_lens'):
            assert torch.equal(getattr(result.next_draft_input, field),
                               getattr(native.next_draft_input, field))
        for field in ('speculative_output_stride', 'cap_lens',
                      'can_run_cuda_graph', 'speculative_num_draft_tokens'):
            assert getattr(result, field) == getattr(native, field)
        assert prediction.local_group.forward_mode == 'IDLE'
        assert prediction.timing is timing and prediction.progress is progress
""",
        bootstrap=True,
        cpu=True,
    )


def test_native_overlap_scheduler_future_map_and_reference_driver_drain_together():
    pytest.importorskip("infercast.sdk.native_sampling")
    # One native request, including a launched successor whose predecessor
    # finishes it. The installed Scheduler owns the entire event sequence.
    run_python(
        """
from array import array
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace as NS
from unittest.mock import patch
import time
import torch
from sglang_simulator.simulation.sglang.hook_bootstrap import install_simulator_hooks
install_simulator_hooks()
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import NextBatchPlan, Scheduler
from sglang.srt.managers.scheduler_components.batch_result_processor import SchedulerBatchResultProcessor
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel, publish
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang_simulator.simulation.manager import StateManager
from sglang_simulator.simulation.sglang.dspark import (
    STEP_ATTRIBUTE, DSparkPhaseCost, DSparkProgress, DSparkSession,
    DSparkStepTiming, DSparkTopology,
)
from sglang_simulator.simulation.sglang.dspark_reference import ReferenceCPUView
from sglang_simulator.simulation.sglang.dspark_reference_driver import DSparkReferenceDriver
from sglang_simulator.simulation.sglang.dspark_runtime import DSparkNativeRuntime
from sglang_simulator.simulation.sglang.req_stats_manager import request_stats_manager
from sglang_simulator.simulation.sglang.scheduler import C_SchedulerHook as Hook
from sglang_simulator.simulation.types import SimulationMode
from sglang_simulator.time_predictor.base import PredictorError

# Model-free configuration: no tokenizer, checkpoint, device or process group.
publish(ServerArgs(model_path='dummy', device='cpu', page_size=1,
                   speculative_algorithm='DSPARK', speculative_dspark_block_size=5,
                   speculative_num_draft_tokens=6, disable_cuda_graph=True,
                   disable_overlap_schedule=False), role='test')
assert not torch.cuda.is_available()
events, built, processed, views, released = [], [], [], [], []
def noop(*args, **kwargs):
    pass

class CPUEvent:
    # Explicit stream/event protocol fixture; no accelerator/timing claim.
    def record(self, *args, **kwargs):
        events.append(('event_record',))
    def wait(self, *args, **kwargs):
        events.append(('event_wait',))
    def synchronize(self):
        events.append(('event_sync',))

class Cost:
    # This fixture is deliberately separate from verified physical policy.
    native_source_policy = NS(reference_needs_cpu_seq_lens=False)
    calls = 0
    def predict_step(self, before):
        self.calls += 1
        return DSparkStepTiming(before.forward_id, tuple(
            DSparkPhaseCost(p, .01, 'CPU protocol fixture')
            for p in before.required_phases))

class Progress:
    calls = 0
    def predict_progress(self, before):
        self.calls += 1
        return DSparkProgress(before.forward_id,
            tuple((r.rid, 2) for r in before.requests), 'CPU protocol fixture')

def resolve(**kwargs):
    view = kwargs['reference_view']
    assert type(view) is ReferenceCPUView
    assert type(kwargs['native_batch']) is ScheduleBatch
    views.append(view)
    return {'path': 'CPU protocol fixture; no qualified cost lookup'}

runtime = DSparkNativeRuntime(session=DSparkSession(Cost(), Progress()),
    topology=DSparkTopology(2, 1), runtime_profile_json='{"source":"CPU protocol fixture"}',
    execution_resolver=resolve, output_token_id=1, reference_events=True)
build = runtime.results.build
def record_build(**kwargs):
    result = build(**kwargs)
    assert type(result) is GenerationBatchResult
    assert type(result.next_draft_input) is DFlashDraftInputV2
    built.append((kwargs['batch'].copy(), result))
    return result
runtime.results.build = record_build
Hook.INFERENCE_PREDICTOR = NS(supports_dspark=True, dspark_runtime=runtime)
Hook.SIM_MODE, Hook.OVERLAP_SCHEDULE, Hook.COORDINATOR = SimulationMode.OFFLINE, True, None
Hook.ITERATION_STATS = []
StateManager.reset()
StateManager.set_last_real_time_ts(time.time())
request_stats_manager.reset()

params = SamplingParams(max_new_tokens=5, ignore_eos=True, top_k=100)
params.normalize(None)
req = Req('native-overlap', '', array('q', [1] * 512), params)
req.extend_end = req.kv.kv_committed_len = req.kv.kv_allocated_len = 512
req.kv.req_pool_idx = 1
pool = ReqToTokenPool(1, 1024, 'cpu', False)
pool.write((1, slice(0, 512)), torch.arange(512, dtype=torch.int32))
allocator = NS(page_size=1, check_decode_capacity=lambda **kw: True,
               free_group_begin=noop, free_group_end=noop)
cache = NS(page_size=1, supports_swa=lambda: False, supports_mamba=lambda: False,
           flush_pending_backups=noop)
model = NS(is_encoder_decoder=False)
algo = SpeculativeAlgorithm.from_string('DSPARK')
prefill = ScheduleBatch(reqs=[req], device='cpu', forward_mode=ForwardMode.EXTEND,
    model_config=model, enable_overlap=True, spec_algorithm=algo,
    req_to_token_pool=pool, token_to_kv_pool_allocator=allocator, tree_cache=cache,
    req_pool_indices=torch.tensor([1]), req_pool_indices_cpu=torch.tensor([1]),
    seq_lens=torch.tensor([512]), seq_lens_cpu=torch.tensor([512]),
    orig_seq_lens=torch.tensor([512], dtype=torch.int32), seq_lens_sum=512,
    prefix_lens=[0], extend_lens=[512], extend_num_tokens=512,
    prefill_input_ids_cpu=torch.ones(512, dtype=torch.int64))
scheduler = Scheduler.__new__(Scheduler)
for name, value in dict(
    enable_overlap=True, is_generation=True, spec_algorithm=algo,
    running_batch=ScheduleBatch(reqs=[], device='cpu'), last_batch=None,
    gracefully_exit=False, _engine_paused=False, _sched_idled=False,
    scheduler_stage_metrics=None, forward_ct=0, processed_tokens_counter=0,
    scripted_scheduler_hook=None, forward_sleep_time=None,
    disaggregation_mode=DisaggregationMode.NULL, enable_unified_memory=False,
    enable_unified_cache_external_linker=False, enable_hierarchical_cache=False,
    enable_lmcache=False, enable_hicache_storage=False, enable_fpm=False,
    enable_hisparse=False, dllm_config=None, chunked_req=None,
    _pending_chunked_abort_req=None, require_mlp_sync=False,
    prefill_decode_interval=0, _prefill_decode_interval_remaining=0,
    waiting_queue=[], _war_barrier_enabled=False, attn_dp_enabled=False,
    _confidence_budget_prepare=None, batch_record_ct=0, batch_record_buf=[None, None],
    enable_async_d2h_copy=False, token_to_kv_pool_allocator=allocator,
    tree_cache=cache, forward_stream_ctx=nullcontext(), copy_stream_ctx=nullcontext(),
    forward_stream=NS(wait_stream=noop), schedule_stream=object(),
    copy_stream=NS(wait_stream=noop), device_module=NS(Event=CPUEvent),
    profiler_manager=NS(_profile_batch_predicate=noop),
    new_token_ratio_tracker=NS(decay_step=noop),
    beam_coordinator=NS(maybe_select_and_relay=noop),
    ngram_embedding_manager=NS(prepare_for_forward=lambda batch, **kw: batch),
    dp_attn_adapter=NS(maybe_prepare_mlp_sync_batch=lambda batch, **kw: batch,
                       maybe_convert_decode_to_extend=lambda batch: batch),
).items():
    setattr(scheduler, name, value)
scheduler.future_map = FutureMap(torch.device('cpu'), algo, pool, needs_cpu_seq_lens=True)
# Use the real native parallel context and Scheduler object, without creating a
# process group or claiming peer communication. Both local attention groups are
# singleton TP0; their independently supplied attention-DP ranks remain0/1.
for rank in (0, 1):
    with get_parallel().override(tp_size=2, attn_dp_size=2, attn_tp_size=1,
                                 attn_cp_size=1, attn_cp_rank=0, moe_ep_size=1,
                                 moe_dp_size=1, moe_tp_size=2, tp_rank=rank,
                                 attn_dp_rank=rank, attn_tp_rank=0):
        local_driver = DSparkReferenceDriver(runtime)
        local_driver.bind(scheduler)
        assert get_parallel().attn_tp_rank == 0
        local_driver.bind_native_dp_rank(get_parallel().attn_dp_rank)
        assert local_driver.owner is scheduler
        assert local_driver.adapter.ledger.local_attention_dp_rank == rank
        local_driver.drained_reset()
        try:
            local_driver.bind_native_dp_rank(1 - rank)
        except PredictorError:
            pass
        else:
            raise AssertionError('native local rank changed after binding/reset')
worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
worker.query_token_num, worker.verify_num_draft_tokens = 5, 6
# Native __getattr__ delegates optional resource hooks to this owning target.
# A CPU target has no MLX prepare_for_kv_cache_release hook; preserve that
# ordinary absence instead of bypassing the native result processor.
target_worker = NS(device='cpu', model_runner=NS(
    req_to_token_pool=pool, token_to_kv_pool_allocator=allocator))
worker._target_worker = target_worker
assert worker.target_worker is target_worker
assert getattr(worker, 'prepare_for_kv_cache_release', None) is None
scheduler.model_worker = worker
metrics = NS(record_scheduler_active=noop, log_batch_result_stats=noop,
             update_device_timer=noop, num_generated_tokens=0, forward_ct_decode=0,
             update_spec_metrics=noop, snapshot_spec_decode_metrics=noop,
             report_decode_stats=noop)
scheduler.metrics_reporter = metrics
scheduler.load_publisher = NS(publish_load_stat=noop)
scheduler.load_inquirer = NS(get_loads=noop)
scheduler.publish_load_snapshot = lambda **kw: None
scheduler._record_step_counters = noop
scheduler.maybe_send_health_check_signal = noop
scheduler.batch_result_processor = SchedulerBatchResultProcessor(
    is_generation=True, disaggregation_mode=DisaggregationMode.NULL,
    enable_overlap=True, enable_overlap_mlx=False, model_config=model,
    token_to_kv_pool_allocator=allocator, tree_cache=cache,
    hisparse_coordinator=None, req_to_token_pool=pool, decode_offload_manager=None,
    metrics_collector=None, metrics_reporter=metrics, draft_worker=worker,
    model_worker=worker, logprob_result_processor=None,
    output_streamer=NS(stream_output=noop), beam_coordinator=scheduler.beam_coordinator,
    abort_request=noop)

# Only initial admission and resource/telemetry boundaries are fixtures. Native
# selection, decode preparation, overlap decision, launch and processing remain.
def admit(running_batch):
    return NextBatchPlan(batch_to_run=prefill if scheduler.forward_ct == 0 else None,
                         running_batch=running_batch)
scheduler.get_new_batch_prefill = admit
def ingest():
    assert scheduler.forward_ct <= 4, 'native loop failed to drain'
    if scheduler.forward_ct and not scheduler.result_queue:
        assert req.finished()
        scheduler.gracefully_exit = True
scheduler.ingest_requests = ingest
scheduler.on_idle = noop

publish_native = scheduler.future_map.publish
def publish_cpu(*args, **kwargs):
    events.append(('publish', runtime._forward_id, len(runtime.reference_driver.adapter.ledger.pending)))
    return publish_native(*args, **kwargs)
scheduler.future_map.publish = publish_cpu
def process_native(original):
    def process(self, batch, result):
        fid = getattr(result, STEP_ATTRIBUTE).step.forward_id
        assert type(batch) is ScheduleBatch
        events.append(('process', fid))
        processed.append(fid)
        return original(self, batch, result)
    return process
next_slot = [512]
def allocate(cache, count):
    start = next_slot[0]
    next_slot[0] += count
    return torch.arange(start, start + count, dtype=torch.int64)

with ExitStack() as patches:
    patches.enter_context(patch.object(torch.get_device_module('cpu'), 'Event', CPUEvent, create=True))
    patches.enter_context(patch('sglang.srt.mem_cache.allocation.alloc_token_slots', side_effect=allocate))
    patches.enter_context(patch('sglang.srt.managers.scheduler_components.batch_result_processor.checkpoint_kv_cache', noop))
    patches.enter_context(patch('sglang.srt.managers.scheduler_components.batch_result_processor.release_kv_cache',
                               side_effect=lambda request, *a, **kw: released.append(request.rid)))
    for name in ('process_batch_result_prefill', 'process_batch_result_decode'):
        patches.enter_context(patch.object(SchedulerBatchResultProcessor, name,
            process_native(getattr(SchedulerBatchResultProcessor, name))))
    scheduler.event_loop_overlap()

driver = runtime.reference_driver
assert driver.owner is scheduler and driver.failure is None
assert processed == [0, 1, 2, 3] and runtime._forward_id == 4
assert len(built) == runtime.session.cost_model.calls == StateManager.get_iteration() == 4
assert runtime.session.progress_model.calls == 3
for fid in (1, 2, 3):
    assert events.index(('publish', fid, 1)) < events.index(('process', fid - 1))
assert events.index(('publish', 0, 0)) < events.index(('publish', 1, 1))
assert sum(e[0] == 'event_wait' for e in events) == 3
assert views[0].seq_lens_cpu == (512,)
assert all(view.seq_lens_cpu is None for view in views[1:])
# CPU relay retained a real mirror even though the explicitly synthetic
# reference policy fixture opts out. Never derive a HIP policy from CPU flags.
assert scheduler.future_map.needs_cpu_seq_lens is True
assert req.finished() and len(req.output_ids) == req.finished_len == 5
assert released == [req.rid] and req.kv.kv_committed_len == 516
assert not scheduler.result_queue and scheduler.running_batch.is_empty()
assert driver.adapter.ledger.pending == driver.adapter.ledger.requests == ()
assert driver.selection is driver.processing is None and driver.retired_request_ids == set()
try:
    runtime.session.history_for(req.rid)
except PredictorError:
    pass
else:
    raise AssertionError('finished native request retained draft history')
assert scheduler._sglang_simulator_dspark_pending_results == 0
assert all(getattr(result, '_sglang_simulator_dspark_accounted') for _, result in built)
assert len(Hook.ITERATION_STATS) == 4
assert sum(row['forward_latency'] for row in Hook.ITERATION_STATS) == sum(
    getattr(result, STEP_ATTRIBUTE).timing.seconds for _, result in built)
# The already-launched successor incurs physical cost once, without a fourth
# output/ITL event. Reprocessing must reject before touching native state/time.
assert len(request_stats_manager.get_req_stats(req.rid).gen_token_latencies) == 3
before = StateManager.get_global_clock()
try:
    scheduler.process_batch_result(*built[-1])
except PredictorError as error:
    assert error.code == 'invalid_dspark_step'
else:
    raise AssertionError('native result accounting allowed replay')
assert StateManager.get_global_clock() == before and processed == [0, 1, 2, 3]
""",
        bootstrap=True,
        cpu=True,
    )
