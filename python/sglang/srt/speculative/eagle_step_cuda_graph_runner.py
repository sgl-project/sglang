# SPDX-License-Identifier: Apache-2.0
"""One CUDA graph per EAGLE v2 decode step (``--enable-speculative-step-cuda-graph``).

At small batch sizes an EAGLE v2 decode step is three CUDA graph replays
(draft, target verify, draft extend) with many small kernels and host work
between them, so the GPU idles between launches. This runner captures the
worker's own decode step once per key (batch size, sampling setup and the
layout of the step's input tensors) and replays it as one graph:

* The capture runs ``EAGLEWorkerV2.forward_batch_generation`` on a copy of the
  batch whose per-step device tensors are runner-owned static buffers, inside
  a ``BreakableCUDAGraphCapture`` on a dedicated stream with a private memory
  pool per key. The three per-stage graphs are spliced in as child-graph nodes
  (``ChildGraphCapture``); the eager work between them is captured from the
  same code that runs it on the regular path.
* A replay copies the live input tensors into the static buffers and launches
  the graph. Host work that must happen every step runs on the host at replay:
  the FutureMap publish, the shared-read fence and, in the simulated
  acceptance benchmarking mode, the acceptance length draw (same function,
  same host RNG stream, once per step).
* Constrained decoding: the verify tree is copied to pinned host buffers inside
  the graph, and the grammar bitmask is built on the host between two graph
  segments (an ``eager_on_graph`` break), still overlapping the target verify
  forward; the captured sampling reads the bitmask from a static buffer.
* Random draws of sampled verification are captured inside the torch graph,
  whose replay advances the CUDA generator exactly like the eager draws.

A key is captured at its first eligible step after ``_NUM_WARMUP_STEPS``
regular steps, if every rank has enough free device memory (the largest key
captured so far plus ``_CAPTURE_MEMORY_HEADROOM``). The last warmup step counts
the per-stage graph replays; a key whose step does not replay exactly the three
per-stage graphs is never captured. Any step the runner cannot capture runs the
regular path; the reason is counted (``/get_server_info``) and logged once, and
a memory refusal is logged once per key with the free and needed memory.

Device memory in use can grow while serving and leave too little for a later
capture, so the HTTP server's start-up runs batches of 1 to ``max_batch_size``
requests with the server's default sampling (``entrypoints/warmup.py``): the
keys of client requests that send no sampling fields are then captured before
serving starts, next to the greedy batch-size-1 key of SGLang's own warm-up
request. A capture refused while serving although PyTorch caches enough unused
memory releases that cache once for its key and checks again.
"""

from __future__ import annotations

import copy
import dataclasses
import gc
import logging
import time
from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

from sglang.srt.constrained.base_grammar_backend import (
    GrammarMask,
    InvalidGrammarObject,
)
from sglang.srt.distributed.parallel_state import get_tp_group, graph_capture
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import SharedReadEnds
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    FullCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    eager_on_graph,
)
from sglang.srt.model_executor.runner_backend_utils.child_graph_capture import (
    ChildGraphCapture,
)
from sglang.srt.model_executor.runner_utils.pool import graph_pool_replay_scope
from sglang.srt.runtime_context import get_stream
from sglang.srt.utils import is_cuda

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.utils import GenerationBatchResult
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2

logger = logging.getLogger(__name__)

# Largest decode batch size captured; larger batches are rarely launch bound.
_MAX_BS = 8
# Captured keys per worker; a new key past this runs the regular step.
_MAX_KEYS = 32
# Regular steps of a key before it is captured; the last one counts the
# per-stage graph replays.
_NUM_WARMUP_STEPS = 2
# Per-stage graphs of one decode step: draft, target verify, draft extend.
_NUM_STAGE_GRAPHS = 3
# Keys whose capture may fail before the runner turns itself off.
_MAX_FAILED_KEYS = 4
# Steps whose outputs may still be read by the scheduler's device-to-host
# copy while the next steps replay.
_OUTPUT_RING_SIZE = 2
_SIM_LEN_SLOTS = 4
# Free device memory a capture needs beyond the largest key captured so far.
_CAPTURE_MEMORY_HEADROOM = 1 << 30
_FIRST_KEY_MEMORY_ESTIMATE = 256 << 20
_MEMORY_RETRY_STEPS = 256
# Releases of PyTorch's cached memory for a refused capture: once per key.
_MAX_CACHE_RELEASES = _MAX_KEYS
_MAX_WARMUP_KEYS = 1024
_CAPTURE_STREAM_NAME = "speculative_step_cuda_graph"

# Device tensors of the schedule batch that a decode step reads and that change
# from step to step. Every CUDA tensor of the draft input and of the sampling
# info is a per-step input as well, except the ones listed as step outputs.
_BATCH_STEP_INPUTS = ("req_pool_indices", "seq_lens", "orig_seq_lens")
# Assigned by the decode step before anything reads them (the capture copy sets
# them to None, so a read before the assignment fails the capture loudly).
_BATCH_STEP_OUTPUTS = ("input_ids", "out_cache_loc")
_DRAFT_INPUT_STEP_OUTPUTS = ("positions", "kv_indptr", "kv_indices")

_UNSUPPORTED_ENVS = (
    envs.SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH,
    envs.SGLANG_ENABLE_CUDA_GRAPH_DEDUP,
    envs.SGLANG_ENABLE_METADATA_GLUE_GRAPH,
    envs.SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH,
    envs.SGLANG_Q35MF_EDGE_SPARSE_VERIFY_SAMPLING,
    envs.SGLANG_ENABLE_GRAPH_POOL_BORROW,
    envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM,
    envs.SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION,
    envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH,
    envs.SGLANG_ENABLE_NAN_LOGITS_CHECK,
    envs.SGLANG_ENABLE_ASYNC_ASSERT,
    envs.SGLANG_SANITIZE_NAN_LOGITS,
)


def _cuda_python_available() -> bool:
    try:
        from cuda.bindings import runtime  # noqa: F401
    except ImportError:
        return False
    return True


def step_cuda_graph_unsupported_reason(server_args: ServerArgs) -> Optional[str]:
    """Why the step CUDA graph cannot run under these server args, or None."""
    from sglang.srt.arg_groups.overrides import resolved_view, resolving_view
    from sglang.srt.runtime_context import attn_dp_enabled_of

    cfg = resolving_view(server_args)
    if cfg.speculative_algorithm not in ("EAGLE", "EAGLE3", "NEXTN"):
        return (
            f"speculative_algorithm={cfg.speculative_algorithm} "
            "(only EAGLE / EAGLE3 / NEXTN are supported)"
        )
    if cfg.enable_multi_layer_eagle:
        return "enable_multi_layer_eagle is not supported"
    if cfg.speculative_eagle_topk is not None and cfg.speculative_eagle_topk != 1:
        return (
            f"speculative_eagle_topk={cfg.speculative_eagle_topk} (only 1 is supported)"
        )
    if cfg.speculative_use_rejection_sampling or cfg.speculative_use_block_verification:
        return "rejection sampling is not supported"
    if cfg.speculative_adaptive:
        return "speculative_adaptive is not supported"
    if not is_cuda():
        return "only CUDA devices are supported"
    if not _cuda_python_available():
        return "the 'cuda-python' package is not installed"
    if torch.cuda.get_allocator_backend() != "native":
        return "only the native CUDA caching allocator is supported"
    if cfg.pp_size > 1:
        return "pipeline parallelism is not supported"
    if attn_dp_enabled_of(resolved_view(server_args)):
        return "attention DP is not supported"
    unsupported_features = (
        ("enable_two_batch_overlap", cfg.enable_two_batch_overlap),
        ("enable_single_batch_overlap", cfg.enable_single_batch_overlap),
        ("enable_pdmux", cfg.enable_pdmux),
        ("enable_lora", cfg.enable_lora or cfg.lora_paths),
        ("disaggregation_mode", cfg.disaggregation_mode != "null"),
        ("enable_hisparse", cfg.enable_hisparse),
        ("enable_memory_saver", cfg.enable_memory_saver),
        ("debug_cuda_graph", cfg.debug_cuda_graph),
        ("disable_cuda_graph", cfg.disable_cuda_graph),
        ("expert_distribution_recorder_mode", cfg.expert_distribution_recorder_mode),
        ("enable_eplb", cfg.enable_eplb),
        ("enable_return_routed_experts", cfg.enable_return_routed_experts),
        ("enable_return_indexer_topk", cfg.enable_return_indexer_topk),
        ("kv_canary", cfg.kv_canary not in (None, "none")),
        ("elastic_ep_backend", cfg.elastic_ep_backend),
    )
    for name, value in unsupported_features:
        if value:
            return f"{name} is not supported"
    for env in _UNSUPPORTED_ENVS:
        if env.get():
            return f"{env.name} is not supported"
    return None


def simulated_accept_index_from_device_length(
    accept_index: torch.Tensor,
    predict: torch.Tensor,
    num_correct_drafts: torch.Tensor,
    candidates: torch.Tensor,
    target_predict: Optional[torch.Tensor],
    spec_steps: int,
    accept_len: torch.Tensor,
    simulate_acc_token_mode: str,
) -> torch.Tensor:
    """``spec_utils.generate_simulated_accept_index`` for an acceptance length
    that was already drawn and is read from ``accept_len`` (a device int32
    tensor of shape [1]), so that every shape is independent of its value."""
    bs = accept_index.shape[0]
    width = spec_steps + 1
    cols = torch.arange(width, dtype=torch.int32, device=accept_index.device).view(
        1, width
    )
    length = accept_len.view(1, 1)
    first = accept_index[:, :1]
    nodes = first + cols
    sim_accept_index = torch.where(cols < length, nodes, torch.full_like(nodes, -1))
    num_correct_drafts.copy_((accept_len - 1).expand_as(num_correct_drafts))
    if simulate_acc_token_mode != "real-draft-token":
        predict.fill_(100)  # some legit token id, as the regular path
        return sim_accept_index

    nodes = nodes.view(-1).long()
    draft_cols = (cols + 1).clamp(max=candidates.shape[1] - 1).view(-1).long()
    drafts = candidates[:, draft_cols].to(predict.dtype)
    bonus = target_predict.gather(1, (length - 1).long().expand(bs, 1))
    current = predict[nodes].view(bs, width)
    updated = torch.where(
        cols < length - 1,
        drafts,
        torch.where(cols == length - 1, bonus.to(predict.dtype), current),
    )
    predict.index_copy_(0, nodes, updated.view(-1))
    return sim_accept_index


def _tp_min(*values: int) -> Tuple[int, ...]:
    """Element-wise MIN over the TP group (host collective)."""
    group = get_tp_group()
    if group.world_size == 1:
        return tuple(int(v) for v in values)
    tensor = torch.tensor([int(v) for v in values], dtype=torch.int64)
    torch.distributed.all_reduce(
        tensor, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group
    )
    return tuple(tensor.tolist())


def _host_copies_in_capture(stream: torch.cuda.Stream) -> List[Tuple[str, int]]:
    """("read" | "write", host address) of every copy node in the graph being
    captured on ``stream`` (child graphs excluded) that touches host memory. A
    replay would repeat such a copy with the values or addresses of capture time."""
    from cuda.bindings import runtime as rt

    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.cuda_utils import (
        checkCudaErrors,
    )

    status, _, graph, _, _, _ = checkCudaErrors(
        rt.cudaStreamGetCaptureInfo(stream.cuda_stream)
    )
    if status != rt.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive:
        return []
    _, num_nodes = checkCudaErrors(rt.cudaGraphGetNodes(graph, 0))
    nodes, _ = checkCudaErrors(rt.cudaGraphGetNodes(graph, num_nodes))
    host_types = (
        rt.cudaMemoryType.cudaMemoryTypeHost,
        rt.cudaMemoryType.cudaMemoryTypeUnregistered,
    )
    found = []
    for node in nodes or []:
        node_type = checkCudaErrors(rt.cudaGraphNodeGetType(node))
        if node_type != rt.cudaGraphNodeType.cudaGraphNodeTypeMemcpy:
            continue
        params = checkCudaErrors(rt.cudaGraphMemcpyNodeGetParams(node))
        for side, ptr in (
            ("read", int(params.srcPtr.ptr)),
            ("write", int(params.dstPtr.ptr)),
        ):
            attrs = checkCudaErrors(rt.cudaPointerGetAttributes(ptr))
            if attrs.type in host_types:
                found.append((side, ptr))
    return found


def _check_host_copies(allowed_writes=()) -> None:
    """Fail the capture if the segment being captured copies from host memory or
    to host memory other than ``allowed_writes`` (runner-owned pinned buffers)."""
    for side, ptr in _host_copies_in_capture(torch.cuda.current_stream()):
        if side == "read" or ptr not in allowed_writes:
            raise RuntimeError(
                f"the captured step would {side} host memory at {ptr:#x} "
                "with its capture-time address or value"
            )


def _is_device_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and value.is_cuda


def _step_tensors(batch: ScheduleBatch) -> List[Tuple[str, str, torch.Tensor]]:
    """(holder, attribute, tensor) for every per-step device input of a decode step."""
    tensors = []
    for name in _BATCH_STEP_INPUTS:
        tensor = getattr(batch, name, None)
        if _is_device_tensor(tensor):
            tensors.append(("batch", name, tensor))
    for holder, obj, outputs in (
        ("spec_info", batch.spec_info, _DRAFT_INPUT_STEP_OUTPUTS),
        ("sampling_info", batch.sampling_info, ()),
    ):
        for name, value in sorted(vars(obj).items()):
            if name not in outputs and _is_device_tensor(value):
                tensors.append((holder, name, value))
    return tensors


def _holders(batch: ScheduleBatch) -> Dict[str, Any]:
    return {
        "batch": batch,
        "spec_info": batch.spec_info,
        "sampling_info": batch.sampling_info,
    }


# try_small_copy's per-tensor element limit.
_SMALL_COPY_MAX_NUMEL = 8192


def _ranges_overlap(a: List[torch.Tensor], b: List[torch.Tensor]) -> bool:
    def span(t):
        storage = t.untyped_storage()
        return storage.data_ptr(), storage.data_ptr() + storage.nbytes()

    spans = sorted(span(t) for t in b if t.numel())
    for t in a:
        if not t.numel():
            continue
        lo, hi = span(t)
        if any(s < hi and lo < e for s, e in spans):
            return True
    return False


def _copy_step_inputs(
    static_inputs: List[torch.Tensor],
    live_inputs: List[torch.Tensor],
    small_copy=None,
) -> str:
    """Copy a replay's live inputs into the graph's static input buffers.

    With ``small_copy`` (``SGLANG_SPEC_STEP_GRAPH_FUSED_INPUT_COPY``) the copies
    are one launch when every tensor qualifies for it ("fused"); otherwise the
    small tensors are one launch and the others one copy each ("partial", only
    when no live tensor shares memory with a static buffer, so the order of the
    copies cannot matter); otherwise one copy per tensor ("per_tensor").
    """
    if small_copy is not None:
        if small_copy(static_inputs, live_inputs):
            return "fused"
        small = [
            i for i, t in enumerate(static_inputs) if t.numel() <= _SMALL_COPY_MAX_NUMEL
        ]
        if 2 <= len(small) < len(static_inputs) and not _ranges_overlap(
            static_inputs, live_inputs
        ):
            if small_copy(
                [static_inputs[i] for i in small], [live_inputs[i] for i in small]
            ):
                chosen = set(small)
                # One copy_ per tensor, as torch._foreach_copy_ does for a
                # mixed-dtype list (a same-dtype list would take its fused
                # path, which is not a plain byte copy).
                for i in range(len(static_inputs)):
                    if i not in chosen:
                        static_inputs[i].copy_(live_inputs[i], non_blocking=True)
                return "partial"
    torch._foreach_copy_(static_inputs, live_inputs, non_blocking=True)
    return "per_tensor"


class _GrammarStep:
    """Capture-time overrides and replay-time host work of constrained decoding."""

    def __init__(self, runner: EagleStepCudaGraphRunner):
        self.runner = runner
        self.host_tree: Optional[Tuple[torch.Tensor, ...]] = None
        self.tree_ready = torch.cuda.Event()
        self.mask: Optional[torch.Tensor] = None

    def overrides(self) -> Dict[str, Any]:
        return {
            "spec_grammar_tree_from_device": self.tree_from_device,
            "spec_grammar_vocab_mask": self.vocab_mask,
        }

    def tree_from_device(
        self, original, cls, retrieve_next_token, retrieve_next_sibling, draft_token
    ):
        tensors = (retrieve_next_token, retrieve_next_sibling, draft_token)
        # Runner-owned pinned buffers, refilled by every replay.
        self.host_tree = tuple(
            torch.empty(t.shape, dtype=t.dtype, pin_memory=True) for t in tensors
        )
        for host, tensor in zip(self.host_tree, tensors):
            host.copy_(tensor, non_blocking=True)
        _check_host_copies({host.data_ptr() for host in self.host_tree})
        eager_on_graph(self._record_tree_ready)()
        return cls(self.host_tree, None)

    def _record_tree_ready(self) -> None:
        self.tree_ready.record()

    def vocab_mask(self, original, *, reqs, tree, sampling_info, device, barrier):
        grammar = reqs[0].grammar
        host_mask = grammar.allocate_vocab_mask(
            vocab_size=sampling_info.vocab_size,
            batch_size=self.host_tree[2].numel(),
            device="cpu",
        )
        self.mask = torch.empty(host_mask.shape, dtype=host_mask.dtype, device=device)
        sampling_info.grammar_mask = None
        _check_host_copies()
        eager_on_graph(self._build_mask)()
        return GrammarMask(grammar, self.mask)

    def _build_mask(self) -> None:
        # Same steps as spec_utils.build_grammar_vocab_mask, on the live batch, into
        # the runner-owned mask (a graph break returns None).
        from sglang.srt.speculative.spec_utils import generate_token_bitmask

        batch = self.runner._live_batch
        barrier = self.runner._live_grammar_barrier
        if barrier is not None:
            barrier()
        self.tree_ready.synchronize()
        vocab_mask, _ = generate_token_bitmask(
            batch.reqs, *self.host_tree, batch.sampling_info.vocab_size
        )
        self.mask.copy_(vocab_mask, non_blocking=True)
        batch.sampling_info.grammar_mask = None


@dataclasses.dataclass
class _StepGraph:
    """One captured decode step and everything its replays read or write."""

    graph: BreakableCUDAGraph
    # (holder, attribute) of each static input, and the static buffers.
    input_names: List[Tuple[str, str]]
    static_inputs: List[torch.Tensor]
    # Capture-time result; replays return shallow copies of it.
    result: GenerationBatchResult
    # (backend, graph epoch) of each spliced per-stage graph.
    stage_graphs: List[Tuple[Any, Optional[int]]]
    ring_slot: torch.Tensor
    ring_next_token_ids: Optional[torch.Tensor] = None
    ring_accept_lens: Optional[torch.Tensor] = None
    ring_results: List[Optional[GenerationBatchResult]] = dataclasses.field(
        default_factory=lambda: [None] * _OUTPUT_RING_SIZE
    )
    # One (next_token_ids, accept_lens) view per ring slot, handed to results.
    ring_views: List[Tuple[torch.Tensor, torch.Tensor]] = dataclasses.field(
        default_factory=list
    )
    # Simulated acceptance: device length read by the graph and the arguments
    # of the host draw.
    sim_accept_len: Optional[torch.Tensor] = None
    sim_draw_args: Optional[Tuple[float, str, int, Optional[torch.Generator]]] = None
    grammar: Optional[_GrammarStep] = None
    num_replays: int = 0
    memory_bytes: int = 0
    # What the key is, for the stats a server reports.
    batch_size: int = 0
    greedy: bool = False
    constrained: bool = False


class EagleStepCudaGraphRunner:
    """Captures and replays whole EAGLE v2 decode steps for one worker."""

    def __init__(self, worker: EAGLEWorkerV2):
        self.worker = worker
        draft_worker = worker.draft_worker
        self.target_model_runner = worker.target_worker.model_runner
        self.draft_model_runner = draft_worker.draft_runner
        self.stage_runners = (
            draft_worker.cuda_graph_runner,
            self.target_model_runner.decode_cuda_graph_runner,
            draft_worker.cuda_graph_runner_for_draft_extend,
        )
        self.device = worker.device

        self._init_state()

        self.disabled_reason = self._construction_unsupported_reason()
        if self.disabled_reason is not None:
            logger.warning(
                "enable_speculative_step_cuda_graph disabled: %s. "
                "Falling back to the per-stage CUDA graphs.",
                self.disabled_reason,
            )
            return

        self._draft_extend_read_ends = self._draft_extend_backend().shared_read_ends(
            ForwardMode.DRAFT_EXTEND_V2
        )
        tp_group = get_tp_group()
        self._sampled_broadcast_ok = tp_group.world_size == 1 or (
            tp_group.pynccl_comm is not None
        )
        self._stream = get_stream(_CAPTURE_STREAM_NAME)
        # A larger batch does not replay every per-stage graph.
        self.max_batch_size = min(
            [_MAX_BS] + [getattr(r, "max_bs", _MAX_BS) for r in self.stage_runners]
        )
        logger.info(
            "Speculative step CUDA graph enabled: each decode step is captured "
            "at its first eligible use (batch size <= %d).",
            _MAX_BS,
        )
        if envs.SGLANG_SPEC_STEP_GRAPH_FUSED_INPUT_COPY.get():
            from sglang.kernels.ops.memory.small_copy import try_small_copy

            self._small_copy = try_small_copy

    # ------------------------------------------------------------------ setup

    def _init_state(self) -> None:
        self._steps: Dict[tuple, _StepGraph] = {}
        # Graphs that must stay alive although they are not replayed (a capture
        # that a peer rank failed); released together with every key.
        self._parked_steps: List[_StepGraph] = []
        self._warmup_counts: Dict[tuple, int] = {}
        self._uncapturable: Dict[tuple, str] = {}
        self._memory_retry_at: Dict[tuple, int] = {}
        self._last_memory_refusal: Optional[Dict[str, Any]] = None
        # Keys that released PyTorch's cached memory for their capture.
        self._cache_released: set = set()
        # Largest batch size the start-up warm-up captures (see __init__).
        self.max_batch_size = _MAX_BS
        self._output_layouts: Dict[tuple, tuple] = {}
        self._failed_keys = 0
        self._num_steps_seen = 0
        self._in_worker_call = False
        self._fallbacks: Counter = Counter()
        self._stats = Counter()
        self._sim_host: Optional[torch.Tensor] = None
        self._sim_events: List[Optional[torch.cuda.Event]] = [None] * _SIM_LEN_SLOTS
        self._sim_slot = 0
        # SGLANG_SPEC_STEP_GRAPH_FUSED_INPUT_COPY: the one-launch input copy.
        self._small_copy = None
        self._logged_input_copy = set()
        self._stream: Optional[torch.cuda.Stream] = None
        self._sampled_broadcast_ok = True
        # The batch and grammar barrier of the step being replayed, for host breaks.
        self._live_batch = None
        self._live_grammar_barrier = None
        self._draft_extend_read_ends = SharedReadEnds.UNKNOWN
        self.disabled_reason: Optional[str] = None

    def _draft_extend_backend(self):
        draft_worker = self.worker.draft_worker
        return (
            draft_worker.draft_extend_attn_backend
            or self.draft_model_runner.attn_backend
        )

    def _construction_unsupported_reason(self) -> Optional[str]:
        names = ("draft", "target verify", "draft extend")
        for name, runner in zip(names, self.stage_runners):
            if runner is None:
                return f"no {name} CUDA graph runner"
            backend = getattr(runner, "backend", None)
            if not isinstance(backend, FullCudaGraphBackend):
                return f"the {name} CUDA graph backend is not the full backend"
            if not backend.keep_raw_graph:
                return f"the {name} CUDA graphs do not keep their raw graphs"
        draft_worker = self.worker.draft_worker
        backends = (
            (draft_worker.draft_attn_backend, ForwardMode.DECODE),
            (self.target_model_runner.attn_backend, ForwardMode.TARGET_VERIFY),
            (self._draft_extend_backend(), ForwardMode.DRAFT_EXTEND_V2),
        )
        for backend, mode in backends:
            capturable = getattr(backend, "out_graph_metadata_capturable", None)
            if capturable is None or not capturable(mode):
                return (
                    f"attention backend {type(backend).__name__} does not declare "
                    f"capturable {mode.name} metadata"
                )
        if self.worker.plan_stream is not None:
            return "the overlap plan stream is not supported"
        for model_runner in (self.target_model_runner, self.draft_model_runner):
            if model_runner.device_timer is not None:
                return "the device timer is not supported"
            if model_runner.canary_manager is not None:
                return "the KV canary is not supported"
            if model_runner.msprobe_debugger is not None:
                return "the msprobe debugger is not supported"
        return None

    # ------------------------------------------------------------- per step

    def _ineligible_reason(
        self, batch: ScheduleBatch, pp_proxy_tensors: Optional[Any]
    ) -> Optional[str]:
        if batch.forward_mode.is_idle():
            return "idle batch"
        bs = batch.seq_lens.shape[0]
        if bs > _MAX_BS:
            return f"batch size above {_MAX_BS}"
        spec_info = batch.spec_info
        if spec_info is None:
            return "no draft input"
        if spec_info.is_verify_input():
            return "prebuilt verify input"
        if self.worker.speculative_num_steps == 0:
            return "speculative_num_steps is 0"
        if batch.return_logprob:
            return "logprobs requested"
        if batch.return_hidden_states:
            return "hidden states requested"
        if pp_proxy_tensors is not None:
            return "pipeline proxy tensors"
        if batch.has_grammar:
            for req in batch.reqs:
                if req.grammar is None:
                    return "constrained and unconstrained requests mixed"
                if isinstance(req.grammar, InvalidGrammarObject):
                    return "invalid grammar"
        if batch.seq_lens_cpu is not None or batch.seq_lens_sum is not None:
            return "host sequence lengths"
        if batch.hisparse_coordinator is not None:
            return "hisparse"
        if batch.mamba_track_indices is not None:
            return "mamba state tracking"
        if batch.multimodal_inputs is not None and any(
            x is not None for x in batch.multimodal_inputs
        ):
            return "multimodal inputs"
        sampling_info = batch.sampling_info
        if sampling_info.has_custom_logit_processor:
            return "custom logit processor"
        if (
            sampling_info.acc_additive_penalties is not None
            or sampling_info.acc_scaling_penalties is not None
        ):
            return "penalties"
        if sampling_info.logit_bias is not None:
            return "logit bias"
        if sampling_info.has_watermark_candidates:
            return "watermark"
        if sampling_info.sampling_mask_batch_indices is not None:
            return "sampling masks requested"
        if not sampling_info.is_all_greedy and not self._sampled_broadcast_ok:
            return "sampled verify without a capturable TP broadcast"
        for name, value in vars(batch).items():
            if (
                name not in _BATCH_STEP_INPUTS
                and name not in _BATCH_STEP_OUTPUTS
                and _is_device_tensor(value)
            ):
                return f"batch tensor {name}"
        return None

    def _key(self, batch: ScheduleBatch, step_tensors) -> tuple:
        sampling_info = batch.sampling_info
        spec_info = batch.spec_info
        return (
            batch.seq_lens.shape[0],
            sampling_info.is_all_greedy,
            sampling_info.need_top_k_sampling,
            sampling_info.need_top_p_sampling,
            sampling_info.need_min_p_sampling,
            spec_info.capture_hidden_mode,
            spec_info.future_dsa_topk_indices_available,
            spec_info.cuda_graph_compatible,
            (
                tuple(type(req.grammar).__name__ for req in batch.reqs)
                if batch.has_grammar
                else None
            ),
            tuple(
                (holder, name, tuple(t.shape), t.dtype)
                for holder, name, t in step_tensors
            ),
        )

    def _fallback(self, reason: str) -> None:
        if reason not in self._fallbacks:
            logger.info(
                "Speculative step CUDA graph: running the regular step (%s); "
                "logged once per reason.",
                reason,
            )
        self._fallbacks[reason] += 1

    def try_replay(
        self,
        batch: ScheduleBatch,
        on_publish=None,
        grammar_barrier=None,
        pp_proxy_tensors=None,
    ) -> Optional[GenerationBatchResult]:
        """Run this decode step as one graph. Returns None when the caller must
        run the regular step instead."""
        if self._in_worker_call or self.disabled_reason is not None:
            return None
        self._num_steps_seen += 1
        reason = self._ineligible_reason(batch, pp_proxy_tensors)
        if reason is not None:
            self._fallback(reason)
            return None
        step_tensors = _step_tensors(batch)
        key = self._key(batch, step_tensors)

        step = self._steps.get(key)
        if step is not None:
            if all(
                backend.graph_epoch == epoch for backend, epoch in step.stage_graphs
            ):
                return self._replay(step, batch, on_publish, grammar_barrier)
            self._reset("the per-stage CUDA graphs were recaptured")

        reason = self._uncapturable.get(key)
        if reason is not None:
            self._fallback(reason)
            return None
        if len(self._steps) >= _MAX_KEYS:
            self._fallback("captured key limit")
            return None

        warmed = self._warmup_counts.pop(key, 0)
        if warmed < _NUM_WARMUP_STEPS:
            self._warmup_counts[key] = warmed + 1
            while len(self._warmup_counts) > _MAX_WARMUP_KEYS:
                self._warmup_counts.pop(next(iter(self._warmup_counts)))
            if warmed + 1 == _NUM_WARMUP_STEPS:
                return self._probe_step(key, batch, on_publish, grammar_barrier)
            return None
        self._warmup_counts[key] = warmed
        if self._memory_retry_at.get(key, 0) > self._num_steps_seen:
            self._fallback("not enough free memory to capture")
            return None
        return self._capture_and_replay(
            key, batch, step_tensors, on_publish, grammar_barrier
        )

    # ------------------------------------------------------- warmup probe

    def _run_regular_step(self, batch, on_publish, grammar_barrier, pp_proxy_tensors):
        self._in_worker_call = True
        try:
            return self.worker.forward_batch_generation(
                batch,
                on_publish=on_publish,
                grammar_barrier=grammar_barrier,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        finally:
            self._in_worker_call = False

    def _probe_step(self, key, batch, on_publish, grammar_barrier):
        """Run a regular step while counting its per-stage graph replays."""
        with ChildGraphCapture(splice=False) as probe:
            result = self._run_regular_step(batch, on_publish, grammar_barrier, None)
        reason = None
        if probe.num_graphs != _NUM_STAGE_GRAPHS:
            reason = (
                f"the step replays {probe.num_graphs} per-stage graphs "
                f"(expected {_NUM_STAGE_GRAPHS})"
            )
        elif any(delta != 0 for delta in probe.generator_offset_deltas):
            reason = "a per-stage graph draws random numbers"
        (ok,) = _tp_min(reason is None)
        if not ok:
            reason = reason or "a peer rank cannot capture this step"
            self._uncapturable[key] = reason
            self._warmup_counts.pop(key, None)
            self._fallback(reason)
        else:
            # The output ring is allocated before the capture, from this layout.
            self._output_layouts[key] = tuple(
                (tuple(t.shape), t.dtype)
                for t in (result.next_token_ids, result.accept_lens)
            )
        return result

    # --------------------------------------------------------------- capture

    def _memory_ok(self, key, batch) -> bool:
        largest = max(
            (s.memory_bytes for s in self._steps.values()),
            default=_FIRST_KEY_MEMORY_ESTIMATE,
        )
        need = largest + _CAPTURE_MEMORY_HEADROOM
        free = torch.cuda.mem_get_info()[0]
        (ok,) = _tp_min(free >= need)
        if (
            not ok
            and key not in self._cache_released
            and len(self._cache_released) < _MAX_CACHE_RELEASES
        ):
            # PyTorch may cache enough unused memory (e.g. after long prefills).
            cached = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
            (releasable,) = _tp_min(free + cached >= need)
            if releasable:
                free = self._release_cached_memory(key, batch, free, need)
                (ok,) = _tp_min(free >= need)
        if ok:
            return True
        # Every rank reports its own numbers: the rank short of memory decides.
        refusal = {
            "batch_size": batch.seq_lens.shape[0],
            "greedy": batch.sampling_info.is_all_greedy,
            "constrained": batch.has_grammar,
            "free_mib": free >> 20,
            "need_mib": need >> 20,
            "reserved_mib": torch.cuda.memory_reserved() >> 20,
            "allocated_mib": torch.cuda.memory_allocated() >> 20,
        }
        self._stats["memory_refusals"] += 1
        self._last_memory_refusal = refusal
        if key not in self._memory_retry_at:
            logger.info(
                "Speculative step CUDA graph: capture memory check failed for batch "
                "size %d, %s, %s: free %d MiB, need %d MiB (largest key + %d MiB "
                "headroom), reserved %d MiB, allocated %d MiB; this key retries "
                "every %d steps.",
                refusal["batch_size"],
                "greedy" if refusal["greedy"] else "sampled",
                "constrained" if refusal["constrained"] else "unconstrained",
                refusal["free_mib"],
                refusal["need_mib"],
                _CAPTURE_MEMORY_HEADROOM >> 20,
                refusal["reserved_mib"],
                refusal["allocated_mib"],
                _MEMORY_RETRY_STEPS,
            )
        return False

    def _release_cached_memory(self, key, batch, free: int, need: int) -> int:
        """Return PyTorch's cached, unused memory to the device, once for this
        key, and give the free memory after it. Unused blocks of live graph
        pools are not released, so the caller checks the need again."""
        self._cache_released.add(key)
        self._stats["cache_releases"] += 1
        torch.cuda.empty_cache()
        free_after = torch.cuda.mem_get_info()[0]
        logger.info(
            "Speculative step CUDA graph: released PyTorch's cached memory for the "
            "capture of batch size %d, %s: free %d -> %d MiB, need %d MiB (once "
            "per key).",
            batch.seq_lens.shape[0],
            "greedy" if batch.sampling_info.is_all_greedy else "sampled",
            free >> 20,
            free_after >> 20,
            need >> 20,
        )
        return free_after

    def _capture_and_replay(
        self, key, batch, step_tensors, on_publish, grammar_barrier
    ):
        if not self._memory_ok(key, batch):
            self._memory_retry_at[key] = self._num_steps_seen + _MEMORY_RETRY_STEPS
            self._fallback("not enough free memory to capture")
            return None

        # Free unreachable graphs now: destroying a graph while a capture is in
        # progress invalidates the capture, and the collector may run at any time.
        gc.collect()
        torch.cuda.synchronize()
        get_tp_group().barrier()
        reserved_before = torch.cuda.memory_reserved()
        start = time.perf_counter()
        step, error = None, None
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            step = self._capture(batch, step_tensors, self._output_layouts.get(key))
        except Exception as e:  # noqa: BLE001 (the key runs the regular step)
            error = e
        finally:
            if gc_was_enabled:
                gc.enable()
        with torch.cuda.stream(self._stream):
            if torch.cuda.is_current_stream_capturing():
                # The capture contexts end their captures, also on errors.
                self.disabled_reason = "a capture left its stream capturing"
        memory = max(0, torch.cuda.memory_reserved() - reserved_before)
        ok, neg_memory = _tp_min(step is not None, -memory)
        self._warmup_counts.pop(key, None)
        if not ok:
            self._failed_keys += 1
            self._stats["capture_failures"] += 1
            reason = (
                f"capture failed: {type(error).__name__}: {error}"
                if error is not None
                else "capture failed on a peer rank"
            )
            self._uncapturable[key] = "capture failed"
            logger.warning(
                "Speculative step CUDA graph: capture failed for batch size %d; "
                "this key runs the regular step (%s).",
                batch.seq_lens.shape[0],
                reason,
                exc_info=error,
            )
            if step is not None:
                # Keep a graph that a peer rank failed to capture: graphs are
                # only released all together (see _reset).
                self._parked_steps.append(step)
            if self._failed_keys >= _MAX_FAILED_KEYS:
                self.disabled_reason = (
                    f"{self._failed_keys} keys failed to capture; last: {reason}"
                )
            if self.disabled_reason is not None:
                logger.warning(
                    "Speculative step CUDA graph turned off for this process: %s",
                    self.disabled_reason,
                )
            return None

        step.memory_bytes = -neg_memory
        step.batch_size = batch.seq_lens.shape[0]
        step.greedy = batch.sampling_info.is_all_greedy
        step.constrained = batch.has_grammar
        self._steps[key] = step
        self._stats["captures"] += 1
        logger.info(
            "Speculative step CUDA graph captured: batch size %d, %s, %s, "
            "%d static inputs, %d graph segment(s), %.1f ms, %.1f MB (%d keys).",
            step.batch_size,
            "greedy" if step.greedy else "sampled",
            "constrained" if step.constrained else "unconstrained",
            len(step.static_inputs),
            len(step.graph._segments),
            (time.perf_counter() - start) * 1e3,
            step.memory_bytes / 2**20,
            len(self._steps),
        )
        logger.debug("Speculative step CUDA graph key: %s", key)
        return self._replay(step, batch, on_publish, grammar_barrier)

    def _capture_batch(self, batch, step_tensors):
        capture_batch = copy.copy(batch)
        capture_batch.spec_info = copy.copy(batch.spec_info)
        capture_batch.sampling_info = copy.copy(batch.sampling_info)
        for name in _BATCH_STEP_OUTPUTS:
            setattr(capture_batch, name, None)
        for name in _DRAFT_INPUT_STEP_OUTPUTS:
            if hasattr(capture_batch.spec_info, name):
                setattr(capture_batch.spec_info, name, None)
        holders = _holders(capture_batch)
        names, static_copies = [], []
        for holder, name, tensor in step_tensors:
            static = tensor.clone()
            setattr(holders[holder], name, static)
            names.append((holder, name))
            static_copies.append(static)
        return capture_batch, names, static_copies

    def _capture(self, batch, step_tensors, output_layout) -> _StepGraph:
        if output_layout is None:
            raise RuntimeError("no output layout from the warm-up step")
        capture_batch, names, static_copies = self._capture_batch(batch, step_tensors)
        # Allocated outside the capture: a replay's outputs must survive the next
        # replay, so they cannot share memory with the graph's intermediates.
        ring_slot = torch.zeros(1, dtype=torch.int64, device=self.device)
        ring_next_token_ids, ring_accept_lens = (
            torch.empty((_OUTPUT_RING_SIZE, *shape), dtype=dtype, device=self.device)
            for shape, dtype in output_layout
        )
        # Simulated acceptance (benchmarking): whenever the step applies it, the
        # length comes from this device scalar, drawn on the host per replay.
        sim_accept_len = torch.ones(1, dtype=torch.int32, device=self.device)
        draw_args: List[Tuple[float, str, int, Optional[torch.Generator]]] = []
        overrides = {
            "spec_simulated_accept_index": self._simulated_accept_override(
                sim_accept_len, draw_args
            )
        }

        grammar = _GrammarStep(self) if batch.has_grammar else None
        if grammar is not None:
            overrides.update(grammar.overrides())

        graph = BreakableCUDAGraph()
        self._in_worker_call = True
        try:
            with (
                graph_capture(stream=self._stream) as context,
                ChildGraphCapture(splice=True, overrides=overrides) as children,
                BreakableCUDAGraphCapture(
                    graph,
                    pool=torch.cuda.graph_pool_handle(),
                    stream=context.stream,
                    capture_error_mode="thread_local",
                ),
            ):
                result = self.worker.forward_batch_generation(
                    capture_batch,
                    on_publish=None,
                    grammar_barrier=None,
                    pp_proxy_tensors=None,
                )
                ring_next_token_ids.index_copy_(
                    0, ring_slot, result.next_token_ids[None]
                )
                ring_accept_lens.index_copy_(0, ring_slot, result.accept_lens[None])
                ring_slot.add_(1).remainder_(_OUTPUT_RING_SIZE)
                _check_host_copies()
        finally:
            self._in_worker_call = False
            # The capture recorded no real event; the scheduler must not wait on it.
            self.draft_model_runner.shared_read_done_event = None

        if children.num_graphs != _NUM_STAGE_GRAPHS:
            raise RuntimeError(
                f"captured {children.num_graphs} per-stage graphs "
                f"(expected {_NUM_STAGE_GRAPHS})"
            )
        num_breaks = 2 if grammar is not None else 0
        if len(graph._break_fns) != num_breaks or (
            grammar is not None and grammar.mask is None
        ):
            raise RuntimeError(
                f"captured {len(graph._break_fns)} host breaks (expected {num_breaks})"
            )
        if len(draw_args) > 1:
            raise RuntimeError(
                f"simulated acceptance applied {len(draw_args)} times (expected 1)"
            )
        result.extra_keep_alive_refs = []
        return _StepGraph(
            graph=graph,
            input_names=names,
            static_inputs=static_copies,
            result=result,
            stage_graphs=list(children.graph_sources),
            ring_slot=ring_slot,
            ring_next_token_ids=ring_next_token_ids,
            ring_accept_lens=ring_accept_lens,
            ring_views=[
                (ring_next_token_ids[i], ring_accept_lens[i])
                for i in range(_OUTPUT_RING_SIZE)
            ],
            sim_accept_len=sim_accept_len if draw_args else None,
            sim_draw_args=draw_args[0] if draw_args else None,
            grammar=grammar,
        )

    @staticmethod
    def _simulated_accept_override(sim_accept_len, draw_args):
        from sglang.srt.speculative import spec_utils

        def generate_simulated_accept_index(
            original,
            accept_index,
            predict,
            num_correct_drafts,
            candidates,
            target_predict,
            bs,
            spec_steps,
            simulate_acc_len: float = spec_utils.SIMULATE_ACC_LEN,
            simulate_acc_method: str = spec_utils.SIMULATE_ACC_METHOD,
            simulate_acc_token_mode: str = spec_utils.SIMULATE_ACC_TOKEN_MODE,
            generator: Optional[torch.Generator] = None,
        ):
            assert simulate_acc_len > 0.0
            # Each replay draws the length from the same generator as the eager step.
            draw_args.append(
                (simulate_acc_len, simulate_acc_method, spec_steps + 1, generator)
            )
            return simulated_accept_index_from_device_length(
                accept_index,
                predict,
                num_correct_drafts,
                candidates,
                target_predict,
                spec_steps,
                sim_accept_len,
                simulate_acc_token_mode,
            )

        return generate_simulated_accept_index

    # ---------------------------------------------------------------- replay

    def _load_simulated_accept_len(self, step: _StepGraph) -> None:
        from sglang.srt.speculative.spec_utils import sample_simulated_acc_len

        if self._sim_host is None:
            self._sim_host = torch.empty(
                _SIM_LEN_SLOTS, dtype=torch.int32, pin_memory=True
            )
        slot = self._sim_slot
        self._sim_slot = (slot + 1) % _SIM_LEN_SLOTS
        event = self._sim_events[slot]
        if event is None:
            event = self._sim_events[slot] = torch.cuda.Event()
        else:
            # The copy that last read this slot was enqueued steps ago.
            event.synchronize()
        self._sim_host[slot] = sample_simulated_acc_len(*step.sim_draw_args)
        step.sim_accept_len.copy_(self._sim_host[slot : slot + 1], non_blocking=True)
        event.record()

    def _replay(
        self, step: _StepGraph, batch, on_publish, grammar_barrier
    ) -> GenerationBatchResult:
        holders = _holders(batch)
        path = _copy_step_inputs(
            step.static_inputs,
            [getattr(holders[holder], name) for holder, name in step.input_names],
            self._small_copy,
        )
        if self._small_copy is not None:
            self._stats[f"{path}_input_copies"] += 1
            key = (path, len(step.static_inputs))
            if key not in self._logged_input_copy:
                self._logged_input_copy.add(key)
                logger.info(
                    "Speculative step CUDA graph: %s input copy for %d tensors.",
                    path.replace("_", "-"),
                    len(step.static_inputs),
                )
        if step.sim_accept_len is not None:
            self._load_simulated_accept_len(step)

        slot = step.num_replays % _OUTPUT_RING_SIZE
        previous = step.ring_results[slot]
        if previous is not None:
            # The scheduler copies a result to the host on its copy stream, right
            # after the forward that produced it. Do not overwrite a slot that a
            # copy may still read: wait for it, or detach a result whose copy was
            # never issued.
            if previous.copy_done is not None:
                if not previous.copy_done.query():
                    self._stats["ring_waits"] += 1
                    previous.copy_done.synchronize()
            elif previous.next_token_ids is step.ring_views[slot][0]:
                self._stats["ring_detaches"] += 1
                previous.next_token_ids = previous.next_token_ids.clone()
                previous.accept_lens = previous.accept_lens.clone()

        self._live_batch, self._live_grammar_barrier = batch, grammar_barrier
        try:
            with graph_pool_replay_scope():
                step.graph.replay()
        finally:
            self._live_batch = self._live_grammar_barrier = None
        step.num_replays += 1

        template = step.result
        result = copy.copy(template)
        result.next_token_ids, result.accept_lens = step.ring_views[slot]
        result.next_draft_input = copy.copy(template.next_draft_input)
        if template.logits_output is not None:
            result.logits_output = copy.copy(template.logits_output)
        result.extra_keep_alive_refs = []
        result.copy_done = None
        step.ring_results[slot] = result

        if on_publish is not None:
            on_publish(result.new_seq_lens)
        if self._draft_extend_read_ends is SharedReadEnds.UNKNOWN:
            self.draft_model_runner.shared_read_done_event = None
        else:
            # One event after the whole step covers every phase's reads.
            read_done = torch.cuda.Event()
            read_done.record()
            self.draft_model_runner.shared_read_done_event = read_done
        self.target_model_runner.forward_pass_id += 1
        self._stats["replays"] += 1
        return result

    def _reset(self, reason: str) -> None:
        """Release every captured key together (never one key alone: a graph
        may reference memory that an earlier capture on the same stream owns)."""
        logger.info(
            "Speculative step CUDA graph: dropping all captured keys (%s).", reason
        )
        torch.cuda.synchronize()
        self._steps.clear()
        self._parked_steps.clear()
        self._warmup_counts.clear()
        gc.collect()
        self._stats["resets"] += 1

    # ----------------------------------------------------------------- stats

    def stats(self) -> Dict[str, Any]:
        allocator = torch.cuda.memory_stats() if torch.cuda.is_initialized() else {}
        return {
            "replays": self._stats["replays"],
            "captures": self._stats["captures"],
            "capture_failures": self._stats["capture_failures"],
            "memory_refusals": self._stats["memory_refusals"],
            "last_memory_refusal": self._last_memory_refusal,
            "cache_releases": self._stats["cache_releases"],
            "num_alloc_retries": allocator.get("num_alloc_retries", 0),
            "num_ooms": allocator.get("num_ooms", 0),
            "resets": self._stats["resets"],
            "ring_waits": self._stats["ring_waits"],
            "ring_detaches": self._stats["ring_detaches"],
            "fused_input_copies": self._stats["fused_input_copies"],
            "partial_input_copies": self._stats["partial_input_copies"],
            "per_tensor_input_copies": self._stats["per_tensor_input_copies"],
            "keys": len(self._steps),
            "captured_keys": [
                {
                    "batch_size": step.batch_size,
                    "greedy": step.greedy,
                    "need_top_k_sampling": key[2],
                    "need_top_p_sampling": key[3],
                    "need_min_p_sampling": key[4],
                    "constrained": step.constrained,
                    "memory_mb": round(step.memory_bytes / 2**20, 1),
                }
                for key, step in self._steps.items()
            ],
            "max_batch_size": self.max_batch_size,
            "fallbacks": dict(self._fallbacks),
            "uncapturable_keys": len(self._uncapturable),
            "disabled_reason": self.disabled_reason,
        }
