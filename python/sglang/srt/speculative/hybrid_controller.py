from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch

from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.runtime_context import get_context, get_spec
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.adaptive_runtime_state import SpecRuntimeState
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.hybrid_info import HybridVerifyInput
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


class HybridSpecWorker(Protocol):
    """Common worker surface managed by :class:`HybridController`."""

    server_args: ServerArgs
    speculative_num_steps: int
    speculative_num_draft_tokens: int

    @property
    def target_worker(self) -> TpModelWorker: ...

    @property
    def draft_worker(self) -> Any: ...

    @property
    def last_shared_read_runner(self) -> Any: ...

    @property
    def spec_v2_attn_backends(self) -> tuple: ...

    def alloc_memory_pool(self, **kwargs) -> None: ...

    def init_attention_backends(self) -> None: ...

    def init_cuda_graphs(self) -> None: ...

    def clear_cache_pool(self) -> None: ...

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
        grammar_barrier=None,
        pp_proxy_tensors=None,
    ) -> Any: ...

    def on_verify_complete_cpu(
        self,
        num_correct_drafts_per_req: list[int],
        batch_size: int = 0,
    ) -> None: ...

    def activate_step_by_batch(self, batch_size: int) -> None: ...

    def sync_hybrid_state(self, batch: ScheduleBatch, batch_result: Any) -> Any: ...


@dataclass(frozen=True)
class HybridWorkerConfig:
    role: str
    algorithm: SpeculativeAlgorithm
    overrides: dict[str, Any]


@dataclass(frozen=True)
class HybridRuntimeState:
    """A route and its complete adaptive-style speculative runtime state."""

    route: str
    worker: HybridSpecWorker
    spec_state: SpecRuntimeState


@dataclass
class HybridRouteStats:
    """Per-route scheduling and acceptance metrics, reset on flush_cache."""

    # Per-route request-level counters. Keep their definition aligned with
    # MetricsReporter.update_spec_metrics(), which backs bench_serving's
    # avg_spec_accept_length: each verified request contributes one forward.
    retrieval_forward_ct: int = 0
    retrieval_num_accept_tokens: float = 0.0
    retrieval_num_draft_tokens: int = 0
    neural_forward_ct: int = 0
    neural_num_accept_tokens: float = 0.0
    neural_num_draft_tokens: int = 0

    # Retrieval routing-decision helpers, accumulated over pure decode requests.
    total_continuation_length: float = 0.0
    total_match_length: float = 0.0
    matching_req_ct: int = 0

    # Batch-level counter: incremented once per pure-decode forward_batch_generation
    # call regardless of batch size, so the debug-log interval
    # (SGLANG_LOG_HYBRID_SPEC) fires every N *batches* as documented, not every
    # N accumulated requests (which would drift with dynamic batch size).
    pure_decode_batch_ct: int = 0

    def reset(self) -> None:
        for f in self.__dataclass_fields__:
            setattr(self, f, 0)

    @property
    def retrieval_accept_length(self) -> float:
        return (
            self.retrieval_num_accept_tokens / self.retrieval_forward_ct
            if self.retrieval_forward_ct
            else 0.0
        )

    @property
    def retrieval_accept_rate(self) -> float:
        return (
            (self.retrieval_num_accept_tokens - self.retrieval_forward_ct)
            / self.retrieval_num_draft_tokens
            if self.retrieval_num_draft_tokens
            else 0.0
        )

    @property
    def neural_accept_length(self) -> float:
        return (
            self.neural_num_accept_tokens / self.neural_forward_ct
            if self.neural_forward_ct
            else 0.0
        )

    @property
    def neural_accept_rate(self) -> float:
        return (
            (self.neural_num_accept_tokens - self.neural_forward_ct)
            / self.neural_num_draft_tokens
            if self.neural_num_draft_tokens
            else 0.0
        )

    @property
    def avg_continuation_length(self) -> float:
        return (
            self.total_continuation_length / self.total_forward_ct
            if self.total_forward_ct
            else 0.0
        )

    @property
    def avg_match_length(self) -> float:
        return (
            self.total_match_length / self.total_forward_ct
            if self.total_forward_ct
            else 0.0
        )

    @property
    def total_forward_ct(self) -> int:
        return self.retrieval_forward_ct + self.neural_forward_ct


class HybridController(BaseSpecWorker):
    """JSON-configured, batch-level speculative worker router.

    Each configured worker owns an independent ServerArgs copy, so draft-token
    counts and other algorithm-specific settings are allowed to differ. The
    controller only assumes that the configured retrieval worker is NGRAM and
    that the neural worker exposes a draft-extend phase; worker classes
    themselves are always resolved through ``SpeculativeAlgorithm.create_worker``.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        ps: ParallelState,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        super().__init__()
        self.server_args = server_args
        self._target_worker = target_worker
        config = self._load_config(get_spec().speculative_hybrid_config)
        self.routing = {
            "min_continuation_ratio": config["min_continuation_ratio"],
            "min_matching_ratio": config["min_matching_ratio"],
        }
        self.worker_configs = self._parse_worker_configs(config)
        self.workers = self._create_workers(gpu_id, ps, nccl_port)
        self.retrieval_worker = self.workers["retrieval"]
        self.neural_worker = self.workers["neural"]
        if not callable(getattr(self.retrieval_worker, "get_retrieval_info", None)):
            raise ValueError(
                "HYBRID retrieval worker must implement get_retrieval_info."
            )
        for role, worker in self.workers.items():
            if not callable(getattr(worker, "sync_hybrid_state", None)):
                raise ValueError(
                    f"HYBRID {role} worker must implement sync_hybrid_state."
                )
        adaptive_controller = getattr(self.neural_worker, "adaptive_controller", None)
        adaptive_widths = (
            {steps + 1 for steps in adaptive_controller.candidate_steps}
            if adaptive_controller is not None
            else set()
        )
        self._runtime_widths = tuple(
            sorted(
                {
                    self.retrieval_worker.speculative_num_draft_tokens,
                    self.neural_worker.speculative_num_draft_tokens,
                    *adaptive_widths,
                }
            )
        )
        self.min_continuation = (
            self.retrieval_worker.speculative_num_draft_tokens - 1
        ) * self.routing["min_continuation_ratio"]

        self._last_route = "neural"
        # Scheduler queries this before preparing the next batch. Track the
        # worker that actually performed the final shared-buffer read; a
        # retrieval verify is followed by neural draft-extend during warm-up.
        self._last_shared_buffer_reader = self.neural_worker

        # Per-route scheduling metrics, reset on flush_cache.
        self._route_stats = HybridRouteStats()
        self._runtime_states: dict[tuple[str, int | None], HybridRuntimeState] = {}
        self._active_runtime_state: HybridRuntimeState | None = None
        self.log_interval = envs.SGLANG_LOG_HYBRID_SPEC.get()

    @staticmethod
    def _load_config(config_json: str) -> dict[str, Any]:
        try:
            config = json.loads(config_json)
        except json.JSONDecodeError as exc:
            raise ValueError("HYBRID config must be a JSON object.") from exc
        if not isinstance(config, dict):
            raise ValueError("HYBRID config must be a JSON object.")
        for role in ("retrieval", "neural"):
            if not isinstance(config.get(role), dict):
                raise ValueError(f"HYBRID config requires a {role} object.")
        for name in ("min_continuation_ratio", "min_matching_ratio"):
            if name not in config:
                raise ValueError(f"HYBRID config requires {name}.")
        for name in ("min_continuation_ratio", "min_matching_ratio"):
            value = config[name]
            if not isinstance(value, (int, float)) or not 0 < value <= 1:
                raise ValueError(f"HYBRID {name} must be in (0, 1].")
        return config

    @staticmethod
    def _parse_worker_configs(config: dict[str, Any]) -> dict[str, HybridWorkerConfig]:
        configs = {}
        for role in ("retrieval", "neural"):
            raw = config[role]
            algorithm_name = raw.get("algorithm")
            if not isinstance(algorithm_name, str):
                raise ValueError(f"HYBRID {role}.algorithm must be a string.")
            algorithm = SpeculativeAlgorithm.from_string(algorithm_name)
            if algorithm.is_none() or algorithm.is_hybrid():
                raise ValueError(
                    f"Unsupported nested HYBRID algorithm: {algorithm_name}."
                )
            overrides = {key: value for key, value in raw.items() if key != "algorithm"}
            configs[role] = HybridWorkerConfig(role, algorithm, overrides)
        return configs

    def _create_workers(
        self, gpu_id: int, ps: ParallelState, nccl_port: int
    ) -> dict[str, HybridSpecWorker]:
        from sglang.srt.arg_groups.speculative_hook import (
            configure_adaptive_speculative_decoding,
        )

        workers: dict[str, HybridSpecWorker] = {}
        base_config = self.server_args.resolved_dict()
        for role, worker_config in self.worker_configs.items():
            unknown = set(worker_config.overrides) - set(
                self.server_args.__dataclass_fields__
            )
            if unknown:
                raise ValueError(
                    f"Unknown ServerArgs override for {role}: {sorted(unknown)}."
                )
            changes = {
                "speculative_algorithm": worker_config.algorithm.name,
                **worker_config.overrides,
            }
            changes["speculative_adaptive"] = bool(
                role == "neural"
                and changes.get(
                    "speculative_adaptive", base_config["speculative_adaptive"]
                )
            )
            worker_server_args = self.server_args.replace_resolved(
                f"hybrid.{role}", **changes
            )
            object.__setattr__(
                worker_server_args, "_hybrid_managed_runtime", role == "neural"
            )
            if changes["speculative_adaptive"]:
                configure_adaptive_speculative_decoding(worker_server_args)
            worker_config.algorithm.handle_server_args(worker_server_args)
            worker_cls = worker_config.algorithm.create_worker(worker_server_args)
            role_config = worker_server_args.resolved_dict()
            context_changes = {
                key: value
                for key, value in role_config.items()
                if base_config.get(key) != value
            }
            restore = {key: base_config[key] for key in context_changes}
            get_context().override(f"hybrid.{role}.construct", **context_changes)
            try:
                workers[role] = cast(
                    HybridSpecWorker,
                    worker_cls(
                        server_args=worker_server_args,
                        gpu_id=gpu_id,
                        ps=ps,
                        nccl_port=nccl_port,
                        target_worker=self._target_worker,
                        spec_stage_span_prefix=role,
                    ),
                )
            finally:
                get_context().override("hybrid.restore", **restore)
        return workers

    @property
    def target_worker(self):
        return self._target_worker

    @property
    def draft_worker(self):
        return self.neural_worker.draft_worker

    @property
    def last_shared_read_runner(self):
        return self._last_shared_buffer_reader.last_shared_read_runner

    @property
    def spec_v2_attn_backends(self) -> tuple:
        # Scheduler setup needs the union because routes may switch at every
        # batch boundary; workers initialize their own backends up front.
        if self._runtime_states:
            return tuple(
                dict.fromkeys(
                    backend
                    for state in self._runtime_states.values()
                    for backend in (
                        state.spec_state.draft_attn_backend,
                        state.spec_state.target_attn_backend,
                        state.spec_state.draft_extend_attn_backend,
                    )
                    if backend is not None
                )
            )
        return tuple(
            dict.fromkeys(
                backend
                for worker in self.workers.values()
                for backend in worker.spec_v2_attn_backends
            )
        )

    def alloc_memory_pool(self, **kwargs):
        for worker in self.workers.values():
            worker.alloc_memory_pool(**kwargs)

    def init_attention_backends(self):
        for worker in self.workers.values():
            worker.init_attention_backends()

    def init_cuda_graphs(self):
        target_runner = self._target_worker.model_runner
        bootstrap_target_graph_runner = target_runner.decode_cuda_graph_runner
        bootstrap_target_attn_backend = target_runner.attn_backend
        # Prefill CUDA graphs are captured against this general-purpose backend.
        # Route-specific target-verify backends may use different static metadata,
        # so replaying a prefill graph while one of them is installed corrupts the
        # graph's cache/index inputs.
        self._target_prefill_attn_backend = bootstrap_target_attn_backend
        bootstrap_width = max(self._runtime_widths)
        for worker in self.workers.values():
            worker.init_cuda_graphs()
        self._init_draft_extend_resources()
        self._init_target_verify_resources(
            bootstrap_width,
            bootstrap_target_graph_runner,
            bootstrap_target_attn_backend,
        )
        self._init_runtime_states()

    def _init_draft_extend_resources(self) -> None:
        draft_worker = self.neural_worker.draft_worker
        neural_width = self.neural_worker.speculative_num_draft_tokens
        adaptive_states = getattr(self.neural_worker, "adaptive_runtime_states", {})
        resources = {
            state.speculative_num_draft_tokens: (
                state.draft_extend_attn_backend,
                state.cuda_graph_runner_for_draft_extend,
            )
            for state in adaptive_states.values()
        }
        resources.setdefault(
            neural_width,
            (
                draft_worker.draft_extend_attn_backend,
                draft_worker.cuda_graph_runner_for_draft_extend,
            ),
        )
        builder = getattr(draft_worker, "build_draft_extend_runtime_resource", None)
        if not callable(builder):
            raise ValueError(
                "HYBRID neural worker must build draft-extend runtime resources."
            )
        for width in self._runtime_widths:
            if width in resources:
                continue
            with (
                draft_worker.draft_tp_context(draft_worker.draft_runner.tp_group),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                resources[width] = builder(num_tokens_per_bs=width)
        self._draft_extend_resources = resources

    def _init_target_verify_resources(
        self,
        bootstrap_width: int,
        bootstrap_graph_runner,
        bootstrap_attn_backend,
    ) -> None:
        """Build route-width target resources with adaptive-style graph buffers."""
        from sglang.srt.model_executor.runner import DecodeCudaGraphRunner

        target_runner = self._target_worker.model_runner
        adaptive_states = getattr(self.neural_worker, "adaptive_runtime_states", {})
        graph_runners = {
            state.speculative_num_draft_tokens: state.target_graph_runner
            for state in adaptive_states.values()
        }
        attn_backends = {
            state.speculative_num_draft_tokens: state.target_attn_backend
            for state in adaptive_states.values()
        }
        max_width = max(self._runtime_widths)
        if bootstrap_width != max_width:
            raise ValueError(
                "HYBRID target bootstrap width must be the maximum runtime width: "
                f"bootstrap={bootstrap_width}, maximum={max_width}."
            )
        graph_runners[bootstrap_width] = bootstrap_graph_runner
        attn_backends[bootstrap_width] = bootstrap_attn_backend
        for width in self._runtime_widths:
            if width in graph_runners:
                continue
            bootstrap_workspace = target_runner.init_new_workspace
            with self.neural_worker._override_worker_state(width - 1, width):
                try:
                    target_attn_backend = target_runner._get_attention_backend(
                        init_new_workspace=True
                    )
                finally:
                    target_runner.init_new_workspace = bootstrap_workspace

                target_graph_runner = None
                if bootstrap_graph_runner is not None:
                    target_graph_runner = DecodeCudaGraphRunner(
                        target_runner,
                        attn_backend=target_attn_backend,
                        speculative_num_steps=width - 1,
                        speculative_num_draft_tokens=width,
                    )
            graph_runners[width] = target_graph_runner
            attn_backends[width] = target_attn_backend

        target_runner.hybrid_target_verify_graph_runners = graph_runners
        target_runner.hybrid_target_verify_attn_backends = attn_backends
        self._target_verify_graph_runners = graph_runners
        self._target_verify_attn_backends = attn_backends

    def _init_runtime_states(self) -> None:
        """Build complete route states after every backend/graph is initialized."""
        target_runner = self._target_worker.model_runner
        draft_worker = self.neural_worker.draft_worker
        graph_runners = self._target_verify_graph_runners
        attn_backends = self._target_verify_attn_backends
        draft_extend_resources = self._draft_extend_resources
        states = {}

        def build_state(route, worker, source_state=None):
            width = (
                source_state.speculative_num_draft_tokens
                if source_state is not None
                else worker.speculative_num_draft_tokens
            )
            target_graph_runner = graph_runners.get(width)
            target_attn_backend = attn_backends.get(width)
            if target_attn_backend is None and target_graph_runner is not None:
                target_attn_backend = target_graph_runner.attn_backend
            if target_attn_backend is None:
                target_attn_backend = target_runner.attn_backend
            draft_extend_attn_backend, draft_extend_graph_runner = (
                draft_extend_resources[width]
            )
            spec_state = SpecRuntimeState(
                speculative_num_steps=(
                    source_state.speculative_num_steps
                    if source_state is not None
                    else worker.speculative_num_steps
                ),
                speculative_num_draft_tokens=width,
                draft_attn_backend=(
                    source_state.draft_attn_backend
                    if source_state is not None
                    else draft_worker.draft_attn_backend
                ),
                cuda_graph_runner=(
                    source_state.cuda_graph_runner
                    if source_state is not None
                    else draft_worker.cuda_graph_runner
                ),
                target_attn_backend=target_attn_backend,
                target_graph_runner=target_graph_runner,
                draft_extend_attn_backend=draft_extend_attn_backend,
                cuda_graph_runner_for_draft_extend=draft_extend_graph_runner,
            )
            key = (
                (route, spec_state.speculative_num_steps)
                if route == "neural"
                else (route, None)
            )
            states[key] = HybridRuntimeState(
                route=route,
                worker=worker,
                spec_state=spec_state,
            )

        build_state("retrieval", self.retrieval_worker)
        adaptive_states = getattr(self.neural_worker, "adaptive_runtime_states", {})
        if adaptive_states:
            for source_state in adaptive_states.values():
                build_state("neural", self.neural_worker, source_state)
        else:
            build_state("neural", self.neural_worker)
        self._runtime_states = states
        self._apply_runtime_state(self._last_route)

    def _apply_runtime_state(self, route: str) -> None:
        """Atomically switch target verify resources at a route boundary.

        Like ``EAGLEWorkerV2.apply_runtime_state``, never combine a backend
        from one width with the graph captured for another width.  The helper
        is deliberately invoked before draft preparation so all later
        metadata writes and target replay use this same state.
        """
        key = (
            (route, self.neural_worker.speculative_num_steps)
            if route == "neural"
            else (route, None)
        )
        state = self._runtime_states.get(key)
        if state is None:
            raise RuntimeError(f"HYBRID runtime state is missing for route={route}.")
        if state is self._active_runtime_state:
            # An extend/mixed batch temporarily installs the backend used to
            # capture target prefill graphs. Restore the route's verify backend
            # even when the rest of the runtime state is already active.
            target_runner = self._target_worker.model_runner
            target_runner.attn_backend = state.spec_state.target_attn_backend
            target_runner.decode_cuda_graph_runner = (
                state.spec_state.target_graph_runner
            )
            return

        spec_state = state.spec_state
        target_runner = self._target_worker.model_runner
        get_context().override(
            "hybrid.route",
            speculative_num_steps=spec_state.speculative_num_steps,
            speculative_num_draft_tokens=spec_state.speculative_num_draft_tokens,
        )

        if route == "neural":
            self.neural_worker.apply_runtime_state(spec_state)
            self._active_runtime_state = state
            return

        target_runner.attn_backend = spec_state.target_attn_backend
        target_runner.decode_cuda_graph_runner = spec_state.target_graph_runner

        draft_worker = self.neural_worker.draft_worker
        draft_worker.draft_attn_backend = spec_state.draft_attn_backend
        draft_worker.draft_runner.draft_attn_backend = spec_state.draft_attn_backend
        draft_worker.cuda_graph_runner = spec_state.cuda_graph_runner
        draft_worker.draft_extend_attn_backend = spec_state.draft_extend_attn_backend
        if spec_state.draft_extend_attn_backend is not None:
            draft_worker.draft_runner.attn_backend = (
                spec_state.draft_extend_attn_backend
            )
        draft_worker.cuda_graph_runner_for_draft_extend = (
            spec_state.cuda_graph_runner_for_draft_extend
        )
        self._active_runtime_state = state

    def _apply_target_prefill_backend(self) -> None:
        """Install the backend paired with the target prefill CUDA graphs."""
        self._target_worker.model_runner.attn_backend = (
            self._target_prefill_attn_backend
        )

    def clear_cache_pool(self):
        for worker in self.workers.values():
            worker.clear_cache_pool()
        self._route_stats.reset()

    def on_verify_complete_cpu(
        self, num_correct_drafts_per_req, batch_size=0, route: str | None = None
    ):
        # Verification statistics belong to the route that produced them. In
        # particular, feeding NGRAM acceptance into neural adaptive tuning
        # would corrupt its step selection after a route switch.
        #
        # Under overlap scheduling this hook runs as a delayed CPU callback
        # (see batch_result_processor._resolve_spec_v2_tokens), one iteration
        # after the forward that produced ``num_correct_drafts_per_req``. A
        # later batch may already have switched ``self._last_route`` by then,
        # so prefer the route stamped on the originating result
        # (``GenerationBatchResult.spec_route``) and only fall back to
        # ``self._last_route`` for callers that predate that field.
        target_route = route if route is not None else self._last_route
        self.workers[target_route].on_verify_complete_cpu(
            num_correct_drafts_per_req, batch_size=batch_size
        )

    def activate_step_by_batch(self, batch_size: int):
        # Give each role a batch-boundary activation opportunity. Retrieval is
        # currently static, while the neural role owns draft steps; calling both
        # is necessary for future retrieval/adaptive workers and avoids keeping
        # inactive route state stale.
        for worker in self.workers.values():
            worker.activate_step_by_batch(batch_size)

    def _match_continuation_lengths(self, batch: ScheduleBatch) -> list[int]:
        """Measure the current suffix's actual retrieval continuation.

        The retrieval worker performs the same lookup it will use for drafting;
        the controller only consumes its per-request continuation lengths.
        """
        return self.retrieval_worker.get_retrieval_info(batch)

    def _should_use_retrieval(
        self, batch: ScheduleBatch, lengths: tuple[list[int], list[int]]
    ) -> bool:
        if not batch.forward_mode.is_decode() or batch.is_extend_in_batch:
            return False
        # Weight each request's continuation by its match_len so a long but
        # shallow (low-confidence) match cannot outweigh a shorter, deeper one.
        cont_lens, match_lens = lengths
        qualifying = sum(
            cont_len * match_len / self.retrieval_worker.speculative_num_draft_tokens
            >= self.min_continuation
            for cont_len, match_len in zip(cont_lens, match_lens)
        )
        min_matching = len(cont_lens) * self.routing["min_matching_ratio"]
        return qualifying >= min_matching

    def _build_ngram_relay_input(
        self,
        batch: ScheduleBatch,
        result: Any,
        source_input: NgramVerifyInput | None = None,
    ) -> NgramVerifyInput:
        """Normalize an accepted path into NGRAM's native fixed-width relay."""
        bs = len(batch.reqs)
        relay_width = max(self._runtime_widths)
        if source_input is not None:
            source_width = source_input.draft_token_num
            accept_rows = source_input.accept_tokens.reshape(bs, source_width)
            accept_lens = source_input.accept_lens
            new_seq_lens = source_input.new_seq_lens
        else:
            if not isinstance(result.next_token_ids, torch.Tensor):
                raise TypeError("HYBRID requires tensor next_token_ids.")
            accept_rows = result.next_token_ids.reshape(bs, -1)
            source_width = accept_rows.shape[1]
            accept_lens = result.accept_lens
            if accept_lens is None:
                accept_lens = torch.ones(
                    bs,
                    dtype=torch.int32,
                    device=result.next_token_ids.device,
                )
            new_seq_lens = result.new_seq_lens

        if source_width > relay_width:
            raise ValueError(
                f"HYBRID accepted-path width {source_width} exceeds relay width "
                f"{relay_width}."
            )
        relay_rows = torch.zeros(
            (bs, relay_width),
            dtype=accept_rows.dtype,
            device=accept_rows.device,
        )
        relay_rows[:, :source_width] = accept_rows
        return NgramVerifyInput(
            draft_token_num=relay_width,
            new_seq_lens=new_seq_lens,
            accept_tokens=relay_rows.flatten(),
            accept_lens=accept_lens,
        )

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
        grammar_barrier=None,
        pp_proxy_tensors=None,
    ):
        is_pure_decode = batch.forward_mode.is_decode() and not batch.is_extend_in_batch
        hybrid_input = (
            batch.spec_info if isinstance(batch.spec_info, HybridVerifyInput) else None
        )
        if is_pure_decode:
            if hybrid_input is not None:
                batch.spec_info = hybrid_input.ngram_verify_input
            lengths = self._match_continuation_lengths(batch)
        else:
            lengths = ([], [])

        if self._should_use_retrieval(batch, lengths):
            self._apply_runtime_state("retrieval")
            batch.capture_hidden_mode = CaptureHiddenMode.FULL
            result = self.retrieval_worker.forward_batch_generation(
                batch,
                on_publish=on_publish,
                grammar_barrier=grammar_barrier,
                pp_proxy_tensors=pp_proxy_tensors,
            )
            ngram_input = result.next_draft_input
            if not isinstance(ngram_input, NgramVerifyInput):
                raise TypeError("HYBRID retrieval route must return NgramVerifyInput.")
            eagle_input = self.neural_worker.sync_hybrid_state(batch, result)
            if not isinstance(eagle_input, EagleDraftInput):
                raise TypeError("HYBRID neural sync must return EagleDraftInput.")
            ngram_input = self._build_ngram_relay_input(
                batch, result, source_input=ngram_input
            )
            last_shared_buffer_reader = self.neural_worker
            route = "retrieval"
        else:
            if hybrid_input is not None:
                batch.spec_info = hybrid_input.eagle_draft_input
            self.neural_worker.activate_step_by_batch(batch.seq_lens.shape[0])
            self._apply_runtime_state("neural")
            if not is_pure_decode:
                self._apply_target_prefill_backend()
            result = self.neural_worker.forward_batch_generation(
                batch,
                on_publish=on_publish,
                grammar_barrier=grammar_barrier,
                pp_proxy_tensors=pp_proxy_tensors,
            )
            eagle_input = result.next_draft_input
            if not isinstance(eagle_input, EagleDraftInput):
                raise TypeError("HYBRID neural route must return EagleDraftInput.")
            self.retrieval_worker.sync_hybrid_state(batch, result)
            ngram_input = self._build_ngram_relay_input(batch, result)
            last_shared_buffer_reader = self.neural_worker
            route = "neural"
        result.next_draft_input = HybridVerifyInput(eagle_input, ngram_input)
        self._last_shared_buffer_reader = last_shared_buffer_reader

        self._update_route_stats(route, result, lengths, is_pure_decode)

        # Stamp the producer route on the result itself. Overlap scheduling
        # processes this result on a delayed CPU callback
        # (on_verify_complete_cpu); by then ``self._last_route`` may already
        # reflect a *later* batch's route, so the attribution must travel
        # with the result rather than be read from live controller state.
        result.spec_route = route
        return result

    def _update_route_stats(
        self,
        route: str,
        result,
        continuation_lengths: tuple[list[int], list[int]],
        is_pure_decode: bool,
    ) -> None:
        """Update all stats except accept tokens during hybrid route"""
        stats = self._route_stats
        bs = len(continuation_lengths[0]) if continuation_lengths[0] is not None else 0

        if is_pure_decode:
            if route == "retrieval":
                stats.retrieval_forward_ct += bs
                stats.retrieval_num_draft_tokens += bs * (
                    self.retrieval_worker.speculative_num_draft_tokens - 1
                )
            else:
                stats.neural_forward_ct += bs
                stats.neural_num_draft_tokens += bs * (
                    self.neural_worker.speculative_num_draft_tokens - 1
                )

            cont_lens, match_lens = continuation_lengths
            stats.total_continuation_length += sum(cont_lens)
            stats.total_match_length += sum(match_lens)
            stats.matching_req_ct += sum(
                cont_len
                * match_len
                / self.retrieval_worker.speculative_num_draft_tokens
                >= self.min_continuation
                for cont_len, match_len in zip(cont_lens, match_lens)
            )
            stats.pure_decode_batch_ct += 1

        if (
            is_pure_decode
            and self.log_interval > 0
            and stats.pure_decode_batch_ct % self.log_interval == 0
        ):
            cont_info = (
                f"cont_len={continuation_lengths[0]} "
                f"match_len={continuation_lengths[1]} "
                f"min_cont={self.min_continuation} "
                if continuation_lengths
                else ""
            )
            logger.info(
                "HybridSpec route=%s bs=%d %s",
                route,
                bs,
                cont_info,
            )

        if route != self._last_route:
            logger.debug("Hybrid speculative route switched to %s", route)
            self._last_route = route

    def update_hybrid_stats(
        self,
        route: str,
        result,
        is_pure_decode: bool,
    ) -> None:
        """Update accept tokens during batch result processing."""
        stats = self._route_stats
        bs = len(result.accept_lens) if result.accept_lens is not None else 0
        accept_lens = result.accept_lens

        if is_pure_decode:
            if route == "retrieval":
                if accept_lens is not None:
                    stats.retrieval_num_accept_tokens += float(accept_lens.sum().item())
            else:
                if accept_lens is not None:
                    stats.neural_num_accept_tokens += float(accept_lens.sum().item())

        if (
            is_pure_decode
            and self.log_interval > 0
            and stats.pure_decode_batch_ct % self.log_interval == 0
        ):
            logger.info(
                "HybridSpec route=%s bs=%d "
                "avg_cont=%.2f avg_match=%.2f matching_req_ct=%d | "
                "retrieval: forward_ct=%d accept_len=%.2f accept_rate=%.2f | "
                "neural: forward_ct=%d accept_len=%.2f accept_rate=%.2f",
                route,
                bs,
                stats.avg_continuation_length,
                stats.avg_match_length,
                stats.matching_req_ct,
                stats.retrieval_forward_ct,
                stats.retrieval_accept_length,
                stats.retrieval_accept_rate,
                stats.neural_forward_ct,
                stats.neural_accept_length,
                stats.neural_accept_rate,
            )
