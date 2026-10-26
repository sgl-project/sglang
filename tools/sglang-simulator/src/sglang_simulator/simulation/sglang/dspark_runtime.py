"""Native CPU result and causal before-state adapter for DSPark simulation."""

from __future__ import annotations

import json
import time
from array import array
from dataclasses import dataclass, field, replace

from sglang_simulator.simulation.sglang.dspark import (
    STEP_ATTRIBUTE,
    DSparkGroup,
    DSparkRequest,
    DSparkSession,
    DSparkStep,
    DSparkTopology,
)
from sglang_simulator.time_predictor.base import PredictorError

INPUT_ATTRIBUTE = "_sglang_simulator_dspark_verify_width"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def cpu_int(value, name):
    device = getattr(value, "device", None)
    if device is not None:
        if getattr(device, "type", None) != "cpu" or getattr(value, "ndim", None) != 0:
            raise PredictorError("invalid_batch", f"{name} must be a CPU integer")
        value = value.item()
    if type(value) is not int or value < 0:
        raise PredictorError("invalid_batch", f"{name} must be a nonnegative integer")
    return value


def cpu_vector(values, size, name):
    device = getattr(values, "device", None)
    if device is None and not isinstance(values, (list, tuple)):
        raise PredictorError("invalid_batch", f"{name} must be a CPU vector")
    if device is not None and (
        getattr(device, "type", None) != "cpu" or values.ndim != 1
    ):
        raise PredictorError("invalid_batch", f"{name} must be a CPU vector")
    if values is None or len(values) != size:
        raise PredictorError("invalid_batch", f"{name} must match the local batch")
    return tuple(cpu_int(value, name) for value in values)


@dataclass(frozen=True)
class DSparkSimulatorRequestFacts:
    rid: str
    origin_input_len: int
    max_new_tokens: int
    to_finish_present: bool
    input_embeds_present: bool
    multimodal_inputs_present: bool


@dataclass(frozen=True)
class DSparkSimulatorSourceFacts:
    """CPU simulator observations, not reference mirror policy or query inputs."""

    forward_mode: str
    requests: tuple[DSparkSimulatorRequestFacts, ...]
    seq_lens_cpu: tuple[int, ...] | None
    draft_input_present: bool
    nxt_kv_lens_cpu: tuple[int, ...] | None
    observation_scope: str = field(default="simulator_cpu_before_worker_v1", init=False)


def capture_simulator_source_facts(native_batch):
    """Optionally freeze actual CPU facts for a future source-owned projector.

    Callers must explicitly opt in. Missing fields are unknown, whereas a present
    None mirror or draft input records known absence. These are simulator facts
    after CPU preparation, which may have overwritten earlier host mirrors; no
    reference execution's mirror availability or graph/planner bound is inferred.
    Token contents and non-CPU tensors are never copied or inspected.
    """

    def required(obj, name):
        try:
            return getattr(obj, name)
        except AttributeError as error:
            raise PredictorError(
                "invalid_batch",
                "simulator source facts are missing a native field",
                field=name,
            ) from error

    if str(required(native_batch, "device")) != "cpu":
        raise PredictorError(
            "invalid_batch", "simulator source facts require a CPU batch"
        )
    mode = required(required(native_batch, "forward_mode"), "name")
    requests = required(native_batch, "reqs")
    if (
        mode not in {"EXTEND", "DECODE"}
        or not isinstance(requests, (list, tuple))
        or not requests
    ):
        raise PredictorError(
            "invalid_batch",
            "simulator source facts require ordered prefill/decode requests",
        )
    facts = []
    for req in requests:
        rid = required(req, "rid")
        original = required(req, "origin_input_ids")
        if (
            type(rid) is not str
            or not rid
            or not isinstance(original, (array, list, tuple))
        ):
            raise PredictorError(
                "invalid_batch",
                "simulator source facts require original CPU request metadata",
            )
        facts.append(
            DSparkSimulatorRequestFacts(
                rid=rid,
                origin_input_len=len(original),
                max_new_tokens=cpu_int(
                    required(required(req, "sampling_params"), "max_new_tokens"),
                    "max_new_tokens",
                ),
                to_finish_present=required(req, "to_finish") is not None,
                input_embeds_present=required(req, "input_embeds") is not None,
                multimodal_inputs_present=required(req, "multimodal_inputs")
                is not None,
            )
        )
    if len({req.rid for req in facts}) != len(facts):
        raise PredictorError(
            "invalid_batch", "simulator source facts repeat a request ID"
        )
    seq = required(native_batch, "seq_lens_cpu")
    draft_input = required(native_batch, "spec_info")
    nxt = None if draft_input is None else required(draft_input, "nxt_kv_lens_cpu")
    return DSparkSimulatorSourceFacts(
        forward_mode=mode,
        requests=tuple(facts),
        seq_lens_cpu=None
        if seq is None
        else cpu_vector(seq, len(facts), "seq_lens_cpu"),
        draft_input_present=draft_input is not None,
        nxt_kv_lens_cpu=None
        if nxt is None
        else cpu_vector(nxt, len(facts), "nxt_kv_lens_cpu"),
    )


def sampling_snapshot(params):
    # The SDK owns the source-pinned field inventory; do not maintain a second
    # simulator sampling schema or discard normalized defaults.
    from infercast.sdk.native_sampling import SAMPLING_FIELDS

    def builtins(value):
        if isinstance(value, dict):
            return {key: builtins(v) for key, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [builtins(v) for v in value]
        if isinstance(value, (set, frozenset)):
            return sorted(builtins(v) for v in value)
        return value

    try:
        value = {name: builtins(getattr(params, name)) for name in SAMPLING_FIELDS}
    except AttributeError as error:
        raise PredictorError(
            "invalid_batch", "native sampling recipe is incomplete"
        ) from error
    if value["is_normalized"] is not True:
        raise PredictorError(
            "invalid_batch", "native sampling recipe must be normalized"
        )
    return canonical(value)


def prepare_cpu_decode(draft_input, batch):
    """Native DFlash reserve semantics through public CPU allocation/writes.

    Reuse the exact source page-aligned sizing helper. The original DFlash
    method uses a device kernel to assign the reserved slots; CPU simulation
    uses ReqToTokenPool.write for the same table update.
    """
    import torch

    from sglang.srt.mem_cache.allocation import (
        alloc_paged_token_slots_extend,
        alloc_token_slots,
    )
    from sglang.srt.mem_cache.allocation_sizing import page_aligned_decode_alloc_lens

    if str(batch.device) != "cpu" or getattr(draft_input, INPUT_ATTRIBUTE, None) != 6:
        raise PredictorError(
            "unsupported_execution_profile",
            "DSPark reserve requires its marked native CPU draft input",
        )
    if getattr(batch, "hisparse_coordinator", None) is not None:
        raise PredictorError(
            "unsupported_execution_profile",
            "DSPark CPU reserve does not model HiSparse mapping",
        )
    batch.maybe_evict_swa()
    size = len(batch.reqs)
    allocator = batch.token_to_kv_pool_allocator
    current, following, needed = page_aligned_decode_alloc_lens(
        batch.reqs,
        reserve=12,
        page_size=allocator.page_size,
    )
    current_cpu = torch.tensor(current, dtype=torch.int64)
    following_cpu = torch.tensor(following, dtype=torch.int64)
    slots = cpu_vector(batch.req_pool_indices, size, "req_pool_indices")
    table = batch.req_to_token_pool.req_to_token
    if getattr(table.device, "type", None) != "cpu" or any(
        nxt > table.shape[1] for nxt in following
    ):
        raise PredictorError(
            "invalid_batch", "native DSPark reserve exceeds the CPU request table"
        )
    if needed:
        if allocator.page_size == 1:
            locations = alloc_token_slots(batch.tree_cache, needed)
        else:
            last = torch.tensor(
                [
                    int(table[slot, cur - 1]) if cur else -1
                    for slot, cur in zip(slots, current, strict=True)
                ],
                dtype=torch.int64,
            )
            locations = alloc_paged_token_slots_extend(
                batch.tree_cache,
                current_cpu,
                current_cpu,
                following_cpu,
                following_cpu,
                last,
                needed,
                req_pool_indices=batch.req_pool_indices,
                batch=batch,
            )
        if locations is None or locations.numel() != needed:
            raise PredictorError(
                "out_of_memory",
                "native DSPark CPU reserve could not allocate requested slots",
            )
        offset = 0
        for slot, cur, nxt in zip(slots, current, following, strict=True):
            batch.req_to_token_pool.write(
                (slot, slice(cur, nxt)), locations[offset : offset + nxt - cur]
            )
            offset += nxt - cur
    committed = []
    topks = []
    for req, nxt in zip(batch.reqs, following, strict=True):
        req.kv.kv_allocated_len = max(req.kv.kv_allocated_len, nxt)
        req.decode_batch_idx += 1
        committed.append(cpu_int(req.kv.kv_committed_len, "kv_committed_len"))
        topks.append(req.sampling_params.top_k)
    batch.seq_lens_cpu = torch.tensor(committed, dtype=torch.int64)
    batch.seq_lens = batch.seq_lens_cpu.clone()
    batch.seq_lens_sum = sum(committed)
    draft_input.nxt_kv_lens_cpu = following_cpu
    draft_input.nxt_kv_lens_sum = sum(following)
    draft_input.max_top_k = max(topks, default=1)
    draft_input.uniform_top_k_value = (
        topks[0] if topks and len(set(topks)) == 1 else None
    )


class NativeDSparkResults:
    """Use native SGLang types and CPU tensors; token content stays synthetic.

    This is the simulator's normal output-content contract. Progress is supplied
    separately; a dummy token ID never determines the committed token count.
    """

    def __init__(self, output_token_id):
        self.output_token_id = cpu_int(output_token_id, "output_token_id")

    def build(self, *, batch, prediction):
        import torch

        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.managers.utils import GenerationBatchResult
        from sglang.srt.speculative.draft_worker_common import make_draft_input_v2

        requests = prediction.local_group.requests
        prefill = prediction.local_group.forward_mode == "EXTEND"
        idle = prediction.local_group.forward_mode == "IDLE"
        counts = dict(prediction.progress.committed_tokens)
        prefix = [request.target_prefix_len for request in requests]
        commits = [counts[request.rid] for request in requests]
        width = 6
        if prefill:
            tokens = torch.full(
                (len(requests),), self.output_token_id, dtype=torch.int64
            )
            new_seq_lens = batch.seq_lens
        else:
            tokens = torch.full(
                (len(requests), width), self.output_token_id, dtype=torch.int64
            )
            new_seq_lens = torch.tensor(
                [p + c for p, c in zip(prefix, commits, strict=True)], dtype=torch.int64
            )
        bonus = torch.full((len(requests),), self.output_token_id, dtype=torch.int64)
        next_input = make_draft_input_v2(bonus_tokens=bonus, new_seq_lens=new_seq_lens)
        setattr(next_input, INPUT_ATTRIBUTE, width)
        result = GenerationBatchResult(
            logits_output=None
            if idle
            else LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=tokens if prefill else tokens.reshape(-1),
            new_seq_lens=new_seq_lens,
            next_draft_input=next_input,
            accept_lens=None if prefill else torch.tensor(commits, dtype=torch.int32),
            # The progress artifact models committed counts. It does not
            # predict pre-trim block acceptance, and the current adapter has
            # no ragged layout. Preserve the native optional/unknown fields.
            block_accept_lens=torch.empty((0,), dtype=torch.int32) if idle else None,
            cap_lens=None,
            speculative_num_draft_tokens=width,
            speculative_output_stride=None if prefill or idle else width,
            num_correct_drafts=0 if prefill else sum(commits) - len(commits),
            can_run_cuda_graph=False,
        )
        setattr(result, STEP_ATTRIBUTE, prediction)
        return result


class DSparkNativeRuntime:
    """Capture native CPU inputs, predict, build native output, then commit.

    ``execution_resolver`` selects graph/padding/fallback from the pinned source
    profile and prospective group, never from an executed proposal or acceptance.
    It must return a canonical JSON object or raise an explicit unsupported gap.
    The runtime passes all complete sampling/history inputs to the UMD cost path.
    """

    def __init__(
        self,
        *,
        session: DSparkSession,
        topology: DSparkTopology,
        runtime_profile_json: str,
        output_token_id: int,
        execution_resolver=None,
        result_factory=None,
        reference_events=False,
        initial_draft_buffer_capacity=0,
        joint_execution_resolver=None,
    ):
        if (
            type(initial_draft_buffer_capacity) is not int
            or initial_draft_buffer_capacity < 0
        ):
            raise PredictorError(
                "invalid_execution_profile",
                "initial draft buffer capacity must be a nonnegative integer",
            )
        self.session = session
        self.topology = topology
        self.runtime_profile_json = runtime_profile_json
        self.execution_resolver = (
            execution_resolver
            if execution_resolver is not None
            else session.cost_model.execution_for
        )
        self.results = (
            result_factory
            if result_factory is not None
            else NativeDSparkResults(output_token_id)
        )
        self._forward_id = 0
        # Cold is the default. A factory-verified warm before-work witness may
        # seed this modeled resource without changing request/progress history.
        self._draft_buffer_capacity = initial_draft_buffer_capacity
        self._peer_failure = None
        self._peer_retirements = set()
        self.reference_driver = None
        if reference_events:
            from sglang_simulator.simulation.sglang.dspark_reference_driver import (
                DSparkReferenceDriver,
            )

            self.reference_driver = DSparkReferenceDriver(self)
        self._joint_execution_resolver = joint_execution_resolver
        self._native_dp_scheduler = None
        if joint_execution_resolver is not None:
            from infercast.sdk.native_dp_progress import NativeDPProgressPolicy
            from sglang_simulator.simulation.sglang.dspark_joint_execution import (
                NativeJointExecutionResolver,
            )
            from sglang_simulator.time_predictor.dspark_golden_progress import (
                SourceGoldenDPProgress,
            )

            policy = getattr(session.cost_model, "native_source_policy", None)
            if not (
                topology == DSparkTopology(2, 2)
                and reference_events is True
                and type(joint_execution_resolver) is NativeJointExecutionResolver
                and type(policy) is NativeDPProgressPolicy
                and joint_execution_resolver.source_policy is policy
                and joint_execution_resolver.runtime_profile_json
                == runtime_profile_json
                and type(session.progress_model) is SourceGoldenDPProgress
                and session.progress_model.identity["source_identity_sha256"]
                == policy.source_identity_sha256
                and session.progress_model.runtime_profile_json == runtime_profile_json
            ):
                raise PredictorError(
                    "invalid_execution_profile",
                    "native DP routing requires matching typed joint source/cost/progress owners",
                )
            if type(getattr(policy, "reference_needs_cpu_seq_lens", None)) is not bool:
                raise PredictorError(
                    "data_unavailable",
                    "native DP routing requires original reference CPU mirror observations",
                    missing_source_facts=["reference_backend_cpu_mirror_policy"],
                )
            self.execution_resolver = joint_execution_resolver.execution_for
            self._native_dp_identity = self._joint_identity()

    def _joint_identity(self):
        resolver = self._joint_execution_resolver
        return (
            resolver.source_policy.source_identity_sha256,
            resolver.source_policy.reference_needs_cpu_seq_lens,
            tuple(
                (mode, resolver.profiles[mode].profile_id)
                for mode in ("EXTEND", "DECODE")
            ),
            resolver.runtime_profile_json,
            self.runtime_profile_json,
            self.session.progress_model.model_ref,
        )

    def _native_dp_route(self, scheduler, batch, *, selecting, reset=False):
        """Read actual native group/ranks and rendezvous local validation errors."""
        from sglang_simulator.simulation.sglang.dspark_native_peers import (
            _exchange,
            _native_group,
            _require,
        )

        from sglang.srt.runtime_context import get_parallel

        parallel = get_parallel()
        group = parallel.tp_group
        _native_group(group)  # A real usable group is the rendezvous prerequisite.

        def local():
            rank = parallel.attn_dp_rank
            # A drained reset can precede the first native selection. Its
            # lifecycle has no native forward and must not invent a counter.
            count = None if reset else scheduler.forward_ct
            _require(
                self._joint_execution_resolver is not None
                and self._native_dp_identity == self._joint_identity()
                and type(rank) is int
                and rank == parallel.tp_rank == group.rank_in_group
                and all(
                    type(getattr(parallel, name)) is int
                    for name in (
                        "tp_size",
                        "attn_dp_size",
                        "attn_tp_size",
                        "pp_size",
                        "attn_cp_size",
                        "nnodes",
                        "attn_tp_rank",
                    )
                )
                and parallel.tp_size == parallel.attn_dp_size == 2
                and parallel.attn_tp_size
                == parallel.pp_size
                == parallel.attn_cp_size
                == parallel.nnodes
                == 1
                and parallel.attn_tp_rank == 0
                and parallel.attn_dp_enabled is True
                and (reset or type(count) is int and count >= 0)
                and (
                    self._native_dp_scheduler is None
                    or self._native_dp_scheduler is scheduler
                )
                and (reset or selecting or self._native_dp_scheduler is scheduler),
                "native DP route differs from its actual scheduler/group/source owner",
            )
            return dict(
                rank=rank,
                forward_ct=count,
                has_batch=batch is not None,
                source=self._native_dp_identity,
                progress=self.session.progress_model.model_ref,
                reset=reset,
            )

        peers = _exchange(
            group,
            "native scheduler route" if selecting else "native worker route",
            local,
        )
        _require(
            peers[0]["forward_ct"] == peers[1]["forward_ct"]
            and peers[0]["has_batch"] == peers[1]["has_batch"]
            and peers[0]["source"] == peers[1]["source"]
            and peers[0]["progress"] == peers[1]["progress"]
            and peers[0]["reset"] == peers[1]["reset"],
            "native DP routes require matching FID/source and actual active/idle batches",
        )
        return group, parallel.attn_dp_rank, peers[group.rank_in_group]["forward_ct"]

    def selected_native_dp(self, scheduler, batch, *, prior_requests=()):
        """Existing native selection entry after its MLP-sync/idle adapter."""
        group, rank, count = self._native_dp_route(scheduler, batch, selecting=True)
        self.reference_driver.bind_native_dp_rank(rank)
        self.synchronize_native_dp_peer_lifecycle(
            tp_group=group, attention_dp_rank=rank
        )
        self._native_dp_scheduler = scheduler
        if batch is None:
            self.reference_driver.selection = self.reference_driver._selection(
                scheduler, batch, prior_requests=prior_requests
            )
            return
        return self.select_native_dp_peer_batch(
            scheduler,
            batch,
            tp_group=group,
            attention_dp_rank=rank,
            native_forward_ct=count + 1,
            prior_requests=prior_requests,
        )

    def select_native_dp_peer_batch(self, scheduler, batch, **kwargs):
        """Explicit library route; public native ADP2 capture stays closed."""
        from sglang_simulator.simulation.sglang.dspark_native_peers import (
            select_native_dp_peer_batch,
        )

        return select_native_dp_peer_batch(self, scheduler, batch, **kwargs)

    def forward_native_dp_peer(self, **kwargs):
        from sglang_simulator.simulation.sglang.dspark_native_peers import (
            forward_native_dp_peer,
        )

        return forward_native_dp_peer(self, **kwargs)

    def synchronize_native_dp_peer_lifecycle(self, **kwargs):
        from sglang_simulator.simulation.sglang.dspark_native_peers import (
            synchronize_native_dp_peer_lifecycle,
        )

        return synchronize_native_dp_peer_lifecycle(self, **kwargs)

    def capture(self, batch, *, resolve=True, reference=False):
        if str(batch.device) != "cpu":
            raise PredictorError(
                "invalid_batch", "DSPark simulator requires native CPU scheduler state"
            )
        if self.topology.attention_dp_size != 1:
            raise PredictorError(
                "unsupported_topology",
                "DSPark ADP2 needs actual peer scheduling partitions and publication",
            )
        group = self._capture_local_group(
            batch, attention_dp_rank=0, resolve=resolve, reference=reference
        )
        return DSparkStep(
            self._forward_id, self.topology, (group,), self.runtime_profile_json
        )

    def _capture_local_group(
        self, batch, *, attention_dp_rank, resolve=False, reference=False
    ):
        """Read one real native partition; this does not resolve a joint cost."""
        if (
            str(batch.device) != "cpu"
            or type(attention_dp_rank) is not int
            or not 0 <= attention_dp_rank < self.topology.attention_dp_size
        ):
            raise PredictorError("invalid_batch", "invalid native CPU DP partition")
        if resolve and self.topology.attention_dp_size != 1:
            raise PredictorError(
                "data_unavailable", "local capture cannot resolve a joint worker cost"
            )
        if getattr(batch, "return_logprob", False) or getattr(
            batch, "return_hidden_states", False
        ):
            raise PredictorError(
                "unsupported_execution_profile",
                "DSPark simulator does not synthesize logprob/hidden payloads",
            )
        mode = getattr(batch.forward_mode, "name", None)
        if mode == "IDLE" and self.topology == DSparkTopology(2, 2):
            if tuple(batch.reqs):
                raise PredictorError(
                    "invalid_batch", "native idle partition has requests"
                )
            return DSparkGroup(attention_dp_rank, "IDLE", (), '{"selection":"pending"}')
        if mode not in {"EXTEND", "DECODE"} or (
            getattr(batch, "is_extend_in_batch", False)
            and (self.topology.attention_dp_size == 1 or mode != "EXTEND")
        ):
            raise PredictorError(
                "unsupported_forward_mode",
                "DSPark runtime needs its homogeneous native prefill or decode batch",
            )
        requests = tuple(batch.reqs)
        seq = cpu_vector(
            batch.seq_lens if reference else batch.seq_lens_cpu,
            len(requests),
            "seq_lens" if reference else "seq_lens_cpu",
        )
        if not requests:
            raise PredictorError(
                "invalid_batch", "single-DP DSPark cannot execute an empty batch"
            )
        if mode == "EXTEND":
            prefix = cpu_vector(batch.prefix_lens, len(requests), "prefix_lens")
            extend = cpu_vector(batch.extend_lens, len(requests), "extend_lens")
            if any(
                s != p + e or e < 1 for s, p, e in zip(seq, prefix, extend, strict=True)
            ):
                raise PredictorError(
                    "invalid_batch",
                    "native prefill sequence differs from actual prefix plus chunk",
                )
        else:
            prefix = (
                tuple(
                    self.session.history_for(req.rid).target_prefix_len
                    for req in requests
                )
                if reference
                else seq
            )
            extend = (6,) * len(requests)
            if not reference and any(
                cpu_int(req.kv.kv_committed_len, "kv_committed_len") != p
                for req, p in zip(requests, prefix, strict=True)
            ):
                raise PredictorError(
                    "invalid_batch",
                    "native decode host mirrors differ from committed KV",
                )
        work = []
        for req, p, e in zip(requests, prefix, extend, strict=True):
            if getattr(req, "grammar", None) is not None:
                raise PredictorError(
                    "unsupported_execution_profile",
                    "DSPark grammar progress is not modeled",
                )
            recipe = sampling_snapshot(req.sampling_params)
            sampling = json.loads(recipe)
            if sampling.get("ignore_eos") is not True or any(
                sampling.get(name) not in (None, [])
                for name in (
                    "stop",
                    "stop_regex",
                    "stop_strs",
                    "stop_regex_strs",
                    "stop_token_ids",
                )
            ):
                raise PredictorError(
                    "unsupported_execution_profile",
                    "synthetic DSPark output requires ignore_eos=True and no stop rules",
                )
            try:
                history = self.session.history_for(req.rid)
            except PredictorError:
                if mode != "EXTEND":
                    raise
                history = None
            if mode == "EXTEND":
                start = (
                    p
                    if history is None or history.draft_valid_from > p
                    else history.draft_valid_from
                )
                end = p if history is None else min(p, history.draft_valid_to)
                if end < start:
                    start = end = p
            else:
                start, end = history.draft_valid_from, history.draft_valid_to
            work.append(
                DSparkRequest(
                    req.rid,
                    p,
                    e,
                    start,
                    end,
                    len(req.output_ids),
                    json.loads(recipe)["max_new_tokens"],
                    recipe,
                    emits_output=(
                        mode == "DECODE"
                        or (
                            batch.chunked_req is not req
                            if reference
                            else cpu_int(
                                req.inflight_middle_chunks, "inflight_middle_chunks"
                            )
                            == 0
                        )
                    ),
                )
            )
        internal_mode = "EXTEND" if mode == "EXTEND" else "TARGET_VERIFY"
        group = DSparkGroup(
            attention_dp_rank, internal_mode, tuple(work), '{"selection":"pending"}'
        )
        if not resolve:
            return group
        buffer_capacity = self._draft_buffer_capacity
        buffer_device_matches = buffer_capacity > 0
        execution = self.execution_resolver(
            group=group,
            runtime_profile_json=self.runtime_profile_json,
            native_batch=batch,
            buffer_capacity=buffer_capacity,
            buffer_device_matches=buffer_device_matches,
        )
        execution = dict(execution)
        # Persistent proposal-buffer state is causal run state. A resolver may
        # not label the first allocation as a warm no-allocation proposal.
        execution["draft_buffer_capacity"] = buffer_capacity
        execution["draft_buffer_device_matches"] = buffer_device_matches
        group = replace(group, execution_json=canonical(execution))
        return group

    def forward_batch_generation(self, *, worker, batch, on_publish=None):
        if self._joint_execution_resolver is not None:
            scheduler = self._native_dp_scheduler
            group, rank, count = self._native_dp_route(
                scheduler, batch, selecting=False
            )
            return self.forward_native_dp_peer(
                worker=worker,
                batch=batch,
                tp_group=group,
                attention_dp_rank=rank,
                native_forward_ct=count,
                joint_execution_resolver=self._joint_execution_resolver,
                on_publish=on_publish,
            )
        started = time.perf_counter()
        if worker.query_token_num != 5 or worker.verify_num_draft_tokens != 6:
            raise PredictorError(
                "unsupported_execution_profile",
                "DSPark worker width differs from runtime",
            )
        driver = self.reference_driver
        transaction = None
        try:
            if driver is None:
                before = self.capture(batch)
            else:
                transaction = driver.prepare_worker(batch)
                before = transaction.step
            prediction = self.session.predict(before)
            result = self.results.build(batch=batch, prediction=prediction)
            candidate = (
                None
                if transaction is None
                else driver.complete_candidate(transaction, prediction)
            )
        except Exception as error:
            if driver is not None:
                driver.fail("worker preparation/cost/build", error)
            raise
        # Native result processing will later update req.kv by accept_lens.
        # Our draft/history ledger is independent of that pending native update.
        try:
            if on_publish is not None:
                on_publish(result.new_seq_lens)
            if driver is not None:
                driver.validate_owner(transaction)
            self.session.commit(prediction)
        except BaseException as error:
            if driver is not None:
                # Publication can already be externally visible. Cancellation
                # must leave the run terminal, while propagating unchanged.
                driver.fail("publication/session commit", error)
            raise
        if driver is not None:
            driver.install(candidate)
        if prediction.local_group.forward_mode == "TARGET_VERIFY":
            self._draft_buffer_capacity = max(
                self._draft_buffer_capacity, len(batch.reqs)
            )
        self._forward_id += 1
        # The scheduler adds native CPU work outside this replacement worker.
        # None of capture/model lookup/result construction/publication/ledger
        # emulation is additional target work to add to a complete worker cost.
        # Exclude the entire replacement call, not just Session.predict(), from
        # the offline wall-clock contribution.
        setattr(
            result,
            STEP_ATTRIBUTE,
            replace(
                prediction,
                prediction_wall_seconds=time.perf_counter() - started,
            ),
        )
        return result

    def note_request_finished(self, *, rid, natural_stop):
        if self.reference_driver is None:
            self.session.forget(rid)
        else:
            started = time.perf_counter()
            try:
                self.reference_driver.retire(rid)
            finally:
                self.reference_driver.callback_wall_seconds += (
                    time.perf_counter() - started
                )

    def clear_cache_pool(self):
        if self._joint_execution_resolver is not None:
            scheduler = self._native_dp_scheduler
            group, rank, _ = self._native_dp_route(
                scheduler, None, selecting=False, reset=True
            )
            return self.synchronize_native_dp_peer_lifecycle(
                tp_group=group, attention_dp_rank=rank, reset=True
            )
        if self.topology.attention_dp_size == 2:
            raise PredictorError(
                "invalid_reference_state",
                "native DP reset requires the explicit joint lifecycle event",
            )
        if self.reference_driver is not None:
            started = time.perf_counter()
            self.reference_driver.drained_reset()
            self.session.clear()
            self.reference_driver.callback_wall_seconds += time.perf_counter() - started
        else:
            self.session.clear()
        # Native clear_cache_pool clears KV, not the proposer's persistent
        # mask-token buffer. Preserve its capacity across the same worker run.
