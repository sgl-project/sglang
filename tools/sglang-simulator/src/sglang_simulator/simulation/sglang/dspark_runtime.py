"""Native CPU result and causal before-state adapter for DSPark simulation."""

from __future__ import annotations

import json
from dataclasses import replace

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
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=tokens if prefill else tokens.reshape(-1),
            new_seq_lens=new_seq_lens,
            next_draft_input=next_input,
            accept_lens=None if prefill else torch.tensor(commits, dtype=torch.int32),
            # The progress artifact models committed counts. It does not
            # predict pre-trim block acceptance, and the current adapter has
            # no ragged layout. Preserve the native optional/unknown fields.
            block_accept_lens=None,
            cap_lens=None,
            speculative_num_draft_tokens=width,
            speculative_output_stride=None if prefill else width,
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
    ):
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
        self._draft_buffer_capacity = 0

    def capture(self, batch):
        if str(batch.device) != "cpu":
            raise PredictorError(
                "invalid_batch", "DSPark simulator requires native CPU scheduler state"
            )
        if self.topology.attention_dp_size != 1:
            raise PredictorError(
                "unsupported_topology",
                "DSPark ADP2 needs actual peer scheduling partitions and publication",
            )
        if getattr(batch, "return_logprob", False) or getattr(
            batch, "return_hidden_states", False
        ):
            raise PredictorError(
                "unsupported_execution_profile",
                "DSPark simulator does not synthesize logprob/hidden payloads",
            )
        mode = getattr(batch.forward_mode, "name", None)
        if mode not in {"EXTEND", "DECODE"} or getattr(
            batch, "is_extend_in_batch", False
        ):
            raise PredictorError(
                "unsupported_forward_mode",
                "DSPark runtime needs its homogeneous native prefill or decode batch",
            )
        requests = tuple(batch.reqs)
        seq = cpu_vector(batch.seq_lens_cpu, len(requests), "seq_lens_cpu")
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
            prefix, extend = seq, (6,) * len(requests)
            if any(
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
                        or cpu_int(req.inflight_middle_chunks, "inflight_middle_chunks")
                        == 0
                    ),
                )
            )
        internal_mode = "EXTEND" if mode == "EXTEND" else "TARGET_VERIFY"
        group = DSparkGroup(0, internal_mode, tuple(work), '{"selection":"pending"}')
        execution = self.execution_resolver(
            group=group, runtime_profile_json=self.runtime_profile_json
        )
        execution = dict(execution)
        # Persistent proposal-buffer state is causal run state. A resolver may
        # not label the first allocation as a warm no-allocation proposal.
        execution["draft_buffer_capacity"] = self._draft_buffer_capacity
        execution["draft_buffer_device_matches"] = self._draft_buffer_capacity > 0
        group = replace(group, execution_json=canonical(execution))
        return DSparkStep(
            self._forward_id, self.topology, (group,), self.runtime_profile_json
        )

    def forward_batch_generation(self, *, worker, batch, on_publish=None):
        if worker.query_token_num != 5 or worker.verify_num_draft_tokens != 6:
            raise PredictorError(
                "unsupported_execution_profile",
                "DSPark worker width differs from runtime",
            )
        before = self.capture(batch)
        prediction = self.session.predict(before)
        result = self.results.build(batch=batch, prediction=prediction)
        # Native synchronous scheduling will later update req.kv by accept_lens.
        # Our draft/history ledger is independent of that pending native update.
        if on_publish is not None:
            on_publish(result.new_seq_lens)
        self.session.commit(prediction)
        if prediction.local_group.forward_mode == "TARGET_VERIFY":
            self._draft_buffer_capacity = max(
                self._draft_buffer_capacity, len(batch.reqs)
            )
        self._forward_id += 1
        return result

    def note_request_finished(self, *, rid, natural_stop):
        self.session.forget(rid)

    def clear_cache_pool(self):
        self.session.clear()
        # Native clear_cache_pool clears KV, not the proposer's persistent
        # mask-token buffer. Preserve its capacity across the same worker run.
