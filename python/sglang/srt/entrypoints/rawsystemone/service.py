"""Request-local prefix scoring over the existing TokenizerManager batch API."""

import asyncio
import logging
import math
import time
import uuid
from dataclasses import dataclass, fields

from .admission import BatchAdmission
from .protocol import (
    CandidateScore,
    RawSystemOneError,
    RawSystemOneRequest,
    RawSystemOneResponse,
    Usage,
)
from .scoring import (
    aggregate,
    invalid_scores,
    native_rows,
    partition_candidates,
    plan_tokens,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RawSystemOneConfig:
    max_options: int = 128
    max_request_tokens: int = 1048576
    max_candidates_per_batch: int = 32
    max_tokens_per_batch: int = 65536
    max_inflight_batches_per_request: int = 2
    max_inflight_batches: int = 8
    max_inflight_candidates: int = 256
    max_inflight_tokens: int = 1048576
    max_pending_requests: int = 64
    timeout_seconds: float = 300

    def __post_init__(self):
        for field in fields(self):
            if (
                not math.isfinite(getattr(self, field.name))
                or getattr(self, field.name) <= 0
            ):
                raise ValueError(f"rawsystemone_{field.name} must be positive")


class RawSystemOneService:
    def __init__(
        self,
        tokenizer_manager,
        config: RawSystemOneConfig | None = None,
        *,
        cache_enabled: bool = True,
        unsupported_reason: str | None = None,
    ):
        self.manager = tokenizer_manager
        self.config = config or RawSystemOneConfig()
        self.cache_enabled = cache_enabled
        self.unsupported_reason = unsupported_reason
        self.admission = BatchAdmission(
            self.config.max_inflight_batches,
            self.config.max_inflight_candidates,
            self.config.max_inflight_tokens,
        )
        self.pending_requests = 0

    @classmethod
    def from_runtime(cls, manager):
        from sglang.srt.runtime_context import (
            get_disagg,
            get_exec,
            get_memory,
            get_parallel,
            get_serving,
            get_spec,
        )

        serving = get_serving()
        config = RawSystemOneConfig(
            **{
                f.name: getattr(serving, "rawsystemone_" + f.name)
                for f in fields(RawSystemOneConfig)
            }
        )
        unsupported = None
        # Single HTTP worker makes the admission bound and runtime epoch global.
        # Other deployments need shared admission/epoch and verified affinity.
        if serving.tokenizer_worker_num != 1:
            unsupported = "Multiple tokenizer workers are not supported."
        elif get_disagg().disaggregation_mode != "null":
            unsupported = "Disaggregated serving is not supported."
        elif get_parallel().dp_size != 1 or get_parallel().pp_size != 1:
            unsupported = "Data and pipeline parallel serving are not supported."
        elif get_spec().speculative_algorithm or get_exec().dllm.dllm_algorithm:
            unsupported = "Speculative and diffusion serving are not supported."
        elif get_exec().features.enable_mis:
            unsupported = "Multi-item scoring mode is not supported."
        model = manager.model_config
        architectures = set(getattr(model.hf_config, "architectures", []) or [])
        if (
            not manager.is_generation
            or model.is_multimodal
            or model.is_encoder_decoder
            or not architectures.intersection(
                {
                    "LlamaForCausalLM",
                    "Qwen2ForCausalLM",
                    "Qwen3ForCausalLM",
                    "MistralForCausalLM",
                }
            )
        ):
            unsupported = "This endpoint requires a supported text-only causal decoder."
        return cls(
            manager,
            config,
            cache_enabled=not get_memory().disable_radix_cache,
            unsupported_reason=unsupported,
        )

    def _check_epoch(self, epoch):
        if self.manager.model_update_epoch != epoch:
            raise RawSystemOneError(
                "model_changed",
                "Model weights changed during scoring; retry the request.",
                409,
            )

    @staticmethod
    def _make_request(**kwargs):
        import msgspec

        from sglang.srt.managers.io_struct import GenerateReqInput
        from sglang.srt.sampling.sampling_params import SamplingParams

        # Override *all* preferred generation defaults, including grammars,
        # penalties, beam search and custom_params. Input scores are vocabulary-
        # wide log-softmax; no answer token is requested or consumed.
        neutral = msgspec.structs.asdict(
            SamplingParams(max_new_tokens=0, temperature=1.0)
        )
        return GenerateReqInput(
            sampling_params=neutral,
            return_logprob=True,
            top_logprobs_num=0,
            return_text_in_logprobs=False,
            stream=False,
            no_logs=True,
            **kwargs,
        )

    async def _batch(self, sequences, start, state, *, phase):
        count, tokens = len(sequences), sum(map(len, sequences))
        queued_at = time.monotonic()
        async with self.admission.lease(count, tokens):
            self._check_epoch(state["epoch"])
            if state["failed"]:
                raise asyncio.CancelledError
            state["inflight"] += 1
            state["max_inflight"] = max(state["max_inflight"], state["inflight"])
            batch_stats = {
                "phase": phase,
                "candidates": count,
                "input_tokens": tokens,
                "queue_seconds": time.monotonic() - queued_at,
                "global_inflight": self.admission.used[0],
            }
            state["batches"].append(batch_stats)
            rids = [f"{state['id']}-{uuid.uuid4().hex}" for _ in sequences]
            generator = None
            success = False
            started_at = time.monotonic()
            try:
                request = self._make_request(
                    input_ids=[list(ids) for ids in sequences],
                    rid=rids,
                    logprob_start_len=start,
                    cache_salt=state["cache_salt"],
                    external_trace_header=state["trace_headers"],
                )
                # As in score_request: one native batch, independent inputs.
                # The parent owns disconnect polling, avoiding multiple readers
                # of the same ASGI receive channel. Trace headers are forwarded.
                generator = self.manager.generate_request(request, None)
                result = None
                async for response in generator:
                    if result is not None:
                        raise invalid_scores(
                            "Unexpected extra non-streaming batch result."
                        )
                    result = response
                self._check_epoch(state["epoch"])
                if not isinstance(result, list) or len(result) != count:
                    raise invalid_scores("Native batch is incomplete.")
                by_id = {}
                for output in result:
                    meta = output.get("meta_info", {})
                    rid = meta.get("id")
                    if rid not in rids or rid in by_id:
                        raise invalid_scores(
                            "Native batch returned an unexpected child ID."
                        )
                    finish = meta.get("finish_reason") or {}
                    if finish.get("type") == "abort":
                        status = finish.get("status_code", 500)
                        raise RawSystemOneError(
                            "overloaded" if status in (429, 503) else "inference_error",
                            "Native scoring request was aborted.",
                            status,
                        )
                    if meta.get("completion_tokens", 0) != 0:
                        raise invalid_scores(
                            "Native scoring unexpectedly generated tokens."
                        )
                    by_id[rid] = meta
                records = [
                    native_rows(ids, start, by_id[rid].get("input_token_logprobs"))
                    for ids, rid in zip(sequences, rids)
                ]
                # Native cache counters, not a claim about actual GPU FLOPs.
                cached = [by_id[rid].get("cached_tokens") for rid in rids]
                if all(isinstance(n, int) for n in cached):
                    batch_stats["cached_tokens"] = sum(cached)
                    batch_stats["uncached_input_tokens"] = tokens - sum(cached)
                success = True
                return records
            except BaseException:
                # Set before any await/slot release, so another worker cannot
                # dequeue a new batch after this parent has failed.
                state["failed"] = True
                raise
            finally:
                try:
                    if not success:
                        for rid in rids:
                            try:
                                self.manager.abort_request(rid)
                            except Exception:
                                logger.error("rawsystemone child abort dispatch failed")
                    if generator is not None:
                        await generator.aclose()
                finally:
                    state["inflight"] -= 1
                    batch_stats["seconds"] = time.monotonic() - started_at

    async def _dispatch(self, batches, sequences, start, state):
        iterator = iter(batches)
        results = {}

        async def worker():
            while not state["failed"]:
                batch = next(iterator, None)
                if batch is None:
                    return
                rows = await self._batch(
                    [sequences[i] for i in batch], start, state, phase="branches"
                )
                results.update(zip(batch, rows))

        tasks = [
            asyncio.create_task(worker())
            for _ in range(
                min(len(batches), self.config.max_inflight_batches_per_request)
            )
        ]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            state["failed"] = True
            raise
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return results

    async def _evaluate(self, request, state, reference):
        cfg = self.config
        if self.unsupported_reason:
            raise RawSystemOneError("unsupported_mode", self.unsupported_reason)
        if self.manager.tokenizer is None:
            raise RawSystemOneError(
                "tokenizer_unavailable", "The native tokenizer is unavailable.", 503
            )
        if len(request.suffixes) > cfg.max_options:
            raise RawSystemOneError(
                "too_many_options", f"At most {cfg.max_options} suffixes are allowed."
            )
        sequences, _ = await self.manager._tokenize_texts(
            [request.prefix + suffix for suffix in request.suffixes]
        )
        if len(sequences) != len(request.suffixes):
            raise invalid_scores("Tokenizer returned an incomplete candidate list.")
        for ids in sequences:
            if len(ids) < 2:
                raise RawSystemOneError(
                    "no_scoreable_tokens",
                    "Every complete candidate needs at least two native tokens.",
                )
            if len(ids) + self.manager.num_reserved_tokens >= self.manager.context_len:
                raise RawSystemOneError(
                    "context_length_exceeded",
                    "A candidate exceeds the native context limit.",
                )
            scheduler_limit = getattr(self.manager, "max_req_input_len", None)
            if scheduler_limit is not None and len(ids) >= scheduler_limit:
                raise RawSystemOneError(
                    "context_length_exceeded",
                    "A candidate exceeds the scheduler input limit.",
                )
            if len(ids) > cfg.max_inflight_tokens:
                raise RawSystemOneError(
                    "token_budget_exceeded",
                    "A candidate exceeds the admission token limit.",
                )
        logical_tokens = sum(map(len, sequences))
        if logical_tokens > cfg.max_request_tokens:
            raise RawSystemOneError(
                "token_budget_exceeded", "Cumulative candidate-token budget exceeded."
            )
        plan = plan_tokens(sequences)
        state.update(
            options=len(sequences),
            distinct_candidates=len(plan.sequences),
            shared_tokens=plan.common_length,
            logical_input_tokens=logical_tokens,
        )
        shared = ()
        shared_total, shared_count = 0.0, 0
        k = plan.common_length
        use_shared = (
            not reference and self.cache_enabled and len(plan.sequences) > 1 and k > 1
        )
        state["mode"] = "shared_prefix" if use_shared else "full_sequence"
        if use_shared:
            state["fallback_reason"] = None
        elif reference:
            state["fallback_reason"] = "reference"
        elif not self.cache_enabled:
            state["fallback_reason"] = "cache_disabled"
        elif len(plan.sequences) == 1:
            state["fallback_reason"] = "one_unique_candidate"
        else:
            state["fallback_reason"] = "short_common_span"
        candidates = [
            i for i, ids in enumerate(plan.sequences) if not use_shared or len(ids) > k
        ]
        # Plan and validate ALL batches before any GPU work, including warm-up.
        batches = partition_candidates(
            candidates,
            plan.sequences,
            min(cfg.max_candidates_per_batch, cfg.max_inflight_candidates),
            min(cfg.max_tokens_per_batch, cfg.max_inflight_tokens),
            cfg.max_inflight_tokens,
        )
        prefix_start = time.monotonic()
        if use_shared:
            (shared,) = await self._batch(
                [plan.sequences[0][:k]], 0, state, phase="shared_prefix"
            )
            shared_total, shared_count = aggregate(plan.sequences[0][:k], shared)
        state["shared_seconds"] = time.monotonic() - prefix_start
        branch_start = time.monotonic()
        scored = await self._dispatch(
            batches, plan.sequences, k - 1 if use_shared else 0, state
        )
        state["branch_seconds"] = time.monotonic() - branch_start
        complete = {}
        for i, ids in enumerate(plan.sequences):
            if use_shared:
                # Drop only the verified placeholder at absolute position K-1.
                records = shared + scored[i][1:] if i in scored else shared
            else:
                records = scored[i]
            if use_shared and i not in scored:
                total, count = shared_total, shared_count
            else:
                # Sum the complete records with fsum so the host result does
                # not depend on the partition into shared and tail spans.
                total, count = aggregate(ids, records)
            complete[i] = (total, count, records)
        self._check_epoch(state["epoch"])
        data = []
        for original, unique in enumerate(plan.original_to_unique):
            total, count, records = complete[unique]
            data.append(
                CandidateScore(
                    index=original,
                    score=total / count,
                    logprob_sum=total,
                    scored_token_count=count,
                    token_logprobs=list(records)
                    if request.return_token_logprobs
                    else None,
                )
            )
        return RawSystemOneResponse(
            id=state["id"],
            model=self.manager.served_model_name,
            data=data,
            best_index=max(range(len(data)), key=lambda i: data[i].score),
            usage=Usage(
                input_tokens=logical_tokens, scored_tokens=logical_tokens - len(data)
            ),
        )

    async def score(
        self,
        request: RawSystemOneRequest,
        raw_request=None,
        *,
        _reference=False,
        _diagnostics=None,
    ):
        """Private reference switch is test-only; never accepted as an HTTP field."""
        if self.pending_requests >= self.config.max_pending_requests:
            raise RawSystemOneError(
                "overloaded", "Rawsystemone admission queue is full.", 429
            )
        started = time.monotonic()
        request_id = "rawsystemone-" + uuid.uuid4().hex
        trace_headers = None
        cache_salt = request_id  # private to this parent, shared by every child
        if raw_request is not None and getattr(self.manager, "enable_trace", False):
            from sglang.srt.observability.trace import extract_trace_headers

            trace_headers = extract_trace_headers(raw_request.headers)
        if raw_request is not None:
            cache_salt = getattr(raw_request.state, "cache_salt", None) or cache_salt
        state = dict(
            id=request_id,
            epoch=self.manager.model_update_epoch,
            cache_salt=cache_salt,
            trace_headers=trace_headers,
            failed=False,
            inflight=0,
            max_inflight=0,
            batches=[],
        )
        task = None
        watcher = None

        async def watch_disconnect():
            while True:
                if await raw_request.is_disconnected():
                    if not state["failed"]:
                        state["failed"] = True
                        task.cancel()
                    return
                await asyncio.sleep(0.1)

        self.pending_requests += 1
        try:
            task = asyncio.create_task(self._evaluate(request, state, _reference))
            if raw_request is not None:
                watcher = asyncio.create_task(watch_disconnect())
            return await asyncio.wait_for(task, timeout=self.config.timeout_seconds)
        except asyncio.TimeoutError:
            state["failed"] = True
            raise RawSystemOneError("timeout", "Scoring timed out.", 504) from None
        except (RawSystemOneError, asyncio.CancelledError):
            state["failed"] = True
            raise
        except Exception as exc:
            state["failed"] = True
            status = getattr(exc, "status_code", 500)
            raise RawSystemOneError(
                "overloaded" if status in (429, 503) else "inference_error",
                "Native scoring failed.",
                status,
            ) from None
        finally:
            self.pending_requests -= 1
            if watcher is not None:
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
            diagnostics = {
                k: v
                for k, v in state.items()
                if k not in ("epoch", "cache_salt", "trace_headers", "inflight")
            }
            diagnostics.update(
                seconds=time.monotonic() - started,
                configured_inflight=self.config.max_inflight_batches_per_request,
                global_batch_limit=self.config.max_inflight_batches,
                submission_batches=len(state["batches"]),
            )
            if _diagnostics is not None:
                _diagnostics.update(diagnostics)
            logger.info("rawsystemone %s", diagnostics)
