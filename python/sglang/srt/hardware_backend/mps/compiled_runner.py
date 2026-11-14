"""Direct MLX decode with bounded, read-only autoregressive lookahead."""

from __future__ import annotations

import logging
import time

import mlx.core as mx
import msgspec
import torch
from torch.export.graph_signature import InputKind

from sglang.kernels.ops.attention.mlx.radix_attention import (
    radix_decode as mlx_radix_decode,
)
from sglang.kernels.ops.attention.mlx.radix_attention_export import (
    radix_decode,
    validate_page_slots,
)
from sglang.srt.compilation.torch_compile_decoration import _to_torch
from sglang.srt.hardware_backend.mps.compiled_graph import (
    CompiledMlxGraph,
    host_alias,
)
from sglang.srt.hardware_backend.mps.compiled_region import (
    RegionBatch,
    UnsupportedMlxRegion,
    discover_decode_region,
    static_metadata_matches,
)
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.forward_mode import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.srt.model_executor.runner.base_runner import BaseRunner
from sglang.srt.runtime_context import get_parallel, get_schedule
from sglang.srt.utils.tensor_bridge import MlxTensorView, _export_evaluated_mlx

logger = logging.getLogger(__name__)


def _causal_sdpa(
    q,
    k,
    v,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
    enable_gqa=False,
):
    if (
        attn_mask is not None
        or dropout_p
        or not is_causal
        or q.shape[-2] != k.shape[-2]
    ):
        raise UnsupportedMlxRegion(
            "MLX prefill requires uncached causal self-attention"
        )
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")


def _plain_prefill(batch):
    tokens = batch.input_ids.numel()
    return (
        batch.forward_mode == ForwardMode.EXTEND
        and batch.batch_size == 1
        and 1 <= tokens <= 32768
        and batch.extend_num_tokens == batch.seq_lens_sum == tokens
        and batch.extend_seq_lens_cpu == [tokens]
        and batch.extend_prefix_lens_cpu == [0]
        and not batch.return_logprob
        and batch.input_embeds is None
        and batch.replace_embeds is None
        and batch.spec_info is None
        and batch.encoder_lens is None
    )


class _PendingDecode(msgspec.Struct, frozen=True, kw_only=True):
    graph: CompiledMlxGraph
    outputs: tuple[mx.array, ...]
    tokens: mx.array
    requests: tuple[int, ...]
    lengths: tuple[int, ...]
    positions: tuple[int, ...]
    generations: tuple[int, ...]
    locations: tuple[int, ...]
    weights: tuple
    owners: tuple[MlxTensorView, ...]


def _plain_greedy(info):
    return (
        info is not None
        and info.is_all_greedy
        and not info.grammars
        and info.grammar_mask is None
        and not info.has_custom_logit_processor
        and info.logit_bias is None
        and info.acc_additive_penalties is None
        and info.acc_scaling_penalties is None
        and (
            info.penalizer_orchestrator is None
            or not info.penalizer_orchestrator.is_required
        )
    )


class _ExportAttention(AttentionBackend):
    def __init__(self, *, cache, kv_pool, layers, prefill=False):
        self.cache = cache
        self.token_to_kv_pool = kv_pool
        self.keys = []
        self.values = []
        self.layers = layers
        self.prefill = prefill

    def forward(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        if not save_kv_cache or kwargs:
            raise UnsupportedMlxRegion(
                "MLX graph decode requires ordinary self-attention"
            )
        index = len(self.keys)
        if index >= len(self.layers) or self.layers[index].layer_id != layer.layer_id:
            raise UnsupportedMlxRegion(
                "MLX attention invocation order differs from decoder stack"
            )
        q = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)
        k = k.reshape(-1, layer.tp_k_head_num, layer.head_dim)
        v = v.reshape_as(k)
        self.keys.append(k)
        self.values.append(v)
        if self.prefill:
            result = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0),
                k.transpose(0, 1).unsqueeze(0),
                v.transpose(0, 1).unsqueeze(0),
                is_causal=True,
                scale=layer.scaling,
                enable_gqa=True,
            )
            return result[0].transpose(0, 1).reshape(q.shape[0], -1)
        result = radix_decode(
            q,
            k,
            v,
            self.cache.get_buffer(f"k_{layer.layer_id}"),
            self.cache.get_buffer(f"v_{layer.layer_id}"),
            self.cache.table,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            layer.scaling,
            page_size=self.token_to_kv_pool.page_size,
        )
        return result.reshape(q.shape[0], -1)


class _DecodeModule(torch.nn.Module):
    def __init__(self, *, model, req_pool, kv_pool, region, prefill=False):
        super().__init__()
        self.model = model
        self.region = region
        self.static_reads = {}
        self.prefill = prefill
        self.cache = torch.nn.Module()
        if not prefill:
            self.cache.register_buffer("table", req_pool.req_to_token)
            for layer in region.layers:
                k, v = kv_pool.get_kv_buffer(layer.layer_id)
                self.cache.register_buffer(f"k_{layer.layer_id}", k)
                self.cache.register_buffer(f"v_{layer.layer_id}", v)
        self.attention = _ExportAttention(
            cache=self.cache,
            kv_pool=kv_pool,
            layers=region.layers,
            prefill=prefill,
        )

    def forward(self, input_ids, positions, requests, lengths, locations):
        batch = RegionBatch(
            forward_mode=ForwardMode.EXTEND if self.prefill else ForwardMode.DECODE,
            batch_size=1 if self.prefill else input_ids.shape[0],
            input_ids=input_ids,
            positions=positions,
            req_pool_indices=requests,
            seq_lens=lengths,
            out_cache_loc=locations,
            seq_lens_sum=input_ids.shape[0] if self.prefill else 0,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            global_num_token_non_padded_cpu=input_ids.shape[0],
        )
        if self.prefill:
            tokens = input_ids.shape[0]
            batch.extend_num_tokens = tokens
            batch.extend_seq_lens = lengths.to(torch.int32)
            batch.extend_seq_lens_cpu = [tokens]
            batch.extend_prefix_lens = torch.zeros_like(lengths, dtype=torch.int32)
            batch.extend_prefix_lens_cpu = [0]
            batch.extend_start_loc = torch.zeros_like(lengths, dtype=torch.int32)
            batch.extend_logprob_start_lens_cpu = [0]
        batch.track_reads(self.static_reads)
        self.attention.keys = []
        self.attention.values = []
        arguments = {
            "input_ids": input_ids,
            self.region.position_arg: positions,
            "forward_batch": batch,
        }
        if self.region.pass_pp_proxy:
            arguments["pp_proxy_tensors"] = None
        result = self.model(**arguments)
        batch._validate_reads()
        if (
            not isinstance(result, LogitsProcessorOutput)
            or result.next_token_logits is None
        ):
            raise UnsupportedMlxRegion(
                "MLX requires next-token logits from the model forward"
            )
        if any(
            value is not None
            for name, value in vars(result).items()
            if name != "next_token_logits"
        ):
            raise UnsupportedMlxRegion("MLX cannot discard auxiliary model outputs")
        if (
            result.next_token_logits.ndim != 2
            or result.next_token_logits.shape[0] != batch.batch_size
        ):
            raise UnsupportedMlxRegion("MLX requires one logits row per decode token")
        if len(self.attention.keys) != len(self.region.layers):
            raise UnsupportedMlxRegion(
                "MLX forward did not invoke every decoder attention"
            )
        return (
            result.next_token_logits,
            torch.stack(self.attention.keys),
            torch.stack(self.attention.values),
        )


def _batch_inputs(batch: ForwardBatch) -> tuple[torch.Tensor, ...]:
    return (
        batch.input_ids,
        batch.positions,
        batch.req_pool_indices,
        batch.seq_lens,
        batch.out_cache_loc,
    )


class CompiledMlxRunner(BaseRunner):
    def __init__(self, model_runner, *, prefill=False):
        super().__init__(model_runner)
        self._prefill = prefill
        self._unsupported_reason = None
        self._region = None
        if model_runner.is_draft_worker or model_runner.spec_algorithm.is_speculative():
            raise ValueError("MLX graph decode does not support speculative decoding")
        if not isinstance(model_runner.token_to_kv_pool, MHATokenToKVPool):
            raise TypeError("MLX graph decode requires the standard MHA KV pool")
        self._layers = ()
        self._init_backend()
        self._max_batch_size = min(16, model_runner.req_to_token_pool.size)
        self._graphs = {}
        self._unsupported_batches = {}
        self._pool_identity = None
        self._model_structure = None
        self._weight_sources = ()
        self._fallback_reasons = set()
        self.execution_count = 0
        self.compile_seconds = 0.0
        self._refresh_region()
        if self._prefill and self._unsupported_reason is None:
            # Export only: no speculative KV writes or hidden warmup inference.
            batch = ForwardBatch(
                forward_mode=ForwardMode.EXTEND,
                batch_size=1,
                input_ids=torch.zeros(8, device="mps", dtype=torch.int64),
                positions=torch.arange(8, device="mps"),
                req_pool_indices=torch.zeros(1, device="mps", dtype=torch.int64),
                seq_lens=torch.tensor([8], device="mps"),
                seq_lens_sum=8,
                out_cache_loc=torch.arange(8, device="mps"),
            )
            try:
                self._graph(batch)
            except UnsupportedMlxRegion as error:
                self._unsupported_reason = str(error)
        logger.info(
            "%s selected for %s %s (batch 1..%d)",
            type(self).__name__,
            type(model_runner.model).__name__,
            "prefill" if prefill else "decode",
            1 if prefill else self._max_batch_size,
        )

    def can_run_graph(self, forward_batch: ForwardBatch) -> bool:
        if self._prefill or forward_batch.forward_mode.is_decode():
            self._refresh_region()
        size = (
            forward_batch.input_ids.numel()
            if self._prefill
            else forward_batch.batch_size
        )
        reason = self._unsupported_reason
        if reason is not None:
            pass
        elif self._prefill and not _plain_prefill(forward_batch):
            reason = "prefill requires an uncached B1 prompt of at most 32768 tokens"
        elif not self._prefill and not forward_batch.forward_mode.is_decode():
            reason = "non-decode mode"
        elif not 1 <= forward_batch.batch_size <= self._max_batch_size:
            reason = "batch outside supported graph range"
        elif (
            forward_batch.input_embeds is not None
            or forward_batch.spec_info is not None
        ):
            reason = "embedding or speculative inputs"
        elif forward_batch.capture_hidden_mode not in (None, CaptureHiddenMode.NULL):
            reason = "hidden-state capture"
        elif forward_batch.lora_ids and any(forward_batch.lora_ids):
            reason = "LoRA request"
        elif (
            forward_batch.global_num_token_non_padded_cpu is not None
            and forward_batch.global_num_token_non_padded_cpu != size
        ):
            reason = "padded decode inputs"
        elif size in self._unsupported_batches:
            reason = self._unsupported_batches[size]
        else:
            try:
                graph = self._graph(forward_batch)
            except UnsupportedMlxRegion as error:
                reason = str(error)
                self._unsupported_batches[size] = reason
            else:
                if not static_metadata_matches(forward_batch, graph.model.static_reads):
                    reason = "batch metadata differs from exported decode contract"
        if reason is not None and reason not in self._fallback_reasons:
            logger.info("%s using eager MPS: %s", type(self).__name__, reason)
            self._fallback_reasons.add(reason)
        if reason is not None:
            self.drain(discard=True, reason="ineligible")
        return reason is None

    def _refresh_region(self):
        model = self.model_runner.model
        previous = self._model_structure
        if (
            previous is not None
            and previous[0] is model
            and all(
                len(module._modules) == len(children)
                and all(module._modules.get(name) is child for name, child in children)
                for module, children in previous[1]
            )
        ):
            return
        structure = (
            model,
            tuple(
                (module, tuple(module._modules.items())) for module in model.modules()
            ),
        )
        self.drain(discard=True, reason="model")
        for graph in self._graphs.values():
            graph.close()
        self._graphs.clear()
        self._unsupported_batches.clear()
        self._model_structure = structure
        self._weight_sources = tuple(
            (module, name, is_parameter)
            for module in model.modules()
            for is_parameter, tensors in (
                (True, module._parameters),
                (False, module._buffers),
            )
            for name, tensor in tensors.items()
            if tensor is not None
        )
        try:
            self._region = discover_decode_region(model)
        except UnsupportedMlxRegion as error:
            self._region = None
            self._unsupported_reason = str(error)
            self._layers = ()
        else:
            self._unsupported_reason = None
            self._layers = self._region.layers

    def load_batch(self, forward_batch: ForwardBatch, **kwargs):
        return forward_batch

    def _validate_metadata(self, batch):
        if self._prefill:
            return self._validate_prefill_metadata(batch)
        if any(tensor.shape != (batch.batch_size,) for tensor in _batch_inputs(batch)):
            raise ValueError("MLX decode requires one input element per request")
        torch.mps.synchronize()
        table = host_alias(self.model_runner.req_to_token_pool.req_to_token)
        requests = host_alias(batch.req_pool_indices).tolist()
        lengths = host_alias(batch.seq_lens).tolist()
        locations = host_alias(batch.out_cache_loc).tolist()
        pool = self.model_runner.token_to_kv_pool
        slots = pool.get_key_buffer(self._layers[0].layer_id).shape[0]
        if len(set(requests)) != len(requests):
            raise ValueError("MLX graph decode requires unique request rows")
        if len(set(locations)) != len(locations):
            raise ValueError("MLX graph decode requires unique KV write locations")
        for request, length, location in zip(requests, lengths, locations):
            if not (
                0 <= request < table.shape[0]
                and 1 <= length <= table.shape[1]
                and 0 < location < slots
            ):
                raise ValueError(
                    "Invalid MLX graph decode request, length or KV location"
                )
            if table[request, length - 1].item() != location:
                raise ValueError(
                    "MLX graph decode KV write location disagrees with table"
                )
            prefix = table[request, : length - 1]
            if bool(((prefix <= 0) | (prefix >= slots)).any()):
                raise ValueError("Invalid MLX graph decode prefix KV slots")
            validate_page_slots(table[request, :length], pool.page_size)
        ids = host_alias(batch.input_ids)
        positions = host_alias(batch.positions)
        if bool(((ids < 0) | (ids >= self.model_runner.model.config.vocab_size)).any()):
            raise ValueError("Compiled MLX input token is outside the vocabulary")
        if bool(
            (
                (positions < 0)
                | (positions >= self.model_runner.model_config.context_len)
            ).any()
        ):
            raise ValueError("Compiled MLX position is outside the context")

    def _validate_prefill_metadata(self, batch):
        tokens = batch.input_ids.numel()
        if any(
            t.shape != (tokens,)
            for t in (batch.input_ids, batch.positions, batch.out_cache_loc)
        ) or any(
            t is None or t.shape != (1,)
            for t in (
                batch.req_pool_indices,
                batch.seq_lens,
                batch.extend_seq_lens,
                batch.extend_prefix_lens,
                batch.extend_start_loc,
            )
        ):
            raise ValueError("Invalid MLX prefill metadata shape")
        torch.mps.synchronize()
        table = host_alias(self.model_runner.req_to_token_pool.req_to_token)
        request = host_alias(batch.req_pool_indices).item()
        locations = host_alias(batch.out_cache_loc)
        pool = self.model_runner.token_to_kv_pool
        slots = pool.get_key_buffer(self._layers[0].layer_id).shape[0]
        if (
            not 0 <= request < table.shape[0]
            or tokens > min(table.shape[1], self.model_runner.model_config.context_len)
            or host_alias(batch.seq_lens).item() != tokens
            or host_alias(batch.extend_seq_lens).item() != tokens
            or host_alias(batch.extend_prefix_lens).item() != 0
            or host_alias(batch.extend_start_loc).item() != 0
            or bool(((locations <= 0) | (locations >= slots)).any())
            or locations.unique().numel() != tokens
            or not torch.equal(table[request, :tokens].long(), locations.long())
        ):
            raise ValueError("Invalid MLX prefill length or KV mapping")
        validate_page_slots(locations, pool.page_size)
        ids = host_alias(batch.input_ids)
        if bool(((ids < 0) | (ids >= self.model_runner.model.config.vocab_size)).any()):
            raise ValueError("Compiled MLX input token is outside the vocabulary")
        if not torch.equal(host_alias(batch.positions), torch.arange(tokens)):
            raise ValueError(
                "MLX uncached prefill requires consecutive positions from zero"
            )

    def _graph(self, batch):
        mr = self.model_runner
        pools = tuple(
            tensor
            for layer in self._layers
            for tensor in mr.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        )
        identity = (id(mr.model),) + tuple(
            (tensor.data_ptr(), tensor.shape, tensor.dtype)
            for tensor in (mr.req_to_token_pool.req_to_token, *pools)
        )
        if self._pool_identity != identity:
            self.drain(discard=True, reason="pool")
            for graph in self._graphs.values():
                graph.close()
            self._graphs.clear()
            self._pool_identity = identity
        size = batch.input_ids.numel() if self._prefill else batch.batch_size
        key = "prefill" if self._prefill and size >= 2 else size
        if key not in self._graphs:
            wrapper = self._make_wrapper()
            start = time.perf_counter()
            _to_torch(mr.model, reverse=False, num_tokens=size)
            try:
                with forward_context(ForwardContext(attn_backend=wrapper.attention)):
                    graph = self._compile(wrapper, batch)
            finally:
                _to_torch(mr.model, reverse=True, num_tokens=size)
                wrapper.attention.keys.clear()
                wrapper.attention.values.clear()
            elapsed = time.perf_counter() - start
            self.compile_seconds += elapsed
            self._graphs[key] = graph
            logger.info(
                "%s %s graph ready: size=%d, page_size=%d, export=%.3fs",
                type(graph).__name__,
                "prefill" if self._prefill else "decode",
                size,
                mr.token_to_kv_pool.page_size,
                elapsed,
            )
        return self._graphs[key]

    def _init_backend(self):
        if (
            get_parallel().tp_size != 1
            or get_parallel().pp_size != 1
            or get_parallel().dp_size != 1
        ):
            raise ValueError("Compiled MLX requires single-rank execution")
        # Only a scheduler that fences ingress and pool reuse may enable lookahead.
        self._async = False
        self._pending = None
        self.prefetch_count = 0
        self.prefetch_hits = 0
        self.prefetch_discards = 0
        self.prefetch_discard_reasons = {}

    def enable_lookahead(self):
        if self._prefill:
            raise RuntimeError("Compiled MLX lookahead is decode-only")
        if not get_schedule().disable_overlap_schedule:
            raise RuntimeError("Compiled MLX lookahead requires the normal scheduler")
        self._async = True

    def _make_wrapper(self):
        mr = self.model_runner
        return _DecodeModule(
            model=mr.model,
            req_pool=mr.req_to_token_pool,
            kv_pool=mr.token_to_kv_pool,
            region=self._region,
            prefill=self._prefill,
        ).eval()

    def _compile(self, wrapper, batch):
        return CompiledMlxGraph(
            model=wrapper,
            example_inputs=_batch_inputs(batch),
            attention=mlx_radix_decode,
            custom_lowerings=(
                {torch.ops.aten.scaled_dot_product_attention.default: _causal_sdpa}
                if self._prefill
                else None
            ),
            dynamic_tokens=self._prefill and batch.input_ids.numel() >= 2,
        )

    def drain(self, *, discard=False, reason="explicit"):
        """Fence before scheduler ingress, pool reuse, weight updates or shutdown."""
        if self._pending is not None:
            mx.eval(*self._pending.outputs)
            if discard:
                self._record_discard(reason)
                self._pending = None

    def _record_discard(self, reason):
        self.prefetch_discards += 1
        self.prefetch_discard_reasons[reason] = (
            self.prefetch_discard_reasons.get(reason, 0) + 1
        )

    def _weights(self):
        return tuple(
            (tensor.data_ptr(), None if tensor.is_inference() else tensor._version)
            for module, name, is_parameter in self._weight_sources
            for tensor in (
                module._parameters[name] if is_parameter else module._buffers[name],
            )
        )

    def _metadata(self, batch):
        requests = tuple(host_alias(batch.req_pool_indices).tolist())
        generations = self.model_runner.req_to_token_pool.req_generation
        return (
            requests,
            tuple(host_alias(batch.seq_lens).tolist()),
            tuple(host_alias(batch.positions).tolist()),
            tuple(generations[list(requests)].tolist()),
        )

    def _consume(self, *, graph, batch, metadata, weights):
        pending = self._pending
        self._pending = None
        if pending is None:
            return None
        table = host_alias(self.model_runner.req_to_token_pool.req_to_token)
        reason = None
        if pending.graph is not graph:
            reason = "graph"
        elif metadata != (
            pending.requests,
            pending.lengths,
            pending.positions,
            pending.generations,
        ):
            reason = "metadata"
        elif weights != pending.weights:
            reason = "weights"
        elif host_alias(batch.input_ids).tolist() != pending.tokens.tolist():
            reason = "tokens"
        elif not all(
            table[request, length - 2].item() == location
            for request, length, location in zip(
                pending.requests, pending.lengths, pending.locations
            )
        ):
            reason = "cache_mapping"
        if reason is None:
            self.prefetch_hits += 1
            return pending.outputs
        self._record_discard(reason)
        return None

    def _prefetch(self, *, graph, arrays, outputs, batch, metadata, weights):
        requests, lengths, positions, generations = metadata
        if (
            not self._async
            or not _plain_greedy(batch.sampling_info)
            or max(lengths) >= self.model_runner.model_config.context_len
        ):
            return
        tokens = mx.argmax(outputs[0], axis=-1).astype(mx.int64)
        following = list(arrays)
        for index, (kind, target) in enumerate(graph.bindings):
            if kind == InputKind.USER_INPUT:
                if target == 0:
                    following[index] = tokens
                elif target in (1, 3):
                    following[index] = following[index] + 1
        # This submission precedes the first wait/export of the current outputs.
        # The previous token's K/V is a graph dependency, not an early pool write.
        next_outputs = graph.launch(tuple(following), tails=outputs[1:])
        self._pending = _PendingDecode(
            graph=graph,
            outputs=next_outputs,
            tokens=tokens,
            requests=requests,
            lengths=tuple(n + 1 for n in lengths),
            positions=tuple(p + 1 for p in positions),
            generations=generations,
            locations=tuple(host_alias(batch.out_cache_loc).tolist()),
            weights=weights,
            owners=tuple(graph.views.values()),
        )
        self.prefetch_count += 1

    @torch.no_grad()
    def execute(self, forward_batch, **kwargs):
        self.drain()
        if not self.can_run_graph(forward_batch):
            raise ValueError("Ineligible batch passed to compiled MLX")
        # A lookahead hit does not validate the new write slot or its table entry.
        self._validate_metadata(forward_batch)
        graph = self._graph(forward_batch)
        arrays = graph.bind(_batch_inputs(forward_batch))
        if self._prefill:
            outputs = graph.launch(arrays)
        else:
            metadata = (
                self._metadata(forward_batch)
                if self._async or self._pending is not None
                else None
            )
            weights = self._weights() if self._async else ()
            outputs = self._consume(
                graph=graph, batch=forward_batch, metadata=metadata, weights=weights
            )
            if outputs is None:
                outputs = graph.launch(arrays)
            if self._async:
                self._prefetch(
                    graph=graph,
                    arrays=arrays,
                    outputs=outputs,
                    batch=forward_batch,
                    metadata=metadata,
                    weights=weights,
                )
        mx.eval(*outputs)
        logits, keys, values = tuple(
            _export_evaluated_mlx(array, torch.device("mps"), mx) for array in outputs
        )
        for index, layer in enumerate(self._layers):
            self.model_runner.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, keys[index], values[index]
            )
        self.execution_count += 1
        if self.execution_count == 1 or self.execution_count % 100 == 0:
            logger.info(
                "Compiled MLX: executions=%d, enqueued=%d, hits=%d, discarded=%d, discard_reasons=%s",
                self.execution_count,
                self.prefetch_count,
                self.prefetch_hits,
                self.prefetch_discards,
                self.prefetch_discard_reasons,
            )
        # The Torch sampler may mutate logits; it must not overwrite a source
        # still being consumed by the dependent MLX argmax/forward.
        return LogitsProcessorOutput(
            next_token_logits=logits.to(torch.float32, copy=True)
        )
