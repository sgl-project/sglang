# SPDX-License-Identifier: Apache-2.0
"""Bounded exact eager text generation for native SenseNova U1."""

from __future__ import annotations

import hashlib
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(slots=True)
class U1ExactTextResult:
    token_ids: list[int]
    prefill_elapsed_s: float
    decode_elapsed_s: float
    total_elapsed_s: float
    prefix_cache_hit: bool
    graph_created: bool
    graph_replayed: bool


@dataclass(slots=True)
class _ExactDecodeGraph:
    graphs: list[torch.cuda.CUDAGraph]
    graph_pool: Any
    initial_caches: list[tuple[torch.Tensor, torch.Tensor]]
    input_token: torch.Tensor
    decode_indexes: list[torch.Tensor]
    generated_tokens: list[torch.Tensor]
    source_prefix_key: str
    prefix_len: int


def _update_hash_with_tensor(hasher, tensor: torch.Tensor) -> None:
    cpu = tensor.detach().cpu().contiguous()
    hasher.update(str(tuple(cpu.shape)).encode("ascii"))
    hasher.update(str(cpu.dtype).encode("ascii"))
    if cpu.dtype == torch.bfloat16:
        cpu = cpu.view(torch.int16)
    elif cpu.dtype == torch.bool:
        cpu = cpu.to(torch.uint8)
    hasher.update(cpu.numpy().tobytes())


class U1ExactTextRuntime:
    def __init__(
        self,
        model,
        *,
        prefix_cache_max_entries: int = 4,
        graph_cache_max_entries: int = 2,
        graph_capture_max_attempts: int = 4,
        prefix_cache_max_tokens: int = 2048,
        graph_max_total_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.prefix_cache_max_entries = prefix_cache_max_entries
        disable_graph = os.environ.get(
            "SENSENOVA_U1_EXACT_TEXT_DISABLE_GRAPH",
            "",
        ).lower() in {"1", "true", "yes", "on"}
        self.graph_cache_max_entries = 0 if disable_graph else graph_cache_max_entries
        self.graph_capture_max_attempts = graph_capture_max_attempts
        self.prefix_cache_max_tokens = prefix_cache_max_tokens
        self.graph_max_total_tokens = graph_max_total_tokens
        self.graph_capture_attempts = 0
        self.prefix_cache: OrderedDict[
            str,
            tuple[
                torch.Tensor,
                list[tuple[torch.Tensor, torch.Tensor]],
                int,
            ],
        ] = OrderedDict()
        self.graph_cache: OrderedDict[tuple[int, ...], _ExactDecodeGraph] = (
            OrderedDict()
        )

    def clear(self) -> None:
        self.prefix_cache.clear()
        while self.graph_cache:
            _, graph = self.graph_cache.popitem(last=False)
            self._reset_graph(graph)

    @staticmethod
    def _reset_graph(graph: _ExactDecodeGraph) -> None:
        for cuda_graph in graph.graphs:
            reset = getattr(cuda_graph, "reset", None)
            if callable(reset):
                reset()

    def _prefix_key(
        self,
        *,
        input_ids: torch.Tensor,
        indexes: torch.Tensor,
        image_token_tag: torch.Tensor,
        input_embeds: torch.Tensor | None,
        compiled_add_rms: bool,
        lm_head_linear: bool,
    ) -> str:
        hasher = hashlib.sha256()
        hasher.update(b"sensenova-u1-exact-text-prefix-v1")
        hasher.update(b":compiled-add-rms" if compiled_add_rms else b":plain-add-rms")
        hasher.update(b":linear-head" if lm_head_linear else b":matmul-head")
        _update_hash_with_tensor(hasher, input_ids.to(dtype=torch.long))
        _update_hash_with_tensor(hasher, indexes.to(dtype=torch.long))
        tag = image_token_tag.to(dtype=torch.bool)
        _update_hash_with_tensor(hasher, tag)
        if bool(tag.any().item()) and input_embeds is not None:
            _update_hash_with_tensor(hasher, input_embeds)
        return hasher.hexdigest()

    def _get_prefix(
        self,
        key: str,
    ) -> (
        tuple[
            torch.Tensor,
            list[tuple[torch.Tensor, torch.Tensor]],
            int,
        ]
        | None
    ):
        value = self.prefix_cache.pop(key, None)
        if value is not None:
            self.prefix_cache[key] = value
        return value

    def _put_prefix(
        self,
        key: str,
        value: tuple[
            torch.Tensor,
            list[tuple[torch.Tensor, torch.Tensor]],
            int,
        ],
    ) -> None:
        self.prefix_cache.pop(key, None)
        self.prefix_cache[key] = value
        while len(self.prefix_cache) > self.prefix_cache_max_entries:
            self.prefix_cache.popitem(last=False)

    def _get_graph(self, key: tuple[int, ...]) -> _ExactDecodeGraph | None:
        value = self.graph_cache.pop(key, None)
        if value is not None:
            self.graph_cache[key] = value
        return value

    def _put_graph(
        self,
        key: tuple[int, ...],
        value: _ExactDecodeGraph,
    ) -> None:
        previous = self.graph_cache.pop(key, None)
        if previous is not None:
            self._reset_graph(previous)
        self.graph_cache[key] = value
        while len(self.graph_cache) > self.graph_cache_max_entries:
            _, evicted = self.graph_cache.popitem(last=False)
            self._reset_graph(evicted)

    def _run_decode_step(
        self,
        *,
        input_token: torch.Tensor,
        caches: list[tuple[torch.Tensor, torch.Tensor]],
        indexes: torch.Tensor,
        cache_position: int,
    ) -> torch.Tensor:
        hidden_states = (
            self.model.language_model.model.eager_text_decode_with_static_cache(
                input_token,
                indexes=indexes,
                caches=caches,
                cache_position=cache_position,
                repeat_kv_cache=True,
            )
        )
        logits = self.model.language_model.eager_text_logits(hidden_states)
        return torch.argmax(logits, dim=-1).to(dtype=torch.long)

    def _capture_graph(
        self,
        *,
        caches: list[tuple[torch.Tensor, torch.Tensor]],
        input_token_id: int,
        start_t_index: int,
        decode_steps: int,
        prefix_key: str,
    ) -> _ExactDecodeGraph:
        device = caches[0][0].device
        prefix_len = int(caches[0][0].shape[0])
        capacity = prefix_len + decode_steps - 1
        initial_caches = [
            (
                torch.empty(
                    (capacity, *k.shape[1:]),
                    device=k.device,
                    dtype=k.dtype,
                ),
                torch.empty(
                    (capacity, *v.shape[1:]),
                    device=v.device,
                    dtype=v.dtype,
                ),
            )
            for k, v in caches
        ]
        for (static_k, static_v), (live_k, live_v) in zip(
            initial_caches,
            caches,
            strict=True,
        ):
            static_k[:prefix_len].copy_(live_k)
            static_v[:prefix_len].copy_(live_v)

        input_token = torch.full(
            (1,),
            input_token_id,
            device=device,
            dtype=torch.long,
        )
        capture_stream = torch.cuda.Stream(device=device)
        graph_pool = torch.cuda.graph_pool_handle()
        capture_stream.wait_stream(torch.cuda.current_stream(device))
        graphs = []
        decode_indexes = []
        generated_tokens = []
        current_token_id = input_token_id

        for step_offset in range(decode_steps - 1):
            indexes = torch.tensor(
                [[start_t_index + step_offset + 1], [0], [0]],
                device=device,
                dtype=torch.long,
            )
            decode_indexes.append(indexes)
            next_token = torch.empty(
                (1,),
                device=device,
                dtype=torch.long,
            )
            input_token.fill_(current_token_id)
            with torch.cuda.stream(capture_stream):
                next_token.copy_(
                    self._run_decode_step(
                        input_token=input_token,
                        caches=initial_caches,
                        indexes=indexes,
                        cache_position=prefix_len + step_offset,
                    )
                )
            capture_stream.synchronize()
            next_token_id = int(next_token.item())
            input_token.fill_(current_token_id)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                graph,
                pool=graph_pool,
                stream=capture_stream,
            ):
                next_token.copy_(
                    self._run_decode_step(
                        input_token=input_token,
                        caches=initial_caches,
                        indexes=indexes,
                        cache_position=prefix_len + step_offset,
                    )
                )
            graphs.append(graph)
            generated_tokens.append(next_token)
            current_token_id = next_token_id

        capture_stream.synchronize()
        torch.cuda.current_stream(device).wait_stream(capture_stream)
        return _ExactDecodeGraph(
            graphs=graphs,
            graph_pool=graph_pool,
            initial_caches=initial_caches,
            input_token=input_token,
            decode_indexes=decode_indexes,
            generated_tokens=generated_tokens,
            source_prefix_key=prefix_key,
            prefix_len=prefix_len,
        )

    def _copy_prefix_caches(
        self,
        graph: _ExactDecodeGraph,
        caches: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        for (static_k, static_v), (live_k, live_v) in zip(
            graph.initial_caches,
            caches,
            strict=True,
        ):
            static_k[: graph.prefix_len].copy_(live_k)
            static_v[: graph.prefix_len].copy_(live_v)

    def _run_eager_tail(
        self,
        *,
        first_token: int,
        caches: list[tuple[torch.Tensor, torch.Tensor]],
        start_t_index: int,
        decode_steps: int,
    ) -> list[int]:
        generated = [first_token]
        capacity = int(caches[0][0].shape[0]) + decode_steps - 1
        static_caches = [
            (
                torch.empty(
                    (capacity, *k.shape[1:]),
                    device=k.device,
                    dtype=k.dtype,
                ),
                torch.empty(
                    (capacity, *v.shape[1:]),
                    device=v.device,
                    dtype=v.dtype,
                ),
            )
            for k, v in caches
        ]
        prefix_len = int(caches[0][0].shape[0])
        for (static_k, static_v), (live_k, live_v) in zip(
            static_caches,
            caches,
            strict=True,
        ):
            static_k[:prefix_len].copy_(live_k)
            static_v[:prefix_len].copy_(live_v)
        token = torch.tensor(
            [first_token],
            device=caches[0][0].device,
            dtype=torch.long,
        )
        for step_offset in range(decode_steps - 1):
            indexes = torch.tensor(
                [[start_t_index + step_offset + 1], [0], [0]],
                device=token.device,
                dtype=torch.long,
            )
            token = self._run_decode_step(
                input_token=token,
                caches=static_caches,
                indexes=indexes,
                cache_position=prefix_len + step_offset,
            )
            generated.append(int(token.item()))
        return generated

    @torch.inference_mode()
    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        indexes: torch.Tensor,
        image_token_tag: torch.Tensor,
        input_embeds: torch.Tensor | None,
        decode_steps: int,
        compiled_add_rms: bool = False,
        lm_head_linear: bool = False,
    ) -> U1ExactTextResult:
        if decode_steps <= 0:
            raise ValueError("decode_steps must be positive")
        if input_ids.ndim != 1 or indexes.shape != (3, input_ids.numel()):
            raise ValueError("invalid exact text input shapes")
        if image_token_tag.numel() != input_ids.numel():
            raise ValueError("image token tag length mismatch")

        device = input_ids.device
        self.model.language_model.exact_lm_head_linear = lm_head_linear
        for layer in self.model.language_model.model.layers:
            layer.exact_compiled_add_rms = compiled_add_rms
        total_started = time.perf_counter()
        prefix_key = self._prefix_key(
            input_ids=input_ids,
            indexes=indexes,
            image_token_tag=image_token_tag,
            input_embeds=input_embeds,
            compiled_add_rms=compiled_add_rms,
            lm_head_linear=lm_head_linear,
        )
        cached_prefix = self._get_prefix(prefix_key)
        prefix_cache_hit = cached_prefix is not None

        torch.cuda.synchronize(device)
        prefill_started = time.perf_counter()
        if cached_prefix is None:
            hidden_states, caches = (
                self.model.language_model.model.eager_text_prefill_with_cache(
                    input_ids,
                    input_embeds=input_embeds,
                    indexes=indexes,
                    image_token_tag=image_token_tag,
                    repeat_kv_cache=True,
                )
            )
            logits = self.model.language_model.eager_text_logits(hidden_states)
            start_t_index = int(indexes[0].max().item())
            if input_ids.numel() <= self.prefix_cache_max_tokens:
                self._put_prefix(
                    prefix_key,
                    (
                        logits.detach(),
                        [(k.detach(), v.detach()) for k, v in caches],
                        start_t_index,
                    ),
                )
        else:
            logits, caches, start_t_index = cached_prefix
        first_token = int(torch.argmax(logits.float(), dim=-1).item())
        prefill_elapsed_s = time.perf_counter() - prefill_started

        decode_elapsed_s = 0.0
        graph_created = False
        graph_replayed = False
        generated = [first_token]
        if decode_steps > 1:
            graph_key = (
                int(input_ids.numel()),
                start_t_index,
                decode_steps,
                int(compiled_add_rms),
                int(lm_head_linear),
            )
            graph = self._get_graph(graph_key)
            graph_eligible = (
                input_ids.numel() + decode_steps <= self.graph_max_total_tokens
                and self.graph_cache_max_entries > 0
            )
            if (
                graph is None
                and graph_eligible
                and self.graph_capture_attempts < self.graph_capture_max_attempts
            ):
                self.graph_capture_attempts += 1
                graph = self._capture_graph(
                    caches=caches,
                    input_token_id=first_token,
                    start_t_index=start_t_index,
                    decode_steps=decode_steps,
                    prefix_key=prefix_key,
                )
                self._put_graph(graph_key, graph)
                graph_created = True

            if graph_created:
                torch.cuda.synchronize(device)
            decode_started = time.perf_counter()
            if graph is not None:
                if graph.source_prefix_key != prefix_key:
                    self._copy_prefix_caches(graph, caches)
                    graph.source_prefix_key = prefix_key
                graph.input_token.fill_(first_token)
                for cuda_graph, next_token in zip(
                    graph.graphs,
                    graph.generated_tokens,
                    strict=True,
                ):
                    cuda_graph.replay()
                    graph.input_token.copy_(next_token)
                torch.cuda.synchronize(device)
                generated.extend(int(token.item()) for token in graph.generated_tokens)
                graph_replayed = True
            else:
                generated = self._run_eager_tail(
                    first_token=first_token,
                    caches=caches,
                    start_t_index=start_t_index,
                    decode_steps=decode_steps,
                )
                torch.cuda.synchronize(device)
            decode_elapsed_s = time.perf_counter() - decode_started

        return U1ExactTextResult(
            token_ids=generated,
            prefill_elapsed_s=prefill_elapsed_s,
            decode_elapsed_s=decode_elapsed_s,
            total_elapsed_s=time.perf_counter() - total_started,
            prefix_cache_hit=prefix_cache_hit,
            graph_created=graph_created,
            graph_replayed=graph_replayed,
        )


__all__ = ["U1ExactTextResult", "U1ExactTextRuntime"]
