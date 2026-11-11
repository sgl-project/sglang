# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Publication preparation and fixed double-buffer transfer/decode/apply scheduling."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

from sglang.srt.weight_sync.gpu_delta import profiling
from sglang.srt.weight_sync.gpu_delta.bindings import (
    _ITEMSIZES,
    _digest,
    validate_ue8m0_payload,
)
from sglang.srt.weight_sync.gpu_delta.derived import prepare_updates
from sglang.srt.weight_sync.gpu_delta.layout import GpuDeltaLayout


class GpuDeltaBackend:
    """Scheduler-owned plan; large decoded masks exist only during apply."""

    def __init__(self, model_runner, identity):
        from sglang.srt.weight_sync.gpu_delta.checkpoint import (
            read_canonical_checkpoint_inventory,
        )

        self._canonical_plan_digest = None
        self.binding_table = None
        self.batch_plan = None
        self.layers_per_batch = int(os.environ.get("GPU_DELTA_LAYERS_PER_BATCH", "1"))
        if self.layers_per_batch < 1:
            raise ValueError("GPU_DELTA_LAYERS_PER_BATCH must be positive")
        inventory = read_canonical_checkpoint_inventory(model_runner)
        self.layout = GpuDeltaLayout(model_runner.model, inventory)
        self.device = next(model_runner.model.parameters()).device
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("direct GPU deltas require an explicit CUDA device")
        from sglang.srt.weight_sync.gpu_delta.host import HostArena

        self.host_arena = HostArena(identity["engine_id"], self.device.index, identity)
        self.decoders = {}
        self.apply_stream = None
        self.io_stream = None

    def prepare(self, request, cancelled=None):
        # Called by the background worker. Small scalar calibration may be read
        # to prepare derived values; no live writes, collectives, loader hooks
        # or generation-stream synchronization occur here.
        with torch.cuda.device(self.device):
            prepared = PreparedDelta(self)
            try:
                prepared.prepare(request, cancelled)
                return prepared
            except BaseException:
                self.discard(prepared)
                raise

    def discard(self, prepared):
        # A failed GPU drain must retain every host mapping its readers may use.
        prepared.close()
        self.close()

    def close(self):
        self.host_arena.close()


@dataclass
class _PreparedBatch:
    decoder: object
    apply: Callable[[], None] | None
    zero_ranges: list[torch.Tensor]
    check_status: Callable[[], None]


class _BindingTable:
    """Rank-local geometry; only publication row values change between updates."""

    def __init__(self, layout, entries):
        self.bindings = tuple(layout.bindings)
        self.ordinals = np.asarray(
            [entries[b.name][0] for b in self.bindings], dtype=np.int64
        )
        self.nbytes = np.asarray(
            [entries[b.name][1]["nbytes"] for b in self.bindings], dtype=np.int64
        )
        self.raw = np.asarray(
            [b.encoding == "raw_bytes" for b in self.bindings], dtype=bool
        )
        self.ue8m0 = np.asarray([b.raw_ue8m0 for b in self.bindings], dtype=bool)
        self.raw_order = np.concatenate(
            (
                np.flatnonzero(self.raw & self.ue8m0),
                np.flatnonzero(self.raw & ~self.ue8m0),
            )
        )
        non_layers = {"embed_tokens": [], "lm_head": [], "standalone": []}
        layers, storage_rows = {}, {}
        for row, binding in enumerate(self.bindings):
            if binding.layer is None:
                module = binding.name.rpartition(".")[0].rpartition(".")[2]
                non_layers.get(module, non_layers["standalone"]).append(row)
            else:
                layers.setdefault(binding.layer, []).append(row)
            for pointer in binding.storage_pointers:
                storage_rows.setdefault(pointer, []).append(row)
        self.non_layers = tuple(
            np.asarray(rows, np.int64) for rows in non_layers.values()
        )
        self.layers = tuple(np.asarray(rows, np.int64) for rows in layers.values())
        self.derived = tuple(layout.derived)
        pairs = [
            (row, index)
            for index, image in enumerate(self.derived)
            for row in storage_rows.get(image.source_pointer, ())
        ]
        self.derived_rows, self.derived_indices = (
            np.asarray(pairs, np.int64).reshape(-1, 2).T
        )

    def select(self, tensors):
        local = tensors[self.ordinals]
        matrices = (local["frame_count"] != 0) & ~self.raw
        raw = (local["changed_bytes"] != 0) & self.raw
        active = matrices | raw
        derived = np.zeros(len(self.derived), dtype=bool)
        derived[self.derived_indices[active[self.derived_rows]]] = True
        return (
            np.flatnonzero(matrices),
            self.raw_order[raw[self.raw_order]],
            [self.derived[row] for row in np.flatnonzero(derived)],
        )


@profiling.annotate("gpu_delta.prepare.layer_plan")
def _plan_layers(backend, table, rows):
    """Cache membership, canonical offsets and apply geometry, never wire frames."""
    key = rows.tobytes()
    if backend.batch_plan is None or backend.batch_plan[0] != key:
        if not len(rows):
            backend.batch_plan = (key, [])
            return []
        from sglang.srt.weight_sync.gpu_delta.apply import plan_apply

        active = np.zeros(len(table.bindings), dtype=bool)
        active[rows] = True
        groups = [group[active[group]] for group in table.non_layers]
        groups = [group for group in groups if len(group)]
        model_layers = [group[active[group]] for group in table.layers]
        model_layers = [group for group in model_layers if len(group)]
        layers_per_batch = backend.layers_per_batch
        groups.extend(
            np.concatenate(model_layers[start : start + layers_per_batch])
            for start in range(0, len(model_layers), layers_per_batch)
        )
        plans = []
        for group in groups:
            outputs, size = [], 0
            for row in group:
                offset = (size + 15) // 16 * 16
                binding, nbytes = table.bindings[row], int(table.nbytes[row])
                outputs.append((binding, offset, nbytes))
                size = offset + nbytes
            # Segment views share one decoded canonical payload. Only the apply
            # plan expands them; DE membership and output offsets stay unique.
            apply_outputs = [
                (view, offset, nbytes)
                for binding, offset, nbytes in outputs
                for view in binding.apply_bindings()
            ]
            tensor_offsets = np.stack(
                (table.ordinals[group], [offset for _, offset, _ in outputs])
            )
            plans.append((tensor_offsets, size, plan_apply(apply_outputs)))
        backend.batch_plan = (key, plans)
    return backend.batch_plan[1]


@profiling.annotate("gpu_delta.prepare.frame_plan")
def _plan_decode(plans, publication, offsets):
    """Rebase shared numeric frame and gap slices into this rank's layer batches."""
    batches = [publication.tensors[outputs[0]] for outputs, _, _ in plans]
    counts = [int(tensors["frame_count"].sum()) for tensors in batches]
    table = np.empty((4, sum(counts)), dtype=np.int64)
    gaps, position = [], 0
    for (outputs, _, _), tensors, count in zip(plans, batches, counts):
        tensor_ids, output_offsets = outputs
        frame_counts = tensors["frame_count"]
        starts = tensors["frame_start"] - (np.cumsum(frame_counts) - frame_counts)
        indices = np.repeat(starts, frame_counts) + np.arange(count, dtype=np.int64)
        destination = table[:, position : position + count]
        destination[:] = publication.frames[:, indices]
        destination[0] += np.repeat(offsets[tensor_ids], frame_counts)
        destination[3] += np.repeat(output_offsets, frame_counts)
        position += count

        gap_counts = tensors["gap_count"]
        starts = tensors["gap_start"] - (np.cumsum(gap_counts) - gap_counts)
        indices = np.repeat(starts, gap_counts) + np.arange(
            int(gap_counts.sum()), dtype=np.int64
        )
        batch_gaps = publication.gaps[indices]
        batch_gaps[:, 0] += np.repeat(output_offsets, gap_counts)
        gaps.append(batch_gaps)
    return table, counts, gaps


@profiling.annotate("gpu_delta.prepare.canonical_plan")
def _qualify_canonical_plan(backend, publication):
    """Bind a canonical publication to the independently prepared local layout."""
    digest = publication.header["plan_digest"]
    if backend._canonical_plan_digest is not None:
        if digest != backend._canonical_plan_digest:
            raise ValueError("canonical delta plan changed")
        return backend.binding_table, True
    catalog = publication.read_catalog()
    if catalog["plan_digest"] != digest:
        raise ValueError("shared canonical catalog plan changed")
    entries = {
        entry["name"]: (ordinal, entry)
        for ordinal, entry in enumerate(catalog["tensors"])
    }
    if len(entries) != len(catalog["tensors"]):
        raise ValueError("duplicate canonical tensor")
    definitions = []
    for name, (_, entry) in entries.items():
        if name not in backend.layout.inventory or backend.layout.excluded.get(
            name
        ) not in {
            None,
            "expert projection owned by another EP/TP rank",
            "static activation calibration",
            "loader ignores computed rotary frequencies",
        }:
            raise ValueError(f"delta publication has an unadmitted tensor: {name}")
        canonical = backend.layout.inventory[name]
        definition = {
            "name": name,
            "dtype": canonical["dtype"],
            "shape": canonical["shape"],
            "encoding": canonical.get(
                "encoding", "raw_bytes" if len(canonical["shape"]) <= 1 else "xor_bytes"
            ),
        }
        for key in ("format", "checkpoint_dtype"):
            if key in canonical:
                definition[key] = canonical[key]
        if any(entry.get(key) != value for key, value in definition.items()) or any(
            key in entry and key not in definition
            for key in ("format", "checkpoint_dtype")
        ):
            raise ValueError(f"canonical tensor metadata mismatch: {name}")
        if entry["byte_order"] != "little" or entry["nbytes"] != (
            math.prod(canonical["shape"]) * _ITEMSIZES[canonical["dtype"]]
        ):
            raise ValueError(f"unsupported canonical tensor size/encoding: {name}")
        definitions.append(definition)
    missing = {binding.name for binding in backend.layout.bindings} - entries.keys()
    if missing:
        raise ValueError(f"publication is missing local tensors: {sorted(missing)}")
    if _digest(sorted(definitions, key=lambda entry: entry["name"])) != digest:
        raise ValueError("publication does not match its canonical tensor plan")
    # Tensor selection and physical layout are receiver-local. The publication
    # never contains TP/EP slices, backend permutations or receiver identities.
    table = _BindingTable(backend.layout, entries)
    backend.binding_table = table
    backend._canonical_plan_digest = digest
    return table, False


class PreparedDelta:
    def __init__(self, backend):
        self.stream = None
        self.io_stream = None
        self.host_snapshot = None
        self._gpu_complete = False
        self.batches, self.raw_copies = [], {}
        self.timings = {}
        self.status_checks = []
        self.apply_bindings, self.apply_source_patches = [], []
        self.apply_fallback_groups = set()
        self.apply_host_view = None
        self.backend, self.device = backend, backend.device

    @profiling.annotate("gpu_delta.prepare.layout")
    def prepare(self, request, cancelled=None):
        from sglang.srt.weight_sync.gpu_delta.host import copy_bytes

        backend = self.backend
        preparation_started = time.perf_counter()
        payload_started = time.perf_counter()
        publication = backend.host_arena.prepare_publication(
            request, self.timings, cancelled
        )
        self.codec = publication.header["codec"]
        from sglang.srt.weight_sync.gpu_delta.codec import NvcompDecoder

        inner_codec = self.codec.removesuffix("-zstd")
        if inner_codec not in backend.decoders:
            backend.decoders[inner_codec] = NvcompDecoder(self.device, inner_codec)
        self.decoder = backend.decoders[inner_codec]
        self.hardware = self.decoder.backend == "hardware"
        self.target_version = publication.header["target_version"]
        plan_started = time.perf_counter()
        table, reused_plan = _qualify_canonical_plan(backend, publication)
        self.timings["host_plan_validate_s"] = time.perf_counter() - plan_started
        self.timings["host_plan_cache_reused"] = int(reused_plan)
        self.host_snapshot = backend.host_arena.prepare_local(
            publication,
            table.ordinals,
            self.timings,
            self.hardware,
        )
        self.timings["host_rank_prepare_body_s"] = time.perf_counter() - payload_started
        self.timings["host_rank_prepare_s"] = self.timings["host_rank_prepare_body_s"]

        tensors_started = time.perf_counter()
        compressed, raw_rows, self.derived = table.select(publication.tensors)
        # Compact scales form one prefix for validation and native GPU expansion.
        self.raw_ue8m0_bytes = int(table.nbytes[raw_rows[table.ue8m0[raw_rows]]].sum())
        previous_plan = backend.batch_plan
        self.static_plans = _plan_layers(backend, table, compressed)
        self.timings["host_batch_plan_reused"] = int(
            previous_plan is not None and backend.batch_plan is previous_plan
        )
        frame_table, frame_counts, self.gaps = _plan_decode(
            self.static_plans,
            publication,
            self.host_snapshot.offsets,
        )
        self.max_decoded = max((plan[1] for plan in self.static_plans), default=0)
        self.matrix_tensor_count = len(compressed)
        self.raw_tensor_count = len(raw_rows)
        self.timings["host_tensor_prepare_s"] = time.perf_counter() - tensors_started

        # Pack the small complete-target bypass once on the host. Decoded-mask
        # storage waits for the actual serving pause; kernel warmup is below.
        raw_started = time.perf_counter()
        raw_targets, raw_h2d_bytes = [], 0
        for row in raw_rows:
            binding, size = table.bindings[row], int(table.nbytes[row])
            position = (
                raw_h2d_bytes if binding.raw_ue8m0 else (raw_h2d_bytes + 7) // 8 * 8
            )
            raw_targets.append((binding, position, size))
            raw_h2d_bytes = position + size
        self.raw_pinned = torch.empty(
            raw_h2d_bytes, dtype=torch.uint8, device="cpu", pin_memory=True
        )
        raw_view = memoryview(self.raw_pinned.numpy())
        for ordinal, (_, position, size) in zip(table.ordinals[raw_rows], raw_targets):
            copy_bytes(
                raw_view[position : position + size],
                self.host_snapshot.get(int(ordinal)),
            )
        if self.raw_ue8m0_bytes:
            validate_ue8m0_payload(self.raw_pinned[: self.raw_ue8m0_bytes])
        self.timings.update(
            host_raw_pack_s=time.perf_counter() - raw_started,
            raw_tensors=len(raw_rows),
            raw_bytes=int(table.nbytes[raw_rows].sum()),
            raw_h2d_bytes=raw_h2d_bytes,
            compressed_batches=len(self.static_plans),
            decode_stages=2,
            layers_per_batch=backend.layers_per_batch,
            compressed_tensors=self.matrix_tensor_count,
            de_host_input_bytes=int(frame_table[1].sum()) if self.hardware else 0,
            decoded_zero_ranges=sum(map(len, self.gaps)),
            decoded_zero_bytes=sum(int(gaps[:, 1].sum()) for gaps in self.gaps),
        )
        self._prepare_gpu_metadata(frame_table, frame_counts, raw_targets)
        self.timings["host_prepare_s"] = time.perf_counter() - preparation_started

    @profiling.annotate("gpu_delta.prepare.gpu_metadata")
    def _prepare_gpu_metadata(self, frame_table, frame_counts, raw_targets):
        """Prepare small GPU inputs; retain their leases, not wire-frame objects."""
        started = time.perf_counter()
        backend = self.backend
        if backend.apply_stream is None:
            apply_stream = torch.cuda.Stream(device=self.device)
            io_stream = torch.cuda.Stream(device=self.device)
            backend.apply_stream, backend.io_stream = apply_stream, io_stream
        self.stream, self.io_stream = backend.apply_stream, backend.io_stream
        self.decoded_ready = [torch.cuda.Event() for _ in range(2)]
        self.decoded_free = [torch.cuda.Event() for _ in range(2)]
        with torch.cuda.stream(self.stream):
            self.raw_device = self.raw_pinned.to(self.device, non_blocking=True)
            expanded_ue8m0 = None
            if self.raw_ue8m0_bytes:
                # Native UE8M0 conversion preserves every finite value, including
                # exponent byte zero (2^-127), in one cast for the entire prefix.
                expanded_ue8m0 = (
                    self.raw_device[: self.raw_ue8m0_bytes]
                    .view(torch.float8_e8m0fnu)
                    .to(torch.float32)
                )
            for binding, position, size in raw_targets:
                payload = self.raw_device[position : position + size]
                expanded = (
                    expanded_ue8m0[position : position + size]
                    if binding.raw_ue8m0
                    else None
                )
                for target, source in binding.raw_sources(payload, expanded):
                    targets, sources = self.raw_copies.setdefault(
                        target.dtype, ([], [])
                    )
                    targets.append(target)
                    sources.append(source)
            self.derived.extend(
                prepare_updates(backend.layout.derived_updates, self.raw_copies)
            )
            self.error = torch.zeros(1, dtype=torch.int32, device=self.device)
            from sglang.srt.weight_sync.gpu_delta.apply import _compiled

            compile_started = time.perf_counter()
            cache_before = _compiled.cache_info()
            metadata_reused = False
            for _, _, group in self.static_plans:
                if group is not None:
                    metadata_reused = (
                        group.prepare_metadata(self.device.index) or metadata_reused
                    )
            cache_after = _compiled.cache_info()
            self.timings.update(
                host_apply_kernel_prepare_s=time.perf_counter() - compile_started,
                apply_kernel_cache_misses=cache_after.misses - cache_before.misses,
                apply_kernel_cache_reused=int(
                    (metadata_reused or cache_after.hits > cache_before.hits)
                    and cache_after.misses == cache_before.misses
                ),
            )
            self.apply_host_metadata = torch.empty(
                sum(
                    2 * len(group.sources) + group.metadata_size
                    for _, _, group in self.static_plans
                    if group is not None
                ),
                dtype=torch.int64,
                pin_memory=True,
            )
            self.apply_metadata = torch.empty(
                self.apply_host_metadata.numel(), dtype=torch.int64, device=self.device
            )
            self._prepare_apply_bindings()
        self.workspace = self.decode_plan = None
        if self.static_plans:
            self.decode_plan = self.decoder.prepare_batches(
                frame_table,
                frame_counts,
                backend.host_arena.tensor,
                self.io_stream if self.hardware else self.stream,
            )
            self.decode_plan.record_stream(self.stream)
            self.workspace = self.decode_plan.workspace
            from sglang.srt.weight_sync.gpu_delta.apply import prepare_status_check

            with torch.cuda.stream(self.stream):
                self.status_checks = [
                    prepare_status_check(decode, self.error)
                    for decode in self.decode_plan.batches
                ]
        # PREPARED includes small input transfers, never output-slot allocation,
        # decoding or synchronization with rollout. The fixed apply kernel
        # is already compiled/loaded; membership never specializes its code.
        ready = []
        for stream in (self.stream, self.io_stream):
            event = torch.cuda.Event()
            event.record(stream)
            ready.append(event)
        waiting = time.perf_counter()
        for event in ready:
            event.synchronize()
        self.timings.update(
            host_metadata_prepare_s=time.perf_counter() - started,
            host_metadata_wait_s=time.perf_counter() - waiting,
            decoder_metadata_h2d_bytes=frame_table.nbytes,
            decoder_workspace_bytes=(
                self.workspace.temporary.numel() if self.workspace else 0
            ),
        )

    @profiling.annotate("gpu_delta.prepare.apply_bindings")
    def _prepare_apply_bindings(self):
        """Prepare fixed descriptors, target pointers and source-index patches."""
        self.apply_host_view = self.apply_host_metadata.numpy()
        indices = [[], []]
        offsets = [[], []]
        position = 0
        for index, (_, _, group) in enumerate(self.static_plans):
            if group is None:
                self.apply_bindings.append(None)
                continue
            count = 2 * len(group.sources)
            start, stop = position + count, position + count + group.metadata_size
            self.apply_host_view[position:start] = list(group.pointer_rows(0))
            self.apply_host_view[start:stop] = group.metadata_rows
            pointers = self.apply_metadata[position:start]
            metadata = self.apply_metadata[start:stop]
            self.apply_bindings.append(
                (
                    start,
                    stop,
                    pointers,
                    metadata,
                    group.bind_launch(pointers, metadata, self.error),
                )
            )
            slot = index % 2
            indices[slot].extend(position + i for i in group.source_indices)
            offsets[slot].extend(group.sources)
            position = stop
        if position != self.apply_host_metadata.numel():
            raise RuntimeError("GPU delta apply metadata slab capacity mismatch")
        self.apply_source_patches = [
            (np.asarray(i, dtype=np.int64), np.asarray(o, dtype=np.int64))
            for i, o in zip(indices, offsets)
        ]

    @profiling.annotate("gpu_delta.pause.bind_apply_metadata")
    def _bind_apply_metadata(self):
        """Patch only actual source bases; misalignment retains the generic path."""
        groups, calls = [], []
        for index, (_, _, group) in enumerate(self.static_plans):
            if group is None:
                calls.append(None)
                continue
            start, stop, pointers, metadata, prepared_call = self.apply_bindings[index]
            aligned = group.bind(self.decoded[index % 2])
            if not aligned or index in self.apply_fallback_groups:
                self.apply_host_view[start:stop] = group.metadata_rows
            if aligned:
                self.apply_fallback_groups.discard(index)
                call = prepared_call
            else:
                self.apply_fallback_groups.add(index)
                call = group.bind_launch(pointers, metadata, self.error)
            groups.append(group)
            calls.append(call)
        for slot, (indices, offsets) in enumerate(self.apply_source_patches):
            if indices.size:
                # Each group's constant-time bound check above proves no int64
                # overflow before this vectorized patch. Targets are untouched.
                self.apply_host_view[indices] = offsets + self.decoded[slot].data_ptr()
        return groups, calls

    @profiling.annotate("gpu_delta.pause.allocate_bind")
    def _allocate_paused(self):
        """Allocate reusable decoded outputs, then bind their pointers."""
        from sglang.srt.weight_sync.gpu_delta.memory import device_empty

        started = time.perf_counter()
        allocate = device_empty if self.hardware else torch.empty
        with torch.cuda.stream(self.stream):
            self.decoded = (
                [
                    allocate(self.max_decoded, dtype=torch.uint8, device=self.device)
                    for _ in range(2)
                ]
                if self.static_plans
                else []
            )
            decoders = (
                self.decode_plan.bind_outputs(self.decoded) if self.decode_plan else []
            )
            apply_groups, apply_calls = self._bind_apply_metadata()
            self.apply_metadata.copy_(self.apply_host_metadata, non_blocking=True)
            for index, decode in enumerate(decoders):
                scratch = self.decoded[index % 2]
                apply = apply_calls[index]
                self.batches.append(
                    _PreparedBatch(
                        decode,
                        apply,
                        [
                            scratch[offset : offset + size]
                            for offset, size in self.gaps[index]
                        ],
                        self.status_checks[index],
                    )
                )
        self.timings.update(
            paused_setup_host_s=time.perf_counter() - started,
            decoder_metadata_uploads=(2 if self.hardware else 3) * int(bool(decoders)),
            compressed_input_buffers=0 if self.hardware or not decoders else 2,
            compressed_input_capacity_bytes=(
                self.decode_plan.input_capacity_bytes if self.decode_plan else 0
            ),
            compressed_input_h2d_bytes=(
                self.decode_plan.input_bytes if self.decode_plan else 0
            ),
            apply_metadata_h2d_bytes=self.apply_metadata.numel() * 8,
            apply_descriptor_h2d_bytes=sum(
                group.metadata_size * 8 for group in apply_groups
            ),
            apply_groups=len(apply_groups),
            apply_grid_ctas=sum(group.grid_ctas for group in apply_groups),
            apply_contracts=sum(len(group.contracts) for group in apply_groups),
            apply_word32_contracts=sum(
                group.word32_contracts for group in apply_groups
            ),
            transformed_tensors=0,
            decoded_buffers=len(self.decoded),
            decoded_scratch_bytes=len(self.decoded) * self.max_decoded,
        )
        self.h2d_bytes = sum(
            self.timings[key]
            for key in (
                "raw_h2d_bytes",
                "decoder_metadata_h2d_bytes",
                "apply_metadata_h2d_bytes",
                "compressed_input_h2d_bytes",
            )
        )

    @contextmanager
    def _phase(self, name):
        if profiling.ENABLED:
            profiling.nvtx.range_push("gpu_delta.pause." + name)
        try:
            yield
        finally:
            if profiling.ENABLED:
                profiling.nvtx.range_pop()

    @profiling.annotate("gpu_delta.pause.apply")
    def apply(self):
        apply_started = time.perf_counter()
        self.backend.layout.check_identity()
        self.timings["host_apply_identity_s"] = time.perf_counter() - apply_started
        with torch.cuda.device(self.device), torch.no_grad():
            self.stream.wait_stream(torch.cuda.default_stream(self.device))
            with torch.cuda.stream(self.stream):
                # Queue raw replacements before allocating and binding matrix buffers.
                with self._phase("raw_apply"):
                    for targets, sources in self.raw_copies.values():
                        torch._foreach_copy_(targets, sources)
                self._allocate_paused()
                with self._phase("paused_gpu_pipeline"):
                    if self.batches:
                        self.io_stream.wait_stream(self.stream)
                        if self.hardware:
                            self._decode_batch(self.batches[0], 0)
                        else:
                            self._copy_batch(self.batches[0], 0)
                    matrices_started = time.perf_counter()
                    if self.batches:
                        if self.hardware:
                            self._apply_hardware_batches()
                        else:
                            self._apply_cuda_batches()
                    self.timings["host_matrix_enqueue_s"] = (
                        time.perf_counter() - matrices_started
                    )
                    with self._phase("derived_refresh"):
                        if self.derived:
                            decode_succeeded = self.error.view(()) == 0
                        for derived in self.derived:
                            torch.where(
                                decode_succeeded,
                                derived.source,
                                derived.destination,
                                out=derived.destination,
                            )
                done = torch.cuda.Event()
                done.record(self.stream)
        completion_started = time.perf_counter()
        done.synchronize()
        self.timings["host_apply_completion_wait_s"] = (
            time.perf_counter() - completion_started
        )
        if self.error.item() != 0:
            raise RuntimeError(
                "direct GPU delta decompression failed; session is poisoned"
            )
        # Release output/workspace leases before generation resumes. DE allocation
        # free/fence overhead remains included in paused host wall time.
        self._release_gpu()
        self._gpu_complete = True
        self.timings["paused_apply_host_wall_s"] = time.perf_counter() - apply_started
        return {
            "applied": True,
            "verification": "manifest-sha256-and-decoder-status",
            "target_version": self.target_version,
            "tensors": self.matrix_tensor_count + self.raw_tensor_count,
            "timings": self.timings,
            "h2d_bytes": self.h2d_bytes,
        }

    def _apply_hardware_batches(self):
        # DE writes one decoded slot while the SMs consume the other.
        for index, batch in enumerate(self.batches):
            slot = index % 2
            self.stream.wait_event(self.decoded_ready[slot])
            self._apply_batch(batch)
            self.decoded_free[slot].record(self.stream)
            # Queue apply before nvCOMP, which may wait on its calling stream.
            if index + 1 < len(self.batches):
                self._decode_batch(self.batches[index + 1], index + 1)

    def _copy_batch(self, batch, index):
        with torch.cuda.stream(self.io_stream), self._phase("h2d"):
            slot = index % 2
            if index >= 2:
                self.io_stream.wait_event(self.decoded_free[slot])
            batch.decoder.copy_input(self.io_stream)
            self.decoded_ready[slot].record(self.io_stream)

    def _apply_cuda_batches(self):
        # CUDA decode and apply share one compute stream. Only compressed H2D
        # overlaps compute; a slot becomes reusable as soon as decoding ends.
        for index, batch in enumerate(self.batches):
            slot = index % 2
            self.stream.wait_event(self.decoded_ready[slot])
            if index + 1 < len(self.batches):
                self._copy_batch(self.batches[index + 1], index + 1)
            with self._phase("decode"):
                if batch.zero_ranges:
                    torch._foreach_zero_(batch.zero_ranges)
                batch.decoder.enqueue()
            self.decoded_free[slot].record(self.stream)
            self._apply_batch(batch)

    def _decode_batch(self, batch, index):
        stream = batch.decoder.stream
        with torch.cuda.stream(stream):
            slot = index % 2
            if index >= 2:
                stream.wait_event(self.decoded_free[slot])
            with self._phase("decode"):
                if batch.zero_ranges:
                    torch._foreach_zero_(batch.zero_ranges)
                batch.decoder.enqueue()
            self.decoded_ready[slot].record(stream)

    def _apply_batch(self, batch):
        # Sticky status and every consumer of it are ordered on the apply
        # stream. DE never races that gate while decoding the next batch.
        with self._phase("layout_apply"):
            batch.check_status()
            if batch.apply is not None:
                batch.apply()

    @profiling.annotate("gpu_delta.pause.release_gpu")
    def _release_gpu(self):
        self.batches.clear()
        self.status_checks.clear()
        self.raw_copies.clear()
        self.derived = []
        self.apply_bindings.clear()
        self.apply_source_patches.clear()
        self.apply_fallback_groups.clear()
        self.apply_host_view = None
        self.apply_metadata = self.apply_host_metadata = None
        self.decoded = self.raw_device = self.error = self.workspace = (
            self.decode_plan
        ) = None

    @profiling.annotate("gpu_delta.release_host")
    def close(self, reuse_host=False):
        # Error/cancel paths must attempt every stream before releasing any
        # storage. Resume can reuse the successful apply fence and scratch
        # release; failure cleanup always drains, even after a completed apply.
        if not reuse_host or not self._gpu_complete:
            failure = None
            for stream in (self.io_stream, self.stream):
                if stream is not None:
                    try:
                        stream.synchronize()
                    except BaseException as error:
                        if failure is None:
                            failure = error
            if failure is not None:
                raise failure
            self._release_gpu()
        if self.host_snapshot is not None:
            self.host_snapshot.close(reuse_host)
            self.host_snapshot = None
        self.raw_pinned = None
