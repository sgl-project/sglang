# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Canonical receiver plans and direct-host DE/application scheduling."""

from __future__ import annotations

import hashlib
import math
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain

import numpy as np
import orjson
import torch

from sglang.srt.weight_sync.gpu_delta import profiling
from sglang.srt.weight_sync.gpu_delta.bindings import (
    _ITEMSIZES,
    BACKEND_LAYOUTS,
    ConsumerSnapshot,
    _digest,
    _tensor_identity,
    validate_ue8m0_payload,
)
from sglang.srt.weight_sync.gpu_delta.derived import (
    nvfp4_derived_update,
    plan_updates,
    prepare_updates,
)
from sglang.srt.weight_sync.gpu_delta.flashinfer import consumer_objects
from sglang.srt.weight_sync.gpu_delta.models import model_mapping


class GpuDeltaLayout:
    """Frozen canonical bindings with generic backend and consumer admission."""

    def __init__(self, model, inventory):
        from sglang.srt.weight_sync.gpu_delta.host import _natural_key

        if not inventory:
            raise ValueError("canonical checkpoint metadata is unavailable")
        # Normalize negotiated wire metadata without changing checkpoint inventory.
        self.inventory = {name: dict(meta) for name, meta in inventory.items()}
        mapping = model_mapping(model)
        self.bindings = []
        self.excluded = mapping.parameters.excluded
        for name, meta in self.inventory.items():
            binding = mapping.bind(name, meta)
            if binding is not None:
                self.bindings.append(binding)
        self.bindings.sort(key=lambda b: _natural_key(b.name))
        if mapping.parameters.moe_layers:
            from sglang.srt.runtime_context import get_exec

            _require_fixed_moe_topology(get_exec().moe)
        self.derived = []
        self.derived_updates = []
        self._consumers = []
        for prefix, layer in mapping.parameters.moe_layers.items():
            layout = mapping.parameters.moe_layouts[prefix]
            if layout.format == "nvfp4":
                self.derived_updates.append(nvfp4_derived_update(layer, layout.backend))
            self._consumers.append(
                ConsumerSnapshot(
                    lambda layer=layer, layout=layout: consumer_objects(layer, layout)
                )
            )
        self.derived_updates = plan_updates(
            self.derived_updates,
            [
                target
                for binding in self.bindings
                if binding.encoding == "raw_bytes"
                for target in binding.storage
            ],
        )
        mapping.finish()
        self.derived.extend(mapping.derived)
        self._consumers.extend(mapping.consumers)
        self._parameter_roots = [
            (module, name, tensor, _tensor_identity(tensor))
            for module in mapping.parameters.modules.values()
            for name, tensor in module._parameters.items()
            if tensor is not None
        ]
        self._prepare_identity_checks()
        self._reject_overlaps(mapping.aliases)
        self.rank_plan_digest = _digest(
            {
                "layouts": BACKEND_LAYOUTS,
                "tensors": [b.describe() for b in self.bindings],
                "excluded": self.excluded,
            }
        )

    def _prepare_identity_checks(self):
        # Deduplicate geometry once, not in the per-update validation loop.
        # Separate views sharing storage still require separate geometry checks.
        checks = chain(
            (
                (tensor, expected, "parameter identity")
                for _, _, tensor, expected in self._parameter_roots
            ),
            (
                (consumer.objects[index], expected, "consumer storage")
                for consumer in self._consumers
                for index, expected in consumer.tensors
            ),
        )
        unique = {}
        for tensor, expected, kind in checks:
            key = id(tensor)
            prior = unique.get(key)
            if prior is None:
                unique[key] = (tensor, expected, kind)
            elif prior[1] != expected:
                raise RuntimeError(
                    f"GPU delta {kind} changed during admission; readmission required"
                )
        self._tensor_checks = tuple(unique.values())

    @profiling.annotate("gpu_delta.check_identity")
    def check_identity(self):
        for module, name, tensor, _ in self._parameter_roots:
            if module._parameters.get(name) is not tensor:
                raise RuntimeError(
                    "GPU delta parameter identity changed; readmission required"
                )
        for consumer in self._consumers:
            consumer.check_objects()
        for tensor, expected, kind in self._tensor_checks:
            if _tensor_identity(tensor) != expected:
                raise RuntimeError(f"GPU delta {kind} changed; readmission required")

    def _reject_overlaps(self, model_aliases):
        # Gate/up intentionally share one interleaved weight/scale allocation;
        # their maps are disjoint. Other shared storage needs an explicit alias
        # adapter so a tied parameter is never XORed twice.
        owners = {}
        for binding in self.bindings:
            for tensor in binding.storage:
                key = tensor.untyped_storage().data_ptr()
                prior = owners.get(key)
                if prior and prior != binding.name:
                    pair = (prior, binding.name)
                    if not all(".experts." in name for name in pair) and not (
                        _gate_up_alias(*pair) or model_aliases(*pair)
                    ):
                        raise ValueError(f"unclassified live weight alias: {pair}")
                owners[key] = binding.name


def _gate_up_alias(a, b):
    return (
        a.replace("gate_proj", "up_proj") == b or b.replace("gate_proj", "up_proj") == a
    )


def _require_fixed_moe_topology(moe):
    # The canonical expert map is valid only while physical expert ownership is
    # the trivial EP partition. EPLB/custom/elastic maps can change ownership
    # without changing tensor pointers, so pointer checks alone cannot admit it.
    if (
        moe.enable_eplb
        or moe.elastic_ep_backend is not None
        or moe.init_expert_location != "trivial"
        or moe.ep_num_redundant_experts != 0
        or (moe.moe_runner_backend, moe.moe_a2a_backend)
        not in {
            ("flashinfer_trtllm", "none"),
            ("flashinfer_trtllm_routed", "none"),
            ("flashinfer_mxfp4", "none"),
            ("flashinfer_cutedsl", "none"),
            ("flashinfer_megamoe", "flashinfer_megamoe"),
        }
    ):
        raise ValueError(
            "direct GPU deltas require fixed trivial EP, no redundant experts or EPLB, "
            "and a native FlashInfer runner with its matching dispatcher"
        )


class GpuDeltaBackend:
    """Scheduler-owned plan; large decoded masks exist only during apply."""

    def __init__(self, model_runner, identity):
        from sglang.srt.weight_sync.gpu_delta.checkpoint import (
            read_canonical_checkpoint_inventory,
        )
        from sglang.srt.weight_sync.gpu_delta.payload import (
            HostPayloadPool,
            configured_cpu_workers,
        )

        self._canonical_plan_digest = None
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

        self.payload_pool = HostPayloadPool(configured_cpu_workers())
        self.host_arena = HostArena(identity["engine_id"], self.device.index)
        self.decoders = {}
        self.apply_stream = None
        self.de_stream = None

    def describe(self):
        self.layout.check_identity()
        return {
            "layouts": dict(BACKEND_LAYOUTS),
            "rank_plan_digest": self.layout.rank_plan_digest,
            "tensors": [binding.describe() for binding in self.layout.bindings],
            "excluded": self.layout.excluded,
        }

    def prepare(self, manifest_path, manifest_sha256, metadata):
        # Called by the background worker. Small scalar calibration may be read
        # to prepare derived values; no live writes, collectives, loader hooks
        # or generation-stream synchronization occur here.
        with torch.cuda.device(self.device):
            prepared = PreparedDelta.__new__(PreparedDelta)
            try:
                prepared.__init__(self, manifest_path, manifest_sha256, metadata)
                return prepared
            except BaseException:
                prepared.close()
                raise

    def close(self):
        self.payload_pool.close()
        self.host_arena.close()


@dataclass
class _PreparedBatch:
    decoder: object
    apply: Callable[[], None] | None
    zero_ranges: list[torch.Tensor]
    check_status: Callable[[], None]


@profiling.annotate("gpu_delta.prepare.layer_plan")
def _plan_layers(backend, bindings, entries, layers_per_batch=1):
    """Cache membership, canonical offsets and apply geometry, never wire frames."""
    key = layers_per_batch, tuple(binding.name for binding in bindings)
    if backend.batch_plan is None or backend.batch_plan[0] != key:
        if not bindings:
            backend.batch_plan = (key, [])
            return []
        from sglang.srt.weight_sync.gpu_delta.apply import plan_apply

        layers = {}
        non_layers = {"embed_tokens": [], "lm_head": [], "standalone": []}
        for binding in bindings:
            if binding.layer is None:
                module = binding.name.rpartition(".")[0].rpartition(".")[2]
                non_layers.get(module, non_layers["standalone"]).append(binding)
            else:
                layers.setdefault(binding.layer, []).append(binding)
        groups = [group for group in non_layers.values() if group]
        model_layers = list(layers.values())
        groups.extend(
            [
                binding
                for layer in model_layers[start : start + layers_per_batch]
                for binding in layer
            ]
            for start in range(0, len(model_layers), layers_per_batch)
        )
        plans = []
        for group in groups:
            outputs, size = [], 0
            for binding in group:
                offset = (size + 15) // 16 * 16
                nbytes = entries[binding.name]["nbytes"]
                outputs.append((binding, offset, nbytes))
                size = offset + nbytes
            # Segment views share one decoded canonical payload. Only the apply
            # plan expands them; DE membership and output offsets stay unique.
            apply_outputs = [
                (view, offset, nbytes)
                for binding, offset, nbytes in outputs
                for view in binding.apply_bindings()
            ]
            plans.append((outputs, size, plan_apply(apply_outputs)))
        backend.batch_plan = (key, plans)
    return backend.batch_plan[1]


@profiling.annotate("gpu_delta.prepare.frame_plan")
def _plan_decode(plans, entries, records):
    """Pack fresh frame rows and omitted ranges for every cached layer batch."""
    counts = [
        sum(len(entries[binding.name]["frames"]) for binding, _, _ in outputs)
        for outputs, _, _ in plans
    ]
    gaps = [[] for _ in plans]

    def rows():
        for batch, (outputs, _, _) in enumerate(plans):
            for binding, output_offset, size in outputs:
                input_offset = records[binding.name]["offset"]
                cursor = 0
                for frame in entries[binding.name]["frames"]:
                    start, decoded = frame["decoded_offset"], frame["decoded_bytes"]
                    if cursor < start:
                        gaps[batch].append((output_offset + cursor, start - cursor))
                    yield (
                        input_offset + frame["encoded_offset"],
                        frame["encoded_bytes"],
                        decoded,
                        output_offset + start,
                    )
                    cursor = start + decoded
                if cursor < size:
                    gaps[batch].append((output_offset + cursor, size - cursor))

    # Exhaust the iterator: its final yield can precede trailing gaps and fully
    # omitted tensors. Contiguous rows match nvCOMP's pointer/size arrays.
    table = np.fromiter(chain.from_iterable(rows()), dtype=np.int64)
    return table.reshape(sum(counts), 4).T.copy(order="C"), counts, gaps


@profiling.annotate("gpu_delta.prepare.canonical_plan")
def _qualify_canonical_plan(backend, manifest):
    """Admit static bindings once; Miles preserves them for the negotiated digest."""
    entries = {entry["name"]: entry for entry in manifest["tensors"]}
    if backend._canonical_plan_digest is not None:
        if manifest["plan_digest"] != backend._canonical_plan_digest:
            raise ValueError("negotiated canonical delta plan changed")
        return entries, True
    definitions = []
    for name, entry in entries.items():
        if name not in backend.layout.inventory or backend.layout.excluded.get(
            name
        ) not in {None, "expert owned by another EP rank"}:
            raise ValueError(f"delta publication has an unadmitted tensor: {name}")
        canonical = backend.layout.inventory[name]
        if (
            entry["shape"] != canonical["shape"]
            or entry["dtype"] != canonical["dtype"]
            or entry["byte_order"] != "little"
        ):
            raise ValueError(f"canonical tensor metadata mismatch: {name}")
        if entry["nbytes"] != math.prod(entry["shape"]) * _ITEMSIZES[
            entry["dtype"]
        ] or entry["encoding"] != (
            canonical.get(
                "encoding", "raw_bytes" if len(entry["shape"]) <= 1 else "xor_bytes"
            )
        ):
            raise ValueError(f"unsupported canonical tensor size/encoding: {name}")
        definition = {
            key: entry[key] for key in ("name", "dtype", "shape", "encoding", "views")
        }
        # Wire format is bound by the negotiated plan digest. Publication frames
        # intentionally carry only byte payload geometry, not another format copy.
        for key in ("format", "checkpoint_dtype"):
            if key in canonical:
                definition[key] = canonical[key]
        definitions.append(definition)
    for binding in backend.layout.bindings:
        views = entries[binding.name]["views"]
        matching = [view for view in views if view["id"] == binding.view_id]
        if len(matching) != 1 or matching[0]["slices"] != binding.slices:
            raise ValueError(f"missing or conflicting rank view for {binding.name}")
    if (
        _digest(sorted(definitions, key=lambda entry: entry["name"]))
        != manifest["plan_digest"]
    ):
        raise ValueError(
            "publication does not match its negotiated canonical view plan"
        )
    # Do not retain a manifest or repeat static schema checks on fitting updates.
    backend._canonical_plan_digest = manifest["plan_digest"]
    return entries, False


class PreparedDelta:
    @profiling.annotate("gpu_delta.prepare.layout")
    def __init__(self, backend, manifest_path, manifest_sha256, metadata):
        from pathlib import Path

        from sglang.srt.weight_sync.gpu_delta.payload import validate_codec

        self.stream = None
        self.de_stream = None
        self.host_snapshot = None
        self.batches, self.raw_copies = [], {}
        self.timings = {}
        self.status_checks = []
        self.apply_bindings, self.apply_source_patches = [], []
        self.apply_fallback_groups = set()
        self.apply_host_view = None
        self.backend, self.device = backend, backend.device
        preparation_started = time.perf_counter()
        layers_per_batch = backend.layers_per_batch
        manifest_started = time.perf_counter()
        path = Path(manifest_path).resolve(strict=True)
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest_sha256:
            raise ValueError("immutable delta manifest SHA256 mismatch")
        manifest = orjson.loads(content)
        self.timings["host_manifest_read_parse_s"] = (
            time.perf_counter() - manifest_started
        )
        plan_started = time.perf_counter()
        validate_codec(manifest)
        self.codec = manifest["codec"]
        self.target_version = manifest["target_version"]
        entries, reused_plan = _qualify_canonical_plan(backend, manifest)
        self.timings["host_plan_validate_s"] = time.perf_counter() - plan_started
        self.timings["host_plan_cache_reused"] = int(reused_plan)
        payload_started = time.perf_counter()
        index, files = backend.host_arena.prepare_encoded(
            path,
            manifest_sha256,
            manifest,
            backend.payload_pool,
            self.timings,
            metadata,
        )
        release_started = time.perf_counter()
        # READY admission drained file reads/hashes; keep only this rank's entries.
        entries = {
            binding.name: entries[binding.name] for binding in backend.layout.bindings
        }
        del manifest, content
        self.timings["host_rank_metadata_release_s"] = (
            time.perf_counter() - release_started
        )
        self.host_snapshot = backend.host_arena.prepare_local(
            index,
            files,
            list(entries.values()),
            backend.payload_pool,
            self.timings,
        )
        self.timings["host_rank_prepare_body_s"] = time.perf_counter() - payload_started
        del files
        self.timings["host_rank_prepare_s"] = time.perf_counter() - payload_started

        tensors_started = time.perf_counter()
        compressed, raw_entries, ue8m0_entries = [], [], []
        for binding in backend.layout.bindings:
            entry = entries[binding.name]
            if binding.encoding == "raw_bytes":
                if entry["changed_bytes"]:
                    group = ue8m0_entries if binding.raw_ue8m0 else raw_entries
                    group.append((binding, entry))
            elif entry["frames"]:
                compressed.append(binding)
        # Compact scales form one prefix for validation and native GPU expansion.
        raw_entries = ue8m0_entries + raw_entries
        self.raw_ue8m0_bytes = sum(entry["nbytes"] for _, entry in ue8m0_entries)
        previous_plan = backend.batch_plan
        self.static_plans = _plan_layers(backend, compressed, entries, layers_per_batch)
        self.timings["host_batch_plan_reused"] = int(
            previous_plan is not None and backend.batch_plan is previous_plan
        )
        frame_table, frame_counts, self.gaps = _plan_decode(
            self.static_plans, entries, self.host_snapshot.index["tensors"]
        )
        self.max_decoded = max((plan[1] for plan in self.static_plans), default=0)
        self.matrix_tensor_count = len(compressed)
        self.raw_tensor_count = len(raw_entries)
        self.timings["host_tensor_prepare_s"] = time.perf_counter() - tensors_started

        # Pack the small complete-target bypass once on the host. Decoded-mask
        # storage waits for the actual serving pause; kernel warmup is below.
        raw_started = time.perf_counter()
        raw_targets, raw_h2d_bytes = [], 0
        for binding, entry in raw_entries:
            position = (
                raw_h2d_bytes if binding.raw_ue8m0 else (raw_h2d_bytes + 7) // 8 * 8
            )
            size = entry["nbytes"]
            raw_targets.append((binding, position, size))
            raw_h2d_bytes = position + size
        self.raw_pinned = torch.empty(
            raw_h2d_bytes, dtype=torch.uint8, device="cpu", pin_memory=True
        )
        raw_view = memoryview(self.raw_pinned.numpy())
        for binding, position, size in raw_targets:
            source = memoryview(self.host_snapshot.get(binding.name).numpy())
            raw_view[position : position + size] = source
        if self.raw_ue8m0_bytes:
            validate_ue8m0_payload(self.raw_pinned[: self.raw_ue8m0_bytes])
        changed_storages = {
            pointer
            for binding in compressed + [binding for binding, _ in raw_entries]
            for pointer in binding.storage_pointers
        }
        self.derived = [
            image
            for image in backend.layout.derived
            if image.source_pointer in changed_storages
        ]
        self.timings.update(
            host_raw_pack_s=time.perf_counter() - raw_started,
            raw_tensors=len(raw_entries),
            raw_bytes=sum(entry["nbytes"] for _, entry in raw_entries),
            raw_h2d_bytes=raw_h2d_bytes,
            compressed_batches=len(self.static_plans),
            decode_stages=2,
            layers_per_batch=layers_per_batch,
            compressed_tensors=self.matrix_tensor_count,
            de_host_input_bytes=int(frame_table[1].sum()),
            decoded_zero_ranges=sum(map(len, self.gaps)),
            decoded_zero_bytes=sum(size for gaps in self.gaps for _, size in gaps),
        )
        self._prepare_gpu_metadata(frame_table, frame_counts, raw_targets)
        self.timings["host_prepare_s"] = time.perf_counter() - preparation_started

    @profiling.annotate("gpu_delta.prepare.gpu_metadata")
    def _prepare_gpu_metadata(self, frame_table, frame_counts, raw_targets):
        """Prepare small GPU inputs; retain their leases, not wire-frame objects."""
        from sglang.srt.weight_sync.gpu_delta.codec import NvcompDecoder

        started = time.perf_counter()
        backend = self.backend
        if backend.apply_stream is None:
            apply_stream = torch.cuda.Stream(device=self.device)
            de_stream = torch.cuda.Stream(device=self.device)
            backend.apply_stream, backend.de_stream = apply_stream, de_stream
        self.stream, self.de_stream = backend.apply_stream, backend.de_stream
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
            inner_codec = self.codec.removesuffix("-zstd")
            if inner_codec not in backend.decoders:
                backend.decoders[inner_codec] = NvcompDecoder(self.device, inner_codec)
            decoder = backend.decoders[inner_codec]
            self.decode_plan = decoder.prepare_batches(
                frame_table,
                frame_counts,
                backend.host_arena.tensor,
                self.de_stream,
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
        # DE execution or synchronization with rollout. The fixed apply kernel
        # is already compiled/loaded; membership never specializes its code.
        ready = []
        for stream in (self.stream, self.de_stream):
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
        self.stream.wait_stream(torch.cuda.default_stream(self.device))
        with torch.cuda.stream(self.stream):
            self.decoded = (
                [
                    device_empty(
                        self.max_decoded, dtype=torch.uint8, device=self.device
                    )
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
            decoder_metadata_uploads=2 * int(bool(decoders)),
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
            self._allocate_paused()
            with torch.cuda.stream(self.stream):
                with self._phase("paused_gpu_pipeline"):
                    # DE and apply alternate between two decoded buffers.
                    # DE first waits for paused allocation and metadata uploads.
                    if self.batches:
                        self.de_stream.wait_stream(self.stream)
                        self._decode_batch(self.batches[0], 0)
                    # Raw-target copies can overlap the first DE operation.
                    with self._phase("raw_apply"):
                        for targets, sources in self.raw_copies.values():
                            torch._foreach_copy_(targets, sources)
                    matrices_started = time.perf_counter()
                    for index, batch in enumerate(self.batches):
                        slot = index % 2
                        self.stream.wait_event(self.decoded_ready[slot])
                        self._apply_batch(batch)
                        self.decoded_free[slot].record(self.stream)
                        # nvCOMP may wait for prior work on its calling stream.
                        # Queue apply BEFORE the next DE call to retain overlap.
                        next_index = index + 1
                        if next_index < len(self.batches):
                            self._decode_batch(self.batches[next_index], next_index)
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
        # The loop-local batch also holds output/workspace leases. Drop it before
        # releasing the dedicated DE allocations, so free/fence overhead is
        # included in paused host wall time and completes before generation resumes.
        batch = None
        self._release_gpu()
        self.timings["paused_apply_host_wall_s"] = time.perf_counter() - apply_started
        return {
            "applied": True,
            "verification": "manifest-sha256-and-decoder-status",
            "target_version": self.target_version,
            "tensors": self.matrix_tensor_count + self.raw_tensor_count,
            "timings": self.timings,
            "h2d_bytes": self.h2d_bytes,
        }

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
    def release_and_close(self):
        try:
            self.host_snapshot.mark_reusable()
        finally:
            self.close()

    def close(self):
        # Error/cancel paths must attempt every stream before releasing any
        # storage. Success already joined and released GPU leases in apply.
        failure = None
        for stream in (self.de_stream, self.stream):
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
            self.host_snapshot.close()
            self.host_snapshot = None
        self.raw_pinned = None
