"""Serve native Engram / PLE table rows from Mooncake MNNVL EGM memory.

Native weight loading builds the compute shards first. This adapter republishes
each shard as an immutable byte table in EGM, then releases the native host
backing. Models keep hashing, masking, quantization interpretation, scales and
collectives; only the row bytes move.
"""

from __future__ import annotations

import atexit
import json
import logging
import math
import operator
import time
import uuid

logger = logging.getLogger(__name__)

_LIMIT = (1 << 63) - 1
_ALLOWED_OPTIONS = {
    "namespace",
    "master_addr",
    "session_id",
    "shard_id",
    "num_shards",
    "num_callers",
    "caller_id",
    "max_lookup_rows",
    "max_egm_bytes_per_numa",
    "local_dir",
    "host_id",
    "rpc_hostname",
    "init_timeout_s",
    "close_timeout_s",
}
_REQUIRED_OPTIONS = {
    "namespace",
    "master_addr",
    "session_id",
    "shard_id",
    "num_shards",
    "num_callers",
    "max_lookup_rows",
    "max_egm_bytes_per_numa",
}

_states = {}


def _integer(value, name, minimum=0):
    if isinstance(value, bool):
        raise ValueError(f"{name} does not accept bool")
    value = operator.index(value)
    if value < minimum or value > _LIMIT:
        raise ValueError(f"{name} is out of range")
    return value


def _timeout(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("timeout must be a finite positive number")
    return value


def parse_config(raw):
    value = json.loads(raw) if isinstance(raw, str) else dict(raw)
    if not isinstance(value, dict):
        raise ValueError("--engram-mnnvl-config must be a JSON object")
    unknown = value.keys() - _ALLOWED_OPTIONS
    if unknown:
        raise ValueError(f"Unknown --engram-mnnvl-config options: {sorted(unknown)}")
    missing = _REQUIRED_OPTIONS - value.keys()
    if missing:
        raise ValueError(f"--engram-mnnvl-config is missing: {sorted(missing)}")
    for key in (
        "num_shards",
        "num_callers",
        "max_lookup_rows",
        "max_egm_bytes_per_numa",
    ):
        value[key] = _integer(value[key], key, 1)
    value["shard_id"] = _integer(value["shard_id"], "shard_id")
    if value["shard_id"] >= value["num_shards"]:
        raise ValueError("shard_id is out of range")
    for key in ("namespace", "session_id"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    for key in ("init_timeout_s", "close_timeout_s"):
        if key in value:
            value[key] = _timeout(value[key])
    return value


def _metadata(endpoint):
    from mooncake.engram_session import Metadata

    return Metadata(endpoint)


def _digest(value):
    from mooncake.engram_session import digest

    return digest(value)


def _store(**kwargs):
    from mooncake.engram import EngramStore

    return EngramStore(**kwargs)


def _reject_reload(*args, **kwargs):
    raise RuntimeError(
        "Engram tables cannot be hot-updated; restart the deployment with a "
        "new session"
    )


class Adapter:
    def __init__(self, path, module, kind, start, count, planes):
        import torch

        self.path, self.module, self.kind = path, module, kind
        self.start, self.count = int(start), int(count)
        self.planes = {}
        if self.start < 0 or self.count < 0:
            raise ValueError("Invalid native Engram compute shard range")
        for name in planes:
            parameter = getattr(module, name)
            if parameter.is_meta or parameter.ndim != 2 or parameter.shape[0] < count:
                raise ValueError(f"{path}.{name} is not a loaded 2-D table")
            table = f"{path}/{name}/{start}:{start + count}"
            self.planes[name] = {
                "name": table,
                "width": parameter.shape[1] * parameter.element_size(),
                "dtype": parameter.dtype,
                "parameter": parameter,
            }
        self.op_id = -1
        self.state = None
        self.store = None

    def manifest(self):
        return {
            plane["name"]: {
                "num_rows": self.count,
                "row_bytes": plane["width"],
                "dtype": str(plane["dtype"]),
                "kind": self.kind,
                "group": f"{self.path}/{self.start}:{self.start + self.count}",
            }
            for plane in self.planes.values()
        }

    def byte_rows(self, plane):
        import torch

        # Reinterpret bytes rather than converting dtype values, so
        # FP8/E8M0/MXFP4/NVFP4 are all supported.
        return (
            self.planes[plane]["parameter"]
            .detach()[: self.count]
            .view(torch.uint8)
            .contiguous()
            .cpu()
            .numpy()
        )

    def fetch(self, plane, ids):
        import torch

        if self.state.closed:
            raise RuntimeError("Engram MNNVL backend is closed")
        entry = self.planes[plane]
        flat = ids.reshape(-1, 1)
        live = (flat >= self.start) & (flat < self.start + self.count)
        selected = torch.where(live, flat, -1)
        out = torch.zeros(
            (flat.shape[0], 1, entry["width"]), dtype=torch.uint8, device=ids.device
        )
        self.store.lookup(
            [entry["name"]],
            selected,
            out,
            offsets=[self.start],
            stream=torch.cuda.current_stream().cuda_stream,
        )
        return out.reshape(*ids.shape, entry["width"])

    def release_native(self):
        import torch

        module = self.module
        host = (
            getattr(module, "host_table", None)
            if self.kind == "sglang_deepseek"
            else None
        )
        # Unpin the native table while the old CPU buffer is still valid; on
        # failure, do not free its backing.
        if host is not None:
            status = torch.cuda.cudart().cudaHostUnregister(host.bytes.data_ptr())
            if int(status) != 0:
                raise RuntimeError(f"Native Engram host unregister failed: {status}")
        for entry in self.planes.values():
            parameter = entry["parameter"]
            parameter.data = torch.empty(
                (0, parameter.shape[1]), dtype=parameter.dtype, device=parameter.device
            )
            parameter.weight_loader = _reject_reload
        if self.kind == "sglang_deepseek":
            module.host_table = None
            module._sort_host_reads = False
            module.staged_by_scheduler = False
            if host is not None:
                host.bytes = None
                host.mm.close()


def discover(model):
    adapters = []
    for path, module in model.named_modules():
        name = type(module).__name__
        origin = type(module).__module__
        if name == "EngramEmbedding" and origin.startswith("sglang."):
            if module.peer is not None or module._shared:
                raise ValueError(
                    "MNNVL Engram requires replication=none and does not use "
                    "peer_read"
                )
            adapters.append(
                Adapter(
                    path,
                    module,
                    "sglang_deepseek",
                    module.row_start,
                    module.rows,
                    ("weight", "scale"),
                )
            )
        elif name == "Qwen4ExpPinnedHostEmbedding" and origin.startswith("sglang."):
            if (
                module._file_prefetcher is not None
                or module._file_rss_trimmer is not None
            ):
                raise ValueError(
                    "MNNVL Engram cannot be combined with PLE file offload"
                )
            shard = module.shard_indices
            adapters.append(
                Adapter(
                    path,
                    module,
                    "sglang_qwen",
                    shard.org_vocab_start_index,
                    shard.org_vocab_end_index - shard.org_vocab_start_index,
                    ("weight",),
                )
            )
    return adapters


def choose_owners(manifests):
    specs, descriptions, candidates = {}, {}, {}
    for index, manifest in enumerate(manifests):
        for name, description in manifest["tables"].items():
            if name in descriptions and descriptions[name] != description:
                raise ValueError("Engine compute shards have inconsistent shape/dtype")
            descriptions[name] = description
            specs[name] = {key: description[key] for key in ("num_rows", "row_bytes")}
            candidates.setdefault(name, []).append(index)
    if not specs:
        raise ValueError(
            "No attachable Engram/PLE tables found in the model; check the "
            "offload config"
        )
    chosen, loads, groups = {}, {}, {}
    for name, description in descriptions.items():
        groups.setdefault(description.get("group", name), []).append(name)

    def group_bytes(group):
        return sum(
            specs[name]["num_rows"] * specs[name]["row_bytes"] for name in groups[group]
        )

    for group in sorted(groups, key=lambda key: (-group_bytes(key), key)):
        holders = set.intersection(*(set(candidates[name]) for name in groups[group]))
        if not holders:
            raise ValueError(
                "The weight/scale of one compute shard have no common data holder"
            )
        holder = min(
            holders,
            key=lambda i: (
                loads.get(manifests[i]["shard"], 0),
                manifests[i]["shard"],
                manifests[i]["caller"],
            ),
        )
        for name in groups[group]:
            chosen[name] = holder
        node = manifests[holder]["shard"]
        loads[node] = loads.get(node, 0) + group_bytes(group)
    return specs, chosen


def exchange(metadata, prefix, local, config, deadline):
    key = f"{prefix}/{config['shard_id']}/{config['caller_id']}"
    metadata.create(key, local, deadline)
    records = {}
    while len(records) != config["num_shards"] * config["num_callers"]:
        for shard in range(config["num_shards"]):
            for caller in range(config["num_callers"]):
                identity = (shard, caller)
                if identity not in records:
                    value = metadata.get(f"{prefix}/{shard}/{caller}", deadline)
                    if value is not None:
                        if value["shard"] != shard or value["caller"] != caller:
                            raise ValueError("Engram caller identity mismatch")
                        records[identity] = value
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Timed out waiting for all engine workers to load Engram weights"
            )
        time.sleep(0.02)
    return [records[key] for key in sorted(records)]


class State:
    """Multiple Stores created by migrating one model layer at a time; shutdown
    order is the reverse of migration order.
    """

    def __init__(self, adapters):
        self.adapters = list(adapters)
        self.stages = []
        self.closed = False

    def stage(self, store, adapters):
        self.stages.append((store, list(adapters)))

    def close(self):
        if self.closed:
            return
        import torch

        torch.cuda.synchronize()
        self.closed = True
        error = None
        # Coordinated release requires all callers in the same order; on a
        # single-layer failure, still release the remaining local resources.
        for store, _ in reversed(self.stages):
            try:
                store.close()
            except Exception as exc:
                error = error or exc
        from . import mnnvl_ops

        mnnvl_ops.forget(self.adapters)
        self.adapters.clear()
        for key, value in list(_states.items()):
            if value is self:
                del _states[key]
        if error is not None:
            raise error


def _verify_rows(store, adapter, local):
    """Sample-read the locally written tables to confirm EGM matches the source
    bytes.

    The native table is released right after migration, so silent byte
    corruption only manifests as model output drift.
    """
    import numpy as np
    import torch

    for name, source in local.items():
        rows, width = source.shape
        if not rows:
            continue
        step = max(1, rows // 32)
        sample = np.unique(np.concatenate(([0, rows - 1], np.arange(0, rows, step))))
        # Store IDs are global row numbers; offsets subtract the table start.
        ids = torch.from_numpy((sample + adapter.start).astype(np.int64))
        ids = ids.reshape(-1, 1).cuda()
        out = torch.zeros((len(sample), 1, width), dtype=torch.uint8, device="cuda")
        store.lookup(
            [name],
            ids,
            out,
            offsets=[adapter.start],
            stream=torch.cuda.current_stream().cuda_stream,
        )
        torch.cuda.synchronize()
        actual = out.reshape(-1, width).cpu().numpy()
        if not np.array_equal(source[sample], actual):
            bad = int(np.count_nonzero(np.any(source[sample] != actual, axis=1)))
            raise RuntimeError(
                f"Engram migration verification failed: {bad}/{len(sample)} "
                f"sampled rows of {name} differ from the source table"
            )


def attach(model, config):
    import torch

    from . import mnnvl_ops

    if hasattr(model, "_engram_mnnvl_state") or hasattr(
        model, "_mooncake_engram_state"
    ):
        return
    adapters = discover(model)
    if not adapters:
        return  # Table-less draft models are distinguished by the main runner's check.
    if not torch.cuda.is_available() or torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "MNNVL Engram must initialize with CUDA available and before graph "
            "capture"
        )
    config = dict(config)
    caller = config.get("caller_id")
    if caller is None:
        if not torch.distributed.is_initialized():
            if config["num_callers"] != 1:
                raise ValueError(
                    "Cannot infer caller_id; provide it in --engram-mnnvl-config"
                )
            caller = 0
        else:
            caller = torch.distributed.get_rank() % config["num_callers"]
    config["caller_id"] = int(caller)
    if not 0 <= config["caller_id"] < config["num_callers"]:
        raise ValueError("caller_id is out of range")
    namespace = config["namespace"] + "/sglang"
    metadata = _metadata(config["master_addr"])
    manifest = {
        "tables": {},
        "shard": config["shard_id"],
        "caller": config["caller_id"],
        "nonce": uuid.uuid4().hex,
    }
    for adapter in adapters:
        manifest["tables"].update(adapter.manifest())
    prefix = "engram-mnnvl/v1/" + _digest([config["session_id"], namespace])
    manifests = exchange(
        metadata,
        prefix,
        manifest,
        config,
        time.monotonic() + config.get("init_timeout_s", 300),
    )
    specs, chosen = choose_owners(manifests)
    identity = config["shard_id"] * config["num_callers"] + config["caller_id"]
    options = {
        key: value
        for key, value in config.items()
        if key not in ("namespace", "local_dir")
    }
    local_dir = config.get("local_dir", "/tmp/mooncake-engram")
    # All caller shards of one layer belong to the same group; each layer's
    # Store must contain every caller's table specs, otherwise the layout digest
    # differs across ranks and coordinated initialization rejects each other.
    stage_of = {
        name: description["group"].rsplit("/", 1)[0]
        for manifest in manifests
        for name, description in manifest["tables"].items()
    }
    # Migrate one layer at a time: the EGM arena coexists only with the current
    # layer's native host table and the pin is released immediately after, cutting
    # the peak on a single NUMA node from the whole model to the largest layer.
    # All callers must migrate and shut down in the same order, determined by
    # table size and module path.
    order = sorted(
        adapters,
        key=lambda adapter: (
            -sum(
                entry["num_rows"] * entry["row_bytes"]
                for entry in adapter.manifest().values()
            ),
            adapter.path,
        ),
    )
    state = State(adapters)
    # Any later wiring failure keeps resources so GC cannot bypass the
    # coordinated shutdown protocol.
    _states[id(model)] = state
    mnnvl_ops.install_ops()
    for adapter in order:
        names = {name for name, stage in stage_of.items() if stage == adapter.path}
        store = _store(
            tables={name: specs[name] for name in names},
            namespace=f"{namespace}/{adapter.path or 'root'}",
            local_dir=local_dir,
            persistent=False,
        )
        local = {
            entry["name"]: adapter.byte_rows(plane)
            for plane, entry in adapter.planes.items()
            if chosen[entry["name"]] == identity
        }
        store.populate_shard_and_sync(
            local,
            backend="mnnvl",
            device=torch.cuda.current_device(),
            table_owners={
                name: manifests[owner]["shard"]
                for name, owner in chosen.items()
                if name in names
            },
            **options,
        )
        _verify_rows(store, adapter, local)
        adapter.state = state
        adapter.store = store
        state.stage(store, [adapter])
        mnnvl_ops.bind(adapter)
        # K is fixed at 1; build a Lookup per plane up front so later changes in
        # token count do not import mappings.
        ids = torch.full(
            (1, 1),
            adapter.start if adapter.count else -1,
            dtype=torch.int64,
            device="cuda",
        )
        for entry in adapter.planes.values():
            output = torch.empty(
                (1, 1, entry["width"]), dtype=torch.uint8, device="cuda"
            )
            store.lookup(
                [entry["name"]],
                ids,
                output,
                offsets=[adapter.start],
                stream=torch.cuda.current_stream().cuda_stream,
            )
        torch.cuda.synchronize()
        adapter.release_native()
    model._engram_mnnvl_state = state
    logger.info(
        "Engram MNNVL enabled: tables=%d, local_writer_tables=%d, session=%s",
        len(specs),
        sum(value == identity for value in chosen.values()),
        config["session_id"],
    )


def shutdown():
    """Call only after workers have stopped submitting and destroyed dependent
    graphs; also best-effort cleanup at process exit.
    """
    errors = []
    for state in list(_states.values()):
        try:
            state.close()
        except Exception as exc:
            errors.append(exc)
    _states.clear()
    if errors:
        raise errors[0]


def _at_exit():
    try:
        shutdown()
    except Exception:
        logger.exception(
            "Engram MNNVL coordinated shutdown did not complete; retaining "
            "resources until process exit"
        )


def maybe_attach(runner):
    raw = getattr(runner.server_args, "engram_mnnvl_config", None)
    if raw is None:
        return
    if runner.is_draft_worker:
        return
    if runner.server_args.pp_size != 1:
        raise ValueError("MNNVL Engram currently supports only pp_size == 1")
    if getattr(runner.server_args, "elastic_ep_backend", None) is not None:
        raise ValueError("MNNVL Engram does not support elastic EP membership changes")
    if runner.startup_weight_load is not None:
        raise ValueError(
            "MNNVL Engram does not support sentinel/asynchronous startup weight "
            "loading"
        )
    config = parse_config(raw)
    attach(runner.model, config)
    if not hasattr(runner.model, "_engram_mnnvl_state"):
        raise RuntimeError(
            "No attachable Engram/PLE table found; enable the native CPU offload "
            "(--engram-config placement=cpu,replication=none or "
            "--ple-offload-embedding --ple-offload-backend pinned)"
        )


def release(scheduler):
    """Release EGM cooperatively only after a graceful shutdown, once graphs no
    longer use it.
    """
    model = getattr(
        getattr(getattr(scheduler, "tp_worker", None), "model_runner", None),
        "model",
        None,
    )
    state = getattr(model, "_engram_mnnvl_state", None)
    if state is not None:
        try:
            state.close()
        except Exception:
            logger.exception(
                "Engram MNNVL coordinated shutdown did not complete; retaining "
                "resources until process exit"
            )


atexit.register(_at_exit)
