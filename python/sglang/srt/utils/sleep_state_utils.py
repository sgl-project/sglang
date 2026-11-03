"""Runtime-state preservation and auditing for sglang sleep mode.

Why this exists
---------------
``release_memory_occupation`` unmaps the memory-pool regions that were active
during model load (tag ``weights``). After ``resume_memory_occupation``:

* parameters are re-filled by the subsequent weight update, and
* registered buffers are restored by ``_export/_import_static_state``,

but **plain tensor attributes** held directly on modules (or inside simple
containers / non-module helper objects on them) are covered by neither path.
After resume they point at freshly committed (zero-filled) pages, so any
non-zero constant they carried is silently lost: a captured CUDA/ACL graph
replays happily against the stable virtual address while reading garbage.

This is the failure class described in vLLM RFC #53343 ("virtual address
stability ≠ value preservation") and fixed in vllm-ascend #13352 /
vllm #53507 by registering such tensors as non-persistent buffers. Doing that
per-tensor requires knowing every offender; this module instead closes the
gap generically by snapshotting and restoring all of them around the pause.

What it provides
----------------
* :func:`export_runtime_state` - snapshot every persistent device tensor that
  is reachable as a plain attribute but is neither a parameter nor a buffer.
  Clones go to CPU so the export does not inflate device memory right before
  the pause.
* :func:`import_runtime_state` - write the saved values back in place.
* :func:`snapshot_sleep_audit` / :func:`diff_sleep_audit` - opt-in
  (``SGLANG_SLEEP_STATE_AUDIT=1``) content audit across parameters + buffers +
  runtime tensors, used to find any state the restore paths still miss.

Environment switches
--------------------
``SGLANG_SLEEP_STATE_RESTORE``  (default ``1``)  enable the generic restore.
Set to ``0`` to observe the raw pollution (useful with the audit to list
exactly which tensors sleep destroys).
``SGLANG_SLEEP_STATE_AUDIT``    (default ``0``)  enable the content audit.
"""

import logging
import os
from typing import Any, Dict, List, Tuple

import torch

logger = logging.getLogger(__name__)

RESTORE_ENABLED = os.environ.get("SGLANG_SLEEP_STATE_RESTORE", "1") == "1"
AUDIT_ENABLED = os.environ.get("SGLANG_SLEEP_STATE_AUDIT", "0") == "1"

# Module dunder-ish dicts that hold parameters/buffers/submodules; those are
# handled by their own (existing) restore paths and must not be re-walked here.
_INTERNAL_MODULE_KEYS = (
    "_parameters",
    "_buffers",
    "_modules",
    "_non_persistent_buffers_set",
)

_MAX_DEPTH = 6
_MAX_TENSORS = 50000
_MAX_REPORT = 30

Path = Tuple[Any, ...]
# A path step is ("mod", module_name) | ("attr", attribute_name) | ("item", key)


def _is_persistent_device_tensor(t: torch.Tensor) -> bool:
    return (
        torch.is_tensor(t) and t.device.type in ("npu", "cuda", "xpu") and t.numel() > 0
    )


def iter_runtime_tensor_entries(
    model: torch.nn.Module,
) -> List[Tuple[Path, torch.Tensor]]:
    """Collect (path, tensor) for plain device-tensor attributes on the model.

    Covers attributes held directly on modules, inside lists/tuples/dicts
    stored on modules, and inside plain (non-``nn.Module``) helper objects
    referenced from modules - e.g. MoE dispatchers, which is where
    vllm-ascend #13352 found its offenders.
    """
    covered = set()
    for p in model.parameters():
        covered.add(id(p))
    for b in model.buffers():
        covered.add(id(b))

    entries: List[Tuple[Path, torch.Tensor]] = []
    seen_ids = set()

    def walk(value: Any, path: Path, depth: int) -> None:
        if len(entries) >= _MAX_TENSORS or depth > _MAX_DEPTH:
            return
        if id(value) in seen_ids or id(value) in covered:
            return
        seen_ids.add(id(value))

        if torch.is_tensor(value):
            if _is_persistent_device_tensor(value):
                entries.append((path, value))
            return
        if isinstance(value, torch.nn.Module) or value is None or callable(value):
            # modules are covered by the named_modules() traversal itself
            return
        if isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                walk(item, path + (("item", i),), depth + 1)
            return
        if isinstance(value, dict):
            for k, item in value.items():
                walk(item, path + (("item", k),), depth + 1)
            return
        if hasattr(value, "__dict__"):
            # plain helper object (dispatcher, allocator wrapper, ...):
            # descend into instance attributes
            for k, v in list(vars(value).items()):
                walk(v, path + (("attr", k),), depth + 1)
            return
        # scalars / strings / other leaves: nothing to do

    for mod_name, module in model.named_modules():
        for key, value in list(vars(module).items()):
            if key in _INTERNAL_MODULE_KEYS:
                continue
            walk(value, (("mod", mod_name), ("attr", key)), 1)
    return entries


def _resolve_path(model: torch.nn.Module, path: Path) -> Any:
    obj: Any = model
    for kind, key in path:
        if kind == "mod":
            obj = dict(model.named_modules())[key]
        elif kind == "attr":
            obj = getattr(obj, key)
        elif kind == "item":
            obj = obj[key]
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown path step {kind}")
    return obj


def _path_str(path: Path) -> str:
    return "/".join(f"{kind}:{key}" for kind, key in path)


def export_runtime_state(model: torch.nn.Module) -> List[Tuple[Path, torch.Tensor]]:
    """Snapshot all runtime tensor attrs to CPU (call BEFORE pausing the pool)."""
    entries = iter_runtime_tensor_entries(model)
    exported: List[Tuple[Path, torch.Tensor]] = []
    total_bytes = 0
    for path, tensor in entries:
        try:
            cpu = tensor.detach().to("cpu", copy=True)
            exported.append((path, cpu))
            total_bytes += cpu.numel() * cpu.element_size()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                "sleep: failed to export runtime tensor %s: %s",
                _path_str(path),
                e,
            )
    logger.info(
        "sleep: exported %d runtime tensor attrs (%.2f MiB) for restore",
        len(exported),
        total_bytes / 1024 / 1024,
    )
    return exported


def import_runtime_state(
    model: torch.nn.Module, entries: List[Tuple[Path, torch.Tensor]]
) -> int:
    """Restore runtime tensor attrs saved by :func:`export_runtime_state`."""
    if not entries:
        return 0
    if not RESTORE_ENABLED:
        logger.warning(
            "sleep: SGLANG_SLEEP_STATE_RESTORE=0, skipping restore of "
            "%d runtime tensor attrs",
            len(entries),
        )
        return 0
    restored = skipped = 0
    with torch.inference_mode():
        for path, saved in entries:
            try:
                target = _resolve_path(model, path)
                if (
                    torch.is_tensor(target)
                    and target.shape == saved.shape
                    and target.dtype == saved.dtype
                ):
                    target.copy_(saved)
                    restored += 1
                else:
                    skipped += 1
                    logger.warning(
                        "sleep: runtime tensor %s no longer matches saved "
                        "shape/dtype, skipped",
                        _path_str(path),
                    )
            except Exception as e:  # pragma: no cover - defensive
                skipped += 1
                logger.warning(
                    "sleep: failed to restore runtime tensor %s: %s",
                    _path_str(path),
                    e,
                )
    logger.info(
        "sleep: restored %d runtime tensor attrs (%d skipped)", restored, skipped
    )
    return restored


def _digest(t: torch.Tensor) -> str:
    with torch.no_grad():
        flat = t.detach().reshape(-1)
        n = flat.numel()
        if n == 0:
            return "empty"
        sample = flat if n <= 8192 else flat[:: n // 8192][:8192]
        f = sample.to(torch.float32)
        head = [round(v, 6) for v in f[:4].tolist()]
        return (
            f"{tuple(t.shape)}|{t.dtype}|n={n}"
            f"|sum={f.sum().item():.6e}|abs={f.abs().sum().item():.6e}|head={head}"
        )


def snapshot_sleep_audit(model: torch.nn.Module) -> Dict[str, str] | None:
    """Digest parameters + buffers + runtime tensors (no-op unless AUDIT=1)."""
    if not AUDIT_ENABLED:
        return None
    state: Dict[str, str] = {}
    try:
        for name, p in model.named_parameters():
            state[f"param:{name}"] = _digest(p)
        for name, b in model.named_buffers():
            state[f"buffer:{name}"] = _digest(b)
        for path, t in iter_runtime_tensor_entries(model):
            state[f"runtime:{_path_str(path)}"] = _digest(t)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("sleep: audit snapshot failed: %s", e)
        return None
    return state


def diff_sleep_audit(
    before: Dict[str, str], after: Dict[str, str]
) -> Tuple[List[str], List[str], List[str]]:
    """Return (changed, added, removed) key lists between two audit snapshots."""
    if not before or not after:
        return [], [], []
    changed = [k for k in before if k in after and before[k] != after[k]]
    added = [k for k in after if k not in before]
    removed = [k for k in before if k not in after]
    return changed, added, removed
