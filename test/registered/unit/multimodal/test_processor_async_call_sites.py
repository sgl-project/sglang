"""Every processor must reach preprocessing through the executor-backed helper.

`process_and_combine_mm_data` is the function the multimodal processor worker
pool actually runs. A processor that calls it directly can never use those
workers: it will build the thread pool and its processor clones on startup and
then route every request past them. That failure is silent -- the model just
serves at one-worker speed -- so pin the call site instead of the symptom.

`process_and_combine_mm_data_async` delegates straight to the sync function when
no executor exists, so using it costs nothing until a model opts into
concurrency.

  async processor entry
    -> await process_and_combine_mm_data_async
      -> worker pool (or direct fallback when no executor exists)
        -> sync process_and_combine_mm_data
  token-space processor entry -> one media task per item -> same worker pool
                              -> expansion and serving assembly -> same worker pool
  startup opt-in AND a token-space processor that supports the model
    True  -> token-space processor -> shared loading -> same worker pool
    False -> original async route -> original HF/native media processor
  request audio/video -> loader; explicit media arguments override request fields
  per-source configs stay with the outer call and reach the member beside loaded media


Source-tree checks reject processor calls that bypass the async worker helper.
"""

import ast
import asyncio
import pathlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=13, suite="base-a-test-cpu")

_MULTIMODAL_ROOT = (
    pathlib.Path(__file__).resolve().parents[4]
    / "python"
    / "sglang"
    / "srt"
    / "multimodal"
)
if not _MULTIMODAL_ROOT.is_dir():
    raise RuntimeError(
        f"multimodal processor tree not found at {_MULTIMODAL_ROOT}; "
        "these tests must run from a full source checkout"
    )
# The async helper and the sync body live side by side here by design.
_EXEMPT = {"base_processor.py"}


def _enclosing_function(node, parents):
    current = parents.get(id(node))
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
        current = parents.get(id(current))
    return None


def _call_sites():
    """Yield (path, lineno, attribute, enclosing_function) for every call."""
    for path in sorted(_MULTIMODAL_ROOT.rglob("*.py")):
        if path.name in _EXEMPT:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parents = {
            id(child): parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr.startswith("process_and_combine_mm_data")
            ):
                continue
            yield path, node.lineno, func.attr, _enclosing_function(node, parents)


def test_no_processor_bypasses_the_worker_pool():
    offenders = [
        f"{path.relative_to(_MULTIMODAL_ROOT)}:{lineno}"
        for path, lineno, attr, _ in _call_sites()
        if not attr.endswith("_async")
    ]
    assert not offenders, (
        "these call sites bypass the multimodal processor worker pool; use "
        "`await self.process_and_combine_mm_data_async(...)`: " + ", ".join(offenders)
    )


def test_every_call_site_can_await():
    """An `await` needs an async def around it, so the migration stays possible."""
    offenders = [
        f"{path.relative_to(_MULTIMODAL_ROOT)}:{lineno}"
        for path, lineno, _, enclosing in _call_sites()
        if not isinstance(enclosing, ast.AsyncFunctionDef)
    ]
    assert not offenders, (
        "preprocessing is reached from a non-async function, so it cannot go "
        "through the worker pool: " + ", ".join(offenders)
    )


@pytest.mark.parametrize("media_source", ["arguments", "request"])
def test_token_space_entry_runs_the_complete_request(
    media_source,
):
    from sglang.srt.multimodal.processors.qwen_vl_token_space import (
        QwenVLTokenSpaceProcessor,
    )

    migrated_processor = object.__new__(QwenVLTokenSpaceProcessor)
    migrated_processor._processor = object()
    migrated_processor.token_space_process_strategy = object()
    loaded = SimpleNamespace(images=[object()], videos=[object()], audios=[object()])
    migrated_processor._load_media_lists = AsyncMock(
        return_value=(loaded.images, loaded.videos, loaded.audios)
    )
    migrated_processor._token_space_media_devices = Mock(
        return_value=("image device", "video device")
    )
    migrated_processor.process_media_async = AsyncMock(return_value=object())
    migrated_processor._run_mm_processor = AsyncMock(return_value=object())
    request_kwargs = {
        "image_data": ["image"],
        "video_data": ["video"],
        "audio_data": ["audio"],
        "input_text": "",
        "input_ids": [1, 101, 2],
        "request_obj": SimpleNamespace(
            video_data=["request video"], audio_data=["request audio"]
        ),
        "mm_token_expansion_start_len": 1,
    }

    call_kwargs = dict(request_kwargs)
    if media_source == "request":
        for name in ("video_data", "audio_data"):
            del call_kwargs[name]
        request_kwargs["video_data"] = request_kwargs["request_obj"].video_data
        request_kwargs["audio_data"] = request_kwargs["request_obj"].audio_data
    output = asyncio.run(migrated_processor.process_mm_data_async(**call_kwargs))

    assert output is migrated_processor._run_mm_processor.return_value
    migrated_processor._load_media_lists.assert_awaited_once_with(
        request_kwargs["image_data"],
        request_kwargs["video_data"],
        request_kwargs["audio_data"],
        audio_sample_rate=None,
        discard_alpha_channel=True,
    )
    migrated_processor.process_media_async.assert_awaited_once_with(
        images=loaded.images,
        videos=loaded.videos,
        audios=loaded.audios,
        image_source_configs=[{}],
        video_source_configs=[{}],
        audio_source_configs=[{}],
        video_metadata=None,
        image_device="image device",
        video_device="video device",
    )
    assert migrated_processor._run_mm_processor.await_args_list == [
        call(
            migrated_processor._build_token_space_output,
            media_features=migrated_processor.process_media_async.return_value,
            input_ids=request_kwargs["input_ids"],
            mm_token_expansion_start_len=1,
            options={},
        ),
        call(
            migrated_processor.sglang_post_process,
            ret=migrated_processor._run_mm_processor.return_value,
        ),
    ]


def _process_mm_data_overrides():
    """Yield (path, node) for every subclass override of `process_mm_data`."""
    for path in sorted(_MULTIMODAL_ROOT.rglob("*.py")):
        if path.name in _EXEMPT:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "process_mm_data":
                yield path, node


def test_overrides_take_the_worker_pools_processor_clone():
    """An override that reaches for `self._processor` puts every worker thread on
    one shared HF processor, which is exactly what the per-thread clone exists to
    prevent. Either accept `processor=` and resolve it, or delegate to super.
    """
    offenders = []
    for path, node in _process_mm_data_overrides():
        args = [a.arg for a in node.args.args] + [a.arg for a in node.args.kwonlyargs]
        body = ast.dump(node)
        reaches_for_shared = "attr='_processor'" in body
        resolves_injected = "_resolve_processor" in body
        if reaches_for_shared and not resolves_injected:
            offenders.append(f"{path.relative_to(_MULTIMODAL_ROOT)}:{node.lineno}")
        elif "processor" not in args and not (
            "'super'" in body or not reaches_for_shared
        ):
            offenders.append(f"{path.relative_to(_MULTIMODAL_ROOT)}:{node.lineno}")
    assert not offenders, (
        "these `process_mm_data` overrides bypass the worker pool's processor "
        "clone; accept `processor=None` and resolve it with "
        "`self._resolve_processor(processor)`: " + ", ".join(offenders)
    )


# Processors that build their whole preprocessing chain themselves and never
# reach `process_and_combine_mm_data`, so the worker pool cannot help them. They
# are not broken by concurrency either -- they simply do not participate. Listed
# explicitly so that adding a processor forces a decision instead of silently
# leaving it at one-worker speed.
_NO_WORKER_POOL_ROUTE = {
    # Runs its own image preprocessing to keep the raw token ids the Engram
    # hasher needs; the shared chain would re-tokenize them.
    "deepseek_v41.py",
    "dots_note_omni.py",
    "inkling.py",
    "lightonocr.py",
    "llava.py",
    "mimo_v2.py",
    "mimo_v2_asr.py",
    "minicpmv4_6.py",
    "moss_vl.py",
    "nano_nemotron_vl.py",
    "voxtral.py",
    "whisper.py",
}


def test_processors_outside_the_worker_pool_are_declared():
    """A new processor must either route through the pool or be listed here.

    Without this, a processor added on the old call site keeps preprocessing on
    the event loop and nobody notices: there is no error, just one-worker
    throughput. Whichever way the list moves, the change should be deliberate.
    """
    unrouted = set()
    for path in sorted(_MULTIMODAL_ROOT.rglob("*.py")):
        if path.name in _EXEMPT:
            continue
        source = path.read_text(encoding="utf-8")
        entry_points = (
            "async def process_mm_data_async" in source
            or "async def _process_special_format" in source
        )
        routed = (
            "process_and_combine_mm_data_async" in source
            or "_run_mm_processor" in source
        )
        if entry_points and not routed:
            unrouted.add(path.name)

    newly_unrouted = unrouted - _NO_WORKER_POOL_ROUTE
    assert not newly_unrouted, (
        "these processors reach preprocessing without going through the worker "
        "pool, so they will serve at one-worker speed; either route them through "
        "`process_and_combine_mm_data_async` or `_run_mm_processor`, or add them to "
        f"_NO_WORKER_POOL_ROUTE with a reason: {sorted(newly_unrouted)}"
    )
    now_routed = _NO_WORKER_POOL_ROUTE - unrouted
    assert not now_routed, (
        "these processors now reach the worker pool, so drop them from "
        f"_NO_WORKER_POOL_ROUTE: {sorted(now_routed)}"
    )


def test_the_scan_actually_finds_call_sites():
    """Guard against the scan silently matching nothing after a rename."""
    assert len(list(_call_sites())) > 20


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
