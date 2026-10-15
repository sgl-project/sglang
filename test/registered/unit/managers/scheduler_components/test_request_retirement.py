"""A request leaves the scheduler only through `RequestRetirer`.

Every way a request leaves (finished, aborted, retracted, a failed PD
transfer) releases up to five parts: media and the streaming session with the
cache request, the beam group, HiSparse rows, the retraction backup, and the
KV row. When each site assembles those parts by hand, some site skips one;
the retirer's entry points run them in one fixed order instead.

This test fails on a call to a part function anywhere under `srt/` outside
the retirer module. The allowlist records the sites still to be removed: a
new site fails, and a removed site fails until its entry is deleted, so it
only shrinks.
"""

import ast
import unittest
import warnings
from collections import Counter
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SRT_DIR = _REPO_ROOT / "python" / "sglang" / "srt"
_RETIRER = "managers/scheduler_components/request_retirer.py"
assert (_SRT_DIR / _RETIRER).is_file(), f"retirer not found under {_SRT_DIR}"

_HISPARSE_RELEASES = ("request_finished", "retract_req", "abort_staging_request")
_MEDIA_RELEASES = ("release_mm_inputs_on_abort", "release_features")

# (module under srt/, scope, part) -> count.
_ALLOWED = {
    # The owners of a part, inside their own release.
    (
        "managers/schedule_batch.py",
        "Req.release_mm_inputs_on_abort",
        "release_features",
    ): 1,
    # Marks a request aborted while it still runs as a stub (the retirer's door
    # rejection, the grammar manager, a session); the stub's commit then
    # retires it.
    (
        "managers/schedule_batch.py",
        "Req.set_finish_with_abort",
        "release_mm_inputs_on_abort",
    ): 1,
    # A streaming session owns its turns' media and its slot's lifetime.
    (
        "session/session_controller.py",
        "SessionController._close",
        "release_features",
    ): 1,
    (
        "session/streaming_session.py",
        "StreamingSession.find_active_slot",
        "release_mm_inputs_on_abort",
    ): 1,
    (
        "session/streaming_session.py",
        "StreamingSession.try_cache_finished_req",
        "release_mm_inputs_on_abort",
    ): 1,
    # A model drops the encoder features it consumed; no request leaves.
    (
        "models/moss_vl.py",
        "MossVLForConditionalGeneration.forward",
        "release_features",
    ): 1,
    # Dummy requests that time a prefill at startup; they never reach a list.
    (
        "managers/scheduler_components/dynamic_chunk_sizer.py",
        "DynamicChunkSizer._profile_prefill_latency",
        "release_kv_cache",
    ): 1,
    # A HiCache restore that failed ends its prefetch; the request stays queued.
    (
        "disaggregation/decode_hicache_mixin.py",
        "DecodeHiCacheTransferMixin._clean_hicache_prefetch_resources",
        "tree_cache.finish",
    ): 1,
    # To be removed: sites that skip a part.
    # A PD decode request aborted before preallocation frees only its early
    # mamba state; it skips the media and the cache request.
    (
        "disaggregation/decode.py",
        "DecodePreallocQueue.pop_preallocated",
        "release_kv_cache",
    ): 1,
}


def _part(call: ast.Call):
    """The release part a call runs, or None."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id if func.id == "release_kv_cache" else None
    if not isinstance(func, ast.Attribute):
        return None
    receiver = ast.unparse(func.value)
    if func.attr == "release_kv_cache":
        return "release_kv_cache"
    if func.attr == "finish" and receiver.endswith("tree_cache"):
        return "tree_cache.finish"
    if func.attr == "retire_group":
        return "retire_group"
    if func.attr in _HISPARSE_RELEASES and "hisparse" in receiver:
        return f"hisparse.{func.attr}"
    if func.attr in _MEDIA_RELEASES:
        return func.attr
    # The retirer's part, outside its entry points.
    if func.attr == "release_aborted_request" and receiver.endswith("request_retirer"):
        return "request_retirer.release_aborted_request"
    return None


def _scan() -> Counter:
    found = Counter()
    for path in sorted(_SRT_DIR.rglob("*.py")):
        rel = path.relative_to(_SRT_DIR).as_posix()
        if rel == _RETIRER:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))

        def visit(node, scope):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                scope = f"{scope}.{node.name}" if scope else node.name
            if isinstance(node, ast.Call) and (part := _part(node)):
                found[(rel, scope, part)] += 1
            for child in ast.iter_child_nodes(node):
                visit(child, scope)

        visit(tree, "")
    return found


class TestRequestRetirement(CustomTestCase):
    def test_release_parts_only_in_the_retirer(self):
        found, allow = _scan(), Counter(_ALLOWED)
        new, gone = found - allow, allow - found
        msg = []
        if new:
            msg.append(
                "Release parts called outside RequestRetirer; call one of its "
                "entry points instead:\n  "
                + "\n  ".join(f"{site} x{n}" for site, n in sorted(new.items()))
            )
        if gone:
            msg.append(
                "Allowlisted sites no longer present (delete the entries):\n  "
                + "\n  ".join(f"{site} x{n}" for site, n in sorted(gone.items()))
            )
        self.assertFalse(msg, "\n\n".join(msg))


if __name__ == "__main__":
    unittest.main()
