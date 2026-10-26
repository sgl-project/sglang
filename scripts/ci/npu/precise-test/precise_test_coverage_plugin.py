"""CI coverage plugin: SIGTERM-first teardown for kill_process_tree.

Loaded ONLY by scripts/ci/npu/precise-test/run_tests_with_coverage.sh
(PYTEST_PLUGINS + PYTHONPATH) and gated on SGLANG_KILL_PROCESS_TREE_GRACEFUL=1,
so it is a strict no-op in every other pytest invocation. Product code is
NOT modified: the patch lives in the pytest process only.

Why: subprocess coverage (COVERAGE_PROCESS_START + site-packages .pth)
traces the server / scheduler / detokenizer processes that tests launch,
but those processes can only flush their coverage data on a graceful
stop. The stock kill_process_tree goes straight to SIGKILL, which skips
atexit and silently loses the data. This wrapper SIGTERMs the tree
first (scheduler/detokenizer flush via coveragerc sigterm=True; the
server drains via its own SIGTERM path), waits a bounded time, then
defers to the original SIGKILL implementation for whatever is left.

All server teardowns in the coverage suites converge on the single
sglang.srt.utils.common.kill_process_tree (re-exported by
sglang.srt.utils, bound into sglang.test.test_utils and the ascend e2e
utils), so patching that one attribute covers base-b and base-c. The
diffusion suites use their own
sglang.multimodal_gen.runtime.utils.process.kill_process_tree, patched
lazily after collection (see _patch_multimodal_kill_process_tree).
"""

import os
import signal
import sys

import psutil


def _enabled() -> bool:
    return os.environ.get("SGLANG_KILL_PROCESS_TREE_GRACEFUL") == "1"


def _make_graceful_kill_process_tree(original):
    graceful_timeout = float(
        os.environ.get("SGLANG_KILL_PROCESS_TREE_GRACEFUL_TIMEOUT", "5")
    )

    def kill_process_tree_graceful(
        parent_pid, include_parent=True, skip_pid=None, wait_timeout=60
    ):
        # Best-effort SIGTERM head start; the original call below is the
        # backstop and also handles the NoSuchProcess race.
        try:
            itself = psutil.Process(parent_pid)
            targets = [c for c in itself.children(recursive=True) if c.pid != skip_pid]
            # Never SIGTERM ourselves: the parent_pid == os.getpid() branch
            # in the original exits via sys.exit(0), which runs atexit (and
            # would flush this process's own coverage data).
            if include_parent and parent_pid != os.getpid():
                targets.append(itself)
            for proc in targets:
                try:
                    proc.send_signal(signal.SIGTERM)
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(targets, timeout=graceful_timeout)
        except psutil.NoSuchProcess:
            pass
        return original(
            parent_pid,
            include_parent=include_parent,
            skip_pid=skip_pid,
            wait_timeout=wait_timeout,
        )

    kill_process_tree_graceful._sglang_ci_graceful = True
    return kill_process_tree_graceful


def _patch_module_kill_process_tree(module):
    original = module.kill_process_tree
    graceful = _make_graceful_kill_process_tree(original)
    module.kill_process_tree = graceful

    # Refresh modules that already bound the original (the sglang.srt.utils
    # re-export, anything an early conftest imported, the diffusion test
    # helpers). Modules imported later pick up the patched attribute on
    # their own.
    for mod in list(sys.modules.values()):
        if getattr(mod, "kill_process_tree", None) is original:
            mod.kill_process_tree = graceful


def _patch_kill_process_tree():
    import sglang.srt.utils.common as common

    _patch_module_kill_process_tree(common)


def _patch_multimodal_kill_process_tree():
    """Patch the diffusion runtime's SIGKILL-only teardown, if it is loaded.

    The diffusion suites tear their servers down via
    sglang.multimodal_gen.runtime.utils.process.kill_process_tree instead of
    the srt helper patched above, so subprocess coverage would be lost on the
    SIGKILL. Unlike srt this module is imported by the diffusion test modules
    during collection, so patch lazily AFTER collection: importing it here
    would pull the whole diffusion runtime into every other suite's pytest
    process for no benefit.
    """
    module = sys.modules.get("sglang.multimodal_gen.runtime.utils.process")
    if module is None:
        return
    if getattr(module.kill_process_tree, "_sglang_ci_graceful", False):
        return
    _patch_module_kill_process_tree(module)


def pytest_configure(config):
    if not _enabled():
        return
    try:
        _patch_kill_process_tree()
    except Exception as exc:
        # E.g. base-a hosted runner, where the NPU stack is not importable
        # outside the tests' own stubs. No servers are launched there, so
        # skipping the patch is fine — just make it visible in the log.
        print(f"precise_test_coverage_plugin: patch skipped: {exc!r}")


def pytest_collection_modifyitems(config, items):
    if not _enabled():
        return
    try:
        _patch_multimodal_kill_process_tree()
    except Exception as exc:
        print(
            f"precise_test_coverage_plugin: multimodal teardown patch skipped: {exc!r}"
        )
