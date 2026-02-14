"""Tests for diffusion `update_weights_from_disk`.

This module verifies the ability to update model weights in place without restarting
the server, which is critical for RL workflows and iterative fine-tuning scenarios.

Author:

Menyang Liu, https://github.com/dreamyang-liu
Chenyang Zhao, https://github.com/zhaochenyang20

We use two model pairs for testing (base model / instruct model pairs):

- FLUX.2-klein-base-4B / FLUX.2-klein-4B
- Qwen/Qwen-Image / Qwen/Qwen-Image-2512

These model pairs share the same architecture, but not every module is
guaranteed to have different weights between base and update models.
Some modules can be identical across the pair.

=============================================================================

Test organization:

7 test cases in 2 classes;
two model pairs are tested locally, one in CI.

=============================================================================

Class 1: TestUpdateWeightsFromDisk                  (6 tests) — API contract, checksum & rollback
Class 2: TestUpdateWeightsFromDiskWithOffload       (1 test) — Offload-aware update + checksum

-----------------------------------------------------------------------------

Class 1: TestUpdateWeightsFromDisk

Validate the update_weights_from_disk API contract, request/response shape,
error handling, checksum verification, and corrupted-weight rollback.

All tests share one class-scoped server (same process, same in-memory weights).
Tests that require "base model then update" should be explicitly reset to
default_model first so behavior is order-independent and updates are real
 (base→update), not no-ops (update→update).

  • test_update_weights_from_disk_default

    base -> instruct with flush_cache=True. Verifies:
    (1) before-update checksum == base model disk checksum;
    (2) after-update checksum == instruct model disk checksum;
    (3) before != after (update actually changed weights).
    rollback to base model after update.

  • test_update_weights_specific_modules

    base -> instruct with flush_cache=False: randomly selects target_modules,
    updates only those from base to instruct model. Verifies:
    (1) updated modules' checksums match instruct model disk checksum;
    (2) non-updated modules' checksums are unchanged (before == after == disk).
    rollback to base model after update.

  • test_update_weights_nonexistent_model

    model_path set to a non-existent path; must fail (400, success=False).

    Ensure server is healthy after inaccurate update and server's checksums
    equals to base model's disk checksums.

  • test_update_weights_missing_model_path

    Request body empty (no model_path); must fail (400, success=False).

    Ensure server is healthy after inaccurate update and server's checksums
    equals to base model's disk checksums.

  • test_update_weights_nonexistent_module

    target_modules=["nonexistent_module"]; must fail (400, success=False).

    Verify server is healthy after inaccurate update and server's checksums
    equals to base model's disk checksums.

  • test_corrupted_weights_rollback

    Verify base -> instruct rollback after loading corrupted instruct model.
    Builds a corrupted model directory by copying the instruct model and
    truncating the vae safetensors. Updates with target_modules=["transformer",
    "vae"]. The transformer updates successfully first; the corrupted vae module
    then fails during safetensors validation, triggering a rollback that restores
    the transformer to its previous weights.

    Ensure server is healthy after rollback and server's checksums equals to
    base model's disk checksums.

-----------------------------------------------------------------------------

Class 2: TestUpdateWeightsFromDiskWithOffload


Ensure weight updates and checksum verification work when layerwise offload is enabled
(--dit-layerwise-offload). With offload, parameters live in CPU buffers and only left
small torch.empty((1,)) as placeholders on GPU; the updater must write into CPU buffers
and update prefetched GPU tensors without shape mismatch.

  • test_update_weights_with_offload_enabled

    Server with --dit-layerwise-offload (base). Update to instruct; must succeed
    (200, success=True), message must not contain "Shape mismatch". Assert server
    checksums == instruct model disk checksums (server healthy).
"""

from __future__ import annotations

import functools
import os
import random
import shutil
import tempfile
import threading

import pytest
import requests

from sglang.multimodal_gen.runtime.loader.utils import (
    _list_safetensors_files,
    find_weights_dir,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    compute_weights_checksum,
    safetensors_weights_iterator,
)
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.server.test_server_utils import (
    ServerContext,
    ServerManager,
)
from sglang.multimodal_gen.test.test_utils import get_dynamic_server_port, is_in_ci

logger = init_logger(__name__)

_ALL_MODEL_PAIRS: list[tuple[str, str, float]] = [
    (
        "black-forest-labs/FLUX.2-klein-base-4B",
        "black-forest-labs/FLUX.2-klein-4B",
        5.0,
    ),
    (
        "Qwen/Qwen-Image",
        "Qwen/Qwen-Image-2512",
        1.0,  # Qwen Image is large; run it less often in CI.
    ),
]


def _select_model_pairs() -> list[tuple[str, str]]:
    """Return the (default, update) model pairs to test.

    When SGLANG_TEST_DIFFUSION_MODEL / SGLANG_TEST_UPDATE_MODEL env vars
    are set, use them as a single explicit pair.  Otherwise, run both
    pairs locally, or randomly pick one in CI (weighted) to save resources.
    """
    default_env = os.environ.get("SGLANG_TEST_DIFFUSION_MODEL")
    update_env = os.environ.get("SGLANG_TEST_UPDATE_MODEL")
    if default_env and update_env:
        return [(default_env, update_env)]
    pairs = [(d, u) for d, u, _ in _ALL_MODEL_PAIRS]
    if is_in_ci():
        weights = [w for _, _, w in _ALL_MODEL_PAIRS]
        return random.choices(pairs, weights=weights, k=1)
    return pairs


_ACTIVE_MODEL_PAIRS = _select_model_pairs()
_PAIR_IDS = [p[0].split("/")[-1] for p in _ACTIVE_MODEL_PAIRS]


@functools.lru_cache(maxsize=None)
def _compute_checksum_from_disk(model_path: str, module_name: str) -> str:
    """Compute SHA-256 checksum from safetensors files on disk.

    Uses the same compute_weights_checksum function as the server,
    so the checksums are directly comparable.

    Results are cached (keyed on model_path and module_name) because the
    same disk checksum is requested multiple times across tests.
    """
    local_path = maybe_download_model(model_path)
    weights_dir = find_weights_dir(local_path, module_name)
    assert weights_dir is not None, f"No weights dir for {module_name} in {local_path}"

    safetensors_files = _list_safetensors_files(weights_dir)
    assert safetensors_files, f"No safetensors files in {weights_dir}"

    return compute_weights_checksum(safetensors_weights_iterator(safetensors_files))


def _get_modules_with_weights_on_disk(
    model_path: str, module_names: list[str]
) -> list[str]:
    """Return module names that have safetensors on disk for the given model."""
    local_path = maybe_download_model(model_path)
    result = []
    for name in module_names:
        weights_dir = find_weights_dir(local_path, name)
        if weights_dir and _list_safetensors_files(weights_dir):
            result.append(name)
    return result


def _get_modules_with_different_checksums(
    base_model: str, update_model: str, module_names: list[str]
) -> list[str]:
    """Return shared modules whose disk checksums differ across model pair."""
    base_modules = set(_get_modules_with_weights_on_disk(base_model, module_names))
    update_modules = set(_get_modules_with_weights_on_disk(update_model, module_names))
    shared_modules = sorted(base_modules & update_modules)

    changed_modules = []
    for name in shared_modules:
        base_cs = _compute_checksum_from_disk(base_model, name)
        update_cs = _compute_checksum_from_disk(update_model, name)
        if base_cs != update_cs:
            changed_modules.append(name)
    return changed_modules


def _prepare_corrupted_model(
    src_model: str, dst_model: str, corrupt_module: str
) -> None:
    """Build a corrupted model directory from src_model.

    Uses symlinks for everything except the corrupt_module directory to
    save disk space and time.  Only the corrupt_module's safetensors are
    physically copied and then truncated so that safetensors_weights_iterator
    detects corruption at load time, triggering a rollback.

    Must be called before every test attempt because the server deletes
    corrupted files on detection.
    """
    # Symlink root-level files (model_index.json, etc.).
    for fname in os.listdir(src_model):
        src_path = os.path.join(src_model, fname)
        dst_path = os.path.join(dst_model, fname)
        if os.path.isfile(src_path) and not os.path.exists(dst_path):
            os.symlink(src_path, dst_path)

    for module_dir in sorted(os.listdir(src_model)):
        src_dir = os.path.join(src_model, module_dir)
        dst_dir = os.path.join(dst_model, module_dir)
        if not os.path.isdir(src_dir):
            continue

        # Non-corrupted modules: symlink the entire directory.
        if module_dir != corrupt_module:
            if not os.path.exists(dst_dir):
                os.symlink(src_dir, dst_dir)
            continue

        # Corrupted module: create a real directory, symlink non-safetensors
        # files, and copy + truncate safetensors files.
        os.makedirs(dst_dir, exist_ok=True)
        for fname in os.listdir(src_dir):
            src_file = os.path.join(src_dir, fname)
            dst_file = os.path.join(dst_dir, fname)
            if not os.path.isfile(src_file):
                continue

            if not fname.endswith(".safetensors"):
                if not os.path.exists(dst_file):
                    os.symlink(src_file, dst_file)
                continue

            # Copy safetensors then truncate to corrupt it.
            shutil.copy2(src_file, dst_file)
            size = os.path.getsize(dst_file)
            with open(dst_file, "r+b") as f:
                f.truncate(size - 1000)
            logger.info(
                "Created corrupted safetensors: %s (%d -> %d bytes)",
                dst_file,
                size,
                size - 1000,
            )


class _UpdateWeightsApiMixin:
    def _get_base_url(self, ctx: ServerContext) -> str:
        return f"http://localhost:{ctx.port}"

    def _update_weights(
        self,
        base_url: str,
        model_path: str,
        flush_cache: bool = True,
        target_modules: list[str] | None = None,
        timeout: int = 300,
    ) -> tuple[dict, int]:
        payload = {"model_path": model_path, "flush_cache": flush_cache}
        if target_modules is not None:
            payload["target_modules"] = target_modules
        response = requests.post(
            f"{base_url}/update_weights_from_disk",
            json=payload,
            timeout=timeout,
        )
        return response.json(), response.status_code

    def _get_weights_checksum(
        self,
        base_url: str,
        module_names: list[str] | None = None,
        timeout: int = 300,
    ) -> dict:
        payload = {}
        if module_names is not None:
            payload["module_names"] = module_names
        response = requests.post(
            f"{base_url}/get_weights_checksum",
            json=payload,
            timeout=timeout,
        )
        assert (
            response.status_code == 200
        ), f"get_weights_checksum failed: {response.status_code} {response.text}"
        return response.json()

    def _assert_server_matches_model_on_changed_modules(
        self,
        base_url: str,
        base_model: str,
        update_model: str,
        expected_model: str,
    ) -> None:
        all_checksums = self._get_weights_checksum(base_url)
        module_names = [k for k, v in all_checksums.items() if v != "not_found"]
        changed_modules = _get_modules_with_different_checksums(
            base_model, update_model, module_names
        )
        if not changed_modules:
            pytest.skip("No checksum-different shared modules in model pair")
        for name in changed_modules:
            server_cs = all_checksums.get(name)
            expected_cs = _compute_checksum_from_disk(expected_model, name)
            assert server_cs == expected_cs, (
                f"Checksum mismatch on '{name}'\n"
                f"  expected({expected_model}): {expected_cs}\n"
                f"  server: {server_cs}"
            )


class TestUpdateWeightsFromDisk(_UpdateWeightsApiMixin):
    """Test suite for update_weights_from_disk API and corrupted-weight rollback.

    Uses a class-scoped server fixture so the server is torn down at class end,
    freeing the port and GPU memory before the offload class starts.
    """

    @pytest.fixture(
        scope="class",
        params=_ACTIVE_MODEL_PAIRS,
        ids=_PAIR_IDS,
    )
    def diffusion_server_no_offload(self, request):
        """Start a diffusion server (no offload) for this test class.

        Precomputes disk checksums for the update model in background threads
        while the server is starting, so they are already cached (via lru_cache)
        by the time tests need them.
        """
        default_model, update_model = request.param
        port = get_dynamic_server_port()
        wait_deadline = float(os.environ.get("SGLANG_TEST_WAIT_SECS", "600"))

        manager = ServerManager(
            model=default_model,
            port=port,
            wait_deadline=wait_deadline,
            extra_args="--num-gpus 1",
        )

        # Warm the lru_cache while the server boots (disk I/O is independent).
        checksum_threads = [
            threading.Thread(
                target=_compute_checksum_from_disk, args=(update_model, module)
            )
            for module in ("transformer", "vae")
        ]
        for t in checksum_threads:
            t.start()

        ctx = manager.start()
        for t in checksum_threads:
            t.join()

        try:
            yield ctx, default_model, update_model
        finally:
            ctx.cleanup()

    @pytest.fixture(scope="class")
    def corrupted_model_dir(self, diffusion_server_no_offload):
        """Create a separate temporary directory per parametrized model pair."""
        tmpdir = tempfile.mkdtemp(prefix="sglang_corrupted_model_")
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_update_weights_from_disk_default(self, diffusion_server_no_offload):
        """Base→instruct with flush_cache=True; verify before/after; rollback to base.

        Resets to base, records before checksum. Updates to instruct with
        flush_cache=True. Asserts: (1) before == base disk; (2) after == instruct
        disk; (3) before != after. Then rollback to base so server ends on base.
        """
        ctx, default_model, update_model = diffusion_server_no_offload
        base_url = self._get_base_url(ctx)

        # Reset to base so we have a real base→instruct.
        self._update_weights(base_url, default_model)

        before_checksum = self._get_weights_checksum(
            base_url, module_names=["transformer"]
        )["transformer"]
        base_disk = _compute_checksum_from_disk(default_model, "transformer")

        result, status_code = self._update_weights(
            base_url,
            update_model,
            flush_cache=True,
        )
        assert status_code == 200 and result.get(
            "success"
        ), f"Update failed: {result.get('message')}"

        after_checksum = self._get_weights_checksum(
            base_url, module_names=["transformer"]
        )["transformer"]
        instruct_disk = _compute_checksum_from_disk(update_model, "transformer")

        assert before_checksum == base_disk, (
            f"Before-update checksum should match base model disk\n"
            f"  base_disk: {base_disk}\n  before:    {before_checksum}"
        )
        assert after_checksum == instruct_disk, (
            f"After-update checksum should match instruct model disk\n"
            f"  instruct_disk: {instruct_disk}\n  after:      {after_checksum}"
        )
        assert (
            before_checksum != after_checksum
        ), "Before and after checksums should differ (update changed weights)"

        # Rollback to base so server ends in known state.
        self._update_weights(base_url, default_model)

    def test_update_weights_specific_modules(self, diffusion_server_no_offload):
        """Partial update base→instruct with flush_cache=False; verify checksums; rollback to base.

        Randomly picks target_modules, updates only those to instruct with
        flush_cache=False. Asserts:
        (1) for modules whose base/update disk checksums differ, updated modules
            match update-model disk and actually change;
        (2) for modules with identical base/update checksums, updating them keeps
            checksums unchanged;
        (3) non-updated modules remain unchanged (before == after).
        Then rollback to base.
        """
        ctx, default_model, update_model = diffusion_server_no_offload
        base_url = self._get_base_url(ctx)

        # Reset to base model so we start from a known state.
        self._update_weights(base_url, default_model)

        # All pipeline module names (from server).
        all_checksums = self._get_weights_checksum(base_url, module_names=None)
        all_module_names = [k for k in all_checksums if all_checksums[k] != "not_found"]
        if not all_module_names:
            pytest.skip("No updatable modules reported by server")

        # Only consider modules that have weights on disk in both models.
        base_modules = set(
            _get_modules_with_weights_on_disk(default_model, all_module_names)
        )
        update_modules = set(
            _get_modules_with_weights_on_disk(update_model, all_module_names)
        )
        candidates = sorted(base_modules & update_modules)
        if not candidates:
            pytest.skip("No shared modules with weights on disk in model pair")

        changed_modules = _get_modules_with_different_checksums(
            default_model, update_model, candidates
        )
        if not changed_modules:
            pytest.skip("No checksum-different shared modules in model pair")

        # Random non-empty subset (fixed seed) that always includes one changed module.
        random.seed(42)
        must_include = random.choice(changed_modules)
        optional = [m for m in candidates if m != must_include]
        k_extra = random.randint(0, len(optional))
        target_modules = [must_include] + random.sample(optional, k_extra)
        target_set = set(target_modules)
        changed_set = set(changed_modules)
        logger.info(
            "Partial update test (flush_cache=False): target_modules=%s (checksum-different modules: %s)",
            target_modules,
            changed_modules,
        )

        before_checksums = self._get_weights_checksum(base_url, module_names=None)

        result, status_code = self._update_weights(
            base_url,
            update_model,
            target_modules=target_modules,
            flush_cache=False,
        )
        assert status_code == 200, f"Update failed: {result}"
        assert result.get("success", False), f"Update failed: {result.get('message')}"

        after_checksums = self._get_weights_checksum(base_url, module_names=None)

        for name in all_module_names:
            if name in target_set:
                if name in changed_set:
                    disk_cs = _compute_checksum_from_disk(update_model, name)
                    assert after_checksums.get(name) == disk_cs, (
                        f"Updated module '{name}': checksum should match update model disk\n"
                        f"  disk: {disk_cs}\n  gpu:  {after_checksums.get(name)}"
                    )
                    assert after_checksums.get(name) != before_checksums.get(name), (
                        f"Updated module '{name}' should change checksum (base != update)\n"
                        f"  before: {before_checksums.get(name)}\n"
                        f"  after:  {after_checksums.get(name)}"
                    )
                else:
                    assert after_checksums.get(name) == before_checksums.get(name), (
                        f"Updated module '{name}' has identical base/update disk checksum, "
                        "so it should remain unchanged\n"
                        f"  before: {before_checksums.get(name)}\n"
                        f"  after:  {after_checksums.get(name)}"
                    )
            else:
                assert after_checksums.get(name) == before_checksums.get(name), (
                    f"Non-updated module '{name}': checksum must be unchanged\n"
                    f"  before: {before_checksums.get(name)}\n"
                    f"  after:  {after_checksums.get(name)}"
                )

        # Rollback to base so server ends in known state.
        self._update_weights(base_url, default_model)

    def test_update_weights_nonexistent_model(self, diffusion_server_no_offload):
        """Nonexistent model path must fail (400). Server healthy, checksums == base disk."""
        ctx, default_model, update_model = diffusion_server_no_offload
        base_url = self._get_base_url(ctx)

        self._update_weights(base_url, default_model)

        result, status_code = self._update_weights(
            base_url,
            "/nonexistent/path/to/model",
            timeout=60,
        )
        logger.info(f"Update result for nonexistent model: {result}")

        assert status_code == 400, f"Expected 400, got {status_code}"
        assert not result.get("success", True), "Should fail for nonexistent model"
        self._assert_server_matches_model_on_changed_modules(
            base_url, default_model, update_model, default_model
        )

    def test_update_weights_missing_model_path(self, diffusion_server_no_offload):
        """Request without model_path must fail (400). Server healthy, checksums == base disk."""
        ctx, default_model, update_model = diffusion_server_no_offload
        base_url = self._get_base_url(ctx)

        self._update_weights(base_url, default_model)

        response = requests.post(
            f"{base_url}/update_weights_from_disk",
            json={},
            timeout=30,
        )

        assert response.status_code == 400, f"Expected 400, got {response.status_code}"
        self._assert_server_matches_model_on_changed_modules(
            base_url, default_model, update_model, default_model
        )

    def test_update_weights_nonexistent_module(self, diffusion_server_no_offload):
        """Nonexistent module must fail (400). Server healthy, checksums == base disk."""
        ctx, default_model, update_model = diffusion_server_no_offload
        base_url = self._get_base_url(ctx)

        self._update_weights(base_url, default_model)

        result, status_code = self._update_weights(
            base_url,
            update_model,
            target_modules=["nonexistent_module"],
            timeout=60,
        )
        logger.info(f"Update nonexistent module result: {result}")

        assert status_code == 400, f"Expected 400, got {status_code}"
        assert not result.get("success", True), "Should fail for nonexistent module"
        assert "not found in pipeline" in result.get("message", "")
        self._assert_server_matches_model_on_changed_modules(
            base_url, default_model, update_model, default_model
        )

    def test_corrupted_weights_rollback(
        self,
        diffusion_server_no_offload,
        corrupted_model_dir: str,
    ):
        """Base→instruct then load corrupted instruct; verify rollback.

        Updates to instruct, then attempts load from corrupted instruct dir
        (vae safetensors truncated). Rollback restores to instruct state.
        Ensures server healthy: reset to base and assert checksums == base disk.
        """
        ctx, default_model, update_model = diffusion_server_no_offload
        base_url = self._get_base_url(ctx)
        rollback_modules = ["transformer", "vae"]

        # --- Step 0: Reset to default model ---
        # Previous tests may have left the server on a different model.
        result, status_code = self._update_weights(base_url, default_model)
        assert status_code == 200 and result.get(
            "success"
        ), f"Failed to reset to default model: {result.get('message')}"

        # --- Step 1: Get base-model checksums for rollback modules ---
        base_checksums = self._get_weights_checksum(
            base_url, module_names=rollback_modules
        )
        logger.info(f"Base model checksums: {base_checksums}")

        # --- Step 2: Update to the update model ---
        result, status_code = self._update_weights(base_url, update_model)
        assert status_code == 200
        assert result.get(
            "success", False
        ), f"Weight update failed: {result.get('message')}"

        # --- Step 3: Record update-model checksums for rollback modules ---
        update_checksums = self._get_weights_checksum(
            base_url, module_names=rollback_modules
        )
        logger.info(f"Update model checksums: {update_checksums}")

        assert (
            update_checksums != base_checksums
        ), "Base and update checksums should differ"

        # --- Step 4: Recreate corrupted model, then attempt load ---
        # Copy all modules from the base model (valid), but corrupt only the
        # vae.  With target_modules=["transformer", "vae"], the transformer
        # updates successfully first, then vae fails, giving a meaningful
        # rollback that actually restores the transformer.
        local_base = maybe_download_model(default_model)
        _prepare_corrupted_model(local_base, corrupted_model_dir, corrupt_module="vae")

        result, status_code = self._update_weights(
            base_url,
            corrupted_model_dir,
            target_modules=rollback_modules,
            timeout=120,
        )
        logger.info(f"Corrupted update result: status={status_code}, body={result}")

        assert not result.get("success", True), "Loading corrupted weights should fail"
        assert (
            "rolled back" in result.get("message", "").lower()
        ), f"Expected rollback message, got: {result.get('message')}"

        # --- Step 5: Verify rollback — rollback module checksums must match update model ---
        post_rollback_checksums = self._get_weights_checksum(
            base_url, module_names=rollback_modules
        )
        logger.info(f"Post-rollback checksums: {post_rollback_checksums}")

        for module in update_checksums:
            assert post_rollback_checksums.get(module) == update_checksums[module], (
                f"Module '{module}' checksum mismatch after rollback\n"
                f"  update:        {update_checksums[module]}\n"
                f"  post-rollback: {post_rollback_checksums.get(module)}"
            )

        assert post_rollback_checksums != base_checksums, (
            "Post-rollback checksums should not match base model "
            "(rollback target is instruct model, not base)"
        )

        # Ensure server healthy: reset to base and verify checksums == base disk.
        result, status_code = self._update_weights(base_url, default_model)
        assert status_code == 200 and result.get(
            "success"
        ), f"Failed to reset to base after rollback: {result.get('message')}"
        self._assert_server_matches_model_on_changed_modules(
            base_url, default_model, update_model, default_model
        )


class TestUpdateWeightsFromDiskWithOffload(_UpdateWeightsApiMixin):
    """Test update_weights_from_disk with layerwise offload enabled."""

    @pytest.fixture(scope="class", params=_ACTIVE_MODEL_PAIRS, ids=_PAIR_IDS)
    def diffusion_server_with_offload(self, request):
        """Start a diffusion server with layerwise offload enabled."""
        default_model, update_model = request.param
        port = get_dynamic_server_port()
        wait_deadline = float(os.environ.get("SGLANG_TEST_WAIT_SECS", "600"))

        manager = ServerManager(
            model=default_model,
            port=port,
            wait_deadline=wait_deadline,
            extra_args="--num-gpus 1 --dit-layerwise-offload true",
        )

        ctx = manager.start()

        try:
            yield ctx, default_model, update_model
        finally:
            ctx.cleanup()

    def test_update_weights_with_offload_enabled(self, diffusion_server_with_offload):
        """Offload: base→instruct update; no Shape mismatch; checksums == instruct disk."""
        ctx, default_model, update_model = diffusion_server_with_offload
        base_url = self._get_base_url(ctx)

        result, status_code = self._update_weights(base_url, update_model)
        assert status_code == 200, f"Expected 200, got {status_code}"
        assert result.get("success", False), f"Update failed: {result.get('message')}"

        message = result.get("message", "")
        assert "Shape mismatch" not in message, f"Shape mismatch detected: {message}"

        self._assert_server_matches_model_on_changed_modules(
            base_url, default_model, update_model, update_model
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
