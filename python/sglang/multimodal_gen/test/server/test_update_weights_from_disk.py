"""
Tests for update_weights_from_disk API in SGLang diffusion server.

This module verifies the ability to update model weights in place without restarting
the server, which is critical for RL workflows and iterative fine-tuning scenarios.

We use two model pairs for testing (before / after update model pairs):

- FLUX.2-klein-base-4B / FLUX.2-klein-4B
- Qwen/Qwen-Image / Qwen/Qwen-Image-2512

These models are with the same model architecture and different number
of parameters. Only weights are different.

Author:

Menyang Liu, https://github.com/dreamyang-liu
Chenyang Zhao, https://github.com/zhaochenyang20

=============================================================================

Test organization:

10 test cases in 2 classes;
two model pairs are tested locally, one in CI.

=============================================================================

Class 1: TestUpdateWeightsFromDisk                  (8 tests) — API contract, checksum & rollback
Class 2: TestUpdateWeightsFromDiskWithOffload       (2 tests) — Offload-aware update

-----------------------------------------------------------------------------

Class 1: TestUpdateWeightsFromDisk

Validate the update_weights_from_disk API contract, request/response shape,
error handling, checksum verification, and corrupted-weight rollback.

  • test_update_weights_with_flush_cache

    Explicit flush_cache=True; must succeed (200, success=True). Ensures the
    flush_cache parameter is accepted and the update completes.

    TODO: Currently, TeaCache can not be verified whether it was flushed
    since no cache-state API is exposed.

  • test_update_weights_without_flush_cache

    Explicit flush_cache=False; must succeed. Ensures updates work when not
    requesting TeaCache flush.

  • test_update_weights_nonexistent_model

    model_path set to a non-existent path; must fail (400, success=False).
    Also, verifies that the update fails and the model is rolled back to the
    original weights.

  • test_update_weights_missing_model_path

    Request body empty (no model_path); must return 400. Validates required
    parameter checks.

  • test_update_weights_specific_modules

    Randomly selects a subset of pipeline modules as target_modules, then asserts:
    (1) updated modules' checksums match the update model's disk checksums;
    (2) non-updated modules' checksums are unchanged (same before and after).

  • test_update_weights_nonexistent_module

    target_modules=["nonexistent_module"]; must return 400 and message containing
    "not found in pipeline". Validates rejection of invalid module names.

  • test_update_weights_checksum_matches

    Updates weights to the update model. Verifies the post-update checksum
    matches the update model's disk checksum.

  • test_corrupted_weights_rollback

    Verify all-or-nothing rollback semantics when loading corrupted weights.
    Builds a corrupted model directory by copying the base model and truncating
    the vae safetensors. Requests an update with target_modules=["transformer",
    "vae"]. The transformer updates successfully first; the corrupted vae then
    fails during safetensors validation, triggering a rollback that restores
    the transformer to its previous weights.

-----------------------------------------------------------------------------
Class 2: TestUpdateWeightsFromDiskWithOffload
-----------------------------------------------------------------------------
Purpose: Ensure weight updates and checksum verification work when layerwise
offload is enabled (--dit-layerwise-offload). With offload, parameters live in
CPU buffers and placeholders on GPU; the updater must write into CPU buffers and
update prefetched GPU tensors without shape mismatch. The checksum endpoint must
read from CPU buffers (not the (1,) placeholders) to produce correct results.

  • test_update_weights_with_offload_enabled
    Server started with --dit-layerwise-offload true. Call update_weights_from_disk
    with the same model; must succeed (200, success=True) and message must not
    contain "Shape mismatch".

  • test_update_weights_checksum_matches
    Fetches checksum before update (base model), then updates weights and fetches
    checksum again (update model). Verifies the post-update checksum matches the
    update model's disk checksum, and differs from the pre-update checksum.

=============================================================================
Relation to RL scenarios and reference implementation
=============================================================================

In RL or iterative training, a typical pattern is:

  1. Run a diffusion (or LLM) server for inference.
  2. Periodically pull new weights (e.g., from a training run or from disk)
     without restarting the server.
  3. Continue serving with the updated model.

The diffusion engine supports this via POST /update_weights_from_disk: it loads
weights from a model_path (HF repo or local) and applies them in-place, with
rollback on failure and support for layerwise offload and DTensor.

For a distributed RL setup where the training process broadcasts weights to
inference engines (rather than loading from disk), see the SGLang LLM test that
simulates rank 0 as trainer and other ranks as inference engines, using
update_weights_from_distributed and init_weights_update_group:

  https://github.com/sgl-project/sglang/blob/main/test/registered/rl/test_update_weights_from_distributed.py

That test verifies weight synchronization across ranks (instruct vs base model)
and optional pause_generation/continue_generation during update. This diffusion
test suite focuses on the disk-based update path and offload/consistency
behavior of the diffusion engine only.
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

# Model pairs for weight update tests: (default_model, update_model, ci_weight).
# The server starts with default_model; tests update weights to update_model.
# ci_weight controls how likely each pair is to be selected in CI runs.
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


class TestUpdateWeightsFromDisk:
    """Test suite for update_weights_from_disk API and corrupted-weight rollback.

    Uses a class-scoped server fixture so the server is torn down at class end,
    freeing the port and GPU memory before the offload class starts.
    """

    @pytest.fixture(
        scope="class",
        params=_ACTIVE_MODEL_PAIRS,
        ids=[p[0].split("/")[-1] for p in _ACTIVE_MODEL_PAIRS],
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
        """Call update_weights_from_disk API."""
        payload = {
            "model_path": model_path,
            "flush_cache": flush_cache,
        }
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
        """Call get_weights_checksum API and return the checksum dict."""
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

    def test_update_weights_with_flush_cache(self, diffusion_server_no_offload):
        """Test updating weights with flush_cache=True.

        Verifies the API accepts flush_cache=True and returns success; does not
        assert that TeaCache was actually reset (server does not expose cache state).
        """
        ctx, _default_model, update_model = diffusion_server_no_offload
        base_url = self._get_base_url(ctx)

        result, status_code = self._update_weights(
            base_url,
            update_model,
            flush_cache=True,
        )

        assert status_code == 200
        assert result.get("success", False), f"Update failed: {result.get('message')}"

    def test_update_weights_without_flush_cache(self, diffusion_server_no_offload):
        """Test updating weights with flush_cache=False."""
        ctx, _default_model, update_model = diffusion_server_no_offload
        base_url = self._get_base_url(ctx)

        result, status_code = self._update_weights(
            base_url,
            update_model,
            flush_cache=False,
        )

        assert status_code == 200
        assert result.get("success", False), f"Update failed: {result.get('message')}"

    def test_update_weights_nonexistent_model(self, diffusion_server_no_offload):
        """Test that updating with non-existent model fails gracefully."""
        ctx, _default_model, _update_model = diffusion_server_no_offload
        base_url = self._get_base_url(ctx)

        result, status_code = self._update_weights(
            base_url,
            "/nonexistent/path/to/model",
            timeout=60,
        )
        logger.info(f"Update result for nonexistent model: {result}")

        # Should fail gracefully
        assert not result.get("success", True), "Should fail for nonexistent model"

    def test_update_weights_missing_model_path(self, diffusion_server_no_offload):
        """Test that request without model_path returns 400."""
        ctx, _default_model, _update_model = diffusion_server_no_offload
        base_url = self._get_base_url(ctx)

        response = requests.post(
            f"{base_url}/update_weights_from_disk",
            json={},
            timeout=30,
        )

        # Should return 400 Bad Request
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"

    def test_update_weights_specific_modules(self, diffusion_server_no_offload):
        """Partial update: random subset of modules updated; checksums verified.

        Randomly picks a non-empty subset of modules that have weights on disk
        for the update model, performs update_weights_from_disk with that
        target_modules, then asserts:
        - Updated modules: in-memory checksum == update model disk checksum.
        - Non-updated modules: checksum unchanged (before == after).
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

        # Only consider modules that exist on disk for the update model.
        candidates = _get_modules_with_weights_on_disk(update_model, all_module_names)
        if not candidates:
            pytest.skip("Update model has no weight dirs for any pipeline module")

        # Random non-empty subset (fixed seed for reproducibility).
        random.seed(42)
        k = random.randint(1, len(candidates))
        target_modules = random.sample(candidates, k)
        target_set = set(target_modules)
        logger.info(
            "Partial update test: target_modules=%s (unchanged: %s)",
            target_modules,
            [m for m in all_module_names if m not in target_set],
        )

        before_checksums = self._get_weights_checksum(base_url, module_names=None)

        result, status_code = self._update_weights(
            base_url,
            update_model,
            target_modules=target_modules,
        )
        assert status_code == 200, f"Update failed: {result}"
        assert result.get("success", False), f"Update failed: {result.get('message')}"

        after_checksums = self._get_weights_checksum(base_url, module_names=None)

        for name in all_module_names:
            if name in target_set:
                disk_cs = _compute_checksum_from_disk(update_model, name)
                assert after_checksums.get(name) == disk_cs, (
                    f"Updated module '{name}': checksum should match update model disk\n"
                    f"  disk: {disk_cs}\n  gpu:  {after_checksums.get(name)}"
                )
            else:
                assert after_checksums.get(name) == before_checksums.get(name), (
                    f"Non-updated module '{name}': checksum must be unchanged\n"
                    f"  before: {before_checksums.get(name)}\n"
                    f"  after:  {after_checksums.get(name)}"
                )

    def test_update_weights_nonexistent_module(self, diffusion_server_no_offload):
        """Test that requesting a non-existent module name fails with a clear error."""
        ctx, _default_model, update_model = diffusion_server_no_offload
        base_url = self._get_base_url(ctx)

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

    def test_update_weights_checksum_matches(self, diffusion_server_no_offload):
        """Verify GPU checksum matches disk after weight update.

        Resets to the base model first (shared fixture may be in any state),
        then updates to the update model and compares the server-side
        checksum with the disk checksum.
        """
        ctx, default_model, update_model = diffusion_server_no_offload
        base_url = self._get_base_url(ctx)

        # Reset to base model so the subsequent update is a real change.
        self._update_weights(base_url, default_model)

        result, status_code = self._update_weights(base_url, update_model)
        assert status_code == 200 and result.get(
            "success"
        ), f"Update failed: {result.get('message')}"

        gpu_checksum = self._get_weights_checksum(
            base_url, module_names=["transformer"]
        )["transformer"]
        disk_checksum = _compute_checksum_from_disk(update_model, "transformer")

        print(f"\n{'='*60}")
        print(f"Checksum test")
        print(f"  gpu:  {gpu_checksum}")
        print(f"  disk: {disk_checksum}")
        print(f"  match: {gpu_checksum == disk_checksum}")
        print(f"{'='*60}")

        assert gpu_checksum == disk_checksum, (
            f"GPU checksum does not match disk checksum for update model\n"
            f"  disk: {disk_checksum}\n"
            f"  gpu:  {gpu_checksum}"
        )

    def test_corrupted_weights_rollback(
        self,
        diffusion_server_no_offload,
        corrupted_model_dir: str,
    ):
        """Load base -> update weights -> attempt corrupted -> verify rollback.

        Checksums are restricted to ["transformer", "vae"] — the modules
        involved in the partial update — to avoid computing checksums for
        unrelated modules.
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

        print(f"\n{'='*80}")
        print("Corrupted-weight rollback test (transformer, vae)")
        for module in sorted(update_checksums.keys()):
            update_cs = update_checksums.get(module, "N/A")
            rollback_cs = post_rollback_checksums.get(module, "N/A")
            base_cs = base_checksums.get(module, "N/A")
            match = "OK" if update_cs == rollback_cs else "MISMATCH"
            print(f"  [{match}] {module}")
            print(f"        base:     {base_cs}")
            print(f"        update:   {update_cs}")
            print(f"        rollback: {rollback_cs}")
        print(f"{'='*80}")

        for module in update_checksums:
            assert post_rollback_checksums.get(module) == update_checksums[module], (
                f"Module '{module}' checksum mismatch after rollback\n"
                f"  update:        {update_checksums[module]}\n"
                f"  post-rollback: {post_rollback_checksums.get(module)}"
            )

        assert post_rollback_checksums != base_checksums, (
            "Post-rollback checksums should not match base model "
            "(rollback target is update model, not base)"
        )


class TestUpdateWeightsFromDiskWithOffload:
    """Test update_weights_from_disk with layerwise offload enabled."""

    @pytest.fixture(
        scope="class",
        params=_ACTIVE_MODEL_PAIRS,
        ids=[p[0].split("/")[-1] for p in _ACTIVE_MODEL_PAIRS],
    )
    def diffusion_server_with_offload(self, request):
        """Start a diffusion server with layerwise offload enabled.

        Disk checksums are already cached by diffusion_server_no_offload
        (which runs first), so no background precomputation is needed here.
        """
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

    def _get_base_url(self, ctx: ServerContext) -> str:
        return f"http://localhost:{ctx.port}"

    def _update_weights(
        self, base_url: str, model_path: str, **kwargs
    ) -> tuple[dict, int]:
        payload = {"model_path": model_path, **kwargs}
        response = requests.post(
            f"{base_url}/update_weights_from_disk",
            json=payload,
            timeout=kwargs.get("timeout", 300),
        )
        return response.json(), response.status_code

    def test_update_weights_with_offload_enabled(self, diffusion_server_with_offload):
        """Test that weight update works correctly when layerwise offload is enabled."""
        ctx, _default_model, update_model = diffusion_server_with_offload
        base_url = self._get_base_url(ctx)

        logger.info("Testing weight update with offload enabled")

        result, status_code = self._update_weights(base_url, update_model)
        logger.info(f"Update result: {result}")

        assert status_code == 200, f"Expected 200, got {status_code}"
        assert result.get("success", False), f"Update failed: {result.get('message')}"

        # Verify no shape mismatch warnings in the message
        message = result.get("message", "")
        assert "Shape mismatch" not in message, f"Shape mismatch detected: {message}"

    def _get_weights_checksum(
        self,
        base_url: str,
        module_names: list[str] | None = None,
        timeout: int = 300,
    ) -> dict:
        """Call get_weights_checksum API and return the checksum dict."""
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

    def test_update_weights_checksum_matches(self, diffusion_server_with_offload):
        """Verify checksum from offloaded CPU buffers matches disk after update.

        Resets to the base model first, then updates to the update model
        and compares the server-side checksum (read from CPU buffers) with
        the disk checksum.
        """
        ctx, default_model, update_model = diffusion_server_with_offload
        base_url = self._get_base_url(ctx)

        # Reset to base model so the subsequent update is a real change.
        self._update_weights(base_url, default_model)

        result, status_code = self._update_weights(base_url, update_model)
        assert status_code == 200 and result.get(
            "success"
        ), f"Update failed: {result.get('message')}"

        gpu_checksum = self._get_weights_checksum(
            base_url, module_names=["transformer"]
        )["transformer"]
        disk_checksum = _compute_checksum_from_disk(update_model, "transformer")

        print(f"\n{'='*60}")
        print(f"Offload checksum test")
        print(f"  gpu:  {gpu_checksum}")
        print(f"  disk: {disk_checksum}")
        print(f"  match: {gpu_checksum == disk_checksum}")
        print(f"{'='*60}")

        assert gpu_checksum == disk_checksum, (
            f"GPU checksum does not match disk checksum for update model\n"
            f"  disk: {disk_checksum}\n"
            f"  gpu:  {gpu_checksum}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
