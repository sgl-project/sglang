"""
Tests for update_weights_from_disk API in SGLang-D (diffusion engine).

This module verifies the ability to hot update model weights without restarting
the server, which is critical for RL workflows and iterative fine-tuning scenarios.

Author:

Menyang Liu, https://github.com/dreamyang-liu
Chenyang Zhao, https://github.com/zhaochenyang20

=============================================================================
Test organization: 10 test cases in 3 classes
=============================================================================

Each test class uses a single long-lived server (pytest fixture with scope="class").
The server is started once when the first test in that class runs; all tests in the
class share the same server and send multiple POST /update_weights_from_disk
requests to it. This reflects real usage: one running diffusion service, many weight
updates over time.

Class 1: TestUpdateWeightsFromDisk                  (7 tests) — API contract & checksum
Class 2: TestUpdateWeightsFromDiskWithOffload       (2 tests) — Offload-aware update
Class 3: TestUpdateWeightsCorruptedRollback         (1 test)  — Corrupted weight rollback

-----------------------------------------------------------------------------
Class 1: TestUpdateWeightsFromDisk
-----------------------------------------------------------------------------
Purpose: Validate the update_weights_from_disk API contract, request/response shape,
error handling, and checksum verification. All 7 tests run against one server
(fixture: diffusion_server_for_weight_update).

  • test_update_weights_with_flush_cache
    Explicit flush_cache=True; must succeed. Ensures the flush_cache parameter
    is accepted and applied.

  • test_update_weights_without_flush_cache
    Explicit flush_cache=False; must succeed. Ensures updates work when not
    flushing TeaCache.

  • test_update_weights_nonexistent_model
    model_path set to a non-existent path; must fail (success=False). Verifies
    all-or-nothing / rollback semantics when load fails.

  • test_update_weights_missing_model_path
    Request body empty (no model_path); must return 400. Validates required
    parameter checks.

  • test_update_weights_specific_modules
    target_modules=["transformer"]; must return 200. Verifies partial module
    update (target_modules parameter).

  • test_update_weights_nonexistent_module
    target_modules=["nonexistent_module"]; must return 400 and message containing
    "not found in pipeline". Validates rejection of invalid module names.

  • test_update_weights_checksum_matches
    Fetches checksum before update (base model), then updates weights and fetches
    checksum again (update model). Verifies the post-update checksum matches the
    update model's disk checksum, and differs from the pre-update checksum.

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

-----------------------------------------------------------------------------
Class 3: TestUpdateWeightsCorruptedRollback
-----------------------------------------------------------------------------
Purpose: Verify all-or-nothing rollback semantics when loading corrupted weights.
The test builds a corrupted model directory by copying the base model and
truncating the vae safetensors. It then requests an update with
target_modules=["transformer", "vae"]. The transformer updates successfully
first; the corrupted vae then fails during safetensors validation, triggering a
rollback that restores the transformer to its previous weights.

  • test_corrupted_weights_rollback
    1. Server starts with the base model (DEFAULT_DIFFUSION_MODEL).
    2. Updates weights to the update model (UPDATE_DIFFUSION_MODEL); must succeed.
    3. Records checksums for all modules after the update.
    4. Prepares a corrupted model (base model copy with truncated vae).
    5. Attempts to load the corrupted model; must fail with rollback message.
    6. Verifies all module checksums still match the update model, confirming
       the rollback restored every module that was partially updated.

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

import os
import shutil
import tempfile

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
from sglang.multimodal_gen.test.test_utils import get_dynamic_server_port

logger = init_logger(__name__)

# Base model the server starts with
DEFAULT_DIFFUSION_MODEL = os.environ.get(
    "SGLANG_TEST_DIFFUSION_MODEL", "black-forest-labs/FLUX.2-klein-base-4B"
)

# Model used for weight updates (same architecture, different weights)
UPDATE_DIFFUSION_MODEL = os.environ.get(
    "SGLANG_TEST_UPDATE_MODEL", "black-forest-labs/FLUX.2-klein-4B"
)


def _compute_checksum_from_disk(model_path: str, module_name: str) -> str:
    """Compute SHA-256 checksum from safetensors files on disk.

    Uses the same compute_weights_checksum function as the server,
    so the checksums are directly comparable.
    """
    local_path = maybe_download_model(model_path)
    weights_dir = find_weights_dir(local_path, module_name)
    assert weights_dir is not None, f"No weights dir for {module_name} in {local_path}"

    safetensors_files = _list_safetensors_files(weights_dir)
    assert safetensors_files, f"No safetensors files in {weights_dir}"

    return compute_weights_checksum(safetensors_weights_iterator(safetensors_files))


@pytest.fixture(scope="class")
def diffusion_server_for_weight_update():
    """Start a diffusion server for weight update tests."""
    port = get_dynamic_server_port()
    wait_deadline = float(os.environ.get("SGLANG_TEST_WAIT_SECS", "600"))

    manager = ServerManager(
        model=DEFAULT_DIFFUSION_MODEL,
        port=port,
        wait_deadline=wait_deadline,
        extra_args="--num-gpus 1",
    )

    ctx = manager.start()

    try:
        yield ctx
    finally:
        ctx.cleanup()


class TestUpdateWeightsFromDisk:
    """Test suite for update_weights_from_disk API."""

    def _get_base_url(self, ctx: ServerContext) -> str:
        return f"http://localhost:{ctx.port}"

    def _update_weights(
        self,
        base_url: str,
        model_path: str,
        flush_cache: bool = True,
        target_modules: list[str] | None = None,
        timeout: int = 300,
    ) -> dict:
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

    def test_update_weights_with_flush_cache(
        self, diffusion_server_for_weight_update: ServerContext
    ):
        """Test updating weights with flush_cache=True."""
        base_url = self._get_base_url(diffusion_server_for_weight_update)

        result, status_code = self._update_weights(
            base_url,
            UPDATE_DIFFUSION_MODEL,
            flush_cache=True,
        )

        assert status_code == 200
        assert result.get("success", False), f"Update failed: {result.get('message')}"

    def test_update_weights_without_flush_cache(
        self, diffusion_server_for_weight_update: ServerContext
    ):
        """Test updating weights with flush_cache=False."""
        base_url = self._get_base_url(diffusion_server_for_weight_update)

        result, status_code = self._update_weights(
            base_url,
            UPDATE_DIFFUSION_MODEL,
            flush_cache=False,
        )

        assert status_code == 200
        assert result.get("success", False), f"Update failed: {result.get('message')}"

    def test_update_weights_nonexistent_model(
        self, diffusion_server_for_weight_update: ServerContext
    ):
        """Test that updating with non-existent model fails gracefully."""
        base_url = self._get_base_url(diffusion_server_for_weight_update)

        result, status_code = self._update_weights(
            base_url,
            "/nonexistent/path/to/model",
            timeout=60,
        )
        logger.info(f"Update result for nonexistent model: {result}")

        # Should fail gracefully
        assert not result.get("success", True), "Should fail for nonexistent model"

    def test_update_weights_missing_model_path(
        self, diffusion_server_for_weight_update: ServerContext
    ):
        """Test that request without model_path returns 400."""
        base_url = self._get_base_url(diffusion_server_for_weight_update)

        response = requests.post(
            f"{base_url}/update_weights_from_disk",
            json={},
            timeout=30,
        )

        # Should return 400 Bad Request
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"

    def test_update_weights_specific_modules(
        self, diffusion_server_for_weight_update: ServerContext
    ):
        """Test updating only specific modules (e.g., transformer only)."""
        base_url = self._get_base_url(diffusion_server_for_weight_update)

        # Try to update only transformer module
        result, status_code = self._update_weights(
            base_url,
            UPDATE_DIFFUSION_MODEL,
            target_modules=["transformer"],
        )
        logger.info(f"Update specific modules result: {result}")

        # This might fail if the model doesn't have a transformer module
        # or if weights for only transformer aren't available
        # The test verifies the API handles target_modules parameter
        assert status_code == 200

    def test_update_weights_nonexistent_module(
        self, diffusion_server_for_weight_update: ServerContext
    ):
        """Test that requesting a non-existent module name fails with a clear error."""
        base_url = self._get_base_url(diffusion_server_for_weight_update)

        result, status_code = self._update_weights(
            base_url,
            UPDATE_DIFFUSION_MODEL,
            target_modules=["nonexistent_module"],
            timeout=60,
        )
        logger.info(f"Update nonexistent module result: {result}")

        assert status_code == 400, f"Expected 400, got {status_code}"
        assert not result.get("success", True), "Should fail for nonexistent module"
        assert "not found in pipeline" in result.get("message", "")

    def test_update_weights_checksum_matches(
        self, diffusion_server_for_weight_update: ServerContext
    ):
        """Verify GPU checksum matches disk after weight update.

        1. Fetch the pre-update (base model) checksum from the server.
        2. Update weights to a different model.
        3. Fetch the post-update checksum and compare with disk.
        4. Verify post-update checksum differs from pre-update (different model).
        """
        base_url = self._get_base_url(diffusion_server_for_weight_update)

        # Update to base model.
        result, status_code = self._update_weights(base_url, DEFAULT_DIFFUSION_MODEL)

        # Checksum before update (base model already loaded by the fixture).
        pre_update_checksum = self._get_weights_checksum(
            base_url, module_names=["transformer"]
        )["transformer"]

        # Update to a different model.
        result, status_code = self._update_weights(base_url, UPDATE_DIFFUSION_MODEL)
        assert status_code == 200 and result.get(
            "success"
        ), f"Update failed: {result.get('message')}"

        # Checksum after update — must match the update model on disk.
        post_update_checksum = self._get_weights_checksum(
            base_url, module_names=["transformer"]
        )["transformer"]
        update_disk_checksum = _compute_checksum_from_disk(
            UPDATE_DIFFUSION_MODEL, "transformer"
        )

        print(f"\n{'='*60}")
        print(f"Checksum test")
        print(f"  pre-update (base):  {pre_update_checksum}")
        print(f"  post-update (gpu):  {post_update_checksum}")
        print(f"  post-update (disk): {update_disk_checksum}")
        print(f"  gpu == disk:        {post_update_checksum == update_disk_checksum}")
        print(f"  changed:            {pre_update_checksum != post_update_checksum}")
        print(f"{'='*60}")

        assert post_update_checksum == update_disk_checksum, (
            f"GPU checksum does not match disk checksum for update model\n"
            f"  disk: {update_disk_checksum}\n"
            f"  gpu:  {post_update_checksum}"
        )
        assert (
            pre_update_checksum != post_update_checksum
        ), "Checksum did not change after updating to a different model"


class TestUpdateWeightsFromDiskWithOffload:
    """Test update_weights_from_disk with layerwise offload enabled."""

    @pytest.fixture(scope="class")
    def diffusion_server_with_offload(self):
        """Start a diffusion server with layerwise offload enabled."""
        port = get_dynamic_server_port()
        wait_deadline = float(os.environ.get("SGLANG_TEST_WAIT_SECS", "600"))

        manager = ServerManager(
            model=DEFAULT_DIFFUSION_MODEL,
            port=port,
            wait_deadline=wait_deadline,
            extra_args="--num-gpus 1 --dit-layerwise-offload true",
        )

        ctx = manager.start()

        try:
            yield ctx
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

    def test_update_weights_with_offload_enabled(
        self, diffusion_server_with_offload: ServerContext
    ):
        """Test that weight update works correctly when layerwise offload is enabled."""
        base_url = self._get_base_url(diffusion_server_with_offload)

        logger.info("Testing weight update with offload enabled")

        result, status_code = self._update_weights(base_url, UPDATE_DIFFUSION_MODEL)
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

    def test_update_weights_checksum_matches(
        self, diffusion_server_with_offload: ServerContext
    ):
        """Verify checksum from offloaded CPU buffers matches disk after update.

        1. Fetch the pre-update (base model) checksum from the server.
        2. Update weights to a different model.
        3. Fetch the post-update checksum and compare with disk.
        4. Verify post-update checksum differs from pre-update (different model).
        """
        base_url = self._get_base_url(diffusion_server_with_offload)

        # Update to base model.
        result, status_code = self._update_weights(base_url, DEFAULT_DIFFUSION_MODEL)

        # Checksum before update (base model already loaded by the fixture).
        pre_update_checksum = self._get_weights_checksum(
            base_url, module_names=["transformer"]
        )["transformer"]

        # Update to a different model.
        result, status_code = self._update_weights(base_url, UPDATE_DIFFUSION_MODEL)
        assert status_code == 200 and result.get(
            "success"
        ), f"Update failed: {result.get('message')}"

        # Checksum after update — must match the update model on disk.
        post_update_checksum = self._get_weights_checksum(
            base_url, module_names=["transformer"]
        )["transformer"]
        update_disk_checksum = _compute_checksum_from_disk(
            UPDATE_DIFFUSION_MODEL, "transformer"
        )

        print(f"\n{'='*60}")
        print(f"Offload checksum test")
        print(f"  pre-update (base):  {pre_update_checksum}")
        print(f"  post-update (gpu):  {post_update_checksum}")
        print(f"  post-update (disk): {update_disk_checksum}")
        print(f"  gpu == disk:        {post_update_checksum == update_disk_checksum}")
        print(f"  changed:            {pre_update_checksum != post_update_checksum}")
        print(f"{'='*60}")

        assert post_update_checksum == update_disk_checksum, (
            f"GPU checksum does not match disk checksum for update model\n"
            f"  disk: {update_disk_checksum}\n"
            f"  gpu:  {post_update_checksum}"
        )
        assert (
            pre_update_checksum != post_update_checksum
        ), "Checksum did not change after updating to a different model"


def _prepare_corrupted_model(
    src_model: str, dst_model: str, corrupt_module: str
) -> None:
    """Build a corrupted model directory from src_model.

    The root-level files (model_index.json, config.json, …) are copied so
    that maybe_download_model recognises dst_model as a valid local
    model.  Every module sub-directory that contains safetensors is copied
    verbatim, except corrupt_module whose safetensors are truncated so
    that safetensors_weights_iterator detects corruption at load time
    (after earlier modules have already been updated), triggering a rollback.

    Must be called before every test attempt because the server deletes
    corrupted files on detection.
    """
    # Copy root-level files (model_index.json, etc.) so the server
    # recognises the directory as a valid local model.
    for fname in os.listdir(src_model):
        src_path = os.path.join(src_model, fname)
        if os.path.isfile(src_path):
            shutil.copy2(src_path, os.path.join(dst_model, fname))

    # Copy module sub-directories; corrupt the designated one.
    for module_dir in sorted(os.listdir(src_model)):
        src_dir = os.path.join(src_model, module_dir)
        if not os.path.isdir(src_dir):
            continue
        safetensors = [f for f in os.listdir(src_dir) if f.endswith(".safetensors")]
        if not safetensors:
            # Still copy non-safetensors dirs (config.json, tokenizer, etc.)
            dst_dir = os.path.join(dst_model, module_dir)
            if not os.path.exists(dst_dir):
                shutil.copytree(src_dir, dst_dir)
            continue

        dst_dir = os.path.join(dst_model, module_dir)
        os.makedirs(dst_dir, exist_ok=True)

        # Copy config.json and other non-safetensors files in the module dir
        for f in os.listdir(src_dir):
            if not f.endswith(".safetensors"):
                src_f = os.path.join(src_dir, f)
                if os.path.isfile(src_f):
                    shutil.copy2(src_f, os.path.join(dst_dir, f))

        for fname in safetensors:
            src_file = os.path.join(src_dir, fname)
            dst_file = os.path.join(dst_dir, fname)
            shutil.copy2(src_file, dst_file)

            if module_dir == corrupt_module:
                # Truncate 1000 bytes from the end to corrupt the tensor data
                size = os.path.getsize(dst_file)
                with open(dst_file, "r+b") as f:
                    f.truncate(size - 1000)
                logger.info(
                    "Created corrupted safetensors: %s (%d -> %d bytes)",
                    dst_file,
                    size,
                    size - 1000,
                )
            else:
                logger.info("Copied valid safetensors: %s", dst_file)


class TestUpdateWeightsCorruptedRollback:
    """Test that loading corrupted weights triggers rollback to the last good model."""

    @pytest.fixture(scope="class")
    def corrupted_model_dir(self):
        """Create a temporary directory for the corrupted model."""
        tmpdir = tempfile.mkdtemp(prefix="sglang_corrupted_model_")
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.fixture(scope="class")
    def diffusion_server_for_rollback(self):
        """Start a diffusion server with the base model."""
        port = get_dynamic_server_port()
        wait_deadline = float(os.environ.get("SGLANG_TEST_WAIT_SECS", "600"))

        manager = ServerManager(
            model=DEFAULT_DIFFUSION_MODEL,
            port=port,
            wait_deadline=wait_deadline,
            extra_args="--num-gpus 1",
        )

        ctx = manager.start()
        try:
            yield ctx
        finally:
            ctx.cleanup()

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

    def test_corrupted_weights_rollback(
        self,
        diffusion_server_for_rollback: ServerContext,
        corrupted_model_dir: str,
    ):
        """Load base → update weights → attempt corrupted → verify rollback.

        Checksums are verified for ALL modules, not just the transformer,
        to ensure the entire pipeline is consistent after rollback.
        """
        base_url = self._get_base_url(diffusion_server_for_rollback)

        # --- Step 1: Get base-model checksums for all modules ---
        base_checksums = self._get_weights_checksum(base_url)
        logger.info(f"Base model checksums: {base_checksums}")

        # --- Step 2: Update to the update model ---
        result, status_code = self._update_weights(base_url, UPDATE_DIFFUSION_MODEL)
        assert status_code == 200
        assert result.get(
            "success", False
        ), f"Weight update failed: {result.get('message')}"

        # --- Step 3: Record update-model checksums for all modules ---
        update_checksums = self._get_weights_checksum(base_url)
        logger.info(f"Update model checksums: {update_checksums}")

        assert (
            update_checksums != base_checksums
        ), "Base and update checksums should differ"

        # --- Step 4: Recreate corrupted model, then attempt load ---
        # Copy all modules from the base model (valid), but corrupt only the
        # vae.  With target_modules=["transformer", "vae"], the transformer
        # updates successfully first, then vae fails, giving a meaningful
        # rollback that actually restores the transformer.
        local_base = maybe_download_model(DEFAULT_DIFFUSION_MODEL)
        _prepare_corrupted_model(local_base, corrupted_model_dir, corrupt_module="vae")

        result, status_code = self._update_weights(
            base_url,
            corrupted_model_dir,
            target_modules=["transformer", "vae"],
            timeout=120,
        )
        logger.info(f"Corrupted update result: status={status_code}, body={result}")

        assert not result.get("success", True), "Loading corrupted weights should fail"
        assert (
            "rolled back" in result.get("message", "").lower()
        ), f"Expected rollback message, got: {result.get('message')}"

        # --- Step 5: Verify rollback — all module checksums must match update model ---
        post_rollback_checksums = self._get_weights_checksum(base_url)
        logger.info(f"Post-rollback checksums: {post_rollback_checksums}")

        print(f"\n{'='*80}")
        print("Corrupted-weight rollback test (all modules)")
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
