"""Tests for diffusion `update_weights_from_disk`.

This module verifies the ability to update model weights in place without restarting
the server, which is critical for RL workflows and iterative fine-tuning scenarios.

Author:

Menyang Liu, https://github.com/dreamyang-liu
Chenyang Zhao, https://github.com/zhaochenyang20

We use two model pairs for testing (base model / instruct model pairs):

- FLUX.2-klein-base-4B / FLUX.2-klein-4B
- Qwen/Qwen-Image / Qwen/Qwen-Image-2512

These model pairs share the same architecture but differ in transformer
weights. The basic testing logic is to refit the instruct model into the
base model and verify the checksum of the transformer weights are the same,
which simulates the real-world RL scenario. However, since these two model
pairs only differ in transformer weights, and we want to verify update a
specific module with update_weights_from_disk API, we need to create a perturbed
instruct model that adds noise to the vae weights. In this sense, the instruct
model differs from the base model in vae and transformer weights, the text
encoder are still the same.

To strictly verify the correctness of the refit API, we compare the checksum in
SHA-256 on the disk and the server.

NOTE and TODO: In the refit a specific module test, we randomly select one module
from the transformer and vae to refit the server and keep other modules the same.
As described above, the vae's weights are perturbed. If we select the vae to be the
target module, ideally speaking, we should assert that the refitted vae's checksum
is the same as directly computed from the perturbed vae weights in the disk. However,
since the there is complex weight-name remapping and QKV merge during model loading,
it is not easy to compare the server-disk checksum for vae and text encoder directly.
Therefore, if the target module is vae, we only verify that the refitted vae's checksum
is different from the base model's vae's checksum.

It should be good issue to solve for the community to adds comparison the server-disk
checksum for vae and text encoder in this test.

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
base model first so behavior is order-independent and updates are real
(base -> perturbed), not no-ops (perturbed -> perturbed).

  • test_update_weights_from_disk_default

    base model -> perturbed model with flush_cache=True.
    Verifies after-update transformer checksum == perturbed model's
    transformer disk checksum


  • test_update_weights_specific_modules

    base -> perturbed with flush_cache=False.  Randomly selects one module
    from _DIFFERING_MODULES (transformer and vae) as target_modules, updates
    only that module. Verifies that:
    (1) targeted module's in-memory checksum changed;
    (2) non-targeted modules' in-memory checksums are unchanged.

  • test_update_weights_nonexistent_model

    model_path set to a non-existent path; must fail (400, success=False).

    Ensure server is healthy after failed update and server's transformer
    checksums equal base model's transformer disk checksum.

  • test_update_weights_missing_model_path

    Request body empty (no model_path); must fail (400, success=False).

    Ensure server is healthy after failed update and server's transformer
    checksums equal base model's transformer disk checksum.

  • test_update_weights_nonexistent_module

    target_modules=["nonexistent_module"]; must fail (400, success=False).

    Verify server is healthy after failed update and server's checksums
    equal base model's transformer disk checksum.

  • test_corrupted_weights_rollback

    All-or-nothing rollback: We first refit the server from base model ->
    perturbed model. We manually truncate the vae weights of the base
    model to get a corrupted model. We then call the refit to update
    the server from the perturbed model -> corrupted model. Verify that:

    1. The update fails due to truncated vae, server should roll back to the
    perturbed model, i.e., server's transformer weights == perturbed model's
    transformer weights != base model's transformer weights.

    2. After the rollback, server's vae weights == perturbed model's vae
    weights != base model's vae weights.

    3. After the rollback, server's text encoder weights == base model's
    text encoder weights == perturbed model's text encoder weights.

-----------------------------------------------------------------------------

Class 2: TestUpdateWeightsFromDiskWithOffload


Ensure weight updates and checksum verification work when layerwise offload is enabled
(--dit-layerwise-offload). With offload, parameters live in CPU buffers and only left
small torch.empty((1,)) as placeholders on GPU; the updater must write into CPU buffers
and update prefetched GPU tensors without shape mismatch.

  • test_update_weights_with_offload_enabled

    Server with --dit-layerwise-offload (base). Load perturbed checkpoint;
    must succeed (200, success=True), no "Shape mismatch". server's transformer checksum
    matches perturbed model's transformer disk checksum.
"""

from __future__ import annotations

import functools
import os
import random
import shutil
import tempfile
import threading
from collections.abc import Callable

import pytest
import requests
from safetensors.torch import load_file, save_file

from sglang.multimodal_gen.runtime.loader.utils import (
    _list_safetensors_files,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    compute_weights_checksum,
    safetensors_weights_iterator,
)
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.server.test_server_utils import (
    ServerManager,
)
from sglang.multimodal_gen.test.test_utils import get_dynamic_server_port, is_in_ci

logger = init_logger(__name__)


_TRANSFORMER_MODULE = "transformer"
_VAE_MODULE = "vae"
_TEXT_ENCODER_MODULE_PREFIX = "text_encoder"


# Modules whose weights differ between the base model and the perturbed
# perturbed checkpoint
_DIFFERING_MODULES: list[str] = [_TRANSFORMER_MODULE, _VAE_MODULE]

_ALL_MODEL_PAIRS: list[tuple[str, str]] = [
    (
        "black-forest-labs/FLUX.2-klein-base-4B",
        "black-forest-labs/FLUX.2-klein-4B",
    ),
    (
        "Qwen/Qwen-Image",
        "Qwen/Qwen-Image-2512",
    ),
]


_CI_MODEL_PAIR_ENV = "SGLANG_MMGEN_UPDATE_WEIGHTS_PAIR"


def _resolve_active_model_pairs() -> list[tuple[str, str]]:
    if not is_in_ci():
        return _ALL_MODEL_PAIRS

    pair_by_id = {pair[0].split("/")[-1]: pair for pair in _ALL_MODEL_PAIRS}
    selected_pair_id = os.environ.get(_CI_MODEL_PAIR_ENV)
    if selected_pair_id is None:
        return [random.choice(_ALL_MODEL_PAIRS)]

    selected_pair = pair_by_id.get(selected_pair_id)
    if selected_pair is None:
        valid_ids = ", ".join(sorted(pair_by_id))
        raise ValueError(
            f"Invalid {_CI_MODEL_PAIR_ENV}={selected_pair_id!r}. "
            f"Expected one of: {valid_ids}."
        )
    return [selected_pair]


_ACTIVE_MODEL_PAIRS = _resolve_active_model_pairs()
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
    weights_dir = os.path.join(local_path, module_name)
    assert os.path.exists(
        weights_dir
    ), f"No weights dir for {module_name} in {local_path}"

    safetensors_files = _list_safetensors_files(weights_dir)
    assert safetensors_files, f"No safetensors files in {weights_dir}"

    return compute_weights_checksum(safetensors_weights_iterator(safetensors_files))


def _clone_model_with_modified_module(
    src_model: str,
    dst_model: str,
    target_module: str,
    transform_safetensor: Callable[[str, str], None],
) -> None:
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

        if module_dir != target_module:
            if not os.path.exists(dst_dir):
                os.symlink(src_dir, dst_dir)
            continue

        os.makedirs(dst_dir, exist_ok=True)
        transformed = False
        for fname in sorted(os.listdir(src_dir)):
            src_file = os.path.join(src_dir, fname)
            dst_file = os.path.join(dst_dir, fname)
            if not os.path.isfile(src_file):
                continue

            if not fname.endswith(".safetensors") or transformed:
                if not os.path.exists(dst_file):
                    os.symlink(src_file, dst_file)
                continue

            transform_safetensor(src_file, dst_file)
            transformed = True


def _truncate_safetensor(src_file: str, dst_file: str) -> None:
    shutil.copy2(src_file, dst_file)
    size = os.path.getsize(dst_file)
    with open(dst_file, "r+b") as f:
        f.truncate(size - 2)
    logger.info(
        "Created corrupted safetensors: %s (%d -> %d bytes)",
        dst_file,
        size,
        size - 2,
    )


def _perturb_safetensor(src_file: str, dst_file: str) -> None:

    tensors = load_file(src_file)
    perturbed = {
        k: (t + 0.01 if t.is_floating_point() else t) for k, t in tensors.items()
    }
    save_file(perturbed, dst_file)
    logger.info("Created perturbed safetensors: %s", dst_file)


class _UpdateWeightsApiMixin:
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

    def _assert_server_matches_model(
        self,
        base_url: str,
        expected_model: str,
    ) -> None:
        server_checksums = self._get_weights_checksum(
            base_url, module_names=[_TRANSFORMER_MODULE]
        )
        expected_cs = _compute_checksum_from_disk(expected_model, _TRANSFORMER_MODULE)
        server_cs = server_checksums.get(_TRANSFORMER_MODULE)
        assert server_cs == expected_cs, (
            f"Checksum mismatch on '{_TRANSFORMER_MODULE}'\n"
            f"  expected({expected_model}): {expected_cs}\n"
            f"  server: {server_cs}"
        )


class TestUpdateWeightsFromDisk(_UpdateWeightsApiMixin):

    @pytest.fixture(
        scope="class",
        params=_ACTIVE_MODEL_PAIRS,
        ids=_PAIR_IDS,
    )
    def diffusion_server_no_offload(self, request):
        default_model, source_model = request.param
        port = get_dynamic_server_port()
        wait_deadline = float(os.environ.get("SGLANG_TEST_WAIT_SECS", "600"))

        manager = ServerManager(
            model=default_model,
            port=port,
            wait_deadline=wait_deadline,
            extra_args="--num-gpus 1",
        )

        # Ensure models are local before spawning threads that need the paths.
        local_default = maybe_download_model(default_model)
        local_source = maybe_download_model(source_model)

        perturbed_vae_model_dir = tempfile.mkdtemp(prefix="sglang_perturbed_vae_")
        corrupted_vae_model_dir = tempfile.mkdtemp(prefix="sglang_corrupted_")

        # Run all disk I/O in background while the server boots.
        bg_threads = [
            threading.Thread(
                target=_compute_checksum_from_disk, args=(default_model, module)
            )
            for module in _DIFFERING_MODULES
        ] + [
            threading.Thread(
                target=_clone_model_with_modified_module,
                args=(
                    local_source,
                    perturbed_vae_model_dir,
                    _VAE_MODULE,
                    _perturb_safetensor,
                ),
            ),
            threading.Thread(
                target=_clone_model_with_modified_module,
                args=(
                    local_default,
                    corrupted_vae_model_dir,
                    _VAE_MODULE,
                    _truncate_safetensor,
                ),
            ),
        ]
        for t in bg_threads:
            t.start()

        ctx = manager.start()
        for t in bg_threads:
            t.join()

        # Sanity: all _DIFFERING_MODULES should differ between base and perturbed.
        for module in _DIFFERING_MODULES:
            assert _compute_checksum_from_disk(
                default_model, module
            ) != _compute_checksum_from_disk(perturbed_vae_model_dir, module), (
                f"Assumption violated: {module} should differ between "
                f"{default_model} and {perturbed_vae_model_dir}"
            )

        try:
            yield ctx, default_model, perturbed_vae_model_dir, corrupted_vae_model_dir
        finally:
            ctx.cleanup()
            shutil.rmtree(perturbed_vae_model_dir, ignore_errors=True)
            shutil.rmtree(corrupted_vae_model_dir, ignore_errors=True)

    def test_update_weights_from_disk_default(self, diffusion_server_no_offload):
        """Default update (target_modules=None, flush_cache=True): all changed modules updated."""
        ctx, default_model, perturbed_model_dir, _ = diffusion_server_no_offload
        base_url = f"http://localhost:{ctx.port}"

        self._update_weights(base_url, default_model, flush_cache=True)

        result, status_code = self._update_weights(
            base_url, perturbed_model_dir, flush_cache=True
        )
        assert status_code == 200
        assert result.get("success", False), f"Update failed: {result.get('message')}"

        self._assert_server_matches_model(base_url, perturbed_model_dir)

    def test_update_weights_specific_modules(self, diffusion_server_no_offload):
        ctx, default_model, perturbed_model_dir, _ = diffusion_server_no_offload
        base_url = f"http://localhost:{ctx.port}"

        # Reset server to default_model.
        self._update_weights(base_url, default_model)
        before_checksums = self._get_weights_checksum(
            base_url, module_names=_DIFFERING_MODULES
        )

        target_modules = [random.choice(_DIFFERING_MODULES)]
        result, status_code = self._update_weights(
            base_url,
            perturbed_model_dir,
            target_modules=target_modules,
            flush_cache=False,
        )
        assert status_code == 200, f"Update failed: {result}"
        assert result.get("success", False), f"Update failed: {result.get('message')}"

        after_checksums = self._get_weights_checksum(
            base_url, module_names=_DIFFERING_MODULES
        )

        # Targeted module should have changed.
        for name in target_modules:
            assert after_checksums.get(name) != before_checksums.get(name), (
                f"Targeted module '{name}' checksum should change after update\n"
                f"  before: {before_checksums.get(name)}\n"
                f"  after:  {after_checksums.get(name)}"
            )

        # Non-targeted modules should be unchanged.
        for name, cs in after_checksums.items():
            if name in target_modules or cs == "not_found":
                continue
            assert cs == before_checksums.get(name), (
                f"Non-targeted module '{name}' should be unchanged\n"
                f"  before: {before_checksums.get(name)}\n"
                f"  after:  {cs}"
            )

    def test_update_weights_nonexistent_model(self, diffusion_server_no_offload):
        """Nonexistent model path must fail (400). Server healthy, checksums == base disk."""
        ctx, default_model, _, _ = diffusion_server_no_offload
        base_url = f"http://localhost:{ctx.port}"

        self._update_weights(base_url, default_model)

        result, status_code = self._update_weights(
            base_url,
            "/nonexistent/path/to/model",
            timeout=60,
        )
        logger.info(f"Update result for nonexistent model: {result}")

        assert status_code == 400, f"Expected 400, got {status_code}"
        assert not result.get("success", True), "Should fail for nonexistent model"
        self._assert_server_matches_model(base_url, default_model)

    def test_update_weights_missing_model_path(self, diffusion_server_no_offload):
        """Request without model_path must fail (400). Server healthy, checksums == base disk."""
        ctx, default_model, _, _ = diffusion_server_no_offload
        base_url = f"http://localhost:{ctx.port}"

        self._update_weights(base_url, default_model)

        response = requests.post(
            f"{base_url}/update_weights_from_disk",
            json={},
            timeout=30,
        )

        assert response.status_code == 400, f"Expected 400, got {response.status_code}"
        result = response.json()
        assert not result.get("success", True), "Should fail when model_path is missing"
        self._assert_server_matches_model(base_url, default_model)

    def test_update_weights_nonexistent_module(self, diffusion_server_no_offload):
        """Nonexistent module must fail (400). Server healthy, checksums == base disk."""
        ctx, default_model, perturbed_model_dir, _ = diffusion_server_no_offload
        base_url = f"http://localhost:{ctx.port}"

        self._update_weights(base_url, default_model)

        result, status_code = self._update_weights(
            base_url,
            perturbed_model_dir,
            target_modules=["nonexistent_module"],
            timeout=60,
        )
        logger.info(f"Update nonexistent module result: {result}")

        assert status_code == 400, f"Expected 400, got {status_code}"
        assert not result.get("success", True), "Should fail for nonexistent module"
        assert "not found in pipeline" in result.get("message", "")
        self._assert_server_matches_model(base_url, default_model)

    def test_corrupted_weights_rollback(self, diffusion_server_no_offload):
        ctx, default_model, perturbed_model_dir, corrupted_vae_model_dir = (
            diffusion_server_no_offload
        )
        base_url = f"http://localhost:{ctx.port}"

        # base → perturbed
        self._update_weights(base_url, default_model)
        base_checksums = self._get_weights_checksum(base_url)

        result, status_code = self._update_weights(base_url, perturbed_model_dir)
        assert status_code == 200 and result.get("success")
        perturbed_checksums = self._get_weights_checksum(base_url)

        text_encoder_modules = sorted(
            name
            for name in perturbed_checksums
            if _TEXT_ENCODER_MODULE_PREFIX in name
            and perturbed_checksums.get(name) != "not_found"
            and base_checksums.get(name) != "not_found"
        )
        assert (
            text_encoder_modules
        ), "Expected at least one text encoder module checksum"

        # perturbed → corrupted (should fail and rollback)
        rollback_targets = [_TRANSFORMER_MODULE, _VAE_MODULE]
        result, status_code = self._update_weights(
            base_url,
            corrupted_vae_model_dir,
            target_modules=rollback_targets,
        )
        assert (
            status_code == 400
        ), f"Expected 400 on corrupted weights, got {status_code}"
        assert not result.get("success", True)
        message = result.get("message", "")
        assert "rolled back" in message.lower()
        # The updater reports the first failing module in the error message.
        # With ordered target_modules=[transformer, vae], this makes the
        # failure point explicit: transformer is processed first, then vae fails.
        assert (
            "Failed to update module 'vae'" in message
        ), f"Expected vae to be the explicit failure point, got: {message}"
        rolled_back_checksums = self._get_weights_checksum(base_url)

        # 1) transformer: server == perturbed != base
        transformer_base = base_checksums.get(_TRANSFORMER_MODULE)
        transformer_perturbed = perturbed_checksums.get(_TRANSFORMER_MODULE)
        transformer_rolled_back = rolled_back_checksums.get(_TRANSFORMER_MODULE)
        assert transformer_rolled_back == transformer_perturbed
        assert transformer_rolled_back != transformer_base

        # 2) vae: server == perturbed != base
        vae_base = base_checksums.get(_VAE_MODULE)
        vae_perturbed = perturbed_checksums.get(_VAE_MODULE)
        vae_rolled_back = rolled_back_checksums.get(_VAE_MODULE)
        assert vae_rolled_back == vae_perturbed
        assert vae_rolled_back != vae_base

        # 3) text encoder(s): server == base == perturbed
        for name in text_encoder_modules:
            assert rolled_back_checksums.get(name) == perturbed_checksums.get(
                name
            ), f"Text encoder module '{name}' should stay equal to perturbed"
            assert rolled_back_checksums.get(name) == base_checksums.get(
                name
            ), f"Text encoder module '{name}' should stay equal to base"


class TestUpdateWeightsFromDiskWithOffload(_UpdateWeightsApiMixin):
    """Test update_weights_from_disk with layerwise offload enabled."""

    @pytest.fixture(scope="class", params=_ACTIVE_MODEL_PAIRS, ids=_PAIR_IDS)
    def diffusion_server_with_offload(self, request):
        default_model, source_model = request.param
        port = get_dynamic_server_port()
        wait_deadline = float(os.environ.get("SGLANG_TEST_WAIT_SECS", "600"))

        local_source = maybe_download_model(source_model)
        perturbed_vae_model_dir = tempfile.mkdtemp(prefix="sglang_perturbed_vae_")

        clone_thread = threading.Thread(
            target=_clone_model_with_modified_module,
            args=(
                local_source,
                perturbed_vae_model_dir,
                _VAE_MODULE,
                _perturb_safetensor,
            ),
        )
        clone_thread.start()

        manager = ServerManager(
            model=default_model,
            port=port,
            wait_deadline=wait_deadline,
            extra_args="--num-gpus 1 --dit-layerwise-offload true",
        )

        ctx = manager.start()
        clone_thread.join()

        try:
            yield ctx, default_model, perturbed_vae_model_dir
        finally:
            ctx.cleanup()
            shutil.rmtree(perturbed_vae_model_dir, ignore_errors=True)

    def test_update_weights_with_offload_enabled(self, diffusion_server_with_offload):
        ctx, _, perturbed_model_dir = diffusion_server_with_offload
        base_url = f"http://localhost:{ctx.port}"

        result, status_code = self._update_weights(base_url, perturbed_model_dir)
        assert status_code == 200, f"Expected 200, got {status_code}"
        assert result.get("success", False), f"Update failed: {result.get('message')}"

        message = result.get("message", "")
        assert "Shape mismatch" not in message, f"Shape mismatch detected: {message}"

        self._assert_server_matches_model(base_url, perturbed_model_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
