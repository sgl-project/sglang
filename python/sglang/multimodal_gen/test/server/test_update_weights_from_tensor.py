"""Tests for diffusion `update_weights_from_tensor`.

This module verifies in-place weight updates from serialized in-memory tensors
without restarting the server.

Test goals:
- API contract and request validation for `/update_weights_from_tensor`
- module-targeted updates (`target_modules`) for transformer/vae
- failure reporting on corrupted tensor update
- flattened-bucket payload compatibility
- offload compatibility (`--dit-layerwise-offload`)

Model-pair selection in CI follows `test_update_weights_from_disk.py`:
- local: run both pairs
- CI: run one pair (random unless `SGLANG_MMGEN_UPDATE_WEIGHTS_PAIR` is set)
"""

import os
import random
import subprocess
from dataclasses import dataclass

import pytest
import requests
import torch

from sglang.multimodal_gen.runtime.loader.utils import _list_safetensors_files
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    compute_weights_checksum,
    safetensors_weights_iterator,
)
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.test.server.test_server_utils import ServerManager
from sglang.multimodal_gen.test.test_utils import (
    DEFAULT_FLUX_2_KLEIN_4B_MODEL_NAME_FOR_TEST,
    DEFAULT_FLUX_2_KLEIN_BASE_4B_MODEL_NAME_FOR_TEST,
    DEFAULT_QWEN_IMAGE_2512_MODEL_NAME_FOR_TEST,
    DEFAULT_QWEN_IMAGE_MODEL_NAME_FOR_TEST,
    get_dynamic_server_port,
    is_in_ci,
)
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

_TRANSFORMER_MODULE = "transformer"

_ALL_MODEL_PAIRS: list[tuple[str, str]] = [
    (
        DEFAULT_FLUX_2_KLEIN_BASE_4B_MODEL_NAME_FOR_TEST,
        DEFAULT_FLUX_2_KLEIN_4B_MODEL_NAME_FOR_TEST,
    ),
    (
        DEFAULT_QWEN_IMAGE_MODEL_NAME_FOR_TEST,
        DEFAULT_QWEN_IMAGE_2512_MODEL_NAME_FOR_TEST,
    ),
]

_CI_MODEL_PAIR_ENV = "SGLANG_MMGEN_UPDATE_WEIGHTS_PAIR"
_BUCKET_SIZE_ENV = "SGLANG_MMGEN_TENSOR_BUCKET_SIZE_BYTES"
_DIT_CPU_OFFLOAD_ENV = "SGLANG_MMGEN_TEST_DIT_CPU_OFFLOAD"


@dataclass
class _NamedTensorUpdate:
    name: str
    tensor: torch.Tensor


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


def _pick_params_from_disk(
    model_path: str,
    module_name: str,
    num_params: int,
    max_numel: int | None = None,
    min_numel: int | None = None,
) -> list[_NamedTensorUpdate]:
    """Pick ``num_params`` parameter tensors from one module on disk.

    Preference order:
    1) floating-point parameters (so ``+delta`` is meaningful)
    2) non-floating parameters as fallback if float params are insufficient
    """
    assert num_params > 0, "num_params must be > 0"
    local_path = maybe_download_model(model_path)
    weights_dir = os.path.join(local_path, module_name)
    assert os.path.exists(
        weights_dir
    ), f"No weights dir for module '{module_name}' in {local_path}"

    safetensors_files = _list_safetensors_files(weights_dir)
    assert safetensors_files, f"No safetensors files in {weights_dir}"

    picked_small_float: list[_NamedTensorUpdate] = []
    picked_small_nonfloat: list[_NamedTensorUpdate] = []
    picked_large_float: list[_NamedTensorUpdate] = []
    picked_large_nonfloat: list[_NamedTensorUpdate] = []
    for name, tensor in safetensors_weights_iterator(safetensors_files):
        if min_numel is not None and tensor.numel() < min_numel:
            continue
        is_small = max_numel is None or tensor.numel() <= max_numel
        if tensor.is_floating_point():
            target = picked_small_float if is_small else picked_large_float
            target.append(_NamedTensorUpdate(name=name, tensor=tensor.clone()))
        else:
            target = picked_small_nonfloat if is_small else picked_large_nonfloat
            target.append(_NamedTensorUpdate(name=name, tensor=tensor.clone()))

    selected: list[_NamedTensorUpdate] = []
    for bucket in (
        picked_small_float,
        picked_small_nonfloat,
        picked_large_float,
        picked_large_nonfloat,
    ):
        if len(selected) >= num_params:
            break
        selected.extend(bucket[: (num_params - len(selected))])

    assert selected, (
        f"No tensor found in module '{module_name}' "
        f"with constraints min_numel={min_numel}, max_numel={max_numel}"
    )
    return selected


def _pick_one_param_from_disk(model_path: str, module_name: str) -> _NamedTensorUpdate:
    return _pick_params_from_disk(model_path, module_name, num_params=1)[0]


def _pick_params_by_names_from_disk(
    model_path: str,
    module_name: str,
    param_names: list[str],
) -> list[_NamedTensorUpdate]:
    """Pick tensors by exact names and keep caller-provided order."""
    assert param_names, "param_names must be non-empty"
    local_path = maybe_download_model(model_path)
    weights_dir = os.path.join(local_path, module_name)
    safetensors_files = _list_safetensors_files(weights_dir)
    assert safetensors_files, f"No safetensors files in {weights_dir}"

    wanted = set(param_names)
    found: dict[str, torch.Tensor] = {}
    for name, tensor in safetensors_weights_iterator(safetensors_files):
        if name in wanted:
            found[name] = tensor.clone()
            if len(found) == len(wanted):
                break
    missing = [name for name in param_names if name not in found]
    assert (
        not missing
    ), f"Requested param(s) not found in module '{module_name}': {missing}"
    return [_NamedTensorUpdate(name=name, tensor=found[name]) for name in param_names]


def _build_direct_update(
    model_path: str,
    module_name: str,
    value: float = 2.0,
    value_sequence: list[float] | None = None,
    num_params: int = 5,
    max_numel: int | None = None,
    min_numel: int | None = None,
    param_names: list[str] | None = None,
) -> list[tuple[str, torch.Tensor]]:
    if param_names is not None:
        picked_params = _pick_params_by_names_from_disk(
            model_path=model_path,
            module_name=module_name,
            param_names=param_names,
        )
    else:
        picked_params = _pick_params_from_disk(
            model_path=model_path,
            module_name=module_name,
            num_params=num_params,
            max_numel=max_numel,
            min_numel=min_numel,
        )
    if value_sequence is not None:
        assert len(value_sequence) >= len(picked_params), (
            f"value_sequence length ({len(value_sequence)}) must be >= "
            f"number of picked params ({len(picked_params)})"
        )
    named_tensors: list[tuple[str, torch.Tensor]] = []
    for i, picked in enumerate(picked_params):
        cur_value = value_sequence[i] if value_sequence is not None else value
        t = picked.tensor.to(device="cuda")
        if t.is_floating_point():
            updated = torch.full_like(t, cur_value)
        else:
            updated = torch.full_like(t, int(cur_value))
        named_tensors.append((picked.name, updated))
    return named_tensors


def _build_invalid_shape_update(
    model_path: str,
    module_name: str,
) -> list[tuple[str, torch.Tensor]]:
    picked = _pick_one_param_from_disk(model_path, module_name)
    t = picked.tensor.to(device="cuda")
    assert t.numel() > 1, "Need tensor with >1 elements to create invalid shape payload"

    bad = t.reshape(-1)[:-1].clone()
    return [(picked.name, bad)]


def _compute_expected_checksum_after_direct_update(
    model_path: str,
    module_name: str,
    named_tensors: list[tuple[str, torch.Tensor]],
) -> str:
    """Compute expected module checksum after applying direct named_tensors update."""
    local_path = maybe_download_model(model_path)
    weights_dir = os.path.join(local_path, module_name)
    safetensors_files = _list_safetensors_files(weights_dir)
    assert safetensors_files, f"No safetensors files in {weights_dir}"

    updates = {name: tensor.detach().to(device="cpu") for name, tensor in named_tensors}
    found: set[str] = set()

    def _iter_expected():
        for name, tensor in safetensors_weights_iterator(safetensors_files):
            if name in updates:
                found.add(name)
                yield name, updates[name]
            else:
                yield name, tensor

    checksum = compute_weights_checksum(_iter_expected())
    missing = sorted(set(updates.keys()) - found)
    assert (
        not missing
    ), f"Updated parameter(s) not found in module '{module_name}': {missing}"
    return checksum


def _iter_module_named_tensors(
    model_path: str,
    module_name: str,
):
    """Iterate all (name, tensor) pairs under a specific module directory."""
    local_path = maybe_download_model(model_path)
    weights_dir = os.path.join(local_path, module_name)
    safetensors_files = _list_safetensors_files(weights_dir)
    assert safetensors_files, f"No safetensors files in {weights_dir}"
    yield from safetensors_weights_iterator(safetensors_files)


def _kill_existing_sglang_serve_processes() -> None:
    """Best-effort cleanup to avoid stale local servers affecting tests."""
    subprocess.run(
        ["bash", "-lc", 'pkill -f "sglang serve" || true'],
        check=False,
    )


class _UpdateWeightsFromTensorApiMixin:
    def _update_weights_from_disk(
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

    def _update_weights_from_tensor(
        self,
        base_url: str,
        named_tensors,
        load_format: str | None = None,
        target_modules: list[str] | None = None,
        weight_version: str | None = None,
        serialized_payloads: list[str] | None = None,
        timeout: int = 300,
    ) -> tuple[dict, int]:
        payload = {
            "serialized_named_tensors": (
                serialized_payloads
                if serialized_payloads is not None
                else [
                    MultiprocessingSerializer.serialize(named_tensors, output_str=True)
                ]
            )
        }
        if load_format is not None:
            payload["load_format"] = load_format
        if target_modules is not None:
            payload["target_modules"] = target_modules
        if weight_version is not None:
            payload["weight_version"] = weight_version

        response = requests.post(
            f"{base_url}/update_weights_from_tensor",
            json=payload,
            timeout=timeout,
        )
        response_json = response.json()
        return response_json, response.status_code

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

    def _reset_to_base_model(self, base_url: str, default_model: str) -> None:
        result, status_code = self._update_weights_from_disk(
            base_url,
            default_model,
            flush_cache=True,
        )
        assert status_code == 200, f"Failed to reset to base model: {result}"
        assert result.get("success", False), f"Failed to reset to base model: {result}"


class TestUpdateWeightsFromTensor(_UpdateWeightsFromTensorApiMixin):
    @pytest.fixture(
        scope="class",
        params=_ACTIVE_MODEL_PAIRS,
        ids=_PAIR_IDS,
    )
    def diffusion_server_no_offload(self, request):
        default_model, _ = request.param
        port = get_dynamic_server_port()
        wait_deadline = float(os.environ.get("SGLANG_TEST_WAIT_SECS", "600"))
        extra_args = "--num-gpus 1"
        dit_cpu_offload_env = os.environ.get(_DIT_CPU_OFFLOAD_ENV)
        if dit_cpu_offload_env is not None and dit_cpu_offload_env.lower() in (
            "0",
            "false",
            "no",
        ):
            extra_args += " --dit-cpu-offload false"

        maybe_download_model(default_model)
        _kill_existing_sglang_serve_processes()

        manager = ServerManager(
            model=default_model,
            port=port,
            wait_deadline=wait_deadline,
            extra_args=extra_args,
        )
        ctx = manager.start()

        try:
            yield ctx, default_model
        finally:
            ctx.cleanup()

    def test_update_weights_from_tensor_direct(self, diffusion_server_no_offload):
        ctx, default_model = diffusion_server_no_offload
        base_url = f"http://localhost:{ctx.port}"

        self._reset_to_base_model(base_url, default_model)
        before = self._get_weights_checksum(base_url)

        payload = _build_direct_update(default_model, _TRANSFORMER_MODULE, value=2.0)
        expected_transformer_checksum = _compute_expected_checksum_after_direct_update(
            model_path=default_model,
            module_name=_TRANSFORMER_MODULE,
            named_tensors=payload,
        )
        result, status_code = self._update_weights_from_tensor(
            base_url,
            named_tensors=payload,
            target_modules=[_TRANSFORMER_MODULE],
        )

        assert status_code == 200, f"Expected 200, got {status_code}: {result}"
        assert result.get("success", False), f"Update failed: {result.get('message')}"

        after = self._get_weights_checksum(base_url)
        assert set(after.keys()) == set(before.keys()), (
            "Module set changed unexpectedly after update.\n"
            f"before_only={sorted(set(before.keys()) - set(after.keys()))}\n"
            f"after_only={sorted(set(after.keys()) - set(before.keys()))}"
        )
        assert after.get(_TRANSFORMER_MODULE) == expected_transformer_checksum, (
            f"Expected transformer checksum to match direct-update payload\n"
            f"  expected: {expected_transformer_checksum}\n"
            f"  actual:   {after.get(_TRANSFORMER_MODULE)}"
        )
        for name in sorted(after.keys()):
            if name == _TRANSFORMER_MODULE:
                continue
            assert after.get(name) == before.get(name), (
                f"Non-targeted module '{name}' should be unchanged\n"
                f"  before: {before.get(name)}\n"
                f"  after:  {after.get(name)}"
            )

    def test_corrupted_tensor_update_fails(self, diffusion_server_no_offload):
        ctx, default_model = diffusion_server_no_offload
        base_url = f"http://localhost:{ctx.port}"

        self._reset_to_base_model(base_url, default_model)
        # Corrupted payload: transformer has invalid shape.
        corrupted_payload = _build_invalid_shape_update(
            default_model, _TRANSFORMER_MODULE
        )
        result, status_code = self._update_weights_from_tensor(
            base_url,
            named_tensors=corrupted_payload,
            target_modules=[_TRANSFORMER_MODULE],
        )
        assert (
            status_code == 400
        ), f"Expected 400 on corrupted update, got {status_code}"
        assert not result.get("success", True)
        message = result.get("message", "")
        assert "Failed to update module 'transformer'" in message
        assert "partially updated" in message.lower()

    def test_update_weights_from_tensor_flattened_bucket_full_transformer(
        self, diffusion_server_no_offload
    ):
        ctx, default_model = diffusion_server_no_offload
        base_url = f"http://localhost:{ctx.port}"

        # Ensure the server starts from the base model.
        self._reset_to_base_model(base_url, default_model)
        base_transformer_checksum = self._get_weights_checksum(
            base_url, module_names=[_TRANSFORMER_MODULE]
        ).get(_TRANSFORMER_MODULE)
        assert base_transformer_checksum, "Missing base transformer checksum"

        # Perturb weights first so this test validates a real restore path.
        perturb_payload = _build_direct_update(
            default_model, _TRANSFORMER_MODULE, value=11.0, num_params=3
        )
        perturb_result, perturb_status = self._update_weights_from_tensor(
            base_url,
            named_tensors=perturb_payload,
            target_modules=[_TRANSFORMER_MODULE],
            timeout=600,
        )
        assert (
            perturb_status == 200
        ), f"Failed to perturb transformer before restore test: {perturb_result}"
        assert perturb_result.get(
            "success", False
        ), f"Failed to perturb transformer before restore test: {perturb_result}"
        perturbed_checksum = self._get_weights_checksum(
            base_url, module_names=[_TRANSFORMER_MODULE]
        ).get(_TRANSFORMER_MODULE)
        assert (
            perturbed_checksum != base_transformer_checksum
        ), "Precondition failed: perturb update did not change transformer checksum"

        bucket_size_bytes = int(
            os.environ.get(_BUCKET_SIZE_ENV, str(4096 * 1024 * 1024))
        )

        current_bucket: list[tuple[str, torch.Tensor]] = []
        current_bucket_bytes = 0

        def _flush_bucket():
            nonlocal current_bucket, current_bucket_bytes
            if not current_bucket:
                return

            bucket = FlattenedTensorBucket(named_tensors=current_bucket)
            bucket_payload = {
                _TRANSFORMER_MODULE: {
                    "flattened_tensor": bucket.get_flattened_tensor(),
                    "metadata": bucket.get_metadata(),
                }
            }
            result, status_code = self._update_weights_from_tensor(
                base_url,
                named_tensors=bucket_payload,
                load_format="flattened_bucket",
                target_modules=[_TRANSFORMER_MODULE],
                timeout=1800,
            )
            assert (
                status_code == 200
            ), f"Bucket update failed, expected 200 got {status_code}: {result}"
            assert result.get(
                "success", False
            ), f"Bucket update failed: {result.get('message')}"
            current_bucket = []
            current_bucket_bytes = 0
            torch.cuda.empty_cache()

        for name, tensor in _iter_module_named_tensors(
            default_model, _TRANSFORMER_MODULE
        ):
            tensor_cuda = tensor.to(device="cuda")
            tensor_bytes = tensor_cuda.numel() * tensor_cuda.element_size()

            if (
                current_bucket
                and current_bucket_bytes + tensor_bytes > bucket_size_bytes
            ):
                _flush_bucket()

            current_bucket.append((name, tensor_cuda))
            current_bucket_bytes += tensor_bytes

        _flush_bucket()

        restored_checksum = self._get_weights_checksum(
            base_url, module_names=[_TRANSFORMER_MODULE]
        ).get(_TRANSFORMER_MODULE)
        assert restored_checksum == base_transformer_checksum, (
            "Full-transformer flattened-bucket restore checksum mismatch\n"
            f"  expected(base): {base_transformer_checksum}\n"
            f"  actual:         {restored_checksum}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
