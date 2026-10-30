# SPDX-License-Identifier: Apache-2.0
"""Disk refits must preserve checkpoint parameters and persistent VAE buffers."""

import hashlib
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.post_training.tensor_update_checker import (
    TensorUpdateChecker,
    build_named_tensor_sha256,
    compute_tensor_sha256,
)
from sglang.multimodal_gen.runtime.post_training.weights_updater import (
    WeightsUpdater,
    _load_weights_into_module,
)
from sglang.multimodal_gen.test.single_test_file.test_update_weights_from_disk import (
    _compute_vae_checksums_from_disk,
)


class _TinyVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2, 2))
        self.norm = torch.nn.BatchNorm1d(2, affine=False)
        self.norm.register_buffer("cache", torch.tensor([17.0]), persistent=False)


def _pipeline(modules, model_path=""):
    return SimpleNamespace(
        modules=modules,
        model_path=str(model_path),
        get_module=modules.get,
    )


def _save_component(model_path, name, state):
    directory = model_path / name
    directory.mkdir(parents=True)
    save_file(state, directory / "diffusion_pytorch_model.safetensors")


def _vae_checkpoint():
    return {
        "weight": torch.full((2, 2), 3.125),
        "norm.running_mean": torch.tensor([1.125, -2.5]),
        "norm.running_var": torch.tensor([4.5, 6.25]),
        "norm.num_batches_tracked": torch.tensor(9, dtype=torch.int64),
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_disk_refit_updates_batchnorm_buffers_and_preserves_integer_dtype(
    tmp_path, dtype
):
    with torch.inference_mode():
        module = _TinyVAE().to(dtype=dtype)
    checkpoint = _vae_checkpoint()
    _save_component(tmp_path, "vae", checkpoint)
    pipeline = _pipeline({"vae": module})

    success, message = WeightsUpdater(pipeline).update_weights_from_disk(str(tmp_path))

    assert success, message
    expected = {
        name: tensor.to(dtype) if tensor.is_floating_point() else tensor
        for name, tensor in checkpoint.items()
    }
    for name, tensor in module.state_dict().items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
    assert module.norm.num_batches_tracked.dtype == torch.int64
    assert module.norm.num_batches_tracked.shape == torch.Size([])
    assert module.norm.cache.item() == 17
    verified, message = TensorUpdateChecker(pipeline).verify(
        "vae", build_named_tensor_sha256(expected.items())
    )
    assert verified, message


def test_refit_does_not_update_nonpersistent_buffers():
    module = _TinyVAE()
    _load_weights_into_module(module, [("norm.cache", torch.tensor([99.0]))])
    assert module.norm.cache.item() == 17


def test_refit_preserves_persistent_alias_of_nonpersistent_buffer():
    module = torch.nn.Module()
    shared = torch.zeros(2)
    module.register_buffer("temporary", shared, persistent=False)
    module.register_buffer("persisted", shared)

    _load_weights_into_module(module, [("persisted", torch.full((2,), 5.0))])

    assert module.persisted is module.temporary
    torch.testing.assert_close(module.persisted, torch.full((2,), 5.0))
    _load_weights_into_module(module, [("temporary", torch.full((2,), 99.0))])
    torch.testing.assert_close(module.persisted, torch.full((2,), 5.0))


def test_vae_disk_manifest_matches_bfloat16_weights_and_integer_buffers(tmp_path):
    checkpoint = _vae_checkpoint()
    _save_component(tmp_path, "vae", checkpoint)
    module = _TinyVAE().to(dtype=torch.bfloat16)
    module.load_state_dict(checkpoint)
    checker = TensorUpdateChecker(_pipeline({"vae": module}))

    manifest = _compute_vae_checksums_from_disk(str(tmp_path))

    assert manifest["norm.num_batches_tracked"] == compute_tensor_sha256(
        checkpoint["norm.num_batches_tracked"]
    )
    verified, message = checker.verify("vae", manifest)
    assert verified, message
    module.norm.running_mean.add_(1)
    verified, message = checker.verify("vae", manifest)
    assert not verified
    assert "checksum mismatch for 1 tensor(s): norm.running_mean" in message


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32, torch.bfloat16])
def test_scalar_checksum_includes_dtype_shape_and_value(dtype):
    scalar = torch.tensor(7, dtype=dtype)
    expected = hashlib.sha256()
    expected.update(str(dtype).encode("utf-8"))
    expected.update(b"()")
    expected.update(scalar.reshape(1).view(torch.uint8).numpy().tobytes())

    assert compute_tensor_sha256(scalar) == expected.hexdigest()
    assert compute_tensor_sha256(scalar) != compute_tensor_sha256(scalar.reshape(1))
    assert compute_tensor_sha256(scalar) != compute_tensor_sha256(scalar + 1)


@pytest.mark.parametrize("mismatch", ["buffer", "missing", "shape", "dtype"])
def test_disk_manifest_rejects_incorrect_or_missing_tensor(mismatch):
    module = _TinyVAE()
    checkpoint = {name: tensor.clone() for name, tensor in module.state_dict().items()}
    if mismatch == "buffer":
        module.norm.running_mean.add_(1)
        expected_error = "checksum mismatch for 1 tensor(s): norm.running_mean"
    elif mismatch == "missing":
        checkpoint["missing_weight"] = torch.ones(1)
        expected_error = "missing 1 tensor(s): missing_weight"
    elif mismatch == "shape":
        checkpoint["weight"] = checkpoint["weight"].reshape(4)
        expected_error = "checksum mismatch for 1 tensor(s): weight"
    else:
        checkpoint["norm.num_batches_tracked"] = checkpoint[
            "norm.num_batches_tracked"
        ].float()
        expected_error = "checksum mismatch for 1 tensor(s): norm.num_batches_tracked"

    verified, message = TensorUpdateChecker(_pipeline({"vae": module})).verify(
        "vae", build_named_tensor_sha256(checkpoint.items())
    )

    assert not verified
    assert expected_error in message


class _OffloadManager:
    enabled = True

    def __init__(self):
        self.weight = torch.zeros(2, 2)

    def update_cpu_weights(self, weights):
        self.weight.copy_(weights["weight"])
        return {"weight"}

    def iter_cpu_weights(self):
        yield "weight", self.weight


class _OffloadedVAE(_TinyVAE, LayerwiseOffloadableModuleMixin):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(1))
        self.layerwise_offload_managers = [_OffloadManager()]


def test_refit_updates_offloaded_parameter_and_resident_buffers():
    module = _OffloadedVAE()
    checkpoint = _vae_checkpoint()

    _load_weights_into_module(module, checkpoint.items())

    assert module.weight.shape == torch.Size([1])
    torch.testing.assert_close(
        module.layerwise_offload_managers[0].weight, checkpoint["weight"]
    )
    torch.testing.assert_close(
        module.norm.running_mean, checkpoint["norm.running_mean"]
    )
    assert module.norm.num_batches_tracked.item() == 9
    verified, message = TensorUpdateChecker(_pipeline({"vae": module})).verify(
        "vae", build_named_tensor_sha256(checkpoint.items())
    )
    assert verified, message


def test_failed_disk_refit_rolls_back_parameters_and_buffers(tmp_path, monkeypatch):
    modules = {"vae": _TinyVAE(), "transformer": torch.nn.Linear(2, 2)}
    original = {
        name: {key: tensor.clone() for key, tensor in module.state_dict().items()}
        for name, module in modules.items()
    }
    base_path = tmp_path / "base"
    for name, state in original.items():
        _save_component(base_path, name, state)
    target_path = tmp_path / "target"
    _save_component(target_path, "vae", _vae_checkpoint())
    _save_component(
        target_path,
        "transformer",
        {"bias": torch.full((2,), 8.0), "weight": torch.zeros(3, 3)},
    )
    pipeline = _pipeline(modules, base_path)
    updater = WeightsUpdater(pipeline)
    rollback = updater._rollback
    rolled_back = []

    def check_partial_update_then_rollback(names):
        # The first component loaded successfully, and the second failed after
        # changing its bias. Both components must return to the old checkpoint.
        assert modules["vae"].norm.num_batches_tracked.item() == 9
        torch.testing.assert_close(
            modules["vae"].norm.running_mean, _vae_checkpoint()["norm.running_mean"]
        )
        torch.testing.assert_close(modules["transformer"].bias, torch.full((2,), 8.0))
        rolled_back.extend(names)
        rollback(names)

    monkeypatch.setattr(updater, "_rollback", check_partial_update_then_rollback)

    success, message = updater.update_weights_from_disk(str(target_path))

    assert not success
    assert "Shape mismatch" in message
    assert rolled_back == ["vae", "transformer"]
    assert pipeline.model_path == str(base_path)
    for name, module in modules.items():
        for key, tensor in module.state_dict().items():
            torch.testing.assert_close(tensor, original[name][key], rtol=0, atol=0)
