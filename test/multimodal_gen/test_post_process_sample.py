import contextlib
import logging
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = PROJECT_ROOT / "python"
for path in (PYTHON_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Stub optional dependencies to avoid heavy imports during test collection.
diffusers_module = types.ModuleType("diffusers")
diffusers_module.__version__ = "0.31.0"
sys.modules.setdefault("diffusers", diffusers_module)

diffusers_image_processor = types.ModuleType("diffusers.image_processor")
diffusers_image_processor.VaeImageProcessor = object
sys.modules.setdefault("diffusers.image_processor", diffusers_image_processor)
diffusers_loaders = types.ModuleType("diffusers.loaders")
sys.modules.setdefault("diffusers.loaders", diffusers_loaders)
diffusers_lora_base = types.ModuleType("diffusers.loaders.lora_base")
diffusers_lora_base._best_guess_weight_name = lambda *_, **__: "weights.safetensors"
sys.modules.setdefault("diffusers.loaders.lora_base", diffusers_lora_base)
diffusers_module.loaders = diffusers_loaders
diffusers_models = types.ModuleType("diffusers.models")
diffusers_models_autoencoders = types.ModuleType("diffusers.models.autoencoders")
diffusers_models_autoencoders_vae = types.ModuleType(
    "diffusers.models.autoencoders.vae"
)
diffusers_models_autoencoders_vae.DiagonalGaussianDistribution = object
sys.modules.setdefault("diffusers.models", diffusers_models)
sys.modules.setdefault("diffusers.models.autoencoders", diffusers_models_autoencoders)
sys.modules.setdefault(
    "diffusers.models.autoencoders.vae", diffusers_models_autoencoders_vae
)
diffusers_module.models = diffusers_models
diffusers_utils = types.ModuleType("diffusers.utils")
diffusers_utils_torch = types.ModuleType("diffusers.utils.torch_utils")
diffusers_utils_torch.randn_tensor = lambda *args, **kwargs: torch.randn(
    *args, **kwargs
)
sys.modules.setdefault("diffusers.utils", diffusers_utils)
sys.modules.setdefault("diffusers.utils.torch_utils", diffusers_utils_torch)
sys.modules.setdefault(
    "imageio",
    SimpleNamespace(
        imwrite=lambda *_, **__: None,
        mimsave=lambda *_, **__: None,
    ),
)
sys.modules.setdefault("remote_pdb", SimpleNamespace(RemotePdb=SimpleNamespace))
sgl_kernel_module = types.ModuleType("sgl_kernel")
sgl_kernel_flash_attn = types.ModuleType("sgl_kernel.flash_attn")
sgl_kernel_flash_attn.flash_attn_varlen_func = lambda *_, **__: None
sys.modules.setdefault("sgl_kernel", sgl_kernel_module)
sys.modules.setdefault("sgl_kernel.flash_attn", sgl_kernel_flash_attn)


def _load_sampling_params():
    import importlib.util

    sglang_module = sys.modules.setdefault("sglang", types.ModuleType("sglang"))
    multimodal_gen_module = sys.modules.setdefault(
        "sglang.multimodal_gen", types.ModuleType("sglang.multimodal_gen")
    )
    sglang_module.multimodal_gen = multimodal_gen_module

    runtime_module = sys.modules.setdefault(
        "sglang.multimodal_gen.runtime",
        types.ModuleType("sglang.multimodal_gen.runtime"),
    )
    multimodal_gen_module.runtime = runtime_module
    server_args_module = types.ModuleType("sglang.multimodal_gen.runtime.server_args")

    class _ServerArgs:
        host: str | None = None

        def __init__(self, *_, **__):
            pass

    server_args_module.ServerArgs = _ServerArgs
    sys.modules["sglang.multimodal_gen.runtime.server_args"] = server_args_module
    runtime_module.server_args = server_args_module
    runtime_utils_module = types.ModuleType("sglang.multimodal_gen.runtime.utils")
    runtime_module.utils = runtime_utils_module
    logging_utils_module = types.ModuleType(
        "sglang.multimodal_gen.runtime.utils.logging_utils"
    )

    def _dummy_logger(name=None):
        return logging.getLogger(name)

    @contextlib.contextmanager
    def _noop_timer(*_args, **_kwargs):
        yield

    logging_utils_module.init_logger = _dummy_logger
    logging_utils_module.log_batch_completion = lambda *_, **__: None
    logging_utils_module.log_generation_timer = _noop_timer
    sys.modules["sglang.multimodal_gen.runtime.utils.logging_utils"] = (
        logging_utils_module
    )
    runtime_utils_module.logging_utils = logging_utils_module

    configs_module = sys.modules.setdefault(
        "sglang.multimodal_gen.configs",
        types.ModuleType("sglang.multimodal_gen.configs"),
    )
    multimodal_gen_module.configs = configs_module

    sample_module = sys.modules.setdefault(
        "sglang.multimodal_gen.configs.sample",
        types.ModuleType("sglang.multimodal_gen.configs.sample"),
    )
    configs_module.sample = sample_module
    utils_module = types.ModuleType("sglang.multimodal_gen.utils")
    utils_module.align_to = lambda value, align: value
    sys.modules["sglang.multimodal_gen.utils"] = utils_module
    multimodal_gen_module.utils = utils_module

    sampling_spec = importlib.util.spec_from_file_location(
        "sglang.multimodal_gen.configs.sample.sampling_params",
        PROJECT_ROOT / "python/sglang/multimodal_gen/configs/sample/sampling_params.py",
    )
    sampling_module = importlib.util.module_from_spec(sampling_spec)
    assert sampling_spec.loader is not None
    sampling_spec.loader.exec_module(sampling_module)
    sys.modules["sglang.multimodal_gen.configs.sample.sampling_params"] = (
        sampling_module
    )
    sample_module.sampling_params = sampling_module

    return sampling_module.DataType


def _load_openai_utils():
    import importlib.util

    DataType = _load_sampling_params()

    sglang_module = sys.modules["sglang"]
    multimodal_gen_module = sys.modules["sglang.multimodal_gen"]
    runtime_module = sys.modules.setdefault(
        "sglang.multimodal_gen.runtime",
        types.ModuleType("sglang.multimodal_gen.runtime"),
    )
    multimodal_gen_module.runtime = runtime_module

    runtime_utils_module = sys.modules.setdefault(
        "sglang.multimodal_gen.runtime.utils",
        types.ModuleType("sglang.multimodal_gen.runtime.utils"),
    )
    runtime_module.utils = runtime_utils_module

    logging_utils_module = types.ModuleType(
        "sglang.multimodal_gen.runtime.utils.logging_utils"
    )

    def _dummy_logger(name=None):
        return logging.getLogger(name)

    @contextlib.contextmanager
    def _noop_timer(*_args, **_kwargs):
        yield

    logging_utils_module.init_logger = _dummy_logger
    logging_utils_module.log_batch_completion = lambda *_, **__: None
    logging_utils_module.log_generation_timer = _noop_timer
    sys.modules["sglang.multimodal_gen.runtime.utils.logging_utils"] = (
        logging_utils_module
    )
    runtime_utils_module.logging_utils = logging_utils_module

    entrypoints_module = sys.modules.setdefault(
        "sglang.multimodal_gen.runtime.entrypoints",
        types.ModuleType("sglang.multimodal_gen.runtime.entrypoints"),
    )
    runtime_module.entrypoints = entrypoints_module

    openai_pkg = sys.modules.setdefault(
        "sglang.multimodal_gen.runtime.entrypoints.openai",
        types.ModuleType("sglang.multimodal_gen.runtime.entrypoints.openai"),
    )
    entrypoints_module.openai = openai_pkg

    utils_spec = importlib.util.spec_from_file_location(
        "sglang.multimodal_gen.runtime.entrypoints.openai.utils",
        PROJECT_ROOT
        / "python/sglang/multimodal_gen/runtime/entrypoints/openai/utils.py",
    )
    utils_module = importlib.util.module_from_spec(utils_spec)
    assert utils_spec.loader is not None
    sys.modules["sglang.multimodal_gen.runtime.entrypoints.openai.utils"] = utils_module
    utils_spec.loader.exec_module(utils_module)

    return DataType, utils_module


def _make_sample(num_frames: int = 1):
    return torch.zeros(3, num_frames, 4, 4)


def test_openai_post_process_video_uses_mimsave(monkeypatch, tmp_path):
    DataType, openai_utils = _load_openai_utils()
    calls = {}

    def fake_mimsave(path, frames, fps, format):  # noqa: A002
        calls["mimsave"] = SimpleNamespace(
            path=path, frames_len=len(frames), fps=fps, format=format
        )

    def fake_imwrite(*_, **__):
        raise AssertionError("imwrite should not be called for video outputs")

    monkeypatch.setattr(openai_utils.imageio, "mimsave", fake_mimsave)
    monkeypatch.setattr(openai_utils.imageio, "imwrite", fake_imwrite)

    out_path = tmp_path / "out.mp4"
    openai_utils.post_process_sample(
        _make_sample(num_frames=2),
        DataType.VIDEO,
        fps=8,
        save_output=True,
        save_file_path=str(out_path),
    )

    assert "mimsave" in calls
    assert calls["mimsave"].path == str(out_path)
    assert calls["mimsave"].frames_len == 2
    assert calls["mimsave"].fps == 8
    assert calls["mimsave"].format == DataType.VIDEO.get_default_extension()


def test_openai_post_process_image_uses_imwrite(monkeypatch, tmp_path):
    DataType, openai_utils = _load_openai_utils()
    calls = {}

    def fake_mimsave(*_, **__):  # noqa: A002
        raise AssertionError("mimsave should not be called for image outputs")

    def fake_imwrite(path, frame, format=None):  # noqa: A002
        calls["imwrite"] = SimpleNamespace(
            path=path, frame_shape=frame.shape, format=format
        )

    monkeypatch.setattr(openai_utils.imageio, "mimsave", fake_mimsave)
    monkeypatch.setattr(openai_utils.imageio, "imwrite", fake_imwrite)

    out_path = tmp_path / "out.jpg"
    openai_utils.post_process_sample(
        _make_sample(num_frames=3),
        DataType.IMAGE,
        fps=8,
        save_output=True,
        save_file_path=str(out_path),
    )

    assert "imwrite" in calls
    assert calls["imwrite"].path == str(out_path)
    assert calls["imwrite"].format in (
        None,
        os.path.splitext(out_path)[1].lstrip("."),
    )
    assert calls["imwrite"].frame_shape[0] > 0
    assert calls["imwrite"].frame_shape[1] > 0
    assert calls["imwrite"].frame_shape[2] == 3
