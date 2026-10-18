"""Tests for releasing the SGLD worker's GPU memory when ComfyUI unloads the model.

Each bug gets a test that fails against the pre-fix code and passes after.
These stub ComfyUI/ torch-adjacent modules the same way test_h3_request.py
does, so they run without a ComfyUI install or a GPU.
"""

import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

import torch

PLUGIN_DIR = Path(__file__).resolve().parents[1]
PKG = "sgld_worker_sleep_under_test"


def _install_comfy_stubs() -> None:
    folder_paths = types.ModuleType("folder_paths")
    _tmp = tempfile.mkdtemp(prefix="sgld_test_temp_")
    folder_paths.get_temp_directory = lambda: _tmp
    folder_paths._diffusion_models_dirs = []
    folder_paths.folder_names_and_paths = {
        "diffusion_models": ([], {".safetensors", ".sft"}),
        "loras": ([], {".safetensors"}),
    }

    def get_filename_list(name):
        names = set()
        for folder in folder_paths.folder_names_and_paths.get(name, ([], set()))[0]:
            if os.path.isdir(folder):
                names.update(os.listdir(folder))
        return sorted(names)

    def get_folder_paths(name):
        return list(folder_paths.folder_names_and_paths.get(name, ([], set()))[0])

    def get_full_path(name, filename):
        for folder in get_folder_paths(name):
            candidate = os.path.join(folder, filename)
            if os.path.isfile(candidate):
                return candidate
        return None

    folder_paths.get_filename_list = get_filename_list
    folder_paths.get_folder_paths = get_folder_paths
    folder_paths.get_full_path = get_full_path
    sys.modules["folder_paths"] = folder_paths

    comfy_api = types.ModuleType("comfy_api")
    comfy_api_input = types.ModuleType("comfy_api.input")

    class VideoInput:
        pass

    comfy_api_input.VideoInput = VideoInput
    comfy_api.input = comfy_api_input
    sys.modules["comfy_api"] = comfy_api
    sys.modules["comfy_api.input"] = comfy_api_input
    # No comfy_api.input_impl: exercises the SGLDVideoInput fallback path.
    sys.modules.pop("comfy_api.input_impl", None)

    comfy = types.ModuleType("comfy")
    comfy.model_detection = types.ModuleType("comfy.model_detection")
    comfy.model_management = types.ModuleType("comfy.model_management")

    comfy_utils = types.ModuleType("comfy.utils")
    for name in (
        "calculate_parameters",
        "load_torch_file",
        "state_dict_prefix_replace",
        "unet_to_diffusers",
    ):
        setattr(comfy_utils, name, lambda *a, **k: None)

    class InterruptProcessingException(Exception):
        pass

    _interrupt = {"flag": False}

    def throw_exception_if_processing_interrupted():
        if _interrupt["flag"]:
            raise InterruptProcessingException()

    comfy.model_management.InterruptProcessingException = InterruptProcessingException
    comfy.model_management.throw_exception_if_processing_interrupted = (
        throw_exception_if_processing_interrupted
    )
    comfy.model_management._test_interrupt = _interrupt

    class ProgressBar:
        instances = []

        def __init__(self, total):
            self.total = total
            self.values = []
            ProgressBar.instances.append(self)

        def update_absolute(self, value, total=None, preview=None):
            self.values.append(value)

    comfy_utils.ProgressBar = ProgressBar
    comfy.utils = comfy_utils

    comfy_model_patcher = types.ModuleType("comfy.model_patcher")

    class ModelPatcher:
        def __init__(self, model, load_device, offload_device, size=0, *a, **k):
            self.model = model
            self.load_device = load_device
            self.offload_device = offload_device
            self.size = size
            self.patches = {}
            self.object_patches = {}
            self.model_options = {}
            self.backup = {}
            self.object_patches_backup = {}
            self.wrappers = {}
            self.callbacks = {}
            self.patches_uuid = None
            self.weight_inplace_update = False

        def add_wrapper_with_key(self, *a, **k):
            pass

        # ComfyUI: loaded_size() reads the model's own weight counter (0 for
        # SGLD, whose weights live in the worker); detach() is what
        # LoadedModel.model_unload always ends in.
        def loaded_size(self):
            return getattr(self.model, "model_loaded_weight_memory", 0)

        def detach(self, unpatch_all=True):
            return self.model

    comfy_model_patcher.ModelPatcher = ModelPatcher
    comfy.model_patcher = comfy_model_patcher

    comfy_patcher_extension = types.ModuleType("comfy.patcher_extension")

    class WrappersMP:
        SAMPLER_SAMPLE = "SAMPLER_SAMPLE"

    comfy_patcher_extension.WrappersMP = WrappersMP
    comfy.patcher_extension = comfy_patcher_extension

    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_detection"] = comfy.model_detection
    sys.modules["comfy.model_management"] = comfy.model_management
    sys.modules["comfy.utils"] = comfy_utils
    sys.modules["comfy.model_patcher"] = comfy_model_patcher
    sys.modules["comfy.patcher_extension"] = comfy_patcher_extension
    return folder_paths


def _load(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(
        module_name, PLUGIN_DIR / relative_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_plugin():
    folder_paths = _install_comfy_stubs()

    package = types.ModuleType(PKG)
    package.__path__ = [str(PLUGIN_DIR)]
    sys.modules[PKG] = package

    core = types.ModuleType(f"{PKG}.core")
    core.__path__ = [str(PLUGIN_DIR / "core")]
    sys.modules[f"{PKG}.core"] = core

    server_api = _load(f"{PKG}.core.server_api", "core/server_api.py")
    core.SGLDiffusionServerAPI = server_api.SGLDiffusionServerAPI
    model_patcher = _load(f"{PKG}.core.model_patcher", "core/model_patcher.py")
    core.SGLDModelPatcher = model_patcher.SGLDModelPatcher
    generator = _load(f"{PKG}.core.generator", "core/generator.py")
    core.SGLDiffusionGenerator = generator.SGLDiffusionGenerator

    _load(f"{PKG}.utils", "utils.py")
    nodes = _load(f"{PKG}.nodes", "nodes.py")
    return (
        server_api,
        model_patcher,
        generator,
        nodes,
        sys.modules[f"{PKG}.utils"],
        folder_paths,
    )


SERVER_API, MODEL_PATCHER, GENERATOR, NODES, UTILS, FOLDER_PATHS = _load_plugin()
SGLDiffusionServerAPI = SERVER_API.SGLDiffusionServerAPI
SGLDModelPatcher = MODEL_PATCHER.SGLDModelPatcher


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload

    def iter_content(self, chunk_size=None):
        yield b"fake-mp4-bytes"


# --- ComfyUI "Unload models" never reached the SGLD worker ------------------


class _FakeWorkerOwner:
    """Stands in for SGLDiffusionGenerator as the patcher's worker handle."""

    def __init__(self, asleep=False):
        self.worker_asleep = asleep
        self.shares_comfy_gpu = True
        self.resident_bytes = 0
        self.calls = []

    def sleep_worker(self, token=None):
        self.calls.append("sleep")
        self.worker_asleep = True

    def wake_worker(self):
        self.calls.append("wake")
        self.worker_asleep = False


def _make_patcher(owner, size=20 * 2**30):
    return SGLDModelPatcher(
        torch.nn.Module(),
        torch.device("cpu"),
        torch.device("cpu"),
        size=size,
        model_type="qwen_image",
        worker=owner,
    )


def test_detach_puts_the_worker_to_sleep():
    owner = _FakeWorkerOwner()
    _make_patcher(owner).detach()
    assert owner.calls == ["sleep"]


def test_loaded_size_counts_worker_vram_only_while_awake():
    owner = _FakeWorkerOwner()
    patcher = _make_patcher(owner, size=7)
    assert patcher.loaded_size() == 7
    owner.worker_asleep = True
    assert patcher.loaded_size() == 0


def test_cloned_patcher_keeps_the_worker_handle():
    owner = _FakeWorkerOwner()
    _make_patcher(owner).clone().detach()
    assert owner.calls == ["sleep"]


def _live_generator(monkeypatch, responses=None):
    gen = GENERATOR.SGLDiffusionGenerator()
    sent = []

    class _Gen:
        def _send_to_scheduler_and_wait_for_response(self, reqs):
            sent.append(type(reqs[0]).__name__)
            return types.SimpleNamespace(
                error=None, output={"success": True, "sleeping": True}
            )

    gen.generator = _Gen()
    gen._patcher = object()
    monkeypatch.setattr(gen, "_is_live", lambda: True)

    io_struct = types.ModuleType(
        "sglang.multimodal_gen.runtime.entrypoints.post_training.io_struct"
    )
    io_struct.ReleaseMemoryOccupationReqInput = type(
        "ReleaseMemoryOccupationReqInput", (), {}
    )
    io_struct.ResumeMemoryOccupationReqInput = type(
        "ResumeMemoryOccupationReqInput", (), {}
    )
    io_struct.ReleaseAllocatorCacheReqInput = type(
        "ReleaseAllocatorCacheReqInput", (), {}
    )
    monkeypatch.setitem(sys.modules, io_struct.__name__, io_struct)
    return gen, sent


def test_generator_sleep_and_wake_send_the_scheduler_requests(monkeypatch):
    gen, sent = _live_generator(monkeypatch)
    gen.sleep_worker(gen.generator)
    gen.sleep_worker(gen.generator)  # already asleep: no second request
    assert sent == ["ReleaseMemoryOccupationReqInput"]
    assert gen.worker_asleep
    gen.wake_worker()
    gen.wake_worker()
    assert sent[1:] == ["ResumeMemoryOccupationReqInput"]
    assert not gen.worker_asleep


def test_generator_ignores_sleep_from_a_stale_patcher(monkeypatch):
    gen, sent = _live_generator(monkeypatch)
    gen.sleep_worker(object())  # token of a worker that was replaced
    assert sent == []
    assert not gen.worker_asleep


def test_generator_does_not_sleep_a_dead_worker(monkeypatch):
    gen, sent = _live_generator(monkeypatch)
    monkeypatch.setattr(gen, "_is_live", lambda: False)
    gen.sleep_worker(gen.generator)
    assert sent == []


def test_ensure_executor_wakes_a_sleeping_worker(monkeypatch):
    gen, sent = _live_generator(monkeypatch)
    executor = types.SimpleNamespace(generator=gen.generator)
    gen.sleep_worker(gen.generator)
    gen.ensure_executor(executor)
    assert sent == [
        "ReleaseMemoryOccupationReqInput",
        "ResumeMemoryOccupationReqInput",
    ]


def test_set_lora_wakes_a_sleeping_worker(monkeypatch):
    gen, sent = _live_generator(monkeypatch)
    gen.generator.set_lora = lambda **kw: sent.append("set_lora")
    executor = object.__new__(GENERATOR.FluxExecutor)
    executor.generator = gen.generator
    executor._ensure_runtime = gen.ensure_executor
    gen.sleep_worker(gen.generator)
    executor.set_lora(lora_nickname="a", lora_path="/x.safetensors")
    assert sent[-2:] == ["ResumeMemoryOccupationReqInput", "set_lora"]


def test_clone_of_live_patcher_still_sleeps_the_real_generator(monkeypatch):
    gen, sent = _live_generator(monkeypatch)
    patcher = SGLDModelPatcher(
        torch.nn.Module(),
        torch.device("cpu"),
        torch.device("cpu"),
        size=1,
        worker=gen,
        worker_token=gen.generator,
    )
    patcher.clone().detach()
    assert sent == ["ReleaseMemoryOccupationReqInput"]


def test_partially_unload_has_no_side_effects():
    patcher = _make_patcher(_FakeWorkerOwner())
    assert patcher.partially_unload(torch.device("cpu"), 123) == 0


def test_off_gpu_worker_reports_zero_and_detach_is_a_noop(monkeypatch):
    gen, sent = _live_generator(monkeypatch)
    gen.shares_comfy_gpu = False
    patcher = SGLDModelPatcher(
        torch.nn.Module(),
        torch.device("cpu"),
        torch.device("cpu"),
        size=9,
        worker=gen,
        worker_token=gen.generator,
    )
    assert patcher.loaded_size() == 0
    # ComfyUI frees model_size() - loaded_size() on its own GPU before load.
    assert patcher.model_size() == 0
    assert patcher.clone().model_size() == 0
    patcher.detach()
    assert sent == []
    assert not gen.worker_asleep


def test_loaded_size_prefers_measured_resident_bytes():
    owner = _FakeWorkerOwner()
    patcher = _make_patcher(owner, size=100)
    owner.resident_bytes = 40
    assert patcher.loaded_size() == 40


def test_worker_gpu_topology(monkeypatch):
    mm = sys.modules["comfy.model_management"]
    monkeypatch.setattr(
        mm, "get_torch_device", lambda: torch.device("cuda", 1), raising=False
    )
    share = GENERATOR.SGLDiffusionGenerator._worker_shares_comfy_gpu
    assert share({"base_gpu_id": 1})
    assert share({"base_gpu_id": 0, "num_gpus": 2})
    assert not share({"base_gpu_id": 2})
    assert not share({})  # worker on GPU 0, ComfyUI on GPU 1
    monkeypatch.setattr(
        mm, "get_torch_device", lambda: torch.device("cpu"), raising=False
    )
    assert not share({})


def test_release_worker_cache_records_resident_bytes(monkeypatch):
    gen, sent = _live_generator(monkeypatch)
    gen.generator._send_to_scheduler_and_wait_for_response = lambda reqs: (
        sent.append(type(reqs[0]).__name__)
        or types.SimpleNamespace(
            error=None, output={"success": True, "resident_bytes": 123}
        )
    )
    gen.release_worker_cache()
    assert sent == ["ReleaseAllocatorCacheReqInput"]
    assert gen.resident_bytes == 123
    gen.shares_comfy_gpu = False
    gen.release_worker_cache()
    assert len(sent) == 1
    gen.shares_comfy_gpu = True
    gen.worker_asleep = True
    gen.release_worker_cache()
    assert len(sent) == 1


# --- VRAM accounting, moved here from #43536 ---------------------------------


def test_model_size_uses_real_computed_size_for_every_model_type():
    for model_type, size in (
        ("minimax_h3", 13_000_000_000),
        ("qwen_image", 20_000_000_000),
    ):
        patcher = SGLDModelPatcher(
            torch.nn.Module(),
            torch.device("cpu"),
            torch.device("cpu"),
            size=size,
            model_type=model_type,
        )
        assert patcher.model_size() == size


def test_model_size_falls_back_to_table_only_when_size_unknown():
    patcher = SGLDModelPatcher(
        torch.nn.Module(),
        torch.device("cpu"),
        torch.device("cpu"),
        size=0,
        model_type="flux",
    )
    assert patcher.model_size() == 27 * 1024 * 1024 * 1024
