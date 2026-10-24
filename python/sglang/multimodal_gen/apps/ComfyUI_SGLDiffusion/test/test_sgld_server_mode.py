"""Regression tests for ComfyUI_SGLDiffusion server-mode bugs found in audit.

Each bug gets a test that fails against the pre-fix code and passes after.
These stub ComfyUI/ torch-adjacent modules the same way test_h3_request.py
does, so they run without a ComfyUI install or a GPU.
"""

import base64
import importlib.util
import io
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

import pytest
import torch
from PIL import Image

PLUGIN_DIR = Path(__file__).resolve().parents[1]
PKG = "sgld_server_mode_under_test"


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


# --- Bug 9: hardcoded 300s timeout on image requests -----------------------


_TMP_FILES = []


def _tmp_file(suffix):
    """A temp file path that the autouse fixture below deletes afterwards."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
        _TMP_FILES.append(fh.name)
        return fh.name


@pytest.fixture(autouse=True)
def _remove_tmp_files():
    yield
    while _TMP_FILES:
        path = _TMP_FILES.pop()
        if os.path.exists(path):
            os.remove(path)


def test_image_timeout_is_configurable():
    client = SGLDiffusionServerAPI(base_url="http://127.0.0.1:1234", image_timeout=900)
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None, **kwargs):
        captured["timeout"] = timeout
        return _Response({"data": [{"b64_json": "x"}]})

    with mock.patch(f"{PKG}.core.server_api.requests.post", side_effect=fake_post):
        client.generate_image(prompt="a cat")

    assert captured["timeout"] == 900


# --- Bug 10: batched image inputs were silently truncated to image[0] ------


def test_get_image_path_uses_first_image_of_a_batch_with_a_warning(caplog):
    batch = torch.zeros(3, 8, 8, 3)
    with caplog.at_level("WARNING"):
        path = UTILS.get_image_path(batch)
    assert os.path.exists(path)
    os.remove(path)
    assert "batch of 3 images" in caplog.text


def test_get_image_path_accepts_single_image():
    single = torch.zeros(1, 8, 8, 3)
    path = UTILS.get_image_path(single)
    assert os.path.exists(path)
    os.remove(path)


# --- Bug 11: LoRA strength could not be set in server mode -----------------


def test_set_lora_sends_strength_to_server():
    client = SGLDiffusionServerAPI(base_url="http://127.0.0.1:1234")
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None, **kwargs):
        captured.update(json)
        return _Response({"status": "ok"})

    with mock.patch(f"{PKG}.core.server_api.requests.post", side_effect=fake_post):
        client.set_lora(
            lora_nickname="style", lora_path="style.safetensors", strength=0.7
        )

    assert captured["strength"] == 0.7


def test_set_lora_node_exposes_strength(monkeypatch):
    client = SGLDiffusionServerAPI(base_url="http://127.0.0.1:1234")
    node = NODES.SGLDiffusionServerSetLora()
    captured = {}

    def fake_set_lora(**kwargs):
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(client, "set_lora", fake_set_lora)
    node.set_lora(sgld_client=client, lora_name="style.safetensors", strength=0.5)
    assert captured["strength"] == 0.5


# --- Bug 6/7: SGLDVideoInput didn't implement VideoInput; dims could be wrong


def test_video_input_fallback_fails_loudly_instead_of_returning_wrong_data():
    video = UTILS.convert_video_to_comfy_video("/tmp/out.mp4", height=720, width=1280)
    assert isinstance(video, UTILS.SGLDVideoInput)  # no comfy_api.input_impl stubbed
    with pytest.raises(NotImplementedError):
        video.get_components()


def test_video_input_save_to_refuses_silent_format_drop():
    video = UTILS.SGLDVideoInput("/tmp/out.mp4", height=720, width=1280)
    with pytest.raises(NotImplementedError):
        video.save_to("/tmp/converted.webm", format="webm")


def test_generic_video_node_reports_server_resolved_size_not_requested_size():
    client = SGLDiffusionServerAPI(base_url="http://127.0.0.1:1234")
    node = NODES.SGLDiffusionGenerateVideo()

    with (
        mock.patch.object(
            client,
            "generate_video",
            return_value={"file_path": "/tmp/x.mp4", "size": "1280x704"},
        ),
    ):
        video, _, _ = node.generate_video(
            sgld_client=client,
            positive_prompt="a cat",
            width=1280,
            height=720,  # requested 720, server actually resolved to 704
        )
    assert video.get_dimensions() == (1280, 704)


# --- Bug 3: server-mode nodes sent local paths a remote server can't read --


def test_generate_video_uploads_local_input_reference_instead_of_sending_path():
    client = SGLDiffusionServerAPI(base_url="http://127.0.0.1:1234")
    local_image = _tmp_file(".png")
    with open(local_image, "wb") as fh:
        fh.write(b"fake-png-bytes")

    seen = {}

    def fake_post(url, json=None, headers=None, timeout=None, files=None, data=None):
        seen["files"] = files
        seen["data"] = data
        seen["json"] = json
        return _Response({"id": "job-1"})

    out_path = _tmp_file(".mp4")

    def fake_get(url, headers=None, timeout=None, **kwargs):
        return _Response({"id": "job-1", "status": "completed", "file_path": out_path})

    with (
        mock.patch(f"{PKG}.core.server_api.requests.post", side_effect=fake_post),
        mock.patch(f"{PKG}.core.server_api.requests.get", side_effect=fake_get),
    ):
        client.generate_video(prompt="a cat", input_reference=local_image)

    os.remove(local_image)
    # A local path must be uploaded as file bytes, not sent as a path string
    # the server (possibly on another machine) has no way to read.
    assert seen["files"] is not None
    assert seen["json"] is None


def _capture_video_request(client, **kwargs):
    seen = {}

    def fake_post(url, json=None, headers=None, timeout=None, files=None, data=None):
        seen["files"] = files and {
            key: (name, fh.read(), content_type)
            for key, (name, fh, content_type) in files.items()
        }
        seen["data"] = data
        seen["json"] = json
        return _Response({"id": "job-1"})

    out_path = _tmp_file(".mp4")

    def fake_get(url, headers=None, timeout=None, **kw):
        return _Response({"id": "job-1", "status": "completed", "file_path": out_path})

    with (
        mock.patch(f"{PKG}.core.server_api.requests.post", side_effect=fake_post),
        mock.patch(f"{PKG}.core.server_api.requests.get", side_effect=fake_get),
    ):
        client.generate_video(prompt="a cat", **kwargs)
    return seen


def test_generate_video_uploads_local_source_video_as_video_reference():
    client = SGLDiffusionServerAPI(base_url="http://127.0.0.1:1234")
    with tempfile.TemporaryDirectory() as tmp:
        clip = os.path.join(tmp, "clip.mp4")
        image = os.path.join(tmp, "ref.png")
        for path, body in ((clip, b"fake-mp4"), (image, b"fake-png")):
            with open(path, "wb") as fh:
                fh.write(body)

        seen = _capture_video_request(client, video_path=clip, input_reference=image)

    # Both local files go up as multipart; neither path string is sent.
    assert seen["json"] is None
    assert seen["files"]["video_reference"] == ("clip.mp4", b"fake-mp4", "video/mp4")
    assert seen["files"]["input_reference"][1] == b"fake-png"
    assert "video_path" not in seen["data"]
    assert "input_reference" not in seen["data"]


def test_generate_video_multipart_fields_are_json_encoded():
    client = SGLDiffusionServerAPI(base_url="http://127.0.0.1:1234")
    with tempfile.TemporaryDirectory() as tmp:
        image = os.path.join(tmp, "ref.png")
        with open(image, "wb") as fh:
            fh.write(b"fake-png")
        seen = _capture_video_request(
            client,
            input_reference=image,
            extra_fields={"target": {"type": "image"}, "flag": False},
        )

    # the server json-decodes form values, so Python repr would break them
    assert seen["data"]["target"] == '{"type": "image"}'
    assert seen["data"]["flag"] == "false"
    assert seen["data"]["prompt"] == "a cat"


def test_generate_video_sends_remote_source_video_as_video_path():
    client = SGLDiffusionServerAPI(base_url="http://127.0.0.1:1234")
    for source in ("https://example.com/clip.mp4", "/srv/sglang/inputs/clip.mp4"):
        seen = _capture_video_request(client, video_path=source)
        # A URL or a path that only exists on the server stays a string.
        assert seen["files"] is None
        assert seen["json"]["video_path"] == source


def test_generate_video_downloads_result_when_server_path_is_not_local():
    client = SGLDiffusionServerAPI(base_url="http://127.0.0.1:1234")

    def fake_post(url, **kwargs):
        return _Response({"id": "job-1"})

    def fake_get(url, headers=None, timeout=None, stream=False, **kwargs):
        if url.endswith("/content"):
            return _Response({}, status_code=200)
        return _Response(
            {
                "id": "job-1",
                "status": "completed",
                # A path on the *server's* filesystem, not this one.
                "file_path": "/srv/sglang/outputs/job-1.mp4",
            }
        )

    with tempfile.TemporaryDirectory() as out_dir:
        with (
            mock.patch(f"{PKG}.core.server_api.requests.post", side_effect=fake_post),
            mock.patch(f"{PKG}.core.server_api.requests.get", side_effect=fake_get),
        ):
            result = client.generate_video(prompt="a cat", output_path=out_dir)

        assert os.path.exists(result["file_path"])
        assert os.path.realpath(
            os.path.dirname(result["file_path"])
        ) == os.path.realpath(out_dir)


@pytest.mark.parametrize("video_id", ["../escape", "a/b", "job.1", ""])
def test_localize_video_result_rejects_unsafe_video_id(video_id):
    client = SGLDiffusionServerAPI(base_url="http://127.0.0.1:1234")
    status = {"status": "completed", "file_path": "/srv/out/x.mp4"}
    with tempfile.TemporaryDirectory() as out_dir:
        with mock.patch(f"{PKG}.core.server_api.requests.get") as get:
            with pytest.raises(RuntimeError, match="unsafe id"):
                client._localize_video_result(status, video_id, out_dir)
        get.assert_not_called()
        assert os.listdir(out_dir) == []


def test_localize_video_result_whitelists_extension_and_cleans_up_on_error():
    client = SGLDiffusionServerAPI(base_url="http://127.0.0.1:1234")

    class _Broken(_Response):
        def iter_content(self, chunk_size=None):
            yield b"partial"
            raise SERVER_API.requests.exceptions.ConnectionError("dropped")

    status = {"status": "completed", "file_path": "/srv/out/job-1.sh"}
    with tempfile.TemporaryDirectory() as out_dir:
        with mock.patch(
            f"{PKG}.core.server_api.requests.get", return_value=_Response({})
        ):
            result = client._localize_video_result(status, "job-1", out_dir)
        assert result["file_path"] == os.path.join(
            os.path.realpath(out_dir), "job-1.mp4"
        )
        os.remove(result["file_path"])
        with mock.patch(
            f"{PKG}.core.server_api.requests.get", return_value=_Broken({})
        ):
            with pytest.raises(SERVER_API.requests.exceptions.ConnectionError):
                client._localize_video_result(status, "job-1", out_dir)
        assert os.listdir(out_dir) == []


class _HTTPErrorResponse(_Response):
    def raise_for_status(self):
        raise SERVER_API.requests.exceptions.HTTPError(
            f"{self.status_code} Client Error", response=self
        )


def _completed_video_client(job, max_polls=3):
    """Client whose job is already completed; /content answers 400, as the
    server does for a job uploaded to cloud storage. Polling past
    `max_polls` fails the test instead of looping until max_wait_time."""
    client = SGLDiffusionServerAPI(base_url="http://127.0.0.1:1234")
    calls = {"status": 0, "content": 0}

    def fake_get(url, **kwargs):
        if url.endswith("/content"):
            calls["content"] += 1
            return _HTTPErrorResponse({"detail": "use the cloud URL"}, 400)
        calls["status"] += 1
        assert calls["status"] <= max_polls, "kept polling a completed job"
        return _Response(job)

    def fake_post(url, **kwargs):
        return _Response({"id": "job-1"})

    patches = (
        mock.patch.object(SERVER_API.requests, "get", fake_get),
        mock.patch.object(SERVER_API.requests, "post", fake_post),
        mock.patch.object(SERVER_API.time, "sleep"),
    )
    return client, calls, patches


def test_generate_video_returns_cloud_url_without_downloading_content():
    url = "https://bucket.example/job-1.mp4"
    client, calls, patches = _completed_video_client(
        {"id": "job-1", "status": "completed", "url": url, "file_path": None}
    )
    with tempfile.TemporaryDirectory() as out_dir:
        with patches[0], patches[1], patches[2]:
            result = client.generate_video(prompt="a cat", output_path=out_dir)
    # resolve_video_path fetches the url, as on main.
    assert result["url"] == url
    assert result["file_path"] is None
    assert calls["content"] == 0


def test_generate_video_fails_fast_when_result_download_fails():
    client, calls, patches = _completed_video_client(
        {"id": "job-1", "status": "completed", "file_path": "/srv/out/job-1.mp4"}
    )
    with tempfile.TemporaryDirectory() as out_dir:
        with patches[0], patches[1], patches[2]:
            with pytest.raises(RuntimeError, match="400"):
                client.generate_video(prompt="a cat", output_path=out_dir)
    assert calls == {"status": 1, "content": 1}


def test_h3_image_conditions_use_data_uri_not_local_file_path():
    node = NODES.SGLDiffusionGenerateH3()
    image = torch.zeros(1, 8, 8, 3)
    uri = node._image_material_uri(image)
    assert uri.startswith("data:image/png;base64,")


def test_h3_rejects_local_video_reference_against_remote_server():
    with pytest.raises(ValueError, match="no way to read or upload"):
        NODES.SGLDiffusionGenerateH3._remote_material_uri(
            "/data/clip.mp4", "reference_video", "http://remote-host.example:3000/v1"
        )


def test_h3_allows_local_video_reference_against_local_server():
    uri = NODES.SGLDiffusionGenerateH3._remote_material_uri(
        "/data/clip.mp4", "reference_video", "http://127.0.0.1:3000/v1"
    )
    assert uri == "file:///data/clip.mp4"


def test_is_local_server_accepts_own_hostname_and_loopback():
    is_local = NODES.SGLDiffusionGenerateH3._is_local_server
    host = NODES.socket.gethostname()
    assert is_local(f"http://{host}:3000/v1")
    assert is_local("http://0.0.0.0:3000/v1")
    assert is_local("http://127.0.0.2:3000/v1")
    assert is_local("http://[::1]:3000/v1")
    with mock.patch.object(
        NODES.socket, "getaddrinfo", return_value=[(0, 0, 0, "", ("10.1.2.3", 0))]
    ):
        assert not is_local("http://gpu-box:3000/v1")


def test_h3_passes_file_uri_through_for_any_server():
    # a file:// URI names a path on the server, so it is never rewritten
    for base_url in ("http://127.0.0.1:3000/v1", "http://remote-host.example:3000/v1"):
        uri = NODES.SGLDiffusionGenerateH3._remote_material_uri(
            "file:///srv/clip.mp4", "reference_video", base_url
        )
        assert uri == "file:///srv/clip.mp4"


# --- server-mode video jobs: no progress bar, Cancel was ignored ------------


def _video_client_with_statuses(statuses):
    client = SGLDiffusionServerAPI(base_url="http://127.0.0.1:1234")
    polled = iter(statuses)

    def fake_get(url, **kwargs):
        return _Response(next(polled))

    def fake_post(url, **kwargs):
        return _Response({"id": "vid1"})

    return client, fake_get, fake_post


def test_generate_video_reports_server_progress():
    client, fake_get, fake_post = _video_client_with_statuses(
        [
            {"status": "queued", "progress": 0},
            {"status": "in_progress", "progress": 40},
            {"status": "completed", "progress": 100, "file_path": "/tmp/x.mp4"},
        ]
    )
    seen = []
    with (
        mock.patch.object(SERVER_API.requests, "get", fake_get),
        mock.patch.object(SERVER_API.requests, "post", fake_post),
        mock.patch.object(SERVER_API.time, "sleep"),
        mock.patch.object(client, "_localize_video_result", lambda s, *a: s),
    ):
        client.generate_video(prompt="a cat", progress_callback=seen.append)
    assert seen == [0, 40, 100]


def test_generate_video_stops_polling_when_interrupted():
    client, fake_get, fake_post = _video_client_with_statuses(
        [{"status": "in_progress", "progress": 10}] * 50
    )
    checks = {"n": 0}

    class Interrupted(Exception):
        pass

    def check_interrupt():
        checks["n"] += 1
        if checks["n"] > 2:
            raise Interrupted()

    with (
        mock.patch.object(SERVER_API.requests, "get", fake_get),
        mock.patch.object(SERVER_API.requests, "post", fake_post),
        mock.patch.object(SERVER_API.time, "sleep"),
    ):
        with pytest.raises(Interrupted):
            client.generate_video(prompt="a cat", check_interrupt=check_interrupt)
    assert checks["n"] == 3  # stopped promptly, did not poll to completion


def test_video_node_wires_comfy_progress_bar_and_interrupt():
    client = SGLDiffusionServerAPI(base_url="http://127.0.0.1:1234")
    node = NODES.SGLDiffusionGenerateVideo()
    captured = {}

    def fake_generate_video(**kwargs):
        captured.update(kwargs)
        return {"file_path": "/tmp/x.mp4", "size": "1280x720"}

    with mock.patch.object(client, "generate_video", fake_generate_video):
        node.generate_video(
            sgld_client=client, positive_prompt="a cat", width=1280, height=720
        )
    captured["progress_callback"](55)
    assert sys.modules["comfy.utils"].ProgressBar.instances[-1].values == [55]
    sys.modules["comfy.model_management"]._test_interrupt["flag"] = True
    try:
        with pytest.raises(
            sys.modules["comfy.model_management"].InterruptProcessingException
        ):
            captured["check_interrupt"]()
    finally:
        sys.modules["comfy.model_management"]._test_interrupt["flag"] = False


@pytest.mark.parametrize("enhance", [True, False])
def test_image_node_enhance_prompt_returns_revised_prompt(enhance):
    client = SGLDiffusionServerAPI(base_url="http://127.0.0.1:1234")
    png = io.BytesIO()
    Image.new("RGB", (8, 8)).save(png, format="PNG")
    b64 = base64.b64encode(png.getvalue()).decode()
    revised = "a fluffy cat" if enhance else None
    captured = {}

    def fake_post(url, json=None, **kwargs):
        captured.update(json)
        return _Response({"data": [{"b64_json": b64, "revised_prompt": revised}]})

    with mock.patch(f"{PKG}.core.server_api.requests.post", side_effect=fake_post):
        image, revised_prompt = NODES.SGLDiffusionGenerateImage().generate_image(
            sgld_client=client, positive_prompt="a cat", enhance_prompt=enhance
        )

    assert captured.get("enhance_prompt", False) == enhance
    assert image.shape == (1, 8, 8, 3)
    assert revised_prompt == ("a fluffy cat" if enhance else "a cat")


def test_video_node_sends_source_video_and_enhance_prompt():
    client = SGLDiffusionServerAPI(base_url="http://127.0.0.1:1234")
    captured = {}

    class SourceVideo:
        def save_to(self, path):
            with open(path, "wb") as fh:
                fh.write(b"source-mp4")

    def fake_generate_video(**kwargs):
        captured.update(kwargs)
        with open(kwargs["video_path"], "rb") as fh:
            captured["video_bytes"] = fh.read()
        return {"file_path": "/tmp/x.mp4", "revised_prompt": "a fluffy cat"}

    with mock.patch.object(client, "generate_video", fake_generate_video):
        _, video_path, revised_prompt = (
            NODES.SGLDiffusionGenerateVideo().generate_video(
                sgld_client=client,
                positive_prompt="a cat",
                video=SourceVideo(),
                enhance_prompt=True,
            )
        )
    os.remove(captured["video_path"])

    assert captured["video_bytes"] == b"source-mp4"
    assert captured["extra_fields"] == {"enhance_prompt": True}
    assert (video_path, revised_prompt) == ("/tmp/x.mp4", "a fluffy cat")
