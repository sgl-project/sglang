"""Common utilities"""

import json
import logging
import os
import random
import ssl
import subprocess
import sys
import time
import urllib.request
import weakref
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property, wraps
from io import BytesIO
from json import dumps
from typing import Any, Optional, Union

import numpy as np
import pybase64
import requests
from tqdm import tqdm

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def load_diffusion_overlay_registry_from_env() -> dict[str, dict[str, Any]]:
    raw_value = os.getenv("SGLANG_DIFFUSION_MODEL_OVERLAY_REGISTRY", "").strip()
    if not raw_value:
        return {}

    if raw_value.startswith("{"):
        payload = json.loads(raw_value)
    else:
        with open(os.path.expanduser(raw_value), encoding="utf-8") as f:
            payload = json.load(f)

    if not isinstance(payload, dict):
        return {}

    normalized: dict[str, dict[str, Any]] = {}
    for source_model_id, spec in payload.items():
        if isinstance(spec, str):
            normalized[source_model_id] = {"overlay_repo_id": spec}
        elif isinstance(spec, dict) and spec.get("overlay_repo_id"):
            normalized[source_model_id] = dict(spec)
    return normalized


def has_diffusion_overlay_registry_match(
    model_path: str, registry: dict[str, dict[str, Any]] | None = None
) -> bool:
    registry = (
        load_diffusion_overlay_registry_from_env() if registry is None else registry
    )
    if model_path in registry:
        return True
    if not os.path.exists(model_path):
        return False
    base_name = os.path.basename(os.path.normpath(model_path))
    return any(base_name == key.rsplit("/", 1)[-1] for key in registry)


def execute_once(func):
    has_run = None

    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal has_run
        if not has_run:
            func(*args, **kwargs)
            has_run = True

    return wrapper


@execute_once
def info_once(message: str):
    logger.info(message)


def is_same_type(values: list):
    """Return whether the elements in values are of the same type."""
    if len(values) <= 1:
        return True
    else:
        t = type(values[0])
        return all(isinstance(v, t) for v in values[1:])


def read_jsonl(filename: str):
    """Read a JSONL file."""
    with open(filename) as fin:
        for line in fin:
            if line.startswith("#"):
                continue
            yield json.loads(line)


def dump_state_text(filename: str, states: list, mode: str = "w"):
    """Dump program state in a text file."""
    from sglang.lang.interpreter import ProgramState

    with open(filename, mode) as fout:
        for i, s in enumerate(states):
            if isinstance(s, str):
                pass
            elif isinstance(s, ProgramState):
                s = s.text()
            else:
                s = str(s)

            fout.write(
                "=" * 40 + f" {i} " + "=" * 40 + "\n" + s + "\n" + "=" * 80 + "\n\n"
            )


def normalize_base_url(host: str, port: int) -> str:
    from sglang.srt.utils.network import NetworkAddress

    return NetworkAddress(host, port).to_url()


class HttpResponse:
    def __init__(self, resp):
        self.resp = resp

    @cached_property
    def _body(self):
        return self.resp.read()

    def json(self):
        return json.loads(self._body)

    @property
    def text(self):
        return self._body.decode("utf-8", errors="replace")

    @property
    def status_code(self):
        return self.resp.status


def http_request(
    url,
    json=None,
    stream=False,
    api_key=None,
    verify=None,
    method: Optional[str] = None,
):
    """A faster version of requests.post with low-level urllib API."""
    headers = {"Content-Type": "application/json; charset=utf-8"}

    # add the Authorization header if an api key is provided
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"

    if stream:
        return requests.post(url, json=json, stream=True, headers=headers)
    else:
        req = urllib.request.Request(url, headers=headers, method=method)
        if json is None:
            data = None
        else:
            data = bytes(dumps(json), encoding="utf-8")

        try:
            if sys.version_info >= (3, 13):
                # Python 3.13+: Use SSL context (cafile removed)
                if verify and isinstance(verify, str):
                    context = ssl.create_default_context(cafile=verify)
                else:
                    context = ssl.create_default_context()
                resp = urllib.request.urlopen(req, data=data, context=context)
            else:
                resp = urllib.request.urlopen(req, data=data, cafile=verify)
            return HttpResponse(resp)
        except urllib.error.HTTPError as e:
            return HttpResponse(e)


def encode_image_base64(image_path: Union[str, bytes]):
    """Encode an image in base64."""
    if isinstance(image_path, str):
        with open(image_path, "rb") as image_file:
            data = image_file.read()
            return pybase64.b64encode(data).decode("utf-8")
    elif isinstance(image_path, bytes):
        return pybase64.b64encode(image_path).decode("utf-8")
    else:
        import torch

        if isinstance(image_path, torch.Tensor):
            # Convert GPU-decoded image tensor (C, H, W) uint8 to PIL Image
            from PIL import Image

            tensor = image_path.cpu() if image_path.device.type != "cpu" else image_path
            image_path = Image.fromarray(tensor.permute(1, 2, 0).numpy())

        # image_path is a PIL Image
        image = image_path
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return pybase64.b64encode(buffered.getvalue()).decode("utf-8")


def encode_frame(frame):
    import cv2  # pip install opencv-python-headless
    from PIL import Image

    # Convert the frame to RGB (OpenCV uses BGR by default)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to PIL Image to easily convert to bytes
    im_pil = Image.fromarray(frame)

    # Convert to bytes
    buffered = BytesIO()

    # frame_format = str(os.getenv('FRAME_FORMAT', "JPEG"))

    im_pil.save(buffered, format="PNG")

    frame_bytes = buffered.getvalue()

    # Return the bytes of the frame
    return frame_bytes


def encode_video_base64(video_path: str, num_frames: int = 16):
    import cv2  # pip install opencv-python-headless

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file:{video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"target_frames: {num_frames}")

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for _ in range(total_frames):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            # Handle the case where the frame could not be read
            # print(f"Warning: Could not read frame at index {i}.")
            pass

    cap.release()

    # Safely select frames based on frame_indices, avoiding IndexError
    frames = [frames[i] for i in frame_indices if i < len(frames)]

    # If there are not enough frames, duplicate the last frame until we reach the target
    while len(frames) < num_frames:
        frames.append(frames[-1])

    # Use ThreadPoolExecutor to process and encode frames in parallel
    with ThreadPoolExecutor() as executor:
        encoded_frames = list(executor.map(encode_frame, frames))

    # encoded_frames = list(map(encode_frame, frames))

    # Concatenate all frames bytes
    video_bytes = b"".join(encoded_frames)

    # Encode the concatenated bytes to base64
    video_base64 = "video:" + pybase64.b64encode(video_bytes).decode("utf-8")

    return video_base64


def download_and_cache_file(url: str, filename: Optional[str] = None):
    """Read and cache a file from a url."""
    if filename is None:
        filename = os.path.join("/tmp", url.split("/")[-1])

    # Check if the cache file already exists
    if os.path.exists(filename):
        return filename

    print(f"Downloading from {url} to {filename}")

    # Stream the response to show the progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check for request errors

    # Total size of the file in bytes
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024  # Download in chunks of 1KB

    # Use tqdm to display the progress bar
    with (
        open(filename, "wb") as f,
        tqdm(
            desc=filename,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    return filename


def is_in_ci() -> bool:
    return envs.SGLANG_IS_IN_CI.get()


def print_highlight(html_content: str):
    if is_in_ci():
        from IPython.display import HTML, display

        html_content = str(html_content).replace("\n", "<br>")
        display(HTML(f"<strong style='color: #00008B;'>{html_content}</strong>"))
    else:
        print(html_content)


process_socket_map = weakref.WeakKeyDictionary()


def reserve_port(host, start=30000, end=40000):
    """
    Reserve an available port by trying to bind a socket.
    Returns a tuple (port, lock_socket) where `lock_socket` is kept open to hold the lock.
    """
    from sglang.srt.utils.network import try_bind_socket

    candidates = list(range(start, end))
    random.shuffle(candidates)
    for port in candidates:
        try:
            sock = try_bind_socket(host, port)
            return port, sock
        except OSError:
            continue
    raise RuntimeError("No free port available.")


def release_port(lock_socket):
    """
    Release the reserved port by closing the lock socket.
    """
    try:
        lock_socket.close()
    except Exception as e:
        print(f"Error closing socket: {e}")


def execute_shell_command(command: str) -> subprocess.Popen:
    """
    Execute a shell command and return its process handle.
    Supports leading KEY=VALUE env vars (e.g. "VAR=1 python script.py") so that
    notebook/CI commands work without requiring shell=True.
    """
    command = command.replace("\\\n", " ").replace("\\", " ")
    parts = command.split()
    env = os.environ.copy()
    i = 0
    while i < len(parts):
        part = parts[i]
        if "=" in part and not part.startswith("-") and not part.startswith("/"):
            key, _, value = part.partition("=")
            if key and value is not None and key.replace("_", "").isalnum():
                env[key] = value
                i += 1
                continue
        break
    parts = parts[i:]
    if not parts:
        raise ValueError(
            "Command contains only environment variable assignments, no executable"
        )
    return subprocess.Popen(parts, text=True, stderr=subprocess.STDOUT, env=env)


def launch_server_cmd(command: str, host: str = "0.0.0.0", port: int = None):
    """
    Launch the server using the given command.
    If no port is specified, a free port is reserved.
    """
    if port is None:
        port, lock_socket = reserve_port(host)
    else:
        lock_socket = None

    full_command = f"{command} --port {port}"
    process = execute_shell_command(full_command)

    if lock_socket is not None:
        process_socket_map[process] = lock_socket

    return process, port


def terminate_process(process):
    """
    Terminate the process and automatically release the reserved port.
    """
    from sglang.srt.utils import kill_process_tree

    kill_process_tree(process.pid)

    lock_socket = process_socket_map.pop(process, None)
    if lock_socket is not None:
        release_port(lock_socket)


def _raise_if_process_exited(process: Optional[Any]) -> None:
    if process is None:
        return

    if hasattr(process, "poll"):
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"Server process exited with code {return_code}")
        return

    if hasattr(process, "is_alive") and not process.is_alive():
        return_code = getattr(process, "exitcode", None)
        if return_code is None:
            raise RuntimeError("Server process exited")
        raise RuntimeError(f"Server process exited with code {return_code}")


def _is_wait_timeout(start_time: float, timeout: Optional[int]) -> bool:
    if timeout is None:
        return False
    return time.perf_counter() - start_time > timeout


def wait_for_http_ready(
    url: str,
    timeout: Optional[int] = None,
    process: Optional[Any] = None,
    headers: Optional[dict] = None,
    request_timeout: int = 5,
) -> None:
    """Wait for an HTTP endpoint to return status 200."""
    start_time = time.perf_counter()
    while True:
        _raise_if_process_exited(process)
        try:
            response = requests.get(url, headers=headers, timeout=request_timeout)
            if response.status_code == 200:
                return
        except requests.exceptions.RequestException:
            _raise_if_process_exited(process)

        if _is_wait_timeout(start_time, timeout):
            raise TimeoutError(
                f"Endpoint {url} did not become ready within timeout period"
            )
        time.sleep(1)


def wait_for_server(
    base_url: str,
    timeout: int = None,
    process: Optional[subprocess.Popen] = None,
) -> None:
    """Wait for the server to be ready by polling the /v1/models endpoint.

    Args:
        base_url: The base URL of the server.
        timeout: Maximum time to wait in seconds. None means wait forever.
        process: Optional server process used for early-exit checks.
    """
    wait_for_http_ready(
        url=f"{base_url}/v1/models",
        timeout=timeout,
        process=process,
        headers={"Authorization": "Bearer None"},
    )
    time.sleep(5)
    print_highlight("""\n
        NOTE: Typically, the server runs in a separate terminal.
        In this notebook, we run the server and notebook code together, so their outputs are combined.
        To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.
        To reduce the log length, we set the log level to warning for the server, the default log level is info.
        We are running those notebooks in a CI environment, so the throughput is not representative of the actual performance.
        """)


def trim_overlap(existing_text, new_chunk):
    """
    Finds the largest suffix of 'existing_text' that is a prefix of 'new_chunk'
    and removes that overlap from the start of 'new_chunk'.
    """
    max_overlap = 0
    max_possible = min(len(existing_text), len(new_chunk))
    for i in range(max_possible, 0, -1):
        if existing_text.endswith(new_chunk[:i]):
            max_overlap = i
            break
    return new_chunk[max_overlap:]


def stream_and_merge(llm, prompt, sampling_params):
    """
    1) Streams the text,
    2) Removes chunk overlaps,
    3) Returns the merged text.
    """
    final_text = ""
    for chunk in llm.generate(prompt, sampling_params, stream=True):
        chunk_text = chunk["text"]
        cleaned_chunk = trim_overlap(final_text, chunk_text)
        final_text += cleaned_chunk
    return final_text


async def async_stream_and_merge(llm, prompt, sampling_params):
    """
    Streams tokens asynchronously, removes chunk overlaps,
    and yields the cleaned chunk in real time for printing.
    """
    final_text = ""
    generator = await llm.async_generate(prompt, sampling_params, stream=True)
    async for chunk in generator:
        chunk_text = chunk["text"]
        cleaned_chunk = trim_overlap(final_text, chunk_text)
        final_text += cleaned_chunk
        yield cleaned_chunk  # yield the non-overlapping portion
