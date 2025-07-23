"""Common utilities"""

import importlib
import json
import logging
import os
import random
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
import weakref
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from json import dumps
from typing import Any, Callable, List, Optional, Tuple, Type, Union

import numpy as np
import pybase64
import requests
from IPython.display import HTML, display
from pydantic import BaseModel
from tqdm import tqdm

logger = logging.getLogger(__name__)


def convert_json_schema_to_str(json_schema: Union[dict, str, Type[BaseModel]]) -> str:
    """Convert a JSON schema to a string.
    Parameters
    ----------
    json_schema
        The JSON schema.
    Returns
    -------
    str
        The JSON schema converted to a string.
    Raises
    ------
    ValueError
        If the schema is not a dictionary, a string or a Pydantic class.
    """
    if isinstance(json_schema, dict):
        schema_str = json.dumps(json_schema)
    elif isinstance(json_schema, str):
        schema_str = json_schema
    elif issubclass(json_schema, BaseModel):
        schema_str = json.dumps(json_schema.model_json_schema())
    else:
        raise ValueError(
            f"Cannot parse schema {json_schema}. The schema must be either "
            + "a Pydantic class, a dictionary or a string that contains the JSON "
            + "schema specification"
        )
    return schema_str


def get_exception_traceback():
    etype, value, tb = sys.exc_info()
    err_str = "".join(traceback.format_exception(etype, value, tb))
    return err_str


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


class HttpResponse:
    def __init__(self, resp):
        self.resp = resp

    def json(self):
        return json.loads(self.resp.read())

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
        # image_path is PIL.WebPImagePlugin.WebPImageFile
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


def _is_chinese_char(cp: int):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def find_printable_text(text: str):
    """Returns the longest printable substring of text that contains only entire words."""
    # Borrowed from https://github.com/huggingface/transformers/blob/061580c82c2db1de9139528243e105953793f7a2/src/transformers/generation/streamers.py#L99

    # After the symbol for a new line, we flush the cache.
    if text.endswith("\n"):
        return text
    # If the last token is a CJK character, we print the characters.
    elif len(text) > 0 and _is_chinese_char(ord(text[-1])):
        return text
    # Otherwise if the penultimate token is a CJK character, we print the characters except for the last one.
    elif len(text) > 1 and _is_chinese_char(ord(text[-2])):
        return text[:-1]
    # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
    # which may change with the subsequent token -- there are probably smarter ways to do this!)
    else:
        return text[: text.rfind(" ") + 1]


def graceful_registry(sub_module_name: str):
    def graceful_shutdown(signum, frame):
        logger.info(
            f"{sub_module_name} Received signal to shutdown. Performing graceful shutdown..."
        )
        if signum == signal.SIGTERM:
            logger.info(f"{sub_module_name} receive sigterm")

    signal.signal(signal.SIGTERM, graceful_shutdown)


class LazyImport:
    """Lazy import to make `import sglang` run faster."""

    def __init__(self, module_name: str, class_name: str):
        self.module_name = module_name
        self.class_name = class_name
        self._module = None

    def _load(self):
        if self._module is None:
            module = importlib.import_module(self.module_name)
            self._module = getattr(module, self.class_name)
        return self._module

    def __getattr__(self, name: str):
        module = self._load()
        return getattr(module, name)

    def __call__(self, *args, **kwargs):
        module = self._load()
        return module(*args, **kwargs)


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
    with open(filename, "wb") as f, tqdm(
        desc=filename,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    return filename


def is_in_ci():
    from sglang.test.test_utils import is_in_ci

    return is_in_ci()


def print_highlight(html_content: str):
    if is_in_ci():
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
    candidates = list(range(start, end))
    random.shuffle(candidates)

    for port in candidates:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            # Attempt to bind to the port on localhost
            sock.bind((host, port))
            return port, sock
        except socket.error:
            sock.close()  # Failed to bind, try next port
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
    """
    command = command.replace("\\\n", " ").replace("\\", " ")
    parts = command.split()
    return subprocess.Popen(parts, text=True, stderr=subprocess.STDOUT)


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


def wait_for_server(base_url: str, timeout: int = None) -> None:
    """Wait for the server to be ready by polling the /v1/models endpoint.

    Args:
        base_url: The base URL of the server
        timeout: Maximum time to wait in seconds. None means wait forever.
    """
    start_time = time.perf_counter()
    while True:
        try:
            response = requests.get(
                f"{base_url}/v1/models",
                headers={"Authorization": "Bearer None"},
            )
            if response.status_code == 200:
                time.sleep(5)
                print_highlight(
                    """\n
                    NOTE: Typically, the server runs in a separate terminal.
                    In this notebook, we run the server and notebook code together, so their outputs are combined.
                    To improve clarity, the server logs are displayed in the original black color, while the notebook outputs are highlighted in blue.
                    We are running those notebooks in a CI parallel environment, so the throughput is not representative of the actual performance.
                    """
                )
                break

            if timeout and time.perf_counter() - start_time > timeout:
                raise TimeoutError("Server did not become ready within timeout period")
        except requests.exceptions.RequestException:
            time.sleep(1)


class TypeBasedDispatcher:
    def __init__(self, mapping: List[Tuple[Type, Callable]]):
        self._mapping = mapping

    def __call__(self, obj: Any):
        for ty, fn in self._mapping:
            if isinstance(obj, ty):
                return fn(obj)
        raise ValueError(f"Invalid object: {obj}")


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


def resolve_obj_by_qualname(qualname: str) -> Any:
    """
    Resolve an object by its fully qualified name.
    """
    module_name, obj_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)
