"""Common utilities."""

import base64
import gc
import importlib
import json
import logging
import os
import signal
import subprocess
import sys
import time
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from json import dumps
from typing import Optional, Union

import numpy as np
import requests
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


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


def http_request(url, json=None, stream=False, api_key=None, verify=None):
    """A faster version of requests.post with low-level urllib API."""
    headers = {"Content-Type": "application/json; charset=utf-8"}

    # add the Authorization header if an api key is provided
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"

    if stream:
        return requests.post(url, json=json, stream=True, headers=headers)
    else:
        req = urllib.request.Request(url, headers=headers)
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
            return base64.b64encode(data).decode("utf-8")
    elif isinstance(image_path, bytes):
        return base64.b64encode(image_path).decode("utf-8")
    else:
        # image_path is PIL.WebPImagePlugin.WebPImageFile
        image = image_path
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


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
    for i in range(total_frames):
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
    video_base64 = "video:" + base64.b64encode(video_bytes).decode("utf-8")

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
            logger.info(f"{sub_module_name} recive sigterm")

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


def execute_shell_command(command: str) -> subprocess.Popen:
    """
    Execute a shell command and return the process handle

    Args:
        command: Shell command as a string (can include \ line continuations)
    Returns:
        subprocess.Popen: Process handle
    """
    # Replace \ newline with space and split
    command = command.replace("\\\n", " ").replace("\\", " ")
    parts = command.split()

    return subprocess.Popen(
        parts,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait_for_server(base_url: str, timeout: int = None) -> None:
    """Wait for the server to be ready by polling the /v1/models endpoint.

    Args:
        base_url: The base URL of the server
        timeout: Maximum time to wait in seconds. None means wait forever.
    """
    start_time = time.time()
    while True:
        try:
            response = requests.get(
                f"{base_url}/v1/models",
                headers={"Authorization": "Bearer None"},
            )
            if response.status_code == 200:
                break

            if timeout and time.time() - start_time > timeout:
                raise TimeoutError("Server did not become ready within timeout period")
        except requests.exceptions.RequestException:
            time.sleep(1)


def terminate_process(process):
    """Safely terminate a process and clean up GPU memory.

    Args:
        process: subprocess.Popen object to terminate
    """
    try:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                try:
                    pgid = os.getpgid(process.pid)
                    os.killpg(pgid, signal.SIGTERM)
                    time.sleep(1)
                    if process.poll() is None:
                        os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait()
    except Exception as e:
        print(f"Warning: {e}")
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        time.sleep(2)
