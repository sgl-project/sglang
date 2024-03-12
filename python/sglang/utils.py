"""Common utilities."""

import base64
import json
import threading
import urllib.request
from io import BytesIO
from json import dumps

import requests


def get_available_gpu_memory(gpu_id, distributed=True):
    """
    Get available memory for cuda:gpu_id device.
    When distributed is True, the available memory is the minimum available memory of all GPUs.
    """
    import torch

    num_gpus = torch.cuda.device_count()
    assert gpu_id < num_gpus

    if torch.cuda.current_device() != gpu_id:
        print(
            f"WARNING: current device is not {gpu_id}, but {torch.cuda.current_device()}, ",
            "which may cause useless memory allocation for torch CUDA context.",
        )

    free_gpu_memory, _ = torch.cuda.mem_get_info(gpu_id)

    if distributed:
        tensor = torch.tensor(free_gpu_memory, dtype=torch.float32).to(
            torch.device("cuda", gpu_id)
        )
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
        free_gpu_memory = tensor.item()

    return free_gpu_memory / (1 << 30)


def is_same_type(values):
    """Return whether the elements in values are of the same type."""
    if len(values) <= 1:
        return True
    else:
        t = type(values[0])
        return all(isinstance(v, t) for v in values[1:])


def read_jsonl(filename: str):
    """Read a JSONL file."""
    rets = []
    with open(filename) as fin:
        for line in fin:
            if line.startswith("#"):
                continue
            rets.append(json.loads(line))
    return rets


def dump_state_text(filename, states, mode="w"):
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
    url, json=None, stream=False, auth_token=None, api_key=None, verify=None
):
    """A faster version of requests.post with low-level urllib API."""
    headers = {"Content-Type": "application/json; charset=utf-8"}

    # add the Authorization header if an auth token is provided
    if auth_token is not None:
        headers["Authorization"] = f"Bearer {auth_token}"

    # add the API Key header if an API key is provided
    if api_key is not None:
        headers["X-API-Key"] = api_key

    if stream:
        return requests.post(url, json=json, stream=True, headers=headers)
    else:
        req = urllib.request.Request(url, headers=headers)
        if json is None:
            data = None
        else:
            data = bytes(dumps(json), encoding="utf-8")
        resp = urllib.request.urlopen(req, data=data, cafile=verify)
        return HttpResponse(resp)


def encode_image_base64(image_path):
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


def _is_chinese_char(cp):
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


def find_printable_text(text):
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


def run_with_timeout(func, args=(), kwargs=None, timeout=None):
    """Run a function with timeout."""
    ret_value = []

    def _target_func():
        ret_value.append(func(*args, **(kwargs or {})))

    t = threading.Thread(target=_target_func)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise TimeoutError()

    if not ret_value:
        raise RuntimeError()

    return ret_value[0]
