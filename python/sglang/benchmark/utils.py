import asyncio
import base64
import io
import json
import os
import resource
from json import JSONDecodeError
from pathlib import Path
from typing import Dict, Optional, Union

import aiohttp
import requests
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast


def create_bench_client_session():
    # When the pressure is big, the read buffer could be full before aio thread read
    # the content. We increase the read_bufsize from 64K to 10M.
    # Define constants for timeout and buffer size for clarity and maintainability
    BENCH_AIOHTTP_TIMEOUT_SECONDS = 6 * 60 * 60  # 6 hours
    BENCH_AIOHTTP_READ_BUFSIZE_BYTES = 10 * 1024**2  # 10 MB
    aiohttp_timeout = aiohttp.ClientTimeout(total=BENCH_AIOHTTP_TIMEOUT_SECONDS)
    return aiohttp.ClientSession(
        timeout=aiohttp_timeout, read_bufsize=BENCH_AIOHTTP_READ_BUFSIZE_BYTES
    )


def remove_prefix(text: str, prefix: str) -> str:
    return text[len(prefix) :] if text.startswith(prefix) else text


def get_auth_headers() -> Dict[str, str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def get_model(pretrained_model_name_or_path: str) -> str:
    if os.getenv("SGLANG_USE_MODELSCOPE", "false").lower() == "true":
        import huggingface_hub.constants
        from modelscope import snapshot_download

        model_path = snapshot_download(
            model_id=pretrained_model_name_or_path,
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            ignore_file_pattern=[".*.pt", ".*.safetensors", ".*.bin"],
        )

        return model_path
    return pretrained_model_name_or_path


def get_tokenizer(
    pretrained_model_name_or_path: str,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    assert (
        pretrained_model_name_or_path is not None
        and pretrained_model_name_or_path != ""
    )
    if pretrained_model_name_or_path.endswith(
        ".json"
    ) or pretrained_model_name_or_path.endswith(".model"):
        from sglang.srt.hf_transformers_utils import get_tokenizer

        return get_tokenizer(pretrained_model_name_or_path)

    if pretrained_model_name_or_path is not None and not os.path.exists(
        pretrained_model_name_or_path
    ):
        pretrained_model_name_or_path = get_model(pretrained_model_name_or_path)
    return AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=True
    )


def is_file_valid_json(path):
    if not os.path.isfile(path):
        return False
    try:
        with open(path) as f:
            json.load(f)
        return True
    except (JSONDecodeError, UnicodeDecodeError) as e:
        print(f"{path} exists but is not valid JSON ({e}), re-downloading.")
        return False


def download_and_cache_file(url: str, filename: Optional[str] = None):
    """Read and cache a file from a url."""
    if filename is None:
        filename = os.path.join("/tmp", url.split("/")[-1])

    # Check if the cache file already exists
    if is_file_valid_json(filename):
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


def set_ulimit(target_soft_limit=65535):
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            print(f"Fail to set RLIMIT_NOFILE: {e}")


def check_chat_template(model_path: str) -> bool:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return "chat_template" in tokenizer.init_kwargs or tokenizer.chat_template
    except Exception as e:
        print(f"Failed to load tokenizer config for {model_path}: {e}")
        return False


def get_bool_env_var(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default)
    return value.lower() in ("true", "1")
