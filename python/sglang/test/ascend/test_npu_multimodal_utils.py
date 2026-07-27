"""
Shared utilities for multimodal NPU test scripts.

"""

import base64
import io
import logging
import os
import shlex
import subprocess
import time
from enum import Enum

from PIL import Image, ImageDraw

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ============================================================
# Color & Shape enums
# ============================================================


class Color(Enum):
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    PURPLE = (128, 0, 128)
    TEAL = (0, 128, 128)


_COLOR_NAME_MAP = {
    "red": Color.RED,
    "green": Color.GREEN,
    "blue": Color.BLUE,
    "purple": Color.PURPLE,
    "teal": Color.TEAL,
}


class Shape(Enum):
    RECTANGLE = "rectangle"
    ELLIPSE = "ellipse"


_SHAPE_NAME_MAP = {
    "rectangle": Shape.RECTANGLE,
    "ellipse": Shape.ELLIPSE,
}

# ============================================================
# Image helpers
# ============================================================


def create_test_image(
    width=256, height=256, color=Color.RED, shape=Shape.RECTANGLE, label=None
):
    """Create a synthetic PNG test image and return (raw_bytes, base64_str).

    The image has a solid-colour background and a contrasting shape,
    giving a VLM enough visual structure to describe.

    Args:
        width, height: Image dimensions in pixels.
        color: A ``Color`` enum value (e.g. ``Color.RED``, ``Color.GREEN``,
            ``Color.BLUE``).
        shape: A ``Shape`` enum value (``Shape.RECTANGLE`` or ``Shape.ELLIPSE``).
            Strings ``"rectangle"`` / ``"ellipse"`` are also accepted for
            backward compatibility.
        label: Optional text label drawn near the center (min 64 px needed).

    Returns:
        Tuple of (raw_bytes, base64_encoded_string)
    """
    rgb = _COLOR_NAME_MAP.get(color, color).value
    shape_enum = _SHAPE_NAME_MAP.get(shape, shape)

    img = Image.new("RGB", (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    if shape_enum == Shape.ELLIPSE:
        margin_w = max(1, width // 6)
        margin_h = max(1, height // 3)
        draw.ellipse(
            [margin_w, margin_h, width - margin_w, height - margin_h],
            fill=rgb,
            outline=(255, 255, 255),
        )
    else:
        draw.rectangle(
            [10, height // 4, width - 10, 3 * height // 4],
            fill=rgb,
            outline=(255, 255, 255),
            width=3,
        )

    if label and min(width, height) >= 64:
        cx, cy = width // 2, height // 2
        draw.text((cx - len(label) * 3, cy - 6), label, fill=(255, 255, 255))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    return img_bytes, base64.b64encode(img_bytes).decode("utf-8")


# ============================================================
# Content verification
# ============================================================

_SHAPE_SYNONYMS = {
    "ellipse": {"ellipse", "oval", "ovoid", "elliptical"},
    "rectangle": {"rectangle", "rectangular", "square"},
}


def assert_color_and_shape(test_case, text, expected_color, expected_shape, prefix=""):
    """Unified assertion for color and shape in VLM output.

    Args:
        test_case: unittest.TestCase instance.
        text: The text output from the model.
        expected_color: Expected color string (e.g. ``"red"``, ``"blue"``).
        expected_shape: Expected shape string (e.g. ``"ellipse"``,
            ``"rectangle"``).
        prefix: Optional label prepended to failure messages.

    Colors are matched with exact lowercase substring.  Shapes are
    matched against a synonym set so that ``"oval"`` is accepted for
    ``"ellipse"`` and ``"rectangular"``/``"square"`` for ``"rectangle"``.
    """
    text_lower = text.lower()
    snippet = text[:300]

    color_lower = expected_color.lower()
    test_case.assertIn(
        color_lower,
        text_lower,
        f"{prefix}Expected color '{expected_color}' not found in output. "
        f"First 300 chars: {snippet}",
    )

    shape_lower = expected_shape.lower()
    synonyms = _SHAPE_SYNONYMS.get(shape_lower, {shape_lower})
    if not any(s in text_lower for s in synonyms):
        test_case.fail(
            f"{prefix}Expected shape '{expected_shape}' "
            f"(synonyms: {sorted(synonyms)}) not found in output. "
            f"First 300 chars: {snippet}"
        )


def assert_text_contains(test_case, text, hints=None):
    """Fail if *text* contains none of the *hints* keywords.

    When *hints* is None, defaults to common image-related keywords
    (colour, shape, etc.).
    """
    if hints is None:
        hints = [
            "image",
            "picture",
            "color",
            "red",
            "blue",
            "green",
            "rectangle",
            "ellipse",
            "circle",
            "shape",
        ]
    if content_has_keywords(text, hints):
        return
    test_case.fail(
        f"Response does not contain any expected keyword {hints}.\n"
        f"First 300 chars: {text[:300]}"
    )


def content_has_keywords(text, keywords=None):
    """Return True if *text* contains at least one keyword."""
    if keywords is None:
        keywords = [
            "image",
            "picture",
            "color",
            "red",
            "blue",
            "green",
            "yellow",
            "rectangle",
            "ellipse",
            "circle",
            "shape",
            "square",
            "triangle",
        ]
    return any(kw in text.lower() for kw in keywords)


# ============================================================
# Server launcher
# ============================================================

NPU_SERVER_ARGS = [
    "--device",
    "npu",
    "--attention-backend",
    "ascend",
    "--trust-remote-code",
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
]


def launch_server(
    model, extra_args=None, timeout=None, env=None, extra_env=None, port=None
):
    """Launch an NPU SGLang server and return (process, base_url).

    If *timeout* is None, imports and uses DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH.
    If *extra_env* is given, its key-value pairs are merged into *env* (or os.environ).
    If *port* is given, the server binds to that port (enabling multiple servers
    to run in parallel); otherwise DEFAULT_URL_FOR_TEST is used.
    """
    if timeout is None:
        from sglang.test.test_utils import DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

        timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

    from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, popen_launch_server

    if extra_env:
        merged = dict(env or os.environ)
        merged.update(extra_env)
        env = merged

    base_url = DEFAULT_URL_FOR_TEST if port is None else f"http://127.0.0.1:{port}"
    args = list(NPU_SERVER_ARGS)
    if extra_args:
        args.extend(extra_args)
    process = popen_launch_server(
        model, base_url, timeout=timeout, other_args=args, env=env
    )
    time.sleep(2)
    return process, base_url


def launch_router(prefill_url, decode_url, host, port):
    """Launch sglang_router for PD disaggregation and return (process, router_url).

    Blocks until the router health-check endpoint responds.
    """
    from sglang.utils import wait_for_http_ready

    lb_command = [
        "python3",
        "-m",
        "sglang_router.launch_router",
        "--pd-disaggregation",
        "--mini-lb",
        "--prefill",
        prefill_url,
        "--decode",
        decode_url,
        "--host",
        host,
        "--port",
        str(port),
    ]
    logger.info(f"Launching router: {shlex.join(lb_command)}")
    process = subprocess.Popen(lb_command)
    router_url = f"http://{host}:{port}"
    wait_for_http_ready(url=router_url + "/health", timeout=300, process=process)
    logger.info(f"Router {router_url} is ready")
    return process, router_url


# ============================================================
# API & Chat helpers
# ============================================================


def data_url(b64_str):
    """Convert a base64-encoded image to a data URL."""
    return f"data:image/png;base64,{b64_str}"


def image_content(b64_str):
    """Build an image_url content block for OpenAI messages."""
    return {
        "type": "image_url",
        "image_url": {"url": data_url(b64_str)},
    }


def text_content(text):
    """Build a text content block for OpenAI messages."""
    return {"type": "text", "text": text}


def chat(base_url, messages, temperature=0, max_tokens=256, seed=None):
    """Send a chat completion request and return the response text."""
    import openai

    client = openai.Client(api_key="sk-123456", base_url=f"{base_url}/v1")
    create_kwargs = dict(
        model="default",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if seed is not None:
        create_kwargs["seed"] = seed
    response = client.chat.completions.create(**create_kwargs)
    text = response.choices[0].message.content
    logger.info(f"VLM response:\n{text}\n")
    return text


def chat_single_image(
    base_url, image_b64, prompt, max_tokens=128, temperature=0, seed=None
):
    """Send a single-image + text request, return response text."""
    return chat(
        base_url,
        [
            {
                "role": "user",
                "content": [image_content(image_b64), text_content(prompt)],
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
    )
