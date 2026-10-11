"""Turn an SGLang-backed ComfyUI graph into a runnable API request.

Pure functions over ComfyUI's API-format graph (``{node_id: {"class_type",
"inputs"}}``); nothing here imports ComfyUI. The payload builders mirror what
the generate nodes send, and ``test_api_recipe.py`` checks that they agree.
"""

import json
import pprint
import shlex
from typing import Any, Dict, List

DEFAULT_BASE_URL = "http://localhost:3000/v1"
IMAGE_NODE = "SGLDiffusionGenerateImage"
VIDEO_NODES = ("SGLDiffusionGenerateVideo", "SGLDiffusionGenerateAudioVideo")
SUPPORTED_NODES = (IMAGE_NODE,) + VIDEO_NODES
SERVER_NODE = "SGLDiffusionServerModel"
FORMATS = ["curl", "python", "sglang generate"]

# Values a linked input is read from when the upstream node is a plain
# primitive / text node.
_LITERAL_KEYS = ("value", "text", "string")
_MAX_LINK_DEPTH = 5

_IMAGE_DEFAULTS = {
    "negative_prompt": "",
    "seed": 1024,
    "steps": 6,
    "cfg": 7.0,
    "width": 1024,
    "height": 1024,
    "enable_teacache": False,
    "enhance_prompt": False,
}
_VIDEO_DEFAULTS = {
    "negative_prompt": "",
    "seed": 1024,
    "steps": 6,
    "cfg": 7.0,
    "width": 1280,
    "height": 720,
    "num_frames": 120,
    "fps": 24,
    "seconds": 5,
    "enable_teacache": False,
    "enhance_prompt": False,
    "extra_request_json": "",
}

_UNRESOLVED = object()


class Recipe:
    def __init__(self, endpoint: str, payload: Dict[str, Any], base_url: str):
        self.endpoint = endpoint
        self.payload = payload
        self.base_url = base_url
        self.files: Dict[str, str] = {}
        self.notes: List[str] = []

    @property
    def is_video(self) -> bool:
        return self.endpoint == "/videos"

    @property
    def is_multipart(self) -> bool:
        return bool(self.files) and not self.is_video


def find_target(graph: dict, node_id: str = "") -> str:
    """Pick the generate node to export; blank means the only one in the graph."""
    candidates = sorted(
        (nid for nid, node in graph.items() if node["class_type"] in SUPPORTED_NODES),
        key=lambda nid: (len(nid), nid),
    )
    if node_id:
        if node_id not in candidates:
            raise ValueError(
                f"Node {node_id} is not an SGLDiffusion generate node; "
                f"candidates: {candidates}"
            )
        return node_id
    if not candidates:
        raise ValueError("No SGLDiffusion generate node in this workflow")
    if len(candidates) > 1:
        raise ValueError(f"Several generate nodes ({candidates}); set node_id")
    return candidates[0]


def build_recipe(graph: dict, node_id: str) -> Recipe:
    node = graph[node_id]
    inputs = {key: _resolve(graph, value) for key, value in node["inputs"].items()}
    base_url = _base_url(graph, node["inputs"].get("sgld_client"))
    kind = node["class_type"]

    if kind == IMAGE_NODE:
        recipe = _image_recipe(inputs, base_url)
    else:
        recipe = _video_recipe(inputs, base_url)
    for key, value in inputs.items():
        if value is _UNRESOLVED and key not in ("sgld_client", "image", "video"):
            recipe.notes.append(f"input {key!r} comes from another node; default used")
    return recipe


def _resolve(graph: dict, value: Any, depth: int = 0) -> Any:
    if not _is_link(value):
        return value
    upstream = graph.get(value[0])
    if upstream is None or depth >= _MAX_LINK_DEPTH:
        return _UNRESOLVED
    for key in _LITERAL_KEYS:
        if key in upstream["inputs"]:
            return _resolve(graph, upstream["inputs"][key], depth + 1)
    return _UNRESOLVED


def _is_link(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[0], str)
    )


def _base_url(graph: dict, client_link: Any) -> str:
    server = graph.get(client_link[0]) if _is_link(client_link) else None
    url = server["inputs"].get("base_url") if server else None
    if (
        server is None
        or server["class_type"] != SERVER_NODE
        or not isinstance(url, str)
    ):
        return DEFAULT_BASE_URL
    url = url.rstrip("/")
    return url if url.endswith("/v1") else f"{url}/v1"


def _get(inputs: dict, defaults: dict, key: str) -> Any:
    value = inputs.get(key, defaults[key])
    return defaults[key] if value is _UNRESOLVED else value


def _prompt(inputs: dict) -> str:
    prompt = inputs.get("positive_prompt")
    if prompt is _UNRESOLVED or prompt is None:
        return "<your prompt>"
    return prompt


def _image_recipe(inputs: dict, base_url: str) -> Recipe:
    get = lambda key: _get(inputs, _IMAGE_DEFAULTS, key)  # noqa: E731
    payload = {
        "prompt": _prompt(inputs),
        "size": f"{get('width')}x{get('height')}",
        "response_format": "b64_json",
    }
    _add_common(payload, get)
    recipe = Recipe("/images/generations", payload, base_url)
    if inputs.get("image") is not None:
        recipe.endpoint = "/images/edits"
        recipe.files["image"] = "input.png"
        recipe.notes.append(
            "image input: replace input.png; an all-one-colour image sets the size instead"
        )
    return recipe


def _video_recipe(inputs: dict, base_url: str) -> Recipe:
    get = lambda key: _get(inputs, _VIDEO_DEFAULTS, key)  # noqa: E731
    payload = {
        "prompt": _prompt(inputs),
        "size": f"{get('width')}x{get('height')}",
        "seconds": get("seconds"),
        "fps": get("fps"),
        "num_frames": get("num_frames"),
    }
    _add_common(payload, get)
    if inputs.get("image") is not None:
        payload["input_reference"] = "input.png"
    if inputs.get("video") is not None:
        payload["video_path"] = "input.mp4"
    extra = get("extra_request_json")
    if isinstance(extra, str) and extra.strip():
        payload.update(json.loads(extra))
    recipe = Recipe("/videos", payload, base_url)
    for key in ("input_reference", "video_path"):
        if key in payload:
            recipe.notes.append(
                f"{key}: replace {payload[key]!r} with a path the server can read"
            )
    return recipe


def _add_common(payload: dict, get) -> None:
    if get("negative_prompt"):
        payload["negative_prompt"] = get("negative_prompt")
    payload["guidance_scale"] = get("cfg")
    payload["num_inference_steps"] = get("steps")
    if get("seed") >= 0:
        payload["seed"] = get("seed")
    if get("enable_teacache"):
        payload["enable_teacache"] = True
    if get("enhance_prompt"):
        payload["enhance_prompt"] = True


def render(recipe: Recipe, fmt: str) -> str:
    renderers = {"curl": _curl, "python": _python, "sglang generate": _cli}
    if fmt not in renderers:
        raise ValueError(f"Unknown format {fmt!r}; choose from {FORMATS}")
    notes = [f"# {note}" for note in recipe.notes]
    return "\n".join(notes + [renderers[fmt](recipe)]) + "\n"


def _form_value(value: Any) -> str:
    return str(value).lower() if isinstance(value, bool) else str(value)


def _curl(recipe: Recipe) -> str:
    url = f"{recipe.base_url}{recipe.endpoint}"
    lines = [f"curl -X POST {shlex.quote(url)}"]
    lines.append('  -H "Authorization: Bearer $SGLANG_API_KEY"')
    if recipe.is_multipart:
        for key, value in recipe.payload.items():
            lines.append(f"  -F {shlex.quote(f'{key}={_form_value(value)}')}")
        for key, path in recipe.files.items():
            lines.append(f"  -F {shlex.quote(f'{key}=@{path}')}")
    else:
        body = json.dumps(recipe.payload)
        lines.append('  -H "Content-Type: application/json"')
        lines.append(f"  -d {shlex.quote(body)}")
    command = " \\\n".join(lines)
    if recipe.is_video:
        poll = shlex.quote(f"{recipe.base_url}/videos/<id from the response>")
        command += (
            "\n# the job runs asynchronously; poll until status is completed:\n"
            f'# curl {poll} -H "Authorization: Bearer $SGLANG_API_KEY"'
        )
    return command


def _python(recipe: Recipe) -> str:
    lines = [
        "import os",
        "import time",
        "",
        "import requests",
        "",
        f"BASE_URL = {recipe.base_url!r}",
        'headers = {"Authorization": f"Bearer {os.environ[\'SGLANG_API_KEY\']}"}',
        f"payload = {pprint.pformat(recipe.payload, sort_dicts=False)}",
        "",
    ]
    url = f'f"{{BASE_URL}}{recipe.endpoint}"'
    if recipe.is_multipart:
        lines += [
            "data = {k: str(v).lower() if isinstance(v, bool) else str(v)",
            "        for k, v in payload.items()}",
            f'with open({next(iter(recipe.files.values()))!r}, "rb") as image:',
            f"    response = requests.post({url}, data=data,",
            '                             files={"image": image}, headers=headers,',
            "                             timeout=300)",
        ]
    else:
        lines.append(
            f"response = requests.post({url}, json=payload, headers=headers, timeout=300)"
        )
    lines += ["response.raise_for_status()", "result = response.json()"]
    if recipe.is_video:
        lines += [
            'while result["status"] not in ("completed", "failed"):',
            "    time.sleep(5)",
            "    result = requests.get(f\"{BASE_URL}/videos/{result['id']}\",",
            "                          headers=headers, timeout=30).json()",
        ]
    lines.append("print(result)")
    return "\n".join(lines)


def _cli(recipe: Recipe) -> str:
    payload = recipe.payload
    args = ["sglang generate", "--model-path <model-path>"]
    flags = {
        "prompt": "--prompt",
        "negative_prompt": "--negative-prompt",
        "seed": "--seed",
        "num_inference_steps": "--num-inference-steps",
        "guidance_scale": "--guidance-scale",
        "num_frames": "--num-frames",
        "fps": "--fps",
        "input_reference": "--image-path",
        "video_path": "--video-path",
    }
    for key, flag in flags.items():
        if key in payload:
            args.append(f"{flag} {shlex.quote(str(payload[key]))}")
    width, _, height = payload["size"].partition("x")
    args += [f"--width {width}", f"--height {height}"]
    if recipe.files:
        args.append(f"--image-path {recipe.files['image']}")
    if payload.get("enable_teacache"):
        args.append("--enable-teacache")
    args.append("--save-output")

    dropped = [
        key
        for key in payload
        if key not in flags
        and key not in ("size", "seconds", "enable_teacache", "response_format")
    ]
    note = ""
    if dropped:
        note = f"# no CLI flag for: {', '.join(dropped)}\n"
    return note + " \\\n  ".join(args)
