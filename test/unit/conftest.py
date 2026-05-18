"""
conftest.py — allow unit tests to import sglang.srt.* sub-packages without
needing the full sglang runtime dependency tree (torch, tqdm, IPython, …).

Strategy
--------
1. Pre-register ``sglang`` and ``sglang.srt`` as proper package namespaces
   (with real ``__path__`` entries) so Python can find sub-packages on disk.
2. Stub specific third-party packages (orjson, fastapi) and sglang sub-modules
   that serving_base.py depends on, so they can be imported without their own
   deep dependency chains.

Because ``sglang`` is placed in sys.modules *before* any import happens,
``sglang/__init__.py`` is never executed, which avoids the torch/tqdm/IPython
imports it triggers.
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

_SGLANG_PYTHON = Path(__file__).parent.parent.parent / "python"
_SGLANG_PKG = _SGLANG_PYTHON / "sglang"

# Ensure python/ is on sys.path so relative sub-package discovery works.
_python_dir = str(_SGLANG_PYTHON)
if _python_dir not in sys.path:
    sys.path.insert(0, _python_dir)


# ---------------------------------------------------------------------------
# 1.  Establish ``sglang`` and ``sglang.srt`` as package namespaces FIRST.
#     This must happen before any other sys.modules manipulation that could
#     create stub parent entries with empty __path__.
# ---------------------------------------------------------------------------

def _make_pkg(dotted_name: str, fs_path: Path) -> types.ModuleType:
    """Create a package-like module pointing at a real directory."""
    mod = types.ModuleType(dotted_name)
    mod.__path__ = [str(fs_path)]
    mod.__package__ = dotted_name
    mod.__file__ = str(fs_path / "__init__.py")
    sys.modules[dotted_name] = mod
    return mod


_make_pkg("sglang", _SGLANG_PKG)
_make_pkg("sglang.srt", _SGLANG_PKG / "srt")
# sglang.srt.iochain is intentionally NOT pre-registered here so that Python
# loads its __init__.py from disk (which exports IOChain, IOContext, IOProcessor,
# RequestLoggingProcessor, etc.).
_make_pkg("sglang.srt.entrypoints", _SGLANG_PKG / "srt" / "entrypoints")
_make_pkg("sglang.srt.entrypoints.openai", _SGLANG_PKG / "srt" / "entrypoints" / "openai")


# ---------------------------------------------------------------------------
# 2.  Stub third-party packages needed by the modules under test.
# ---------------------------------------------------------------------------

def _leaf_stub(name: str, **attrs) -> types.ModuleType:
    """Register a leaf module stub (parents must already exist)."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# orjson
_orjson = _leaf_stub("orjson")
_orjson.loads = json.loads
_orjson.dumps = json.dumps
_orjson.JSONDecodeError = json.JSONDecodeError

# fastapi
class _HTTPException(Exception):
    def __init__(self, status_code: int = 400, detail: str = ""):
        self.status_code = status_code
        self.detail = detail

_fastapi = _leaf_stub("fastapi")
_fastapi.HTTPException = _HTTPException
_fastapi.Request = object

class _ORJSONResponse:
    def __init__(self, content=None, status_code: int = 200):
        self.content = content
        self.status_code = status_code

class _StreamingResponse:
    def __init__(self, body_iterator, *, status_code: int = 200,
                 media_type=None, headers=None, background=None):
        self.body_iterator = body_iterator
        self.status_code = status_code
        self.media_type = media_type
        self.headers = headers or {}
        self.background = background

_fastapi_responses = _leaf_stub("fastapi.responses")
_fastapi_responses.ORJSONResponse = _ORJSONResponse
_fastapi_responses.StreamingResponse = _StreamingResponse
_fastapi.responses = _fastapi_responses


# ---------------------------------------------------------------------------
# 3.  Stub sglang.srt sub-modules that serving_base.py imports directly.
#     These stubs are registered as leaf entries so they are used instead of
#     loading the real files (which have their own heavy dependency trees).
# ---------------------------------------------------------------------------

class _DS32EncodingError(Exception):
    pass

_leaf_stub(
    "sglang.srt.entrypoints.openai.encoding_dsv32",
    DS32EncodingError=_DS32EncodingError,
)

class _ErrorResponse:
    def __init__(self, **kw):
        self.__dict__.update(kw)
    def model_dump(self):
        return self.__dict__

class _OpenAIServingRequest:
    pass

_leaf_stub(
    "sglang.srt.entrypoints.openai.protocol",
    ErrorResponse=_ErrorResponse,
    OpenAIServingRequest=_OpenAIServingRequest,
)

class _GenerateReqInput:
    received_time = None

class _EmbeddingReqInput:
    received_time = None

_leaf_stub(
    "sglang.srt.managers.io_struct",
    GenerateReqInput=_GenerateReqInput,
    EmbeddingReqInput=_EmbeddingReqInput,
)

class _TokenizerManager:
    pass

_leaf_stub(
    "sglang.srt.managers.tokenizer_manager",
    TokenizerManager=_TokenizerManager,
)

_leaf_stub(
    "sglang.srt.observability.req_time_stats",
    monotonic_time=time.monotonic,
)

class _ServerArgs:
    tokenizer_metrics_allowed_custom_labels = None
    tokenizer_metrics_custom_labels_header = None
    io_processors: list = []

_leaf_stub("sglang.srt.server_args", ServerArgs=_ServerArgs)
