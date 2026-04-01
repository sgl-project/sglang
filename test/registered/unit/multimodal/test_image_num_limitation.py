"""Tests for the image count limitation feature (issue #8540).

Validates that:
- The server arg --max-images-per-request is respected
- Per-processor IMAGE_MAX_NUM defaults work as fallback
- Server arg takes precedence over processor default
- Requests at or below the limit pass through
- Requests with no images never trigger the limit

Tests call BaseMultimodalProcessor._enforce_image_limit directly,
imported via lightweight module stubs to avoid the heavy sglang
dependency chain (GPU libs, IPython, etc.).
"""

import enum
import importlib
import importlib.util
import logging
import os
import sys
import types
import unittest
from unittest.mock import MagicMock


def _stub_module(name, attrs=None):
    """Register a stub module in sys.modules if not already present."""
    if name not in sys.modules:
        mod = types.ModuleType(name)
        if attrs:
            for k, v in attrs.items():
                setattr(mod, k, v)
        sys.modules[name] = mod
    elif attrs:
        mod = sys.modules[name]
        for k, v in attrs.items():
            if not hasattr(mod, k):
                setattr(mod, k, v)
    return sys.modules[name]


def _setup_stubs():
    """Set up minimal stubs so base_processor.py can be loaded in isolation."""

    class Modality(enum.Enum):
        IMAGE = enum.auto()
        MULTI_IMAGES = enum.auto()
        VIDEO = enum.auto()
        AUDIO = enum.auto()

    # Core package stubs
    _stub_module("sglang")
    _stub_module("sglang.srt")
    _stub_module("sglang.srt.managers")
    _stub_module(
        "sglang.srt.managers.schedule_batch",
        {
            "Modality": Modality,
            "MultimodalDataItem": MagicMock,
            "MultimodalInputFormat": MagicMock,
        },
    )
    _stub_module(
        "sglang.srt.server_args",
        {"get_global_server_args": MagicMock(return_value=None)},
    )

    # sglang.srt.utils is a package with submodules
    utils_pkg = _stub_module("sglang.srt.utils", {
        "envs": MagicMock(),
        "is_cpu": lambda: False,
        "is_npu": lambda: False,
        "is_xpu": lambda: False,
        "load_audio": MagicMock(),
        "load_image": MagicMock(),
        "load_video": MagicMock(),
        "logger": logging.getLogger("sglang.test"),
    })
    utils_pkg.__path__ = []  # Mark as package so submodule imports work

    _stub_module(
        "sglang.srt.utils.cuda_ipc_transport_utils",
        {
            "CudaIPCTransportEngine": MagicMock,
            "CudaIpcTensorTransportProxy": MagicMock,
            "MmItemMemoryPool": MagicMock,
            "MM_FEATURE_CACHE_SIZE": 128,
            "MM_ITEM_MEMORY_POOL_RECYCLE_INTERVAL": 60,
        },
    )

    _stub_module("sglang.srt.multimodal")
    _stub_module("sglang.srt.multimodal.processors")


def _load_enforce_image_limit():
    """Import _enforce_image_limit from base_processor.py."""
    base_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "python")
    )
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    _setup_stubs()

    spec = importlib.util.spec_from_file_location(
        "sglang.srt.multimodal.processors.base_processor",
        os.path.join(
            base_dir, "sglang", "srt", "multimodal", "processors", "base_processor.py"
        ),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.BaseMultimodalProcessor._enforce_image_limit


_enforce_image_limit = _load_enforce_image_limit()


def _validate_image_count(server_args, image_max_num, n_image):
    """Call the real validation logic from BaseMultimodalProcessor."""
    _enforce_image_limit(n_image, server_args.max_images_per_request, image_max_num)


def _make_server_args(max_images_per_request=None):
    """Create a minimal server_args-like object for testing."""
    args = MagicMock()
    args.max_images_per_request = max_images_per_request
    return args


class TestImageNumLimitation(unittest.TestCase):
    """Unit tests for the image count validation logic."""

    def test_no_limit_when_unset(self):
        """No error when neither server arg nor processor default is set."""
        server_args = _make_server_args(max_images_per_request=None)
        _validate_image_count(server_args, image_max_num=None, n_image=100)

    def test_server_arg_rejects_over_limit(self):
        """Requests exceeding --max-images-per-request are rejected."""
        server_args = _make_server_args(max_images_per_request=5)
        with self.assertRaises(ValueError) as ctx:
            _validate_image_count(server_args, image_max_num=None, n_image=6)
        self.assertIn("6", str(ctx.exception))
        self.assertIn("5", str(ctx.exception))
        self.assertIn("--max-images-per-request", str(ctx.exception))

    def test_server_arg_allows_at_limit(self):
        """Requests at exactly the limit should pass."""
        server_args = _make_server_args(max_images_per_request=5)
        _validate_image_count(server_args, image_max_num=None, n_image=5)

    def test_server_arg_allows_below_limit(self):
        """Requests below the limit should pass."""
        server_args = _make_server_args(max_images_per_request=10)
        _validate_image_count(server_args, image_max_num=None, n_image=3)

    def test_processor_default_rejects_over_limit(self):
        """Per-processor IMAGE_MAX_NUM is enforced when server arg is not set."""
        server_args = _make_server_args(max_images_per_request=None)
        with self.assertRaises(ValueError) as ctx:
            _validate_image_count(server_args, image_max_num=12, n_image=13)
        self.assertIn("13", str(ctx.exception))
        self.assertIn("12", str(ctx.exception))

    def test_processor_default_allows_at_limit(self):
        """Requests at exactly the processor default should pass."""
        server_args = _make_server_args(max_images_per_request=None)
        _validate_image_count(server_args, image_max_num=12, n_image=12)

    def test_server_arg_overrides_processor_default(self):
        """--max-images-per-request takes precedence over IMAGE_MAX_NUM."""
        server_args = _make_server_args(max_images_per_request=3)
        with self.assertRaises(ValueError) as ctx:
            _validate_image_count(server_args, image_max_num=12, n_image=4)
        self.assertIn("3", str(ctx.exception))

    def test_server_arg_can_raise_processor_default(self):
        """Server arg can allow more images than the processor default."""
        server_args = _make_server_args(max_images_per_request=20)
        _validate_image_count(server_args, image_max_num=12, n_image=15)

    def test_no_images_always_passes(self):
        """Zero images should never trigger the limit."""
        server_args = _make_server_args(max_images_per_request=1)
        _validate_image_count(server_args, image_max_num=1, n_image=0)

    def test_single_image_with_limit_one(self):
        """A single image with limit=1 should pass."""
        server_args = _make_server_args(max_images_per_request=1)
        _validate_image_count(server_args, image_max_num=None, n_image=1)

    def test_error_message_includes_actionable_hint(self):
        """Error message should tell users how to fix the issue."""
        server_args = _make_server_args(max_images_per_request=2)
        with self.assertRaises(ValueError) as ctx:
            _validate_image_count(server_args, image_max_num=None, n_image=5)
        msg = str(ctx.exception)
        self.assertIn("--max-images-per-request", msg)
        self.assertIn("5", msg)
        self.assertIn("2", msg)


if __name__ == "__main__":
    unittest.main()
