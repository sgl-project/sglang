"""Unit tests for the per-processor default image count limit.

`IMAGE_NUM_LIMITATION` is a safety net against OOM from oversized image
batches (each 4K image can cost ~1 GiB of GPU memory in the fast image
processor). An explicit ``--limit-mm-data-per-request`` image entry takes
priority over the processor default.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase


def _make_processor(
    image_num_limitation=None,
    cli_limit=None,
):
    """Create a BaseMultimodalProcessor via the real __init__ with mocked deps.

    cli_limit, when set, is published as --limit-mm-data-per-request
    {"image": N} so the priority order against the class default is exercised.
    """
    if image_num_limitation is not None:

        class _Processor(BaseMultimodalProcessor):
            IMAGE_NUM_LIMITATION = image_num_limitation

    else:
        _Processor = BaseMultimodalProcessor

    override = get_context().override_server_args(
        mm_process_config={},
        allowed_media_domains=[],
        mm_processor_worker_num=0,
        mm_io_worker_num=0,
        mm_preprocess_cache_size_mb=None,
        tokenizer_worker_num=1,
        trust_mm_content_hashes=False,
        media_url_max_file_size_mb=64,
        limit_mm_data_per_request=({"image": cli_limit} if cli_limit else None),
    )
    override.install()

    server_args = ServerArgs(
        model_path="dummy",
        mm_process_config={},
        allowed_media_domains=[],
        mm_processor_worker_num=0,
        mm_io_worker_num=0,
        mm_preprocess_cache_size_mb=None,
        tokenizer_worker_num=1,
        trust_mm_content_hashes=False,
        media_url_max_file_size_mb=64,
        disable_fast_image_processor=False,
        limit_mm_data_per_request=({"image": cli_limit} if cli_limit else None),
    )

    with patch.object(_Processor, "__abstractmethods__", set()):
        proc = _Processor(
            hf_config=MagicMock(),
            server_args=server_args,
            _processor=MagicMock(),
            transport_mode=None,
        )
    if proc.mm_processor_executor is not None:
        proc.mm_processor_executor.shutdown()
    return override, proc


def _image_list(count):
    # Opaque non-dict items: the limit check runs before any decoding, so the
    # items never need to be real images.
    return [f"image-{i}" for i in range(count)]


class TestImageNumLimitation(CustomTestCase):
    def setUp(self):
        super().setUp()
        self._overrides = []

    def _make(self, **kwargs):
        override, proc = _make_processor(**kwargs)
        self._overrides.append(override)
        return proc

    def tearDown(self):
        for override in reversed(self._overrides):
            override.restore()
        super().tearDown()

    def test_default_limit_enforced(self):
        proc = self._make()
        with self.assertRaisesRegex(ValueError, "at most 5 image"):
            proc.validate_image_num_limitation(_image_list(6))

    def test_within_default_limit_passes(self):
        proc = self._make()
        # Must not raise.
        proc.validate_image_num_limitation(_image_list(5))
        proc.validate_image_num_limitation(_image_list(0))
        proc.validate_image_num_limitation(None)

    def test_subclass_override_enforced(self):
        proc = self._make(image_num_limitation=12)
        with self.assertRaisesRegex(ValueError, "at most 12 image"):
            proc.validate_image_num_limitation(_image_list(13))
        proc.validate_image_num_limitation(_image_list(12))

    def test_cli_limit_takes_priority(self):
        # CLI asks for 2: it must win over both the class default (5) and a
        # subclass override (12).
        proc = self._make(image_num_limitation=12, cli_limit=2)
        with self.assertRaisesRegex(ValueError, "at most 2 image"):
            proc.validate_image_num_limitation(_image_list(3))
        proc.validate_image_num_limitation(_image_list(2))

    def test_cli_limit_can_raise_default(self):
        proc = self._make(cli_limit=8)
        proc.validate_image_num_limitation(_image_list(8))
        with self.assertRaisesRegex(ValueError, "at most 8 image"):
            proc.validate_image_num_limitation(_image_list(9))

    def test_error_message_mentions_cli_flag(self):
        proc = self._make()
        with self.assertRaisesRegex(ValueError, r"--limit-mm-data-per-request"):
            proc.validate_image_num_limitation(_image_list(99))

    def test_preprocessed_single_item_exempt(self):
        # A single processor_output / precomputed_embedding dict carries no
        # decode cost, so the limit must not reject it.
        proc = self._make()
        proc.validate_image_num_limitation(
            [{"format": "processor_output", "pixel_values": None}]
        )


class TestLoadMmDataRejectsOversized(CustomTestCase):
    """The limit must fire inside load_mm_data, before any image is fetched."""

    def test_load_mm_data_rejects_oversized_batch(self):
        override, proc = _make_processor()
        self.addCleanup(override.restore)
        mm_tokens = SimpleNamespace(
            image_token="<image>",
            video_token="<video>",
            audio_token=None,
        )
        with patch.object(
            BaseMultimodalProcessor,
            "fast_load_mm_data",
            new=lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not load")),
        ):
            with self.assertRaisesRegex(ValueError, "at most 5 image"):
                import asyncio

                asyncio.run(
                    proc.load_mm_data(
                        prompt="hello <image> world",
                        multimodal_tokens=mm_tokens,
                        image_data=_image_list(6),
                    )
                )


if __name__ == "__main__":
    unittest.main()
