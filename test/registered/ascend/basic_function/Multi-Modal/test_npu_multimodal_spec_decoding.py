"""
NPU multimodal Speculative decoding -> speedup without correctness loss.

"""

import base64
import io
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image, ImageDraw

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_5_9B_WEIGHTS_PATH
from sglang.test.ascend.test_npu_multimodal_utils import (
    chat_single_image,
    content_has_keywords,
    launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=200, suite="full-2-npu-a3", nightly=True)


def _create_multi_object_image():
    """Create a 640x480 image with four distinct coloured objects.

    Layout:
        - Red circle   (top-left)
        - Blue rectangle  (top-right)
        - Green triangle  (bottom-left)
        - Yellow circle   (bottom-right)

    Designed for P1-001 where the prompt asks the model to enumerate each
    object individually.
    """
    img = Image.new("RGB", (640, 480), color=(30, 30, 120))
    draw = ImageDraw.Draw(img)

    # Red circle
    draw.ellipse([50, 50, 200, 200], fill=(255, 0, 0), outline=(255, 255, 255), width=2)
    # Blue rectangle
    draw.rectangle(
        [300, 50, 500, 200], fill=(0, 0, 255), outline=(255, 255, 255), width=2
    )
    # Green triangle (approximated as polygon)
    draw.polygon(
        [(150, 300), (50, 450), (250, 450)],
        fill=(0, 255, 0),
        outline=(255, 255, 255),
        width=2,
    )
    # Yellow circle
    draw.ellipse(
        [350, 300, 500, 450], fill=(255, 255, 0), outline=(255, 255, 255), width=2
    )

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    return img_bytes, base64.b64encode(img_bytes).decode("utf-8")


# ===================================================================
# P1-001: Speculative decoding variants (shared test logic)
# ===================================================================


class _SpecDecBase(CustomTestCase):
    """Shared infrastructure for speculative-decoding test variants."""

    _model = QWEN3_5_9B_WEIGHTS_PATH

    _common_args = [
        "--mem-fraction-static",
        "0.78",
        "--mamba-radix-cache-strategy",
        "extra_buffer",
        "--tp-size",
        1,
        "--dtype",
        "bfloat16",
        "--mamba-ssm-dtype",
        "bfloat16",
    ]

    _spec_args = [
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
    ]

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = _create_multi_object_image()
        cls._prompt = "Describe each object in the image"

        cls._baseline_process = None
        cls._spec_process = None

        # ---- Launch baseline + speculative servers in parallel ----
        # They run on separate chips (--base-gpu-id 1 vs 0) and separate
        # ports, so both can load and serve concurrently.
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                fut_baseline = executor.submit(
                    launch_server,
                    cls._model,
                    extra_args=cls._common_args
                    + cls._extra_args
                    + ["--base-gpu-id", "1"],
                    port=cls._baseline_port,
                )
                fut_spec = executor.submit(
                    launch_server,
                    cls._model,
                    extra_args=cls._common_args
                    + cls._extra_args
                    + cls._spec_args
                    + ["--base-gpu-id", "0"],
                    port=cls._spec_port,
                )
                cls._baseline_process, cls._baseline_url = fut_baseline.result()
                cls._spec_process, cls._spec_url = fut_spec.result()
        except Exception:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls):
        if cls._spec_process is not None:
            kill_process_tree(cls._spec_process.pid)
        if cls._baseline_process is not None:
            kill_process_tree(cls._baseline_process.pid)

    def _run_spec_dec_test(self):
        tag = self.__class__.__name__
        baseline_url = self._baseline_url
        spec_url = self._spec_url

        # ---- Baseline (non-speculative) output ----
        output_bl = chat_single_image(
            baseline_url,
            self._image_b64,
            self._prompt,
            max_tokens=64,
            temperature=0,
        )
        self.assertTrue(output_bl, f"{tag}: Baseline returned empty output")
        self.assertGreater(
            len(output_bl), 5, f"{tag}: Baseline too short: '{output_bl}'"
        )
        self.assertTrue(
            content_has_keywords(output_bl),
            f"{tag}: Baseline output doesn't reference image: '{output_bl[:200]}'",
        )

        # ---- Speculative decoding output ----
        output_spec = chat_single_image(
            spec_url,
            self._image_b64,
            self._prompt,
            max_tokens=64,
            temperature=0,
        )
        self.assertTrue(output_spec, f"{tag}: Spec returned empty output")
        self.assertGreater(
            len(output_spec), 5, f"{tag}: Spec too short: '{output_spec}'"
        )
        self.assertTrue(
            content_has_keywords(output_spec),
            f"{tag}: Spec output doesn't reference image: '{output_spec[:200]}'",
        )

        # At temperature=0 (greedy), speculative decoding must produce
        # identical output to the non-speculative baseline.  The rejection
        # sampling algorithm guarantees the output distribution matches the
        # target model, and greedy collapses that to a deterministic sequence.
        self.assertEqual(
            output_bl,
            output_spec,
            f"{tag}: Spec output differs from baseline at temperature=0:\n"
            f"  baseline: '{output_bl[:200]}'\n"
            f"  spec:     '{output_spec[:200]}'",
        )

        # ---- Verify MTP is actually accepting drafts ----
        for _ in range(3):
            chat_single_image(
                spec_url,
                self._image_b64,
                self._prompt,
                max_tokens=64,
                temperature=0,
            )

        # Poll for avg_spec_accept_length to stabilise instead of a fixed sleep.
        avg_spec_accept_length = 1.0
        for _ in range(15):
            time.sleep(1)
            try:
                server_info = requests.get(spec_url + "/server_info", timeout=10).json()
                avg_spec_accept_length = server_info["internal_states"][0].get(
                    "avg_spec_accept_length", 1.0
                )
            except Exception:
                continue
            if avg_spec_accept_length > 1.5:
                break

        print(f"  [{tag}] avg_spec_accept_length={avg_spec_accept_length:.2f}")
        self.assertGreater(
            avg_spec_accept_length,
            1.5,
            f"{tag}: accept_length={avg_spec_accept_length:.2f} <= 1.5 — "
            f"MTP drafts are mostly rejected, speculative decoding is ineffective",
        )


class TestMultimodalSpeculativeDecoding(_SpecDecBase):
    """Enable cuda-graph: verify MTP speculative decoding produces correct multimodal
    output vs a non-speculative baseline.

    Uses Qwen3.5-9B with built-in MTP heads (no external draft model).
    Qwen3.5-9B has GDN attention + DeepStack ViT + native NEXTN support.

    [Test Category] multimodal
    [Test Target] multimodal + speculative decoding + cuda-graph enabled
    """

    _baseline_port = 21000
    _spec_port = 21001
    _extra_args = ["--cuda-graph-bs-decode", "1", "2"]

    def test_speculative_decoding_speedup_and_correctness(self):
        """Compare MTP speculative decoding vs non-speculative baseline."""
        self._run_spec_dec_test()


class TestMultimodalSpeculativeDecodingNoGraph(_SpecDecBase):
    """Disable cuda-graph: verify MTP speculative decoding produces
    correct multimodal output with cuda-graph disabled.

    Uses Qwen3.5-9B with built-in MTP heads (no external draft model).
    Qwen3.5-9B has GDN attention + DeepStack ViT + native NEXTN support.

    [Test Category] multimodal
    [Test Target] multimodal + speculative decoding + cuda-graph disabled
    """

    _baseline_port = 21002
    _spec_port = 21003
    _extra_args = [
        "--cuda-graph-backend-prefill",
        "disabled",
        "--cuda-graph-backend-decode",
        "disabled",
    ]

    def test_speculative_decoding_no_cuda_graph(self):
        """Compare MTP speculative decoding vs non-speculative baseline, both with cuda-graph disabled."""
        self._run_spec_dec_test()


if __name__ == "__main__":
    unittest.main()
