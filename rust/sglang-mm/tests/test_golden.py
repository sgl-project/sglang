import glob
import os

import numpy as np
import pytest

import sglang.srt.multimodal._core.inkling

GOLDEN_DIR = os.environ.get(
    "INKLING_MM_GOLDEN_DIR",
    os.path.join(os.path.dirname(__file__), "..", "tests", "golden"),
)
GOLDENS = sorted(glob.glob(os.path.join(GOLDEN_DIR, "golden_*.npz")))


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


@pytest.mark.parametrize("path", GOLDENS, ids=[os.path.basename(p) for p in GOLDENS])
def test_patchify_rgb_bit_exact(path):
    g = np.load(path)
    got = sglang.srt.multimodal._core.inkling.patchify_rgb(
        g["arr"], int(g["patch_size"])
    )
    np.testing.assert_array_equal(got, g["bits"].reshape(-1))


@pytest.mark.parametrize("path", GOLDENS, ids=[os.path.basename(p) for p in GOLDENS])
def test_decode_patchify_png_bit_exact(path):
    g = np.load(path)
    h_ref, w_ref = g["arr"].shape[:2]
    h, w, got = sglang.srt.multimodal._core.inkling.decode_patchify(
        g["png"].tobytes(), int(g["patch_size"])
    )
    assert (h, w) == (h_ref, w_ref)
    np.testing.assert_array_equal(got, g["bits"].reshape(-1))


def test_batch_matches_single():
    gs = [np.load(p) for p in GOLDENS]
    data = [g["png"].tobytes() for g in gs]
    ps = int(gs[0]["patch_size"])
    for (h, w, bits), g in zip(
        sglang.srt.multimodal._core.inkling.decode_patchify_batch(data, ps), gs
    ):
        np.testing.assert_array_equal(bits, g["bits"].reshape(-1))


def test_golden_fixtures_exist():
    assert len(GOLDENS) >= 4, f"expected golden fixtures in {GOLDEN_DIR}"
