#!/usr/bin/env python3
"""
Test script to verify CLIP consistency checking works correctly.

This script tests:
1. Loading GT embeddings from .npy files
2. Computing CLIP embeddings for images
3. Computing cosine similarity between embeddings

Usage:
    python test_clip_consistency.py

    # Test with a specific image
    python test_clip_consistency.py --image /path/to/test.png --case qwen_image_t2i --num-gpus 1

    # Just verify embeddings can be loaded
    python test_clip_consistency.py --verify-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def test_load_embeddings():
    """Test that all GT embeddings can be loaded."""
    print("=" * 60)
    print("Test 1: Loading GT embeddings")
    print("=" * 60)

    # Add sglang to path
    script_dir = Path(__file__).parent
    sglang_root = script_dir.parent.parent.parent.parent.parent
    sys.path.insert(0, str(sglang_root))

    from sglang.multimodal_gen.test.server.consistency_utils import (
        EMBEDDINGS_DIR,
        gt_exists,
        load_gt_embeddings,
    )

    print(f"Embeddings directory: {EMBEDDINGS_DIR}")

    # Find all .npy files
    npy_files = list(EMBEDDINGS_DIR.rglob("*.npy"))
    print(f"Found {len(npy_files)} embedding files\n")

    success_count = 0
    fail_count = 0

    for npy_file in sorted(npy_files):
        # Parse path to get num_gpus and case_id
        gpu_dir = npy_file.parent.name  # e.g., "1-gpu"
        num_gpus = int(gpu_dir.split("-")[0])
        case_id = npy_file.stem  # e.g., "qwen_image_t2i"

        try:
            # Test gt_exists
            exists = gt_exists(case_id, num_gpus)
            assert exists, f"gt_exists returned False for {case_id}"

            # Check raw embedding shape to determine if video or image
            raw_emb = np.load(npy_file)
            is_video = raw_emb.ndim == 2 and raw_emb.shape[0] > 1

            # Test loading
            embeddings = load_gt_embeddings(case_id, num_gpus, is_video=is_video)

            # Verify shape based on actual data
            expected_count = raw_emb.shape[0] if raw_emb.ndim == 2 else 1
            assert (
                len(embeddings) == expected_count
            ), f"Expected {expected_count} embeddings, got {len(embeddings)}"

            for emb in embeddings:
                assert emb.shape == (768,), f"Expected shape (768,), got {emb.shape}"
                # Verify L2 normalized (norm should be ~1.0)
                norm = np.linalg.norm(emb)
                assert 0.99 < norm < 1.01, f"Embedding not normalized, norm={norm}"

            print(f"  ✓ {gpu_dir}/{case_id}: {len(embeddings)} embedding(s), shape OK")
            success_count += 1

        except Exception as e:
            print(f"  ✗ {gpu_dir}/{case_id}: {e}")
            fail_count += 1

    print(f"\nResults: {success_count} passed, {fail_count} failed")
    return fail_count == 0


def test_clip_model():
    """Test that CLIP model can be loaded and used."""
    print("\n" + "=" * 60)
    print("Test 2: Loading CLIP model")
    print("=" * 60)

    script_dir = Path(__file__).parent
    sglang_root = script_dir.parent.parent.parent.parent.parent
    sys.path.insert(0, str(sglang_root))

    from sglang.multimodal_gen.test.server.consistency_utils import (
        compute_clip_embedding,
        get_clip_model,
    )

    try:
        model, processor = get_clip_model()
        print("  ✓ CLIP model loaded successfully")

        # Create a dummy image to test embedding computation
        dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        emb = compute_clip_embedding(dummy_image)

        assert emb.shape == (768,), f"Expected shape (768,), got {emb.shape}"
        norm = np.linalg.norm(emb)
        assert 0.99 < norm < 1.01, f"Embedding not normalized, norm={norm}"

        print(f"  ✓ Embedding computation works, shape={emb.shape}, norm={norm:.4f}")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_similarity_computation():
    """Test cosine similarity computation."""
    print("\n" + "=" * 60)
    print("Test 3: Similarity computation")
    print("=" * 60)

    script_dir = Path(__file__).parent
    sglang_root = script_dir.parent.parent.parent.parent.parent
    sys.path.insert(0, str(sglang_root))

    from sglang.multimodal_gen.test.server.consistency_utils import (
        compute_clip_similarity,
    )

    try:
        # Test with identical embeddings (should be 1.0)
        emb1 = np.random.randn(768).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)  # Normalize

        sim_identical = compute_clip_similarity(emb1, emb1)
        assert abs(sim_identical - 1.0) < 0.001, f"Expected 1.0, got {sim_identical}"
        print(f"  ✓ Identical embeddings: similarity = {sim_identical:.4f}")

        # Test with orthogonal embeddings (should be ~0.0)
        emb2 = np.random.randn(768).astype(np.float32)
        emb2 = emb2 - np.dot(emb2, emb1) * emb1  # Make orthogonal
        emb2 = emb2 / np.linalg.norm(emb2)  # Normalize

        sim_orthogonal = compute_clip_similarity(emb1, emb2)
        assert abs(sim_orthogonal) < 0.1, f"Expected ~0.0, got {sim_orthogonal}"
        print(f"  ✓ Orthogonal embeddings: similarity = {sim_orthogonal:.4f}")

        # Test with opposite embeddings (should be -1.0)
        sim_opposite = compute_clip_similarity(emb1, -emb1)
        assert abs(sim_opposite + 1.0) < 0.001, f"Expected -1.0, got {sim_opposite}"
        print(f"  ✓ Opposite embeddings: similarity = {sim_opposite:.4f}")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_compare_with_gt():
    """Test the full comparison workflow."""
    print("\n" + "=" * 60)
    print("Test 4: Full comparison workflow (with dummy image)")
    print("=" * 60)

    script_dir = Path(__file__).parent
    sglang_root = script_dir.parent.parent.parent.parent.parent
    sys.path.insert(0, str(sglang_root))

    from sglang.multimodal_gen.test.server.consistency_utils import (
        EMBEDDINGS_DIR,
        compare_with_gt,
        load_gt_embeddings,
    )

    try:
        # Find a case to test with
        npy_files = list((EMBEDDINGS_DIR / "1-gpu").glob("*.npy"))
        if not npy_files:
            print("  ⚠ No 1-gpu embeddings found, skipping")
            return True

        # Use the first non-video case
        test_file = None
        for f in npy_files:
            case_id = f.stem
            if "t2v" not in case_id and "i2v" not in case_id and "ti2v" not in case_id:
                test_file = f
                break

        if test_file is None:
            test_file = npy_files[0]

        case_id = test_file.stem
        is_video = "t2v" in case_id or "i2v" in case_id or "ti2v" in case_id

        print(f"  Testing with case: {case_id} (is_video={is_video})")

        # Load GT embeddings
        gt_embeddings = load_gt_embeddings(case_id, num_gpus=1, is_video=is_video)
        print(f"  Loaded {len(gt_embeddings)} GT embedding(s)")

        # Create dummy frames (random images won't match GT)
        num_frames = len(gt_embeddings)
        dummy_frames = [
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            for _ in range(num_frames)
        ]

        # Compare (should fail since dummy images won't match GT)
        result = compare_with_gt(
            output_frames=dummy_frames,
            gt_embeddings=gt_embeddings,
            threshold=0.92,
            case_id=case_id,
        )

        print(
            f"  Result: passed={result.passed}, min_similarity={result.min_similarity:.4f}"
        )
        print(f"  (Expected to fail since using random dummy images)")

        # Now test with GT embeddings directly (simulate perfect match)
        # Create frames that would produce the same embeddings
        # This is a sanity check - we just verify the workflow runs

        print("  ✓ Comparison workflow executed successfully")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_with_real_image(image_path: Path, case_id: str, num_gpus: int):
    """Test with a real image against GT."""
    print("\n" + "=" * 60)
    print(f"Test: Real image comparison")
    print(f"  Image: {image_path}")
    print(f"  Case: {case_id}")
    print(f"  GPUs: {num_gpus}")
    print("=" * 60)

    script_dir = Path(__file__).parent
    sglang_root = script_dir.parent.parent.parent.parent.parent
    sys.path.insert(0, str(sglang_root))

    from PIL import Image

    from sglang.multimodal_gen.test.server.consistency_utils import (
        compare_with_gt,
        gt_exists,
        load_gt_embeddings,
    )

    if not image_path.exists():
        print(f"  ✗ Image not found: {image_path}")
        return False

    if not gt_exists(case_id, num_gpus):
        print(f"  ✗ GT not found for case: {case_id} ({num_gpus}-gpu)")
        return False

    try:
        # Load image
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        print(f"  Image shape: {img_array.shape}")

        # Load GT
        is_video = "t2v" in case_id or "i2v" in case_id or "ti2v" in case_id
        gt_embeddings = load_gt_embeddings(case_id, num_gpus, is_video=is_video)

        # Compare
        result = compare_with_gt(
            output_frames=[img_array],
            gt_embeddings=gt_embeddings[:1],  # Use first GT embedding only
            threshold=0.92,
            case_id=case_id,
        )

        print(f"\n  Result: {'PASSED' if result.passed else 'FAILED'}")
        print(f"  Similarity: {result.similarity_scores[0]:.4f}")
        print(f"  Threshold: {result.threshold}")

        return result.passed

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test CLIP consistency checking")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify embeddings can be loaded",
    )
    parser.add_argument("--image", type=Path, help="Path to test image")
    parser.add_argument("--case", type=str, help="Case ID to test against")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs")

    args = parser.parse_args()

    if args.image:
        if not args.case:
            print("Error: --case is required when using --image")
            sys.exit(1)
        success = test_with_real_image(args.image, args.case, args.num_gpus)
        sys.exit(0 if success else 1)

    # Run all tests
    all_passed = True

    all_passed &= test_load_embeddings()

    if not args.verify_only:
        all_passed &= test_clip_model()
        all_passed &= test_similarity_computation()
        all_passed &= test_compare_with_gt()

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED ✓")
    else:
        print("Some tests FAILED ✗")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
