"""Parallel, low-memory preprocessing for large image batches (PIL Qwen2-VL family).

``Qwen2VLImageProcessorPil._preprocess`` handles a request's images one after
another on a single thread, keeps every image's flattened patches in a list and
then ``np.concatenate``s the list. A request carrying N images therefore peaks
near three times the size of its final ``pixel_values`` while pinning one core:
a 250-page request built ~13 GiB of ``pixel_values`` through a ~34 GiB peak and
spent most of its time in page reclaim rather than in preprocessing.

When ``SGLANG_MM_IMAGE_PREPROCESS_THREADS`` is 2 or more, the image processor is
swapped for :class:`ParallelQwen2VLImageProcessorPil`. Its rows per image are
known before any pixel work (``smart_resize`` of the input size), so it allocates
the final tensor once and has worker threads run the stock single-image path on
their own processor clone, copying each result into its slice. The output is
bit-identical to the stock call; peak memory is the output plus one image of
scratch per thread. Inputs it does not understand use the stock path unchanged.
"""

from __future__ import annotations

import copy
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional

import torch
from PIL import Image

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

try:
    from transformers.image_processing_utils import BatchFeature
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
        Qwen2VLImageProcessorPil,
        smart_resize,
    )
except ImportError:  # transformers without the split PIL backend
    Qwen2VLImageProcessorPil = None

_pool: Optional[ThreadPoolExecutor] = None
_pool_lock = threading.Lock()
_thread_state = threading.local()


def _get_pool(threads: int) -> ThreadPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(
                max_workers=threads, thread_name_prefix="sglang-mm-image"
            )
        return _pool


def _thread_clone(processor: Any) -> Any:
    """A per-thread deep copy, so no two threads ever share processor state."""
    clones = getattr(_thread_state, "clones", None)
    if clones is None:
        clones = _thread_state.clones = {}
    entry = clones.get(id(processor))
    if entry is None or entry[0] is not processor:
        entry = clones[id(processor)] = (processor, copy.deepcopy(processor))
    return entry[1]


def _plan_rows(images: Any, args: tuple, kwargs: dict) -> Optional[List[int]]:
    """Rows of ``pixel_values`` per image, or None to use the stock path."""
    if args or not isinstance(images, (list, tuple)) or len(images) < 2:
        return None
    if not all(isinstance(image, Image.Image) for image in images):
        return None
    return_tensors = kwargs.get("return_tensors")
    if getattr(return_tensors, "value", return_tensors) != "pt":
        return None
    if not kwargs.get("do_resize"):
        return None
    patch, merge = kwargs.get("patch_size"), kwargs.get("merge_size")
    temporal = kwargs.get("temporal_patch_size")
    if not all(isinstance(v, int) and v > 0 for v in (patch, merge, temporal)):
        return None
    size = kwargs.get("size")
    shortest = getattr(size, "shortest_edge", None)
    longest = getattr(size, "longest_edge", None)
    if not shortest or not longest:
        return None
    rows = []
    for image in images:
        width, height = image.size
        resized_height, resized_width = smart_resize(
            height, width, factor=patch * merge, min_pixels=shortest, max_pixels=longest
        )
        rows.append((resized_height // patch) * (resized_width // patch))
    return rows


def _stock(processor: Any, images: list, kwargs: dict) -> Any:
    # Call the unmodified method explicitly so a clone never re-enters the
    # parallel override.
    return Qwen2VLImageProcessorPil._preprocess_image_like_inputs(
        processor, images, **dict(kwargs)
    )


def _preprocess_parallel(
    processor: Any, images: list, kwargs: dict, rows: List[int], threads: int
) -> Optional[Any]:
    starts = [0]
    for count in rows:
        starts.append(starts[-1] + count)

    # The first image fixes the row width and dtype of the output.
    first = _stock(processor, images[:1], kwargs)
    first_values = first["pixel_values"]
    if first_values.shape[0] != rows[0]:
        return None
    output = torch.empty((starts[-1], first_values.shape[1]), dtype=first_values.dtype)
    output[: rows[0]].copy_(first_values)
    grids = [None] * len(images)
    grids[0] = first["image_grid_thw"][0]
    del first, first_values

    mismatched = []

    def work(index: int) -> None:
        result = _stock(_thread_clone(processor), images[index : index + 1], kwargs)
        values = result["pixel_values"]
        if (
            values.shape[0] != rows[index]
            or values.shape[1] != output.shape[1]
            or values.dtype != output.dtype
        ):
            mismatched.append(index)
            return
        output[starts[index] : starts[index + 1]].copy_(values)
        grids[index] = result["image_grid_thw"][0]

    list(_get_pool(threads).map(work, range(1, len(images))))
    if mismatched:
        logger.warning(
            "Parallel image preprocessing planned the wrong row count for %d of %d "
            "images; reprocessing this batch on the stock path.",
            len(mismatched),
            len(images),
        )
        return None
    return BatchFeature(
        data={"pixel_values": output, "image_grid_thw": torch.stack(grids)},
        tensor_type=None,
    )


if Qwen2VLImageProcessorPil is not None:

    class ParallelQwen2VLImageProcessorPil(Qwen2VLImageProcessorPil):
        """``Qwen2VLImageProcessorPil`` that preprocesses large batches across threads."""

        def _preprocess_image_like_inputs(self, images, *args, **kwargs):
            threads = envs.SGLANG_MM_IMAGE_PREPROCESS_THREADS.get()
            rows = _plan_rows(images, args, kwargs) if threads >= 2 else None
            if rows is not None:
                result = _preprocess_parallel(self, list(images), kwargs, rows, threads)
                if result is not None:
                    return result
            return super()._preprocess_image_like_inputs(images, *args, **kwargs)


def maybe_enable_parallel_image_preprocess(processor: Any) -> bool:
    """Swap in the parallel image processor when enabled and applicable.

    Must run before the multimodal processor clones ``processor`` for its worker
    pool, so every clone carries the parallel class.
    """
    threads = envs.SGLANG_MM_IMAGE_PREPROCESS_THREADS.get()
    if threads < 2 or Qwen2VLImageProcessorPil is None:
        return False
    image_processor = getattr(processor, "image_processor", None)
    if type(image_processor) is not Qwen2VLImageProcessorPil:
        return False
    image_processor.__class__ = ParallelQwen2VLImageProcessorPil
    logger.info(
        "Parallel image preprocessing enabled with %d threads (%s).",
        threads,
        Qwen2VLImageProcessorPil.__name__,
    )
    return True
