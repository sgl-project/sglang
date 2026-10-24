"""Multimodal support for the embedded Rust server."""

from __future__ import annotations

import importlib
import json
import logging
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, Optional, Tuple

import msgspec

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_mm, get_model, get_parallel, get_serving

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class RustMmSpec(msgspec.Struct, frozen=True, kw_only=True):
    """The Rust MM pipeline for one model: the sources Rust resolves it from
    (`sglang_processor::select_mm_processor`, reached through the extension's
    ``MmSpec``, see :meth:`RustServer._build_mm_spec`) and the token ids the
    drain adapter (:meth:`RustMmProcessor.wrap_encoded`) reads."""

    family: str
    feature_shm: bool
    # Where Rust reads config.json and the HF processor config, as Python's
    # `get_processor` does.
    model_path: str
    tokenizer_path: str
    revision: Optional[str]
    image_processor_backend: str
    # `--mm-process-config`'s `image` object as JSON, or None when unset.
    image_overrides: Optional[str]
    image_token_id: int
    vision_start_token_id: Optional[int]
    vision_end_token_id: Optional[int]
    video_token_id: Optional[int]


class RustMmFamily(msgspec.Struct, frozen=True, kw_only=True):
    """The Python half of one Rust MM family (a `dynamo-multimodal` family
    behind `sglang_processor::mm`): which models it serves. Supporting a new
    model family = one entry in :data:`RUST_MM_FAMILIES` plus its Rust family
    (the `processor-mm-parity` skill) — the launch gate is data-driven."""

    name: str
    # The registered Python MM processor the Rust pipeline replaces, as
    # "module:Class". Compared by identity, so an
    # SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE override still disables the Rust path.
    mm_processor: str
    # Model types whose image-only M-RoPE matches the family's fast path.
    model_types: FrozenSet[str]

    def serves(self, mm_processor_cls: Any, model_type: Optional[str]) -> bool:
        module_name, _, class_name = self.mm_processor.partition(":")
        cls = getattr(importlib.import_module(module_name), class_name)
        return mm_processor_cls is cls and model_type in self.model_types


RUST_MM_FAMILIES: Tuple[RustMmFamily, ...] = (
    RustMmFamily(
        name="qwen_vl",
        mm_processor="sglang.srt.multimodal.processors.qwen_vl:QwenVLImageProcessor",
        model_types=frozenset(
            (
                "qwen2_vl",
                "qwen2_5_vl",
                "qwen3_vl",
                "qwen3_vl_moe",
                "qwen3_5",
                "qwen3_5_moe",
            )
        ),
    ),
)


def rust_mm_family_for(
    mm_processor_cls: Any, model_type: Optional[str]
) -> Optional[RustMmFamily]:
    """The declared family serving this model, or ``None`` — which
    :meth:`RustServer.launch` turns into a hard error (no Python fallback)."""
    return next(
        (f for f in RUST_MM_FAMILIES if f.serves(mm_processor_cls, model_type)), None
    )


class _ModalityTable(dict):
    """``mm.meta`` modality names -> scheduler ``Modality``, resolved lazily so
    this module keeps its light import footprint."""

    def __missing__(self, name):
        from sglang.srt.managers.schedule_batch import Modality

        self.update(
            {"image": Modality.IMAGE, "video": Modality.VIDEO, "audio": Modality.AUDIO}
        )
        # An unknown name must raise here, not recurse.
        if name not in self:
            raise KeyError(name)
        return dict.__getitem__(self, name)


_MODALITIES = _ModalityTable()


def settle_shm_buffers(buffers, *, admitted: bool) -> None:
    """Decide the unlink duty for every segment a drained request parked in
    POSIX shm (any buffer that is not an inline numpy array is the Rust
    extension's ``ShmBuffer``, whatever its name). Admitted: Rust hands the
    duty to the ``ShmPointerMMData`` stubs, which unlink after the
    post-broadcast open on every rank. Rejected: unlink now. Both are
    idempotent on the Rust side. This is the drain loop's job, never a
    wrapper's, so a package's own ``_wrap_mm_result`` needs no knowledge of it."""
    import numpy as np

    for value in buffers.values():
        if isinstance(value, np.ndarray):
            continue
        if admitted:
            value.release()
        else:
            value.discard()


class MmTransportStats:
    """Zero-copy accounting at the Rust-to-Python multimodal hand-off.

    A feature tensor crosses in one of two shapes: ``inline`` -- a numpy array
    that owns the Rust vector, viewed by ``torch.from_numpy`` -- or ``shm`` -- a
    POSIX segment the worker wrote, named in the request so every TP rank maps
    it after the broadcast."""

    __slots__ = (
        "inline_features",
        "inline_bytes",
        "shm_features",
        "shm_bytes",
        "copies",
    )

    def __init__(self) -> None:
        self.inline_features = 0
        self.inline_bytes = 0
        self.shm_features = 0
        self.shm_bytes = 0
        self.copies = 0

    def record_inline(self, array, tensor) -> None:
        self.inline_features += 1
        self.inline_bytes += array.nbytes
        if array.flags["OWNDATA"] or tensor.data_ptr() != array.ctypes.data:
            self.copies += 1

    def record_shm(self, stub) -> None:
        import numpy as np

        self.shm_features += 1
        self.shm_bytes += int(np.prod(stub.shape)) * np.dtype(stub.dtype).itemsize

    def snapshot(self) -> Dict[str, int]:
        return {name: getattr(self, name) for name in self.__slots__}


MM_TRANSPORT_STATS = MmTransportStats()


class RustMmProcessor:
    """Selects the Rust MM pipeline for one model.

    Construction registers the same ``mm_processor`` mapping the Python
    TokenizerManager would build — not to process requests (the Rust worker pool
    does that, GIL-free) but so :meth:`resolve_spec` can tell whether a Rust
    family serves the model. At drain time :meth:`wrap_encoded` wraps the
    Rust-produced buffers into the scheduler's ``MultimodalProcessorOutput``.

    There is no Python fallback: a model without a Rust MM spec fails at launch,
    and inputs outside the pipeline's scope are rejected per request.
    """

    # Rust mm-worker threads when --mm-processor-worker-num is 0. They are
    # GIL-free, so unlike the Python processor pool more than one always helps.
    AUTO_MM_WORKERS = 8

    def __init__(self, *, server_args: ServerArgs, model_config: ModelConfig):
        # Lazy: this class exists only for multimodal models under
        # SGLANG_RUST_SERVER.
        from sglang.srt.managers.multimodal_processor import import_processors

        self.server_args = server_args
        self.model_config = model_config
        # Worker threads == max concurrently-processed mm requests.
        self.mm_workers = get_mm().mm_processor_worker_num or self.AUTO_MM_WORKERS

        # The mapping the Python TokenizerManager builds in
        # init_tokenizer_and_processor.
        import_processors("sglang.srt.multimodal.processors")
        if mm_process_pkg := envs.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE.get():
            import_processors(mm_process_pkg, overwrite=True)

    def resolve_spec(self) -> Optional[RustMmSpec]:
        """The :class:`RustMmSpec` for this model, or ``None`` when no Rust
        family serves it (the launch gate turns that into a hard error).

        Only eligibility is decided here. Rust resolves the processor parameters
        from the model's HF config files and refuses, when ``MmSpec`` is built,
        any it cannot reproduce bit for bit."""
        from sglang.srt.managers.multimodal_processor import get_mm_processor_cls
        from sglang.srt.utils.hf_transformers.processor import (
            resolve_image_processor_backend,
        )

        hf_config = self.model_config.hf_config
        mm_processor_cls = get_mm_processor_cls(
            hf_config, model_config=self.model_config
        )
        family = rust_mm_family_for(
            mm_processor_cls, getattr(hf_config, "model_type", None)
        )
        if family is None:
            return None
        image_overrides = (get_mm().mm_process_config or {}).get("image")
        spec = RustMmSpec(
            family=family.name,
            feature_shm=self._use_feature_shm(),
            model_path=get_model().model_path,
            tokenizer_path=get_serving().tokenizer_path,
            revision=get_model().revision,
            image_processor_backend=resolve_image_processor_backend(get_mm()),
            image_overrides=(
                None if image_overrides is None else json.dumps(image_overrides)
            ),
            image_token_id=hf_config.image_token_id,
            vision_start_token_id=getattr(hf_config, "vision_start_token_id", None),
            vision_end_token_id=getattr(hf_config, "vision_end_token_id", None),
            video_token_id=getattr(hf_config, "video_token_id", None),
        )
        logger.info("rust server: Rust MM pipeline enabled (family=%s)", family.name)
        return spec

    def _use_feature_shm(self) -> bool:
        """Whether to park feature buffers in POSIX shm rather than inline.

        On exactly when the drained request is broadcast across TP ranks *and*
        the receiver's ``unwrap_shm_features`` will materialize the stubs (its
        gates: non-default tensor transport, no ``skip_tokenizer_init``).

        Inline, the whole ~20 MB/image buffer rides ``broadcast_pyobj`` serially
        on the scheduler loop, so ranks 1..n start the TP-sharded ViT ~30 ms
        after rank 0 and every rank then stalls that long at the first
        collective. With shm the broadcast carries a ~100-byte stub and all ranks
        map in parallel — the transport the Python TokenizerManager already uses.
        Single-rank serving stays inline, where shm would only add a copy.
        """
        from sglang.srt.multimodal.transport import (
            determine_tensor_transport_mode,
        )

        return (
            get_parallel().tp_size > 1
            and determine_tensor_transport_mode() != "default"
            and not get_serving().skip_tokenizer_init
        )

    @staticmethod
    def wrap_encoded(spec: RustMmSpec, buffers):
        """Drain-time adapter: wrap one request's ``mm.*`` buffers (a
        ``{name: numpy array | ShmBuffer}`` mapping from ``recv_requests``) into
        the scheduler's ``MultimodalProcessorOutput``. Wrapping only -- load,
        resize, patchify, token expansion and M-RoPE all ran in Rust.

        The buffers are ``mm.feature.{i}`` per item (shaped ``[rows,
        feature_dim]`` f32, inline or shm), ``mm.mrope`` (``[3, seq_len]``
        int64), and ``mm.meta``, a msgpack sidecar decoding to ``{"items":
        [{"modality", "hash", "offsets", "model_specific_data"}, ...],
        "token_ids", "mrope_delta"}`` with the items in buffer order. External
        model packages receive the same layout and wrap it their own way.

        Runs on the scheduler loop, so it must stay copy-free *and* hash-free:
        the inline numpy arrays own the Rust buffers, ``torch.from_numpy`` just
        views them, and each item's ``hash`` is worker-precomputed so
        ``set_pad_value`` skips ``hash_feature``. Any per-byte work here — memcpy,
        sha256, tens of MB per image-heavy request — measurably inflates every
        running request's inter-token latency."""
        import msgspec
        import torch

        from sglang.srt.managers.schedule_batch import (
            MultimodalProcessorOutput,
        )

        # Ownership of the segments is not decided here: the drain loop
        # releases or discards every `ShmBuffer` once it knows whether the
        # request is admitted (`settle_shm_buffers`). A failure part-way only
        # has to close the stubs built so far, which hold an open mapping each.
        stubs: list = []
        try:
            # Decoded straight from the numpy buffer (a memoryview over the
            # Rust-owned bytes), not from a `.tobytes()` copy of it.
            meta = msgspec.msgpack.decode(memoryview(buffers["mm.meta"]))
            items = [
                RustMmProcessor._wrap_item(
                    index=index, item=item, buffers=buffers, stubs=stubs
                )
                for index, item in enumerate(meta["items"])
            ]
            if envs.SGLANG_MM_PRECOMPUTE_HASH.get():
                for item in items:
                    item.set_pad_value()
            output = MultimodalProcessorOutput(
                mm_items=items,
                im_token_id=spec.image_token_id,
                im_start_id=spec.vision_start_token_id,
                im_end_id=spec.vision_end_token_id,
                video_token_id=spec.video_token_id,
                mrope_positions=torch.from_numpy(buffers["mm.mrope"]),
                mrope_position_delta=torch.tensor(
                    [[meta["mrope_delta"]]], dtype=torch.long
                ),
            )
        except BaseException:
            for stub in stubs:
                stub.close_and_unlink()
            raise
        return output

    @staticmethod
    def _wrap_item(*, index: int, item: dict, buffers, stubs: list):
        import numpy as np
        import torch

        from sglang.srt.managers.mm_utils import ShmPointerMMData
        from sglang.srt.managers.schedule_batch import MultimodalDataItem

        feature = buffers[f"mm.feature.{index}"]
        if isinstance(feature, np.ndarray):
            array = feature
            feature = torch.from_numpy(array)
            MM_TRANSPORT_STATS.record_inline(array=array, tensor=feature)
        else:
            MM_TRANSPORT_STATS.record_shm(stub=feature)
            # The worker placed this item's buffer in a named POSIX segment
            # (see `_use_feature_shm`). Build the stub in its
            # post-`__setstate__` form: rank 0 never pickle-roundtrips its
            # own copy, and `materialize()` needs the mapped view.
            stub = ShmPointerMMData.__new__(ShmPointerMMData)
            stub.__setstate__(
                {
                    "shm_name": feature.name,
                    "shape": tuple(feature.shape),
                    "dtype": torch.float32,
                    "precomputed_hash": item["hash"],
                }
            )
            stubs.append(stub)
            feature = stub
        t, h, w = item["model_specific_data"]["image_grid_thw"]
        return MultimodalDataItem(
            modality=_MODALITIES[item["modality"]],
            feature=feature,
            hash=item["hash"],
            offsets=[tuple(span) for span in item["offsets"]],
            model_specific_data={
                "image_grid_thw": torch.tensor([[t, h, w]], dtype=torch.long)
            },
        )
