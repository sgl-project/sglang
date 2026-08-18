# SPDX-License-Identifier: Apache-2.0
"""GGUF checkpoint reading for diffusion transformers.

A GGUF file replaces only the transformer component of a diffusion pipeline;
every other component (VAE, text encoder, scheduler) still loads from the base
model repository.

Unlike the safetensors path, tensor shapes here are not knowable from the model
config alone: a quantized tensor is stored as packed bytes whose row length
depends on its GGML type. ``read_gguf_tensor_meta`` is therefore called *before*
model construction so that ``GGUFLinearMethod.create_weights`` can register
parameters with their exact packed byte shapes. That keeps the generic weight
loader in ``fsdp_load`` working unchanged -- it casts each loaded tensor to the
meta parameter's dtype, which is already ``uint8`` for packed weights, making
the cast a no-op.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Generator

import msgspec
import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# GGMLQuantizationType ids this module names directly.
GGML_F32, GGML_F16, GGML_BF16 = 0, 1, 30

# GGML types that store plain, unquantized values. Everything else is packed and
# must go through ``ggml_dequantize`` before use.
GGML_UNQUANTIZED_TYPES = frozenset({GGML_F32, GGML_F16, GGML_BF16})

# Types whose dequantize kernel walks 32-element blocks and bounds-checks each
# thread (``dequantize_block`` in dequantize.cuh). Every other quantized type is
# handled by a 256-element super-block kernel with no bounds check, which
# requires the tensor's element count to be a whole number of super-blocks:
# the truncating variants would leave the tail uninitialized, and the two
# ceil-rounding ones (IQ4_NL, IQ4_XS) would write past the output.
_GGML_BOUNDS_CHECKED_TYPES = frozenset({2, 3, 6, 7, 8})  # Q4_0 Q4_1 Q5_0 Q5_1 Q8_0
_GGML_SUPER_BLOCK = 256

GGUF_MAGIC = b"GGUF"


class GGUFTensorMeta(msgspec.Struct, frozen=True):
    """Layout of one GGUF tensor, read from the file header.

    Attributes:
        ggml_type: The numeric ``GGMLQuantizationType``.
        logical_shape: Shape the dequantized tensor has, in torch order --
            ``(out_features, in_features)`` for a weight matrix. GGUF stores
            dimensions fastest-varying first, so this is the reverse of the
            file's ``ne`` array.
        stored_shape: Shape the parameter is registered and loaded with. Equal
            to ``logical_shape`` for unquantized tensors; for quantized ones it
            is ``(out_features, row_bytes)`` of packed ``uint8``.
        stored_dtype: dtype the parameter is registered with.
        param_name: Model parameter this tensor loads into. A quantized weight
            targets ``qweight`` -- the name ``GGUFLinearMethod`` registers --
            while everything else keeps its checkpoint name.
    """

    ggml_type: int
    logical_shape: tuple[int, ...]
    stored_shape: tuple[int, ...]
    stored_dtype: torch.dtype
    param_name: str

    @property
    def is_quantized(self) -> bool:
        return self.ggml_type not in GGML_UNQUANTIZED_TYPES


_WEIGHT_SUFFIX = ".weight"


def _param_name_for(gguf_name: str, is_quantized: bool) -> str:
    """Map a GGUF tensor name to the model parameter it loads into."""
    if is_quantized and gguf_name.endswith(_WEIGHT_SUFFIX):
        return gguf_name[: -len(_WEIGHT_SUFFIX)] + ".qweight"
    return gguf_name


def is_gguf_file(path: str | os.PathLike) -> bool:
    """Return whether ``path`` is a GGUF file, by suffix or magic bytes."""
    if not path:
        return False
    path = str(path)
    if not os.path.isfile(path):
        return False
    if path.endswith(".gguf"):
        return True
    try:
        with open(path, "rb") as f:
            return f.read(4) == GGUF_MAGIC
    except OSError:
        return False


def _require_gguf():
    try:
        import gguf
    except ImportError as exc:
        raise ImportError(
            "Reading a GGUF checkpoint requires the `gguf` package. "
            "Install it with `pip install gguf`."
        ) from exc
    return gguf


def _torch_dtype_for_ggml(ggml_type: int) -> torch.dtype:
    if ggml_type == GGML_F32:
        return torch.float32
    if ggml_type == GGML_F16:
        return torch.float16
    if ggml_type == GGML_BF16:
        return torch.bfloat16
    # Packed types are carried as raw bytes.
    return torch.uint8


def _open_reader(gguf_file: str):
    """Open a GGUF file, rejecting one whose byte order is not the host's.

    gguf-py records endianness per file and reports ``"S"`` when it differs, but
    it only hands back views with a swapped NumPy dtype. That is not something
    this loader can fix up: ``torch.from_numpy`` refuses a non-native dtype, and
    a quantized tensor stays ``uint8`` with its scale fields embedded inside each
    block, so a whole-buffer swap would silently corrupt dequantization instead
    of failing. Rejecting is the honest option until there is a reason -- and a
    sample file -- to convert every block layout properly.
    """
    gguf = _require_gguf()

    try:
        reader = gguf.GGUFReader(gguf_file)
    except Exception as exc:
        # gguf-py builds every tensor view up front, so an incomplete download
        # surfaces here as a bare reshape error that reads like a layout bug.
        size = os.path.getsize(gguf_file) if os.path.isfile(gguf_file) else 0
        raise ValueError(
            f"Failed to read GGUF {gguf_file} ({size} bytes). An incomplete or "
            f"corrupt download is the usual cause. Underlying error: {exc}"
        ) from exc
    if reader.byte_order == "S":
        raise ValueError(
            f"GGUF file {gguf_file} is stored in the opposite byte order from "
            "this host, which is not supported. Convert it to a native-endian "
            "GGUF first."
        )
    return reader


def read_gguf_tensor_meta(gguf_file: str) -> dict[str, GGUFTensorMeta]:
    """Read every tensor's layout from a GGUF file header.

    Only the header is parsed; tensor data is not read.
    """
    from gguf.constants import GGML_QUANT_SIZES

    reader = _open_reader(gguf_file)
    meta: dict[str, GGUFTensorMeta] = {}
    for tensor in reader.tensors:
        ggml_type = int(tensor.tensor_type)
        is_quantized = ggml_type not in GGML_UNQUANTIZED_TYPES
        # GGUF stores `ne` fastest-varying first; torch wants the reverse.
        logical_shape = tuple(int(d) for d in reversed(tensor.shape))
        if not is_quantized:
            stored_shape = logical_shape
        else:
            block_size, type_size = GGML_QUANT_SIZES[tensor.tensor_type]
            inner = logical_shape[-1]
            if inner % block_size:
                raise ValueError(
                    f"GGUF tensor {tensor.name} has inner dimension {inner} that is "
                    f"not a multiple of the block size {block_size} for type "
                    f"{ggml_type}."
                )
            row_bytes = inner // block_size * type_size
            stored_shape = (*logical_shape[:-1], row_bytes)
            if ggml_type not in _GGML_BOUNDS_CHECKED_TYPES:
                numel = math.prod(logical_shape)
                if numel % _GGML_SUPER_BLOCK:
                    raise ValueError(
                        f"GGUF tensor {tensor.name} has {numel} elements, which is "
                        f"not a whole number of {_GGML_SUPER_BLOCK}-element super "
                        f"blocks required by the type-{ggml_type} dequantize "
                        "kernel. Loading it would read or write out of bounds."
                    )
        meta[tensor.name] = GGUFTensorMeta(
            ggml_type=ggml_type,
            logical_shape=logical_shape,
            stored_shape=stored_shape,
            stored_dtype=_torch_dtype_for_ggml(ggml_type),
            param_name=_param_name_for(tensor.name, is_quantized),
        )
    return meta


def _tensor_from_reader(tensor, meta: GGUFTensorMeta) -> torch.Tensor:
    """Materialize one GGUF tensor as a torch tensor in its stored layout."""
    import numpy as np

    data: np.ndarray = tensor.data
    if meta.ggml_type == GGML_BF16:
        # gguf-py hands BF16 back as raw bytes rather than a typed array, so
        # reinterpret rather than cast (a cast would read the byte values as
        # numbers).
        out = torch.from_numpy(data.view(np.uint16).copy()).view(torch.bfloat16)
    else:
        out = torch.from_numpy(np.asarray(data).copy())
    return out.reshape(meta.stored_shape)


def gguf_weights_iterator(
    gguf_file: str,
    tensor_meta: dict[str, GGUFTensorMeta],
    key_filter: Callable[[str], bool] | None = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield ``(param_name, tensor)`` for every tensor in a GGUF file.

    Quantized tensors are yielded as packed ``uint8`` in their stored layout and
    under the ``qweight`` name the layer registered; ``GGUFLinearMethod``
    dequantizes them at use time.

    ``key_filter`` matches the same way as the safetensors iterator's: it sees
    the checkpoint's own tensor name and keeps the ones it returns true for.
    """
    reader = _open_reader(gguf_file)
    for tensor in reader.tensors:
        if key_filter is not None and not key_filter(tensor.name):
            continue
        meta = tensor_meta.get(tensor.name)
        if meta is None:
            # The header was read from this same file, so a miss means the
            # caller passed metadata from a different checkpoint.
            raise KeyError(
                f"GGUF tensor {tensor.name} is missing from the supplied metadata; "
                "the metadata and the checkpoint do not match."
            )
        yield meta.param_name, _tensor_from_reader(tensor, meta)


def _is_local_path(reference: str) -> bool:
    """Whether ``reference`` is meant as a filesystem path, not a Hub reference.

    Hub references are always ``owner/repo...``, so anything absolute or
    explicitly relative names a local file even when it does not exist.
    """
    return os.path.isabs(reference) or reference.startswith((".", "~"))


def names_gguf_checkpoint(reference: str) -> bool:
    """Whether ``reference`` names a GGUF checkpoint, without any network I/O.

    Lets callers validate runtime support *before* a Hub reference is downloaded,
    so an unsupported configuration is rejected in a second rather than after a
    multi-gigabyte fetch.
    """
    if not reference:
        return False
    if is_gguf_file(reference):
        return True
    if os.path.exists(reference):
        # An existing path that is not a GGUF file is something else entirely.
        return False
    if _is_local_path(reference):
        # A missing local path is judged by intent alone, so the error names the
        # file rather than depending on how many directories deep it sits.
        return reference.endswith(".gguf")
    if ":" in reference:
        repo_id, _, quant_type = reference.rpartition(":")
        return repo_id.count("/") == 1 and bool(quant_type)
    # Hub file reference: owner/repo/path/inside/repo.gguf
    return reference.endswith(".gguf") and len(reference.strip("/").split("/")) >= 3


def resolve_gguf_reference(reference: str, revision: str | None = None) -> str | None:
    """Resolve a Hub reference to a local ``.gguf`` path.

    Accepts ``owner/repo/path/inside/repo.gguf`` and ``owner/repo:QUANT_TYPE``
    (e.g. ``leejet/MiniMax-H3-GGUF:Q4_K_M``). Returns ``None`` when the
    reference is not a Hub GGUF reference, so callers can fall through to
    treating it as a local path.
    """
    if not reference or os.path.exists(reference):
        return None
    if _is_local_path(reference):
        # A missing local file is a typo, not a repo named after its directories.
        return None

    from huggingface_hub import hf_hub_download

    if ":" in reference:
        repo_id, _, quant_type = reference.rpartition(":")
        if repo_id.count("/") != 1 or not quant_type:
            return None
        return _download_gguf_by_quant_type(repo_id, quant_type, revision)

    if not reference.endswith(".gguf"):
        return None
    parts = reference.strip("/").split("/")
    if len(parts) < 3:
        return None
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])
    logger.info("Downloading GGUF %s from %s", filename, repo_id)
    return hf_hub_download(repo_id, filename, revision=revision)


def _download_gguf_by_quant_type(
    repo_id: str, quant_type: str, revision: str | None
) -> str:
    """Download the single ``.gguf`` in ``repo_id`` matching ``quant_type``."""
    from huggingface_hub import HfApi, hf_hub_download

    files = [
        sibling.rfilename
        for sibling in HfApi().model_info(repo_id, revision=revision).siblings
    ]
    suffix = f"-{quant_type}.gguf"
    matches = [f for f in files if f.endswith(suffix)]
    if not matches:
        candidates = sorted(f for f in files if f.endswith(".gguf"))
        raise ValueError(
            f"No file matching quant type {quant_type!r} in {repo_id}. "
            f"Available GGUF files: {candidates}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Quant type {quant_type!r} is ambiguous in {repo_id}: {sorted(matches)}. "
            "Pass the full path instead, e.g. owner/repo/subdir/file.gguf."
        )
    logger.info("Downloading GGUF %s from %s", matches[0], repo_id)
    return hf_hub_download(repo_id, matches[0], revision=revision)
