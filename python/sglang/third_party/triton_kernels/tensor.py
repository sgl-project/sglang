from dataclasses import dataclass, fields
from typing import Type

import torch
from triton.tools.ragged_tma import create_ragged_descriptor
from triton.tools.tensor_descriptor import TensorDescriptor

from .target_info import cuda_capability_geq
from .tensor_details import bitmatrix as bitmatrix_details
from .tensor_details import ragged_tensor as ragged_tensor_details
from .tensor_details.layout import BlackwellMXValueLayout, Layout, StridedLayout
from .tensor_details.ragged_tensor import RaggedTensorMetadata


# storage
# ---------------------------------------------------------------------------- #
@dataclass
class Storage:
    data: torch.Tensor
    layout: Layout = None

    def __post_init__(self):
        assert isinstance(self.data, torch.Tensor)
        if self.layout is None:
            self.layout = StridedLayout(self.data.shape)

    @property
    def device(self):
        return self.data.device

    def is_tma_compliant(self):
        # TMAs didn't exist until Hopper
        if not cuda_capability_geq(9, 0):
            return False
        # TMAs only exist for 2D, 3D, 5D inputs
        if len(self.data.shape) not in [2, 3, 5]:
            return False
        # TMAs need at most one stride equal to 1
        # and all other strides divisble by 16
        strides = list(self.data.stride())
        try:
            major_dim = strides.index(1)
        except ValueError:
            major_dim = -1
        ndim = self.data.ndim
        bitwidth = 4 if self.data.dtype == torch.uint8 else self.data.element_size() * 8
        compliant = [strides[i] * bitwidth % 128 == 0 for i in range(ndim) if i != major_dim]
        return all(compliant)

    def make_dense_tma(self, block_shape, is_scale):
        strides = list(self.data.stride())
        shape = list(self.data.shape)
        transpose = strides[-1] != 1
        if transpose:
            # Need to transpose since tensor descriptor expects strides except for the last dimension 16-byte aligned
            # https://github.com/triton-lang/triton/blob/e5e0081db3335e7755e2c67c784cb1c92769812f/python/triton/tools/tensor_descriptor.py#L26
            block_shape = block_shape[:-2] + [block_shape[-1], block_shape[-2]]
            shape = shape[:-2] + [shape[-1], shape[-2]]
            strides = strides[:-2] + [strides[-1], strides[-2]]
        if self.data.dtype == torch.uint8 and not is_scale:
            indx = strides.index(1)
            block_shape[indx] = block_shape[indx] // 2
            if isinstance(self.layout, BlackwellMXValueLayout) and shape[-1] % 128 != 0:
                raise ValueError(
                    "inner shape need to be multiple of 128 for mxfp4 (CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B) TMAs.")
        block_shape = self.layout.swizzle_block_shape(block_shape)
        return TensorDescriptor(self.data, shape, strides, block_shape)

    def make_tma(self, block_shape, mode, is_scale=False):
        if mode in ["dense", "gather", "scatter"]:
            return self.make_dense_tma(block_shape, is_scale)
        assert mode == "ragged"
        ragged_dim = len(self.data.shape) - 2
        return create_ragged_descriptor(self.data, block_shape, ragged_dim=ragged_dim)


# data types
# ---------------------------------------------------------------------------- #
@dataclass
class IntegerType:
    bitwidth: int


@dataclass
class FloatType:
    bitwidth_exponent: int
    bitwidth_mantissa: int
    is_signed: bool

    def __post_init__(self):
        self.bitwidth = int(self.is_signed) + self.bitwidth_exponent + self.bitwidth_mantissa


BIT = IntegerType(1)
FP4 = FloatType(bitwidth_exponent=2, bitwidth_mantissa=1, is_signed=True)


def bitwidth(type: IntegerType | FloatType | torch.dtype):
    if isinstance(type, torch.dtype):
        return type.itemsize * 8
    return type.bitwidth


# main tensor class
# ---------------------------------------------------------------------------- #


@dataclass
class Tensor:
    storage: Storage | torch.Tensor
    dtype: IntegerType | FloatType | torch.dtype = None
    shape: list[int] | None = None
    shape_max: list[int] | None = None

    def __post_init__(self):
        # set storage
        if isinstance(self.storage, torch.Tensor):
            self.storage = Storage(self.storage)
        # initialize dtype
        if self.dtype is None:
            self.dtype = self.storage.data.dtype
        if bitwidth(self.dtype) < 8 and self.shape is None:
            raise ValueError("shape must be provided for sub-byte types")
        # initialize shape
        if self.shape is None:
            self.shape = list(self.storage.data.shape)
        # validate shape: all elements must be `int` or numel-1 `torch.Tensor`
        is_int = lambda s: isinstance(s, int)
        is_item = lambda s: hasattr(s, "numel") and s.numel() == 1
        assert all(map(lambda s: is_int(s) or is_item(s), self.shape))
        # initialize shape_max
        if self.shape_max is None:
            self.shape_max = [None] * len(self.shape)
        for i, (s, smax) in enumerate(zip(self.shape, self.shape_max)):
            if smax is not None and not is_int(smax):
                raise ValueError(f"shape_max[{i}] must be `int` or `None`; got {type(smax)}")
            if smax is None:
                self.shape_max[i] = s
        # validate shape_max: all elements must be `int`
        assert all(map(is_int, self.shape_max))

    # torch compatibility layer
    @property
    def ndim(self):
        return len(self.shape)

    @property
    def device(self):
        return self.storage.device

    def stride(self, i=None):
        return self.storage.data.stride() if i is None else self.storage.data.stride(i)

    def data_ptr(self):
        return self.storage.data.data_ptr()

    def numel(self):
        return self.storage.data.numel()

    def element_size(self):
        return bitwidth(self.dtype) // 8

    @property
    def data(self):
        t = self.storage
        return t.data if isinstance(t, Storage) else t

    def dim(self):
        return self.ndim

    def size(self, i=None):
        if i is None:
            return self.shape
        return self.shape[i]


# ---------------------------------------------------------------------------- #
# bitmatrix
# ---------------------------------------------------------------------------- #
@dataclass
class Bitmatrix(Tensor):

    def __post_init__(self):
        assert self.dtype == BIT
        super().__post_init__()


make_bitmatrix_metadata = bitmatrix_details.make_bitmatrix_metadata
make_bitmatrix_metadata_torch = bitmatrix_details.make_bitmatrix_metadata_torch


# ---------------------------------------------------------------------------- #
# ragged tensor
# ---------------------------------------------------------------------------- #
@dataclass
class RaggedTensor:
    """
    A ragged `tensor` is a collection of 2D tensors that share the same number of columns.
    Each tensor in this collection is called a `slice`.
    """

    # slice_sizes[i] is the number of rows in slice `i`
    slice_sizes: torch.Tensor
    # ragged tensors are stored in memory as (potentially padded) 2D tensors of shape
    # [num_total_rows, num_cols]
    # where `num_total_rows` >= sum(slice_sizes)
    data: torch.Tensor
    # `metadata`` contains information about the ragged tensor
    # see `tensor_details/ragged_tensor.py` for more details
    metadata: RaggedTensorMetadata


# construct ragged tensor metadata from `slice_sizes` and `max_n_blocks`
make_ragged_tensor_metadata = ragged_tensor_details.make_ragged_tensor_metadata
make_ragged_tensor_metadata_torch = ragged_tensor_details.make_ragged_tensor_metadata_torch

# remap ragged tensor metadata to a new slice assignment
remap_ragged_tensor_metadata = ragged_tensor_details.remap_ragged_tensor_metadata
remap_ragged_tensor_metadata_torch = ragged_tensor_details.remap_ragged_tensor_metadata_torch

# ---------------------------------------------------------------------------- #
# sparse matrix
# ---------------------------------------------------------------------------- #


@dataclass
class SparseMatrix:
    indx: torch.Tensor
    vals: torch.Tensor
    mask: Bitmatrix

    def __post_init__(self):
        self.mask_metadata = make_bitmatrix_metadata(self.indx, self.mask)


# layout utilities
# ---------------------------------------------------------------------------- #


def get_layout(tensor: torch.Tensor | Tensor | None):
    if tensor is None:
        return None
    if isinstance(tensor, Tensor):
        return tensor.storage.layout
    return StridedLayout


def wrap_torch_tensor(torch_tensor, dtype=None):
    if dtype is None:
        dtype = torch_tensor.dtype
    shape = list(torch_tensor.shape)
    shape[torch_tensor.stride().index(1)] *= bitwidth(torch_tensor.dtype) // bitwidth(dtype)
    return Tensor(Storage(torch_tensor), dtype=dtype, shape=shape)


def convert_layout(tensor: Tensor, layout_cls: Type[Layout], **layout_kwargs):
    assert isinstance(tensor, Tensor)
    old_storage = tensor.storage
    old_data = old_storage.layout.unswizzle_data(old_storage.data)
    new_layout = layout_cls(old_data.shape, **layout_kwargs)
    new_data = new_layout.swizzle_data(old_data)
    attrs = {k.name: getattr(tensor, k.name) for k in fields(tensor) if k.name != "storage"}
    return Tensor(Storage(new_data, new_layout), **attrs)
