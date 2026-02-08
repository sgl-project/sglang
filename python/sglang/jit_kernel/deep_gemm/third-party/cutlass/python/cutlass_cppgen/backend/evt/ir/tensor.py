#################################################################################################
#
# Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

"""
High-level class for tensor
"""

from cutlass_library import LayoutType

from cutlass_cppgen.backend.evt.ir.layout_algorithm import (
    Layout,
    broadcast,
    canonicalization,
    permutation,
    reshape,
    _reverse_tuple
)
from cutlass_cppgen.utils.datatypes import get_datatype_and_layout, get_tensor_shape, library_type


class Tensor:
    """
    The tensor abstracts the data type
    """
    def __init__(self, tensor=None, element=None, shape=None, stride=None,layout_tag=None, is_constant=False) -> None:
        if element is not None and tensor is not None:
            raise Exception(f"Must not specify both element and tensor")
        elif shape is not None and tensor is not None:
            raise Exception(f"Must not specify both shape and tensor")
        elif layout_tag is not None and tensor is not None:
            raise Exception(f"Must not specify both layout_tag and tensor")
        elif (element is None or (layout_tag is None and stride is None) or shape is None) and (tensor is None) :
            raise Exception(f"Must specify one of (element, shape, layout/stride) or (tensor)")
        elif stride is not None and tensor is not None:
            raise Exception(f"Must not specify both stride and tensor")
        elif stride is not None and layout_tag is not None:
            raise Exception(f"Must not specify layout_tag when stride is provided")

        if isinstance(tensor, Tensor):
            # Directly copy all the attributes
            self.__dict__.update(vars(tensor))
        else:
            if tensor is None:
                self.element = library_type(element)
            else:
                self.element, layout_tag = get_datatype_and_layout(tensor)
                shape = get_tensor_shape(tensor)
            if stride is not None:
                self.layout = Layout(shape[::-1], stride[::-1])
            else:
                if layout_tag == LayoutType.RowMajor:
                    self.layout = Layout(shape[::-1])
                elif layout_tag == LayoutType.ColumnMajor:
                    self.layout = permutation(Layout(shape), [idx for idx in reversed(range(len(shape)))])
            self.layout = canonicalization(self.layout)

            self.is_constant = is_constant
            # Save the tensor value if it is constant
            if is_constant and tensor is not None:
                self.value = tensor

    @property
    def shape(self):
        """
        Returns the RowMajor layout shape
        """
        return _reverse_tuple(self.layout.shape)

    @property
    def stride(self):
        """
        Returns the RowMajor layout stride
        """
        return _reverse_tuple(self.layout.stride)

    @property
    def rank(self):
        """
        Returns the rank of the tensor
        """
        return len(self.shape)

    #
    # Layout Algorithms
    #

    def broadcast(self, shape):
        """
        Broadcast self.layout to shape
        """
        assert isinstance(shape, tuple)
        self.layout = broadcast(self.layout, _reverse_tuple(shape))

    def reshape(self, shape):
        """
        Reshape self.layout to shape
        """
        assert isinstance(shape, tuple)
        reverse_shape = _reverse_tuple(shape)
        self.layout = reshape(self.layout, reverse_shape)

    def permute(self, indices):
        """
        Permute self.layout according to indices
        """
        length = len(indices)
        indices = [length - idx - 1 for idx in indices]
        self.layout = permutation(self.layout, indices[::-1])
