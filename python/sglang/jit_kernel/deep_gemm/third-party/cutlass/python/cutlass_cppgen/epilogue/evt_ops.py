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
Collection of builtin functions used for host reference in EVT
"""

import numpy as np

from cutlass_cppgen.utils.datatypes import is_cupy_tensor, is_numpy_tensor, is_torch_available, is_torch_tensor

if is_torch_available():
    import torch


def multiply_add(x, y, z):
    return x * y + z


def sum(x, dim):
    if is_numpy_tensor(x):
        return x.sum(axis=tuple(dim))
    elif is_torch_tensor(x):
        return torch.sum(x, dim)


def max(x, dim):
    if is_numpy_tensor(x):
        return x.max(axis=tuple(dim))
    elif is_torch_tensor(x):
        return torch.amax(x, dim)


def maximum(x, y):
    if is_numpy_tensor(x):
        return np.maximum(x, y)
    elif is_torch_tensor(x):
        return torch.maximum(x, torch.tensor(y))


def minimum(x, y):
    if is_numpy_tensor(x):
        return np.minimum(x, y)
    elif is_torch_tensor(x):
        return torch.minimum(x, torch.tensor(y))

def exp(x):
    if is_numpy_tensor(x):
        return np.exp(x)
    elif is_torch_tensor(x):
        return torch.exp(x)


##############################################################################
# Layout manipulate nodes
##############################################################################

def permute(x, indices: tuple):
    if is_numpy_tensor(x):
        return np.transpose(x, axes=indices)
    elif is_torch_tensor(x):
        return x.permute(*indices)


def reshape(x, new_shape: tuple):
    if is_numpy_tensor(x):
        return np.reshape(x, newshape=new_shape)
    elif is_torch_tensor(x):
        return x.view(new_shape)
