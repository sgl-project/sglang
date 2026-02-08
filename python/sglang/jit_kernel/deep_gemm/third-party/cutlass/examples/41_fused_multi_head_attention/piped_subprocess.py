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

from typing import List
import torch
import subprocess
import sys
import tempfile
import os
import numpy as np


TORCH_DTYPE_NAME = {
    torch.float32: "f32",
    torch.float16: "f16",
    torch.bfloat16: "b16"
}
NAME_TORCH_DTYPE = {v: k for k, v in TORCH_DTYPE_NAME.items()}

def _tensor_from_storage(tensor: torch.Tensor, dtype) -> torch.Tensor:
    # PyTorch >= 2.0
    if hasattr(tensor, 'untyped_storage'):
        return torch.tensor([], dtype=dtype).set_(tensor.untyped_storage())
    return torch.tensor([], dtype=dtype).set_(tensor.storage().untyped())

class PipedSubprocess:
    def __init__(self, binary: str) -> None:
        self.binary = binary
        self.tempdir_ctx = tempfile.TemporaryDirectory()

    def __enter__(self) -> "PipedSubprocess":
        self.subp = subprocess.Popen(self.binary, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr, text=True, bufsize=0)
        self.tempdir = self.tempdir_ctx.__enter__()
        self.file_counter = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.tempdir_ctx.__exit__(exc_type, exc_val, exc_tb)

    def temp_filename(self, suffix: str) -> str:
        self.file_counter += 1
        return os.path.join(self.tempdir, f"{self.file_counter}{suffix}")

    def write(self, *args) -> None:
        for a in args:
            self.subp.stdin.write(str(a) + " ")

    def writeTensor(self, tensor: torch.Tensor, name: str, stride_names: List[str]) -> None:
        print(f"Py ->C++: {TORCH_DTYPE_NAME[tensor.dtype]}:{name}")
        tensor_u8 = _tensor_from_storage(tensor, torch.uint8)
        self.write("tensor_begin", f"{TORCH_DTYPE_NAME[tensor.dtype]}:{name}", tensor_u8.shape[0])
        filename = self.temp_filename(f"{name}.tensor")
        assert tensor.storage_offset() == 0
        with open(filename, "wb+") as fd:
            fd.write(bytes(tensor_u8.numpy()))
        self.write("file", filename)
        self.write("tensor_end")

        for stride_name, stride_value in zip(stride_names, tensor.stride()):
            self.write(stride_name, stride_value)

    def readTensor(self, name, stride_name, shape) -> torch.Tensor:
        tmpfile = self.temp_filename(f"{name}.tensor")
        self.write("tmpfile", tmpfile)

        self.readExpect("tensor_begin")
        dtype_str, name = self.read().split(":")
        print(f"C++->Py : {dtype_str}:{name}")
        u8len = int(self.read())
        dtype = NAME_TORCH_DTYPE[dtype_str]

        self.readExpect("file")
        self.readExpect(tmpfile)

        with open(tmpfile, "rb") as fd:
            data = fd.read(u8len)
            # `np.array` is not strictly needed, but avoids a torch warning
            tensor_u8 = torch.frombuffer(np.array(data), dtype=torch.uint8, count=u8len)
        self.readExpect("tensor_end")
        
        tensor = _tensor_from_storage(tensor_u8, dtype)
        strides = []
        for sn in stride_name:
            self.readExpect(sn)
            strides.append(int(self.read()))
        if len(strides) != shape:
            strides.append(1)
        assert len(strides) == len(shape), name
        return torch.as_strided(tensor, shape, strides)

    def readNamed(self, name: str):
        self.readExpect(name)
        return self.read()

    def readExpect(self, what: str) -> None:
        r = self.read()
        if r != what:
            raise ValueError(f"Read {r} but expected {what}")

    def read(self):
        read_all = []
        # Skip initial whitespace
        while True:
            r = self.subp.stdout.read(1)
            if r not in [' ', "\n"]:
                read_all.append(r)
                break
        # Read data
        while True:
            r = self.subp.stdout.read(1)
            if r in [' ', "\n"]:
                break
            read_all.append(r)
        return ''.join(read_all)
        
