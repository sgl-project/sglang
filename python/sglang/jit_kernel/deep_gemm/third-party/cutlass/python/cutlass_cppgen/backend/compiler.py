#################################################################################################
#
# Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import ctypes
import json
import os
import sqlite3
import subprocess
import tempfile

from cutlass_cppgen.utils.lazy_import import lazy_import
cuda = lazy_import("cuda.cuda")
cudart = lazy_import("cuda.cudart")
nvrtc = lazy_import("cuda.nvrtc")
from cutlass_library import SubstituteTemplate

import cutlass_cppgen
from cutlass_cppgen import CACHE_FILE, CUTLASS_PATH, cuda_install_path, logger
from cutlass_cppgen.backend.gemm_operation import GemmOperationUniversal
from cutlass_cppgen.backend.library import ApiVersion
from cutlass_cppgen.backend.utils.device import device_cc

IncludeTemplate = r"""#include "${include}"
"""


def compile_with_nvcc(cmd, source, error_file):
    succeed = True
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        error_message = e.output.decode()
        with open(error_file, "w") as error_out:
            error_log = "Compilation error for the following kernel: \n"
            error_log += source
            error_log += "\nError Message:\n"
            error_log += error_message
            error_out.write(error_log)
        succeed = False
    if not succeed:
        # Print the error log to stdout if log level is set to warning or higher
        # verbosity. Otherwise, simply point to the error log file.
        logger.warning(error_log)
        raise Exception(f"Invalid Kernel. See '{error_file}' for details.")


class CompilationOptions:
    """
    Compilation options.
    """

    def __init__(self, flags, arch, include_paths=[]):
        self.includes = []
        self.include_paths = include_paths
        self.flags = flags
        self.arch = arch

    def get_str(self):
        opts = []
        for flag in self.flags:
            opts.append(flag)

        for incl in self.include_paths:
            opts.append(f"--include-path={incl}")

        arch_flag = f"-arch=sm_{self.arch}"
        if self.arch in [90, 100, 101, 103, 120, 121] and int(cutlass_cppgen.nvcc_version().split('.')[0]) >= 12:
            arch_flag += "a"
        opts.append(arch_flag)

        return " ".join(opts)

    def get(self):
        options = []

        for flag in self.flags:
            options.append(bytes(str.encode(flag)))

        for incl in self.include_paths:
            options.append(bytes(str.encode(f" --include-path={incl}")))

        arch_flag = f" -arch=sm_{self.arch}"
        if self.arch in [90, 100, 101, 103, 120, 121]:
            arch_flag += "a"

        options.append(bytes(str.encode(arch_flag)))

        return options


def convertToBinaryData(filename):
    with open(filename, "rb") as file:
        blobData = file.read()
    return blobData


def CDLLBin(host_binary):
    tempfile.tempdir = "./"
    temp_so = tempfile.NamedTemporaryFile(prefix="host_func", suffix=".so", delete=True)
    with open(temp_so.name, "wb") as file:
        file.write(host_binary)
    host_lib = ctypes.CDLL(temp_so.name)
    return host_lib


class ArtifactManager:
    """
    Artifact manager
    """

    def __init__(self) -> None:
        connection = sqlite3.connect(CACHE_FILE)
        cursor = connection.cursor()
        # Create the table if it does not already exist
        sqlite_create_table_query = """
        CREATE TABLE IF NOT EXISTS compiled_operations(op_key TEXT NOT NULL UNIQUE,
                                                        cubin BLOB NOT NULL,
                                                        hostbin BLOB NOT NULL,
                                                        op_name TEXT NOT NULL,
                                                        op_attrs TEXT NOT NULL)
        """
        cursor.execute(sqlite_create_table_query)
        connection.commit()
        cursor.close()

        self._nvrtc_compile_options = ["-std=c++17", "-default-device"]
        self._nvcc_compile_options = [
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-Xcudafe --diag_suppress=esa_on_defaulted_function_ignored",
        ]
        self.nvcc()
        self.compiled_cache_device = {}
        self.compiled_cache_host = {}

    def nvrtc(self):
        self.backend = "nvrtc"
        self.default_compile_options = self._nvrtc_compile_options

    def nvcc(self):
        self.backend = "nvcc"
        self.default_compile_options = self._nvcc_compile_options

    def insert_operation(self, op_key, cubin, hostfile, op_name, op_attrs):
        connection = sqlite3.connect(CACHE_FILE)
        cursor = connection.cursor()
        sqlite_insert_blob_query = """ INSERT OR IGNORE INTO compiled_operations (op_key, cubin, hostbin, op_name, op_attrs) VALUES (?, ?, ?, ?, ?)"""

        hostbin = convertToBinaryData(hostfile)

        data_tuple = (op_key, cubin, hostbin, op_name, json.dumps(op_attrs))

        cursor.execute(sqlite_insert_blob_query, data_tuple)
        connection.commit()
        cursor.close()

    def load_operation(self, op_key, extra_funcs):
        connection = sqlite3.connect(CACHE_FILE)
        cursor = connection.cursor()
        sqlite_fetch_blob_query = """SELECT * from compiled_operations where op_key = ?"""
        cursor.execute(sqlite_fetch_blob_query, (op_key,))
        record = cursor.fetchall()
        if len(record) == 0:
            return False
        for row in record:
            key, cubin_image, host_binary, operation_name, op_attr = row
            op_attr = json.loads(op_attr)
            err, module = cuda.cuModuleLoadData(cubin_image)
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError("Cuda Error: {}".format(err))

            err, kernel = cuda.cuModuleGetFunction(module, bytes(str.encode(operation_name)))
            self.compiled_cache_device[key] = kernel

            compiled_host_fns = {}
            host_lib = CDLLBin(host_binary)

            func_name = operation_name + "_get_params"
            func = getattr(host_lib, func_name)
            func.restype = ctypes.POINTER(ctypes.c_char * op_attr[0])
            compiled_host_fns["get_args"] = func

            func_name = operation_name + "_shared_memory_size"
            func = getattr(host_lib, func_name)
            compiled_host_fns["shared_memory_capacity"] = func()

            for attr in op_attr:
                if isinstance(attr, str):
                    func_name = operation_name + "_" + attr
                    func = getattr(host_lib, func_name)

                    # Set the return type of the function
                    if attr in extra_funcs and extra_funcs[attr] != None:
                        func.restype = extra_funcs[attr]

                    compiled_host_fns[attr] = func

            self.compiled_cache_host[key] = compiled_host_fns
        return True

    def emit_compile_(self, operation_list, compilation_options, host_compilation_options):
        """
        Compile a list of kernels and store them into database
        """
        source_buffer_device = ""
        source_buffer_host = ""
        # 1. include
        includes = []
        for operation in operation_list:
            for incl in operation.emitter.includes:
                if incl not in includes:
                    includes.append(incl)

        includes_host = ["builtin_types.h", "device_launch_parameters.h", "cstddef"] + includes
        for incl in includes:
            source_buffer_device += SubstituteTemplate(
                IncludeTemplate,
                {"include": incl},
            )

        for incl in includes_host:
            source_buffer_host += SubstituteTemplate(
                IncludeTemplate,
                {"include": incl},
            )

        # 2. Operations
        for operation in operation_list:
            source_buffer_device += operation.emit()
            source_buffer_host += operation.emit()
            values = {
                "operation_name": operation.name(),
                "operation_suffix": operation.emitter.operation_suffix,
            }
            source_buffer_device += SubstituteTemplate(
                operation.KernelTemplate,
                values,
            )
            source_buffer_host += SubstituteTemplate(operation.HostTemplate, values)

        if self.backend == "nvrtc":
            # 3. compile
            err, program = nvrtc.nvrtcCreateProgram(
                str.encode(source_buffer_device),
                bytes(str.encode("module.cu")),
                0, [], [])

            if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                raise RuntimeError("NVRTC Error: {}".format(err))

            # Compile program
            options = compilation_options.get()

            err, = nvrtc.nvrtcCompileProgram(program, len(options), options)
            if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                error_string = "NVRTC Error: {}\n".format(err)

                # Get log from compilation
                err, logSize = nvrtc.nvrtcGetProgramLogSize(program)
                if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                    raise RuntimeError("NVRTC Error: {}".format(err))

                log = b" " * logSize
                err, = nvrtc.nvrtcGetProgramLog(program, log)
                if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                    raise RuntimeError("NVRTC Error: {}".format(err))

                raise RuntimeError(error_string + log.decode() + source_buffer_device)

            # Get data from compilation
            err, dataSize = nvrtc.nvrtcGetCUBINSize(program)
            if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                raise RuntimeError("NVRTC Error: {}".format(err))

            cubin_image = b" " * dataSize
            (err,) = nvrtc.nvrtcGetCUBIN(program, cubin_image)
            if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                raise RuntimeError("NVRTC Error: {}".format(err))

        else:  # with nvcc backend
            # emit code
            tempfile.tempdir = "./"
            temp_cu = tempfile.NamedTemporaryFile(
                prefix="kernel", suffix=".cu", delete=True)
            temp_cubin = tempfile.NamedTemporaryFile(
                prefix="kernel", suffix=".cubin", delete=True)
            with open(temp_cu.name, "w") as file:
                file.write(source_buffer_device)

            # compile with nvcc
            cmd_template = "${cuda_install_path}/bin/nvcc ${options} -cubin ${srcfile} -o ${tarfile}"
            values = {
                "cuda_install_path": cuda_install_path(),
                "options": compilation_options.get_str(),
                "srcfile": temp_cu.name,
                "tarfile": temp_cubin.name,
            }
            cmd = SubstituteTemplate(cmd_template, values)
            compile_with_nvcc(cmd.split(" "), source_buffer_device, "./cutlass_python_compilation_device_error.txt")

            # load the cubin image
            with open(temp_cubin.name, "rb") as file:
                cubin_image = file.read()

        tempfile.tempdir = "./"
        temp_src = tempfile.NamedTemporaryFile(
            prefix="host_src", suffix=".cu", delete=True)

        # Write the host source
        with open(temp_src.name, "w") as outfile:
            outfile.write(source_buffer_host)

        temp_dst = tempfile.NamedTemporaryFile(
            prefix="host_func", suffix=".so", delete=True)

        # Set up host compilation arguments
        cmd = []
        cmd.append(f"{cuda_install_path()}/bin/nvcc")
        cmd.extend(["-x", "cu", "-Xcompiler=-fpermissive", "-Xcompiler=-w", "-Xcompiler=-fPIC"])
        cmd.extend(host_compilation_options.get_str().split(" "))
        cmd.extend(["-shared", "-o", temp_dst.name, temp_src.name, "-lcudart", "-lcuda"])

        # Comile and load the library
        compile_with_nvcc( cmd, source_buffer_host, error_file="./cutlass_python_compilation_host_error.txt")
        host_lib = ctypes.CDLL(temp_dst.name)

        return cubin_image, host_lib, temp_dst

    def add_module(self, operations, compile_options=None, bypass_cache=False):
        """
        Insert a new compiled device module
        """
        include_paths = [
            cuda_install_path() + "/include",
            CUTLASS_PATH + "/include",
            CUTLASS_PATH + "/tools/util/include",
            CUTLASS_PATH + "/python/cutlass/cpp/include",
        ]

        cutlass_cppgen.initialize_cuda_context()
        arch = device_cc()

        host_compile_options = CompilationOptions(
            self._nvcc_compile_options, arch, include_paths)
        if compile_options is None:
            compile_options = CompilationOptions(
                self.default_compile_options, arch, include_paths)
        # save the cubin
        operation_key = []
        operation_list = []
        for operation in operations:
            # step 1: get kernel string as key
            key = operation.rt_module.emit() + operation.procedural_name() + self.backend
            # step 1: check if the operation is in cache
            compiled_kernel = self.compiled_cache_device.get(key)

            if compiled_kernel is None and not bypass_cache:
                hit = self.load_operation(key, getattr( operation.rt_module, "extra_funcs", {}))
                if hit:
                    compiled_kernel = self.compiled_cache_device.get(key)
                    assert compiled_kernel is not None
            if compiled_kernel is not None:
                operation.rt_module.kernel = compiled_kernel
                compiled_host_fns = self.compiled_cache_host.get(key)
                assert compiled_host_fns is not None
                for key in compiled_host_fns.keys():
                    setattr(operation.rt_module, key, compiled_host_fns[key])
                operation.rt_module.initialize()
            else:
                operation_list.append(operation.rt_module)
                operation_key.append(key)

        if len(operation_list) > 0:
            cubin_image, host_lib, host_file = self.emit_compile_(
                operation_list, compile_options, host_compile_options)

            err, module = cuda.cuModuleLoadData(cubin_image)
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError("Cuda Error: {}".format(err))

            operation_name = []
            operation_attr = []
            for operation, key in zip(operation_list, operation_key):
                # get device kernels
                err, operation.kernel = cuda.cuModuleGetFunction(
                    module,
                    bytes(str.encode(operation.name()))
                )
                operation_name.append(operation.name())
                self.compiled_cache_device[key] = operation.kernel
                # get host functions
                compiled_host_fns = {}
                op_attr = []

                # get param size
                func_name = operation.name() + "_get_param_size"
                func = getattr(host_lib, func_name)
                param_size = func()

                func_name = operation.name() + "_get_params"
                func = getattr(host_lib, func_name)
                func.argtype = operation.argtype
                func.restype = ctypes.POINTER(ctypes.c_char * param_size)
                setattr(operation, "get_args", func)
                compiled_host_fns["get_args"] = func

                # set shared memory size
                func_name = operation.name() + "_shared_memory_size"
                func = getattr(host_lib, func_name)
                setattr(operation, "shared_memory_capacity", func())
                compiled_host_fns["shared_memory_capacity"] = func()
                # set the maximum dynamic shared size
                operation.initialize()

                # get extra functions
                op_attr.append(param_size)

                if hasattr(operation, "extra_funcs"):
                    for suffix, ret_type  in operation.extra_funcs.items():
                        func_name = operation.name() + "_" + suffix
                        func = getattr(host_lib, func_name)
                        if ret_type is not None:
                            func.restype = ret_type
                        setattr(operation, suffix, func)
                        compiled_host_fns[suffix] = func
                        op_attr.append(suffix)

                operation_attr.append(op_attr)
                self.compiled_cache_host[key] = compiled_host_fns

            for (key, operation_name, operation_attr,) in zip(operation_key, operation_name, operation_attr):
                self.insert_operation(
                    key, cubin_image, host_file.name, operation_name, operation_attr)
