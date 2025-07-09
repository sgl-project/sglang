#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import torch
import logging

import nixl._utils as nixl_utils
from nixl._api import nixl_agent, nixl_agent_config

logger = logging.getLogger(__name__)

class Gds:
    def __init__(
        self,
        gds_file_path: str,
        buf_size: int,
    ):
        self.gds_file_path=gds_file_path
        self.buf_size = buf_size
        self.agent_config = nixl_agent_config(backends=[])
        self.x_agent = nixl_agent("GDSTester", self.agent_config)
        self.plugin_list = self.x_agent.get_plugin_list()
        assert "GDS" in self.plugin_list
        self.x_agent.create_backend("GDS")

    def tensor_nixl_obj(self, device_indices: torch.Tensor):
        # process for tensor to initial nixlXferDList object
        tensor_reg_descs = self.x_agent.register_memory(device_indices)
        if tensor_reg_descs is not None:
            return self.x_agent.get_xfer_descs(device_indices)
        else:
            raise ValueError("tensor register nixl failed")

    def file_nixl_obj(self, index:int, buff_size:int, fd):
        # process for gds file to initial nixlXferDList object
        agent_file_list = [(index, buff_size, fd, "b")]
        agent_file_descs = self.x_agent.register_memory(agent_file_list, "FILE")
        if agent_file_descs is not None:
            return agent_file_descs
        else:
            raise ValueError("file register nixl failed")

    def d2s(self, mode: str, file_path: str, device_indices: torch.Tensor, index: int, buff_size: int):
        valid_options = {"READ", "WRITE"}
        if mode not in valid_options:
            raise ValueError("mode has to be write/read")
        logger.info(f"total len for the tensor: {device_indices.numel()} * {device_indices.element_size()}")
        gpu_reg_descs = self.x_agent.get_reg_descs(device_indices)
        agent_xfer_tensor = self.tensor_nixl_obj(device_indices)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if not os.path.exists(file_path):
            with open(file_path, 'w') as file:
                logger.info("##create##" + file_path)
                pass
        fd = os.open(file_path, os.O_RDWR | os.O_CREAT)
        try:
            file_descs = self.file_nixl_obj(index, buff_size, fd)
            agent_xfer_files = file_descs.trim()
            # initialize transfer mode
            xfer_hdl = self.x_agent.initialize_xfer(
                mode, agent_xfer_tensor, agent_xfer_files, "GDSTester"
            )
            if not xfer_hdl:
                raise ValueError("Creating transfer failed.")
            state = self.x_agent.transfer(xfer_hdl)
            if state == "ERR":
                raise ValueError("Transfer got to Error state.")
            done = False
            while not done:
                state = self.x_agent.check_xfer_state(xfer_hdl)
                if state == "ERR":
                    raise ValueError("Transfer got to Error state.")
                elif state == "DONE":
                    done = True
                    logger.info("d2s Initiator done")
            self.x_agent.release_xfer_handle(xfer_hdl)
            self.x_agent.deregister_memory(gpu_reg_descs)
            self.x_agent.deregister_memory(file_descs)
        finally:
            os.close(fd)

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=None
        )
    head_num = 4
    head_dim = 8
    bacth_size = 6
    k_buffer = [
            torch.zeros((bacth_size, head_num, head_dim),dtype=torch.bfloat16,device="cuda")
            #torch.zeros((35907, 8, 128),dtype=torch.bfloat16,device="cuda")
            #assume that number of layer is 2 for test only
            for _ in range(2)
            ]
    logger.info("####GDS k_buffer###")
    k_buffer[0][0, 0, 0] = 2.0
    test_device_indices = k_buffer[0][0:3]
    occupy_size = test_device_indices.numel() * test_device_indices.element_size()
    logger.info("Write tensor to k buffer 0 and gds to local file")
    logger.info(k_buffer)
    gds_backend = Gds(gds_file_path="/tmp/gds/try.txt",buf_size=128)
    gds_backend.d2s("WRITE", "/tmp/gds/k_buffer_layer1", test_device_indices, 0, occupy_size)
    test2_device_indices = k_buffer[1][0:3]
    gds_backend.d2s(test2_device_indices, "READ", "/tmp/gds/k_buffer_layer1", 0, occupy_size)
    logger.info("GDS Read tensor back from local file to k buffer 1")
    logger.info(k_buffer)
    if k_buffer[1][0, 0, 0] != 2.0:
        raise ValueError("Data inconsistent after GDS")
    logger.info("######Test pass!######")


if __name__ == "__main__":
    os.environ["NIXL_PLUGIN_DIR"] = "/opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu/plugins"
    main()
