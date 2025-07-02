import logging
import multiprocessing as mp
import pickle
import struct
from multiprocessing import shared_memory
from typing import Dict, List

logger = logging.getLogger(__name__)

"""
This class will be use in scheduler and dp controller
If this class is placed in the dp controller,
it will cause circular references, so it is placed in a separate file.
"""


class DPBalanceMeta:
    def __init__(self, num_workers: int, do_init: bool = False):
        self.mutex = mp.Lock()
        self.shm_name_onfly_info = "sglang_dp_balance_onfly_info"
        self.shm_name_local_tokens = "sglang_dp_balance_local_tokens"
        self.onfly_info_size = (
            512 * num_workers * 8
        )  # max_onfly_req_per_worker * num_workers * dByte
        self.local_tokens_size = num_workers * 8 + 512
        self.num_workers = num_workers

        if do_init:
            self.shm1 = shared_memory.SharedMemory(
                name=self.shm_name_onfly_info, create=True, size=self.onfly_info_size
            )
            self.shm2 = shared_memory.SharedMemory(
                name=self.shm_name_local_tokens,
                create=True,
                size=self.local_tokens_size,
            )
            init_local_tokens = [0 for _ in range(num_workers)]
            init_onfly_req = [{} for _ in range(num_workers)]
            self.set_shared_local_tokens(init_local_tokens)
            self.set_shared_onfly_info(init_onfly_req)
            self.shm1.name

    def destructor(self):
        # we must destructor this class manually, otherwise will cause shm leak
        self.shm1.close()
        self.shm1.unlink()
        self.shm2.close()
        self.shm2.unlink()

    def get_shared_onfly(self) -> List[Dict[int, int]]:
        """Retrieve data from shared memory and deserialize it into List[Dict[int, int]]"""
        shm = shared_memory.SharedMemory(name=self.shm_name_onfly_info)

        header_size = struct.calcsize("Q")
        data_size = struct.unpack("Q", shm.buf[:header_size])[0]
        assert 0 <= data_size <= self.onfly_info_size, "no valid data in shared memory"

        serialized_data = bytes(shm.buf[header_size : header_size + data_size])
        onfly_info = pickle.loads(serialized_data)

        shm.close()
        return onfly_info

    def set_shared_onfly_info(self, data: List[Dict[int, int]]):
        """Serialize the data and write it to shared memory."""
        serialized_data = pickle.dumps(data)
        data_size = len(serialized_data)

        assert data_size < self.onfly_info_size, (
            f"The size of the serialized data {data_size} "
            f"exceeds the shared memory capacity {self.onfly_info_size} bytes. "
            "Please increase onfly_info_size."
        )

        shm = shared_memory.SharedMemory(name=self.shm_name_onfly_info)
        shm.buf[: struct.calcsize("Q")] = struct.pack("Q", data_size)
        shm.buf[struct.calcsize("Q") : struct.calcsize("Q") + data_size] = (
            serialized_data
        )
        shm.close()

    def get_shared_local_tokens(self) -> List[int]:
        shm = shared_memory.SharedMemory(name=self.shm_name_local_tokens)
        serialized_data = bytes(shm.buf)
        worker_onfly_data = pickle.loads(serialized_data)
        shm.close()
        return worker_onfly_data

    def set_shared_local_tokens(self, data: List[int]):
        serialized_data = pickle.dumps(data)
        data_size = len(serialized_data)

        shm = shared_memory.SharedMemory(name=self.shm_name_local_tokens)
        shm.buf[:data_size] = serialized_data

        shm.close()
