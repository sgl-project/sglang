import time

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPoolHost

if __name__ == "__main__":
    # todo: update the test according to the latest changes in the cache controller
    mem_pool_device = MHATokenToKVPool(
        size=10000,
        dtype=torch.float16,
        head_num=12,
        head_dim=512,
        layer_num=12,
        device="cuda:0",
    )
    # todo: move into a separate test file

    mem_pool_host = MLATokenToKVPoolHost(mem_pool_device)
    controller = HiCacheController(mem_pool_device, mem_pool_host)

    allocations = []
    host_backups = []

    def evict_device(need_size):
        i = 0
        while need_size > 0:
            if i >= len(allocations):
                time.sleep(0.1)
                i = 0  # Reset index to start over
                continue
            device_indices, host_indices = allocations[i]
            num_evicted = controller.evict_device(device_indices, host_indices)
            if num_evicted > 0:
                need_size -= num_evicted
                if mem_pool_host.is_backup(host_indices):
                    host_backups.append(host_indices)
                allocations.pop(i)
            else:
                i += 1  # Only increment if no eviction happened
            if need_size <= 0:
                break

    def evict_host(need_size):
        i = 0
        while need_size > 0:
            if i >= len(host_backups):
                time.sleep(0.1)
                i = 0  # Reset index to start over
                continue
            host_indices = host_backups[i]
            num_evicted = controller.evict_host(host_indices)
            if num_evicted > 0:
                need_size -= num_evicted
                # Remove the evicted host backup
                host_backups.pop(i)
            else:
                i += 1  # Only increment if no eviction happened
            if need_size <= 0:
                break

    import random

    for i in range(100):
        input_size = random.randint(100, 1000)
        device_indices = mem_pool_device.alloc(input_size)
        if device_indices is None:
            # no sufficient device memory available
            need_size = input_size - mem_pool_device.available_size()
            evict_device(need_size)
            device_indices = mem_pool_device.alloc(input_size)
        host_indices = controller.write_through(device_indices=device_indices)
        if host_indices is None:
            # no sufficient host memory available
            need_size = input_size - mem_pool_host.available_size()
            evict_host(need_size)

            host_indices = controller.write_through(device_indices=device_indices)
        allocations.append((device_indices, host_indices))
        time.sleep(0.01)

    host_backup_copy = [i for i in host_backups]
    for i, host_indices in enumerate(host_backup_copy):
        if mem_pool_host.is_backup(host_indices):
            device_indices = controller.load_back(host_indices=host_indices)
            if device_indices is None:
                need_size = len(host_indices) - mem_pool_device.available_size()
                evict_device(need_size)
                device_indices = controller.load_back(host_indices=host_indices)
            allocations.append((device_indices, host_indices))

    time.sleep(1)
