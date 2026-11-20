import ctypes
import multiprocessing as mp

import torch
from cuda.bindings import driver as d

from sglang.srt.mem_cache.pool_host.egm import EgmAllocation, _check

N = 4096


def reader(conn):
    torch.cuda.set_device(2)
    raw, size = conn.recv()
    handle = _check(
        d.cuMemImportFromShareableHandle(
            raw, d.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        ),
        "import",
    )
    addr = int(_check(d.cuMemAddressReserve(size, 0, 0, 0), "reserve"))
    _check(d.cuMemMap(addr, size, 0, handle, 0), "map")
    desc = d.CUmemAccessDesc()
    desc.location.type = d.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    desc.location.id = 2
    desc.flags = d.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READ
    _check(d.cuMemSetAccess(addr, size, [desc], 1), "access")
    buf = (ctypes.c_ubyte * N)()
    _check(d.cuMemcpyDtoH(ctypes.addressof(buf), addr, N), "copy to host")
    assert bytes(buf) == bytes(i % 251 for i in range(N))
    _check(d.cuMemUnmap(addr, size), "unmap")
    _check(d.cuMemAddressFree(addr, size), "free")
    _check(d.cuMemRelease(handle), "release")
    conn.send("byte-exact")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch.cuda.set_device(0)
    a = EgmAllocation(N, 0)
    try:
        ctypes.memmove(a.address, bytes(i % 251 for i in range(N)), N)
        pconn, cconn = mp.Pipe()
        p = mp.Process(target=reader, args=(cconn,))
        p.start()
        pconn.send((a.export(), a.size))
        if not pconn.poll(30):
            p.terminate()
            p.join()
            raise TimeoutError("reader did not finish")
        print("reader:", pconn.recv(), flush=True)
        p.join(10)
        if p.exitcode != 0:
            raise RuntimeError(f"reader exit: {p.exitcode}")
    finally:
        a.close()
