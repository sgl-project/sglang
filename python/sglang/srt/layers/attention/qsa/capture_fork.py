"""Fork a captured side stream from completed projection, without a main event."""
import ctypes as c

_driver = c.CDLL("libcuda.so.1")
_info = _driver.cuStreamGetCaptureInfo_v3
_info.argtypes = [c.c_void_p, c.POINTER(c.c_int), c.POINTER(c.c_uint64),
                  c.POINTER(c.c_void_p), c.POINTER(c.POINTER(c.c_void_p)),
                  c.POINTER(c.c_void_p), c.POINTER(c.c_size_t)]
_info.restype = c.c_int
_set = _driver.cuStreamUpdateCaptureDependencies_v2
_set.argtypes = [c.c_void_p, c.POINTER(c.c_void_p), c.c_void_p, c.c_size_t, c.c_uint]
_set.restype = c.c_int


def projection_frontier(stream):
    status, capture_id, graph = c.c_int(), c.c_uint64(), c.c_void_p()
    nodes, edge_data, count = c.POINTER(c.c_void_p)(), c.c_void_p(), c.c_size_t()
    error = _info(stream.cuda_stream, c.byref(status), c.byref(capture_id),
                  c.byref(graph), c.byref(nodes), c.byref(edge_data), c.byref(count))
    if error:
        raise RuntimeError(f"cuStreamGetCaptureInfo_v3: {error}")
    if status.value == 0:
        return None
    if status.value != 1 or not count.value:
        raise RuntimeError("Active projection capture has no dependency frontier")
    # The driver-owned array expires at the next stream API. Copy node handles
    # now; the nodes live until the graph is destroyed. Query edge data as well
    # to avoid a lossy query of the main stream's programmatic dependencies.
    return tuple(nodes[i] for i in range(count.value))


def fork_after_projection(stream, nodes):
    array = (c.c_void_p * len(nodes))(*nodes)
    # NULL edge data gives FULL completion dependencies: Qprep reads projected
    # qk only after it is written. Replace the temporary QKV event dependency.
    error = _set(stream.cuda_stream, array, None, len(nodes), 1)
    if error:
        raise RuntimeError(f"cuStreamUpdateCaptureDependencies_v2: {error}")
