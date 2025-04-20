from typing import List, Tuple

import torch

if torch.version.hip is not None:
    # ROCM custom allreduce
    def init_custom_ar(
        meta: torch.Tensor,
        rank_data: torch.Tensor,
        handles: List[str],
        offsets: List[int],
        rank: int,
        full_nvlink: bool,
    ) -> int:
        return torch.ops.sgl_kernel.init_custom_ar.default(
            meta, rank_data, handles, offsets, rank, full_nvlink
        )

    def all_reduce_reg(fa: int, inp: torch.Tensor, out: torch.Tensor) -> None:
        torch.ops.sgl_kernel.all_reduce_reg.default(fa, inp, out)

    def all_reduce_unreg(
        fa: int, inp: torch.Tensor, reg_buffer: torch.Tensor, out: torch.Tensor
    ) -> None:
        torch.ops.sgl_kernel.all_reduce_unreg.default(fa, inp, reg_buffer, out)

    def dispose(fa: int) -> None:
        torch.ops.sgl_kernel.dispose.default(fa)

    def meta_size() -> int:
        return torch.ops.sgl_kernel.meta_size.default()

    def register_buffer(
        fa: int, t: torch.Tensor, handles: List[str], offsets: List[int]
    ) -> None:
        return torch.ops.sgl_kernel.register_buffer.default(fa, t, handles, offsets)

    def get_graph_buffer_ipc_meta(fa: int) -> Tuple[torch.Tensor, List[int]]:
        return torch.ops.sgl_kernel.get_graph_buffer_ipc_meta.default(fa)

    def register_graph_buffers(
        fa: int, handles: List[str], offsets: List[List[int]]
    ) -> None:
        torch.ops.sgl_kernel.register_graph_buffers.default(fa, handles, offsets)

    def allocate_meta_buffer(size: int) -> torch.Tensor:
        return torch.ops.sgl_kernel.allocate_meta_buffer.default(size)

    def get_meta_buffer_ipc_handle(inp: torch.Tensor) -> torch.Tensor:
        return torch.ops.sgl_kernel.get_meta_buffer_ipc_handle.default(inp)

else:

    def init_custom_ar(
        ipc_tensors: List[int], rank_data: torch.Tensor, rank: int, full_nvlink: bool
    ) -> int:
        r"""
        Initialize custom All-Reduce (AR) communication resources for distributed training.

        Parameters
        ----------
        ipc_tensors : List[int]
            List of IPC (Inter-Process Communication) handles or identifiers, used for
            inter-process or inter-GPU tensor sharing.
            Each item typically represents a handle to a GPU resource that can be
            shared across processes or nodes.
        rank_data : torch.Tensor
            Tensor containing data relevant to the current process's rank, which may
            include metadata or buffer pointers necessary for AR setup.
            Shape and dtype depend on the distributed backend and communication pattern.
        rank : int
            The rank (unique integer identifier) of the current process within the
            distributed group. Usually ranges from 0 to (world_size - 1).
        full_nvlink : bool
            Whether to enable full NVLink topology optimizations for communication.
            If True, assumes all participating GPUs are fully connected via NVLink for
            potentially higher bandwidth and lower latency.

        Returns
        -------
        status : int
            Status code indicating whether initialization was successful.
            Typically, 0 indicates success and non-zero indicates failure.

        Note
        ----
        This function is typically used in distributed deep learning setups where
        custom AR (All-Reduce) implementations are required to maximize performance
        on multi-GPU systems, especially when leveraging advanced interconnects
        such as NVLink.
        """
        return torch.ops.sgl_kernel.init_custom_ar.default(
            ipc_tensors, rank_data, rank, full_nvlink
        )

    def dispose(fa: int) -> None:
        r"""
        Release or dispose of a resource identified by its handle.

        Parameters
        ----------
        fa : int
            File or resource handle (identifier) to be disposed of or released.
            Typically, this is an integer value returned by a resource allocation
            or initialization function, such as a file descriptor, memory handle,
            or GPU resource identifier.

        Returns
        -------
        None

        Note
        ----
        This function is used to explicitly release or clean up resources
        associated with the given handle to prevent resource leaks.
        After calling this function, the handle `fa` should not be used again.
        """
        torch.ops.sgl_kernel.dispose.default(fa)

    def all_reduce(
        fa: int,
        inp: torch.Tensor,
        out: torch.Tensor,
        reg_buffer: int,
        reg_buffer_sz_bytes: int,
    ) -> None:
        r"""
        Perform an All-Reduce operation across distributed processes or devices.

        Parameters
        ----------
        fa : int
            Resource or communicator handle used to identify the All-Reduce context,
            typically obtained from an initialization function. This handle manages
            the underlying communication resources.
        inp : torch.Tensor
            Input tensor containing the local data to be reduced. Must be a
            contiguous tensor on the appropriate device (e.g., GPU).
            Shape and dtype should be compatible with the collective operation.
        out : torch.Tensor
            Output tensor to store the result of the All-Reduce operation.
            Must be pre-allocated and have the same shape and dtype as `inp`.
        reg_buffer : int
            Registered buffer handle or memory address used for communication.
            This may refer to a memory region registered with the backend for
            efficient transfers (e.g., for RDMA or GPU direct communication).
        reg_buffer_sz_bytes : int
            Size of the registered buffer in bytes. Should be large enough to
            accommodate the data being communicated in the All-Reduce operation.

        Returns
        -------
        None

        Note
        ----
        The All-Reduce operation aggregates data (e.g., sum, max) from all
        participating processes/devices and distributes the result back to each.
        This function assumes that all arguments are correctly initialized and
        compatible with the communication backend associated with `fa`.
        """
        torch.ops.sgl_kernel.all_reduce.default(
            fa, inp, out, reg_buffer, reg_buffer_sz_bytes
        )

    def get_graph_buffer_ipc_meta(fa) -> Tuple[List[int], List[int]]:
        r"""
        Retrieve IPC (Inter-Process Communication) metadata for a graph buffer.

        Parameters
        ----------
        fa : int
            Resource or graph buffer handle. This integer identifier is typically obtained
            from a prior initialization function and refers to a specific graph buffer for
            which IPC metadata is needed.

        Returns
        -------
        ipc_handles : List[int]
            A list of IPC handles (e.g., file descriptors or memory handles) required for
            sharing the graph buffer across processes or devices.
        ipc_sizes : List[int]
            A list of sizes in bytes corresponding to each IPC handle in `ipc_handles`.
            Each entry specifies the size of the buffer or memory region that can be
            accessed via the corresponding IPC handle.

        Note
        ----
        This function is commonly used in distributed or multi-process environments where
        direct memory access to graph buffers is required. The returned IPC metadata allows
        other processes to map and access the same memory regions efficiently, facilitating
        zero-copy communication between processes or GPUs.
        """
        return torch.ops.sgl_kernel.get_graph_buffer_ipc_meta.default(fa)

    def register_buffer(fa: int, fake_ipc_ptrs: List[int]) -> None:
        r"""
        Register external buffer pointers with a given resource or communication handle.

        Parameters
        ----------
        fa : int
            Resource handle or context identifier, typically obtained from an initialization
            function. This handle manages the context for which the buffers are being registered.
        fake_ipc_ptrs : List[int]
            List of external (possibly fake or simulated) IPC buffer pointers or memory addresses.
            Each integer in the list represents a pointer or handle to a memory region that
            needs to be registered with the backend for future communication or computation.

        Returns
        -------
        None

        Note
        ----
        Registering buffer pointers is often necessary in distributed or multi-process
        environments to enable efficient data sharing, memory mapping, or direct access.
        The term "fake" may indicate that these IPC pointers are placeholders, stubs,
        or simulated values used for testing, prototyping, or on platforms where true
        IPC handles are not available.
        """
        return torch.ops.sgl_kernel.register_buffer.default(fa, fake_ipc_ptrs)

    def register_graph_buffers(
        fa: int, handles: List[List[int]], offsets: List[List[int]]
    ) -> None:
        r"""
        Register multiple graph buffer regions with the specified resource or graph context.

        Parameters
        ----------
        fa : int
            Resource or graph context handle, typically representing a graph buffer manager
            or communication context. Usually obtained from an initialization function.
        handles : List[List[int]]
            A nested list of resource handles or buffer identifiers. Each sublist corresponds
            to a set of handles associated with a particular graph buffer or memory region.
            These handles may represent file descriptors, IPC handles, or device pointers
            used for inter-process or inter-device communication.
        offsets : List[List[int]]
            A nested list of integer offsets. Each sublist corresponds to the offsets within
            the buffer regions specified in the `handles` argument. Offsets are typically
            specified in bytes and indicate the start position of each sub-buffer within
            the larger buffer or memory region.

        Returns
        -------
        None

        Note
        ----
        This function is typically used in distributed or multi-process graph computations
        where multiple buffer regions need to be registered for efficient memory access
        and communication. The combination of `handles` and `offsets` allows for precise
        registration of sub-regions within larger graph buffers, enabling optimized
        memory mapping and data transfer.
        """
        torch.ops.sgl_kernel.register_graph_buffers.default(fa, handles, offsets)

    def meta_size() -> int:
        r"""
        Get the size (in bytes) of the metadata structure used internally.

        Returns
        -------
        size : int
            The size of the metadata structure, in bytes. This value indicates how
            much memory is required to store the internal metadata for buffers,
            resources, or communication contexts.

        Note
        ----
        The metadata structure is typically used to store information required for
        managing buffers, resources, or distributed communication. The returned size
        can be used for memory allocation or serialization purposes.
        """
        return torch.ops.sgl_kernel.meta_size.default()
