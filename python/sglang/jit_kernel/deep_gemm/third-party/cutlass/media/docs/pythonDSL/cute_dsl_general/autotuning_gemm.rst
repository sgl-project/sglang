.. _autotuning_gemm:

Guidance for Auto-Tuning
============================= 

Numerous GEMM kernel code examples are offered within our codebase. 
When integrating these kernels into frameworks, auto-tuning becomes essential 
for achieving optimal performance. This involves selecting the appropriate 
kernel parameters based on the inputs of real applications.
Next, we'll briefly introduce some tips on how to perform auto-tuning.

The auto-tuning process typically involves the following steps:

1. Define search space
2. Benchmark each configuration and select the kernel with the best performance
3. Enable caching to reduce the tuning cost

The search space defines the valid combinations of kernel parameters that can be used to run the kernels. 
Different inputs (shapes, data types, etc.) typically require different kernel parameters to achieve optimal performance.
The search space is related to the kernel. We take the Blackwell GEMM persistent kernel as an example. 
The search space is as follows:

- ``mma_tiler_mn``: Defines the dimensions of the matrix tile that each Matrix Multiply-Accumulate (MMA) instruction processes in a single operation. 
- ``cluster_shape_mn``: Specifies the number of CTAs along each dimension within a cluster. Refer `Parallel Thread Execution ISA documentation <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-family-instructions>`_ for the possible mma tiler size and cluster shape for different tensor data types.
- ``use_2cta_instrs``: Whether to utilize Blackwell's 2 CTA instructions for MMA/Copy.
- ``use_tma_store``: Whether to use Tensor Memory Access (TMA) instructions to store the result back to global memory.

After defining the search space, we could traverse all parameter combinations to find the optimal kernel. 
The ``autotune_gemm`` function below demonstrates a simple exhaustive search approach - it iterates 
through configurations, compiles and benchmarks each kernel, and returns the best performing one.
Since kernel compilation incurs overhead, it's important to cache and reuse compiled kernels 
to minimize host launch latency. CuTe DSL facilitates this through its separate compilation 
and execution workflow. More details can be found in :ref:`JIT_Caching`.
As demonstrated in the ``autotune_gemm`` function 
(between the ``begin of cache the compiled GEMM kernel`` and ``end of cache the compiled GEMM kernel`` comments), 
we can use ``cute.compile()`` to compile a kernel once, cache the compiled result, and reuse the cached JIT executor for multiple kernel 
executions. We could maintain a global configuration-to-kernel dictionary (``config_kernel_dict``) to cache the compiled GEMM kernels, 
where each key (``kernel_cache_key``) uniquely identifies a kernel based on its characteristics.
Usually we could use the {dtype + kernel configs} as the cached key for GEMM compilation. For example, 

.. code-block:: python

    kernel_cache_key = f"{ab_dtype}x{c_dtype}x{acc_dtype}x{use_2cta_instrs}x{mma_tiler}x{cluster_shape_mn}x{use_tma_store}"

If the input tensor's layout is static, we should add the shape in the cached key too.
Users can customize the ``benchmark`` function to measure kernel execution time.
For stable and reliable performance measurements:

1. Run a few warmup iterations (e.g., 5-10) to stabilize GPU temperature
2. Execute multiple timed iterations (e.g., 100-1000) for statistical significance
3. Use CUDA events and synchronization for precise timing
4. Lock GPU frequencies (SM and memory frequencies) with nvidia-smi
5. Process results by removing outliers and using min/avg statistics as measurements.

This ensures reliable kernel selection through proper benchmarking.

.. code-block:: python

    # get the best GEMM kernel for given input tensors
    def autotune_gemm(
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        stream: cuda.CUstream,
        use_2cta_instrs_list: List[bool] = [True],
        use_tma_store_list: List[bool] = [True],
        mma_tiler_m_list: List[int] = [256],
        mma_tiler_n_list: List[int] = [256],
        cluster_shape_m_list: List[int] = [2],
        cluster_shape_n_list: List[int] = [1],
    ):
        best_kernel = None
        min_time = float("inf")
        # traverse the search space
        for use_2cta_instrs in use_2cta_instrs_list:
            for use_tma_store in use_tma_store_list:
                for mma_tiler_mn in product(mma_tiler_m_list, mma_tiler_n_list):
                    for cluster_shape_mn in product(cluster_shape_m_list, cluster_shape_n_list):
                        acc_dtype = cutlass.Float32
                        hardware_info = cutlass.utils.HardwareInfo()
                        max_active_clusters = hardware_info.get_max_active_clusters(
                            cluster_shape_mn[0] * cluster_shape_mn[1]
                        )
                        # instance a GEMM kernel
                        gemm = PersistentDenseGemmKernel(
                            acc_dtype,
                            use_2cta_instrs,
                            mma_tiler_mn,
                            cluster_shape_mn,
                            use_tma_store,
                        )
                        # begin of cache the compiled GEMM kernel
                        if kernel_cache_key not in config_kernel_dict:
                            # compile gemm kernel
                            compiled_gemm = cute.compile(
                                gemm,
                                a,
                                b,
                                c,
                                max_active_clusters,
                                stream,
                            )
                            config_kernel_dict[kernel_cache_key] = compiled_gemm
                        else:
                            compiled_gemm = config_kernel_dict[kernel_cache_key]
                        # end of cache the compiled GEMM kernel
                        try:
                            # define a benchmark function to measure the execution time of the compiled GEMM kernel
                            cur_time = benchmark(
                                partial(compiled_gemm, a, b, c, stream),
                            )
                        except Exception as e:
                            print(f"Execution error: {e}")
                            cur_time = float("inf")
                        if cur_time < min_time:
                            min_time = cur_time
                            best_kernel = compiled_gemm
        if best_kernel is None:
            raise ValueError("No best kernel found")
        return best_kernel

This brute-force approach ensures we could find the optimal parameters, though at the cost of trying every possibilities.
For more advanced use cases, users can explore sophisticated optimization 
techniques like search space pruning and genetic algorithms to reduce tuning overhead and discover better 
configurations more efficiently.

To further optimize tuning performance, we can utilize caching mechanisms to avoid redundant computations.
We could cache the tuning results in a input-to-kernel dictionary (e.g., ``input_kernel_dict``). 
When processing inputs with matching ``config_key`` values, the cached kernel can be reused directly without re-tuning. 
The ``config_key`` is related with the input tensor's characteristics, such as the shape, data type, etc. 
The setup of ``config_key`` is very flexible, users can customize it based on their own application.
For instance, if the data type is fixed in users' application, we could use the input tensor's shape as the key, i.e., ``(m, n, k)``. 
To further reduce tuning overhead, we could consider using a simplified key like ``config_key = (power_of_2(m), power_of_2(n), power_of_2(k))``, 
where ``m``, ``n``, and ``k`` are rounded up to the nearest power of 2. This simplification can significantly reduce the number 
of unique keys while still maintaining good performance in most cases. However, it's important to validate that this 
approximation doesn't negatively impact performance for your specific use case. 

.. code-block:: python

    config_key = (m, n, k)
    if config_key in input_kernel_dict:
        compiled_gemm = input_kernel_dict[config_key]
    else:
        compiled_gemm = autotune_gemm(...)
        input_kernel_dict[config_key] = compiled_gemm
    # launch gemm kernel
    compiled_gemm(a_tensor, b_tensor, c_tensor, stream)

By following the methods above, you can customize your own auto-tuner to find the optimal GEMM kernel configuration 
for specific matrix dimensions and data types, significantly improving computational performance for models.
