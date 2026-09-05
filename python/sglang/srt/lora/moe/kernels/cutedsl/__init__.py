"""CuTeDSL grouped GEMM for the MoE LoRA base path: two architectures
(``kernel_sm90_bf16`` WGMMA, ``kernel_sm100_bf16`` tcgen05), two row modes
(masked ``[E, m_max, *]`` slabs; contiguous flat segments, whose
``seg_offsets`` ride the ``group_m`` argument slot). ``api`` compiles and
binds; ``schedule_builder`` writes the packed schedules and owns their ABI;
``scheduler`` decodes them on device. The mainloops derive from the CUTLASS
dense persistent examples and keep their NVIDIA BSD-3 headers.
"""
