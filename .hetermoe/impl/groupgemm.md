specifically, the two groupgemms to use are fused moe (groupgemm) kernels for a8w8 and a16w4 (fused marlin)

implementation in the system:
    assume the input is already sorted and dispatched correctly for each groupgemm
    a8w8 has a quantization step for activation, you should be able to handle that elegantly
        if the input for the groupgemm is bf16, then no extra work is required
        if the input for the groupgemm is int8, then we need to quantize the weights accordingly
            if the a8w8 uses techniques like AWQ that requires scaling some columns, mark this work and remind me to impl later
    then just call these kernels

the reference (in reference.md) includes one mixed groupgemm usage of bf16 and nvfp4 with different tensor cores