# Notes 
- auto moe backend is flashinfer trtllm
- inductor[moe] test total time is misleading as prefill uses triton_kernel Moe backend and not flashinfer_trtllm
- dataset = sharegpt
- OSL 8k