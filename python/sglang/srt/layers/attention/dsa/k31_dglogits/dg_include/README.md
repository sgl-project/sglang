# DeepGEMM headers for dg_logits.cuh and dg_logits_1p.cuh

Unmodified copies of the headers under `deep_gemm/include/deep_gemm/` of
sgl-deep-gemm 0.2.3.post2 (https://github.com/sgl-project/DeepGEMM, a fork of
https://github.com/deepseek-ai/DeepGEMM; MIT License, Copyright (c) 2025 DeepSeek;
the license text is in `LICENSE` in this directory).

They are the DeepGEMM headers that `../dg_logits.cuh` and `../dg_logits_1p.cuh`
include, directly or through each other. `../dg_logits.py` puts this directory on
nvcc's include path ahead of the installed `deep_gemm` package, so the vendored
logits kernels compile from the same DeepGEMM sources whatever sgl-deep-gemm version
is installed (0.2.6 moved `epilogue/clean_logits.cuh` to `legacy/`). CUTLASS and CuTe
still come from the installed package.

SHA-256 of each file (equal to the files in the sgl-deep-gemm 0.2.3.post2 package):

```
a13373053608b286c1ac39ce771cdd6ab133f44ec4f016bd6fcbc5032132a1a9  deep_gemm/common/compile.cuh
872af0b873a9b367c01eaf68c321ddcd9d0f3c26ce7b8d3abfc8a66d89b8ad65  deep_gemm/common/cute_tie.cuh
ab6b2bd75c06c054fbbe97d5897efa7909d380ad531620770c031b3debdf2123  deep_gemm/common/exception.cuh
809b3d5e162f01b2d72248fd1040b2392eb4a9577b9776acd10b7b8ca5b29a4f  deep_gemm/common/math.cuh
7b216c4a4f94e6691e2f553b204f697257eac1312b71fd7d24e7b420815c67b8  deep_gemm/common/packing.cuh
d231e0ceb5e41d3128a82b3e326b6cfc5c46b58013edf0bfd48bf1704859711e  deep_gemm/common/ring_pipeline.cuh
56d1da681f345a9fe59a850f35ad580bcb6b4defab695e42e50d0ed919ba5daa  deep_gemm/common/tma_copy.cuh
8b1fc343a774f3025855249eecf3bf9402ccd6e688b285a1956e320e8e2faeb2  deep_gemm/common/types.cuh
55e53fc49a096c6b24774b0c99ce947dbb3f81a55ccf61a20581ed78a9c98702  deep_gemm/common/utils.cuh
06331e5023fa6209c8a6c903ac0bd296308def4278ec53ad93ca60786314620d  deep_gemm/epilogue/clean_logits.cuh
151de0f8dfb2d9e8a7b587c9fb5bedbbcb34fc885aa1d297a656281ab5d0c4f2  deep_gemm/layout/mqa_logits.cuh
439c57d8aa4914c01dcb55a874ace4fdaecec4682ea1099ba36e3cb8b23f41ab  deep_gemm/mma/sm100.cuh
9d5238875771a171521bd5ea2d1d46f599623dceb2da75419d4fbe9c0b9a55c3  deep_gemm/ptx/ld_st.cuh
69a3f9a8c6291aa44b16ab5e3c8fee9a66c64bab569561ad3cbc860ab53aae53  deep_gemm/ptx/tcgen05.cuh
8a9d5e03afdeb20f53b1027e9f6f3a2ef45bb33152dba2b61d778684e1716412  deep_gemm/ptx/utils.cuh
07aadc4c093521d1d3fc5ea8c4da2417db8aa9b4010fc1ea62fbf826fe297a74  deep_gemm/scheduler/sm100_mqa_logits.cuh
7b594c493abf8d21d2512a93d93c5abd1d8bac8485ea434b8630131c86e53b5a  deep_gemm/scheduler/sm100_paged_mqa_logits.cuh
```
