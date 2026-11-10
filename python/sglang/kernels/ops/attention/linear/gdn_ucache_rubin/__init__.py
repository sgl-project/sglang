# SPDX-License-Identifier: Apache-2.0
"""FlashInfer's Rubin (sm_107a) build of the UMMA ReplaySSM u-cache flush kernel, vendored.

The three modules beside this file are byte-identical copies of
``flashinfer/gdn_kernels/{gdn_replay_mtp_umma_rubin,_umma_helpers_rubin,device_target}.py``
at flashinfer-ai/flashinfer#6791 head 8c3701fb052fafece6a39e1ae41b53f540cac963
(Apache-2.0); ``device_target.py`` is also identical to FlashInfer v0.7.2's.
sha256:

* gdn_replay_mtp_umma_rubin.py  ed45e8879a572369689090e29601c201e8f27b32e2bba3301a864877274c5e74
* _umma_helpers_rubin.py         e8c479dced4af16788d4fdcd326eb39b536d6dc18d58af29e331e89fa0b4794d
* device_target.py               7e8390150de1d6e2348161145110dec292c3a8aec68e43f016287c7cddda8b17

Local change on top of those: an fp16 k ring build (``k_f16``, chosen by the k ring's
dtype; ``mma_f16_16816`` in the helpers). With a bf16 k ring the kernel is the one above.
Also local: rings with rows of their own (``ring_indices``, a ``ring_rows`` build; without
it the kernel takes the state indices in that parameter and never reads it).

They are CuTe-DSL (nvidia-cutlass-dsl >= 4.8.0) and compile at first use; they import
nothing from FlashInfer except, lazily on the uGPU path only, ``flashinfer.green_ctx``.
``gdn_replayssm_ucache`` loads them on SM107 for images whose FlashInfer predates the
Rubin build (0.7.2). Drop this package once the image's FlashInfer ships #6791.
"""
