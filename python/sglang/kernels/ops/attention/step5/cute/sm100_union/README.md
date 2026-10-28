# Step5 SM100 union prefill

Private dependencies from StepFun commit b3055a1c0e5a17da35bbf36d8f60651e073706c3. Original source hashes and local modifications are recorded in MANIFEST.json. Copyright and license headers are retained.

This fork does not import or replace the public SGLang FA4 implementation. Package initializers are inert. FakeTensor descriptors are constructed locally without monkeypatching CUTLASS or modifying installed files. Compilation uses SGLang's cache owner with recursive source fingerprinting.

The candidate consumes compact Q/O rows. Query group count comes from union metadata, not ceil(T/qgroup). Private load and output hooks apply the group's compact-row offset. WorkQGlobal is the native per-query position vector; WorkQLocal/WorkQLen are compact start/end rows. The schedule computes union-relative causal limits, and retains the tail-load validity mask.

The public entry is step5_sparse_prefill. Backend-owned typed schedule buffers are allocated before decoder layers. Enabled by default on compute capability 10.x; disable it with SPARSE_PREFILL_SM100 in step5/options.py. No padded Q/O buffers or adapter packing kernels are used. Native SWA remains unchanged.
