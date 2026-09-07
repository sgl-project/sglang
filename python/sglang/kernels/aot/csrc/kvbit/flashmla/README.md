# KVBit FlashMLA specialization

This directory contains the SM90 sparse-decode specialization used by the
DeepSeek V4 INT4 KV cache. It is derived from FlashMLA commit
`98751d47134c8f2f1a4df5b07875144c3d8075d1` and is distributed under the
MIT license in `LICENSE`.

The code is intentionally compiled into `kvbit_flashmla_ops`, separately from
the regular `flashmla_ops` extension. The regular extension is built from the
official `sgl-project/FlashMLA` dependency and retains its public ABI.

The `kvbit_int4_sparse_decode_fwd` specialization consumes a 368-byte row:

- 224 bytes: 448 signed two's-complement int4 values, even dimension in the
  low nibble
- 14 bytes: fourteen E4M3 `absmax / 7` steps, one per 32 NoPE dimensions
- 2 bytes: zero padding
- 128 bytes: 64 BF16 RoPE values

The specialization uses MODEL1/H64 and consumes NoPE directly without a
Hadamard transform. SWA and optional extra KV use the same row format and scale
contract. The official FlashMLA ABI remains unchanged.
