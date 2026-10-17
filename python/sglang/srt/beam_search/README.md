# Trie-conditioned beam search

The model-native trie output head constrains generated semantic IDs (SIDs) to
a three-codebook trie. Beam width is selected independently for each request.
Models without the optional trie artifact keep the ordinary beam-search path.

## Model artifact

Place `trie_output_head.safetensors` at the root of a local model directory or
Hugging Face model repository. Token IDs for depth `d` are
`token_start + d * codebook_size + code`.

The safetensors metadata must contain these integer-valued strings:

| Metadata | Value |
|---|---|
| `format_version` | `"1"` |
| `token_start` | First SID token ID, non-negative |
| `codebook_size` | Number of codes per codebook, positive |
| `num_codebooks` | `"3"` |

The artifact contains one-dimensional integer tensors in CSR order:

| Tensor | Contents |
|---|---|
| `root_token_ids` | Sorted, unique first-codebook codes |
| `level1_offsets` | Offsets into `level1_token_ids`, one per root plus a terminal offset |
| `level1_token_ids` | Sorted, unique second-codebook children of each root |
| `level2_offsets` | Offsets into `level2_token_ids`, one per second-level entry plus a terminal offset |
| `level2_token_ids` | Sorted third-codebook children of each two-code prefix |

Codes are relative to their codebooks, in `[0, codebook_size)`. Offsets start
at zero, end at the corresponding child-array length, and cannot describe
empty child groups. The configured token range must fit the model vocabulary.

## Request constraints and projection

Requests must use `max_new_tokens=3`; stop tokens inside the SID token range
are rejected. `beam_width` is selected independently for each request and
cannot exceed the expandable trie frontier or available request/KV capacity.

The compact projection supports NVIDIA CUDA, TP=1, an unquantized LM head,
and no LoRA or RL target. Other configurations use dense projection with the
same trie constraints. Full-vocabulary normalization is preserved by default.
`SGLANG_OPT_BEAM_TRIE_TOPK_LOGPROB=true` skips that normalizer and normalizes
only the retained valid candidates instead; this is an explicit approximation
that can change beam scores and ranking.

## Shared-context attention

Set `SGLANG_ENABLE_BEAM_SHARED_CONTEXT_ATTENTION=true` before starting the
server to opt into the native shared-prompt decode kernel for all-beam batches
using FlashInfer MHA, FP16/BF16 KV, and no sliding-window or encoder-decoder
attention. It buckets groups by their request-specific beam width.

Shared-context batches use eager execution; ordinary batches and supported
compact-trie-only batches retain their normal CUDA-graph paths. Attention logit
caps are not supported by the shared-context kernel.
