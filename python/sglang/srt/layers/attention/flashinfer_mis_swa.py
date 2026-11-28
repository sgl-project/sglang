"""FlashInfer FA2 variant for item-logical sliding-window MIS attention.

The variant owns the entire mask, so physical packed offsets never prune KV
traversal. It consumes the same linear-size metadata as ordinary MIS.
"""

import hashlib

import torch
from flashinfer import BatchPrefillWithPagedKVCacheWrapper

_VARIANT = r"""
struct MultiItemSlidingWindowAttention : AttentionVariantBase {
  static constexpr bool use_softmax = true;
  uint32_t qo_len, kv_len, window_left;
  float sm_scale_log2, soft_cap, soft_cap_scale;

  template <typename Params>
  __device__ __host__ MultiItemSlidingWindowAttention(
      const Params& params, uint32_t batch_idx, uint8_t* smem_ptr) {
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    // Traversal must retain the shared prefix even for the last packed item.
    window_left = kv_len;
    soft_cap = params.logits_soft_cap;
    soft_cap_scale = soft_cap > 0 ? params.sm_scale / soft_cap : 0;
    sm_scale_log2 = (soft_cap > 0 ? soft_cap : params.sm_scale) * math::log2e;
  }

  REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    return soft_cap > 0 ? float(math::tanh(logits * soft_cap_scale)) : logits;
  })

  REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
    // Reject tile padding before loading request-specific metadata.
    if (qo_idx >= qo_len || kv_idx >= kv_len) return false;
    const uint32_t q = kv_len - qo_len + qo_idx;
    const uint32_t prefix = params.mis_prefix[batch_idx];
    int64_t logical_q = q;
    int64_t logical_k = kv_idx;
    if (q < prefix) {
      if (kv_idx > q) return false;
    } else {
      const uint32_t pos = params.mis_positions[
          uint64_t(batch_idx) * params.mis_stride + q - prefix];
      logical_q = int64_t(prefix) + pos - 1;
      if (kv_idx >= prefix) {
        // q-pos is the preceding delimiter, which is not a visible key.
        if (kv_idx <= q - pos || kv_idx > q) return false;
        logical_k = logical_q - (q - kv_idx);
      }
    }
    return params.logical_window_left < 0 ||
           logical_q - logical_k <= params.logical_window_left;
  })
};
"""


class MultiItemSWAWrapper(BatchPrefillWithPagedKVCacheWrapper):
    def __init__(self, workspace, dtype_q, dtype_kv, head_dim, v_head_dim):
        uri = (
            "sglang_mis_swa_"
            + hashlib.sha256(_VARIANT.encode()).hexdigest()[:12]
            + f"_{dtype_q}_{dtype_kv}_{head_dim}_{v_head_dim}"
        ).replace("torch.", "")
        super().__init__(
            workspace,
            "NHD",
            backend="fa2",
            variant_owns_mask=True,
            jit_args=[
                uri,
                dtype_q,
                dtype_kv,
                dtype_q,
                torch.int32,
                head_dim,
                v_head_dim,
                ["mis_prefix", "mis_positions"],
                ["uint32_t", "uint16_t"],
                ["mis_stride", "logical_window_left", "sm_scale", "logits_soft_cap"],
                ["uint32_t", "int32_t", "double", "double"],
                "MultiItemSlidingWindowAttention",
                _VARIANT,
            ],
            jit_kwargs={
                "pos_encoding_mode": 0,
                "use_sliding_window": False,
                "use_logits_soft_cap": False,
                "use_fp16_qk_reduction": False,
            },
        )
        self.multi_item_params = None

    def forward(
        self,
        q,
        kv_cache,
        *,
        causal,
        window_left,
        sm_scale,
        logits_soft_cap,
        k_scale=None,
        v_scale=None,
    ):
        assert causal, "MIS requires causal attention"
        p = self.multi_item_params
        return self.run(
            q,
            kv_cache,
            p.prefix_len_ptr,
            p.token_pos_in_items_ptr,
            p.token_pos_in_items_len,
            window_left,
            sm_scale * (k_scale if k_scale is not None else 1.0),
            logits_soft_cap or 0.0,
            # Physical windows stay off; the variant uses logical_window_left.
            window_left=-1,
            v_scale=v_scale,
        )
