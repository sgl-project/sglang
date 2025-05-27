#include "common.h"
#include "vec.h"

namespace {

template <typename scalar_t>
void rope_kernel_impl(
    scalar_t* __restrict__ q_pe_out,
    scalar_t* __restrict__ k_pe_out,
    int64_t* __restrict__ t_pos,
    scalar_t* __restrict__ q_pe,
    scalar_t* __restrict__ k_pe,
    scalar_t* __restrict__ t_emb_pos,
    int64_t seq_len,
    int64_t num_head,
    int64_t rotary_dim,
    int64_t HR,
    int64_t q_pe_stride_s,
    int64_t out_stride_qs,
    int64_t out_stride_ks,
    int64_t HK,
    int64_t k_pe_stride_s,
    int64_t q_pe_stride_n,
    int64_t out_stride_qn) {
  int64_t COFF = HR / 2;
  at::parallel_for(0, seq_len * num_head, GRAIN_SIZE / rotary_dim, [&](int64_t begin, int64_t end) {
    int64_t seq{0}, head_id{0};
    data_index_init(begin, seq, seq_len, head_id, num_head);
    for (int64_t i = begin; i < end; ++i) {
      int64_t in_offset_q = seq * q_pe_stride_s + head_id * q_pe_stride_n;
      int64_t out_offset_q = seq * out_stride_qs + head_id * out_stride_qn;
      int64_t out_offset_k = seq * out_stride_ks;
      int64_t p = 0;
      scalar_t* sin_start = nullptr;
      scalar_t* cos_start = nullptr;
      // step 0) get the rotary position embedding for the current position
      p = t_pos[seq];
      sin_start = t_emb_pos + p * HR + COFF;
      cos_start = t_emb_pos + p * HR;
      // step 1) apply_rotary_pos_emb for the rotary_dim elements in every
      // head of query/key
      for (int64_t h = 0; h < rotary_dim; h += 2) {
        scalar_t cos = cos_start[h >> 1];
        scalar_t sin = sin_start[h >> 1];
        scalar_t in1 = q_pe[in_offset_q + h];
        scalar_t in2 = q_pe[in_offset_q + h + 1];
        scalar_t out1 = in1 * cos - in2 * sin;
        scalar_t out2 = in2 * cos + in1 * sin;
        q_pe_out[out_offset_q + h] = out1;
        q_pe_out[out_offset_q + h + 1] = out2;
      }
      for (int64_t h = 0; h < HK; h += 2) {
        scalar_t cos = cos_start[h >> 1];
        scalar_t sin = sin_start[h >> 1];
        int64_t k_pe_offset = seq * k_pe_stride_s;
        scalar_t in1_k = k_pe[k_pe_offset + h];
        scalar_t in2_k = k_pe[k_pe_offset + h + 1];
        scalar_t out1_k = in1_k * cos - in2_k * sin;
        scalar_t out2_k = in2_k * cos + in1_k * sin;
        k_pe_out[out_offset_k + h] = out1_k;
        k_pe_out[out_offset_k + h + 1] = out2_k;
      }
      // move to the next index
      data_index_step(seq, seq_len, head_id, num_head);
    }
  });
}

template <typename scalar_t>
void rotary_embedding_origin_impl(
    const int64_t* __restrict__ positions,       // [batch_size, seq_len] or
                                                 // [num_tokens]
    scalar_t* __restrict__ query,                /// [batch_size, seq_len, num_heads,
                                                 /// head_size] or [num_tokens, num_heads,
                                                 /// head_size]
    scalar_t* __restrict__ key,                  // [batch_size, seq_len, num_kv_heads,
                                                 // head_size] or [num_tokens, num_kv_heads,
                                                 // head_size]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim //
                                                 // 2]
    const int rot_dim,
    const int64_t query_stride,
    const int64_t key_stride,
    const int num_heads,
    const int num_kv_heads,
    const int head_size,
    const int num_tokens) {
  using bVec = at::vec::Vectorized<scalar_t>;
  using fVec = at::vec::Vectorized<float>;
  constexpr int bVecSize = bVec::size();

  const int embed_dim = rot_dim / 2;
  bool flag = (embed_dim % bVecSize == 0);
  const int loop_upper = flag ? embed_dim : embed_dim - bVecSize;

  auto compute_loop = [&](const int64_t token_head, const scalar_t* cache_ptr, scalar_t* qk) {
    int j = 0;
    for (; j < loop_upper; j += bVecSize) {
      const int rot_offset = j;
      const int x_index = rot_offset;
      const int y_index = embed_dim + rot_offset;

      const int64_t out_x = token_head + x_index;
      const int64_t out_y = token_head + y_index;

      bVec _cos = bVec::loadu(cache_ptr + x_index);
      bVec _sin = bVec::loadu(cache_ptr + y_index);

      bVec _q_x = bVec::loadu(qk + out_x);
      bVec _q_y = bVec::loadu(qk + out_y);
      fVec _cos_0, _cos_1;
      std::tie(_cos_0, _cos_1) = at::vec::convert_to_float(_cos);
      fVec _sin_0, _sin_1;
      std::tie(_sin_0, _sin_1) = at::vec::convert_to_float(_sin);
      fVec _q_x_0, _q_x_1;
      std::tie(_q_x_0, _q_x_1) = at::vec::convert_to_float(_q_x);
      fVec _q_y_0, _q_y_1;
      std::tie(_q_y_0, _q_y_1) = at::vec::convert_to_float(_q_y);

      auto out1_0 = _q_x_0 * _cos_0 - _q_y_0 * _sin_0;
      auto out1_1 = _q_x_1 * _cos_1 - _q_y_1 * _sin_1;
      auto out1 = convert_from_float_ext<scalar_t>(out1_0, out1_1);
      out1.store(qk + out_x);

      auto out2_0 = _q_y_0 * _cos_0 + _q_x_0 * _sin_0;
      auto out2_1 = _q_y_1 * _cos_1 + _q_x_1 * _sin_1;
      auto out2 = convert_from_float_ext<scalar_t>(out2_0, out2_1);
      out2.store(qk + out_y);
    }
    if (!flag) {
      for (; j < embed_dim; ++j) {
        const int x_index = j;
        const int y_index = embed_dim + j;

        const int64_t out_x = token_head + x_index;
        const int64_t out_y = token_head + y_index;

        const float _cos = cache_ptr[x_index];
        const float _sin = cache_ptr[y_index];

        const float _q_x = qk[out_x];
        const float _q_y = qk[out_y];

        qk[out_x] = _q_x * _cos - _q_y * _sin;
        qk[out_y] = _q_y * _cos + _q_x * _sin;
      }
    }
  };

#pragma omp parallel for
  for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
    int64_t pos = positions[token_idx];
    const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;

    for (int i = 0; i < num_heads; ++i) {
      const int head_idx = i;
      const int64_t token_head = token_idx * query_stride + head_idx * head_size;
      compute_loop(token_head, cache_ptr, query);
    }

    for (int i = 0; i < num_kv_heads; ++i) {
      const int head_idx = i;
      const int64_t token_head = token_idx * key_stride + head_idx * head_size;
      compute_loop(token_head, cache_ptr, key);
    }
  }
}

template <typename scalar_t>
void rotary_embedding_origin_gptj_impl(
    const int64_t* __restrict__ positions,       // [batch_size, seq_len] or
                                                 // [num_tokens]
    scalar_t* __restrict__ query,                /// [batch_size, seq_len, num_heads,
                                                 /// head_size] or [num_tokens, num_heads,
                                                 /// head_size]
    scalar_t* __restrict__ key,                  // [batch_size, seq_len, num_kv_heads,
                                                 // head_size] or [num_tokens, num_kv_heads,
                                                 // head_size]
    const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim //
                                                 // 2]
    const int rot_dim,
    const int64_t query_stride,
    const int64_t key_stride,
    const int num_heads,
    const int num_kv_heads,
    const int head_size,
    const int num_tokens) {
  const int embed_dim = rot_dim / 2;

#pragma omp parallel for collapse(2)
  for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
    for (int i = 0; i < num_heads; ++i) {
      int64_t pos = positions[token_idx];
      const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;
      const scalar_t* cos_cache_ptr = cache_ptr;
      const scalar_t* sin_cache_ptr = cache_ptr + embed_dim;
      const int head_idx = i;
      const int64_t token_head = token_idx * query_stride + head_idx * head_size;
      scalar_t* head_query = token_head + query;
      for (int j = 0; j < embed_dim; j += 1) {
        const int rot_offset = j;
        const int x_index = 2 * rot_offset;
        const int y_index = 2 * rot_offset + 1;

        const float cos = cos_cache_ptr[rot_offset];
        const float sin = sin_cache_ptr[rot_offset];

        const float x = head_query[x_index];
        const float y = head_query[y_index];

        head_query[x_index] = x * cos - y * sin;
        head_query[y_index] = y * cos + x * sin;
      }
    }
  }

#pragma omp parallel for collapse(2)
  for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
    for (int i = 0; i < num_kv_heads; ++i) {
      int64_t pos = positions[token_idx];
      const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;
      const scalar_t* cos_cache_ptr = cache_ptr;
      const scalar_t* sin_cache_ptr = cache_ptr + embed_dim;
      const int head_idx = i;
      const int64_t token_head = token_idx * key_stride + head_idx * head_size;
      scalar_t* head_key = key + token_head;
      for (int j = 0; j < embed_dim; j += 1) {
        const int rot_offset = j;
        const int x_index = 2 * rot_offset;
        const int y_index = 2 * rot_offset + 1;

        const float cos = cos_cache_ptr[rot_offset];
        const float sin = sin_cache_ptr[rot_offset];

        const float x = head_key[x_index];
        const float y = head_key[y_index];

        head_key[x_index] = x * cos - y * sin;
        head_key[y_index] = y * cos + x * sin;
      }
    }
  }
}

}  // namespace

void rotary_embedding_origin_cpu(
    at::Tensor& positions,
    at::Tensor& query,
    at::Tensor& key,
    int64_t head_size,
    at::Tensor& cos_sin_cache,
    bool is_neox) {
  RECORD_FUNCTION("sgl-kernel::rotary_position_embedding_origin_cpu", std::vector<c10::IValue>({query, key}));
  int num_tokens = positions.numel();
  int rot_dim = cos_sin_cache.size(1);
  int num_heads = query.size(-1) / head_size;
  int num_kv_heads = key.size(-1) / head_size;
  int64_t key_stride = key.stride(-2);
  int64_t query_stride = query.stride(-2);

  AT_DISPATCH_REDUCED_FLOATING_TYPES(query.scalar_type(), "rotary_embedding_origin_impl", [&] {
    if (is_neox) {
      rotary_embedding_origin_impl(
          positions.data_ptr<int64_t>(),
          query.data_ptr<scalar_t>(),
          key.data_ptr<scalar_t>(),
          cos_sin_cache.data_ptr<scalar_t>(),
          rot_dim,
          query_stride,
          key_stride,
          num_heads,
          num_kv_heads,
          head_size,
          num_tokens);
    } else {
      rotary_embedding_origin_gptj_impl(
          positions.data_ptr<int64_t>(),
          query.data_ptr<scalar_t>(),
          key.data_ptr<scalar_t>(),
          cos_sin_cache.data_ptr<scalar_t>(),
          rot_dim,
          query_stride,
          key_stride,
          num_heads,
          num_kv_heads,
          head_size,
          num_tokens);
    }
  });
}
std::tuple<at::Tensor, at::Tensor>
rotary_position_embedding_cpu(at::Tensor& t_pos, at::Tensor& q_pe, at::Tensor& k_pe, at::Tensor& t_emb_pos) {
  RECORD_FUNCTION(
      "sgl-kernel::rotary_position_embedding_cpu", std::vector<c10::IValue>({t_pos, q_pe, k_pe, t_emb_pos}));
  CHECK_INPUT(t_pos);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q_pe);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(k_pe);
  CHECK_INPUT(t_emb_pos);
  CHECK_DIM(1, t_pos);
  CHECK_DIM(3, q_pe);
  CHECK_DIM(3, k_pe);
  CHECK_DIM(2, t_emb_pos);

  int64_t seq_len = q_pe.size(0);
  int64_t num_head = q_pe.size(1);
  int64_t rotary_dim = q_pe.size(2);
  int64_t HK = k_pe.size(2);
  int64_t HR = t_emb_pos.size(1);
  CHECK_EQ(HR, rotary_dim);
  CHECK_EQ(k_pe.size(0), seq_len);
  CHECK_EQ(k_pe.size(1), 1);
  CHECK_EQ(t_pos.size(0), seq_len);
  CHECK_EQ(HK, rotary_dim);

  at::Tensor q_pe_out = at::empty_like(q_pe);
  at::Tensor k_pe_out = at::empty_like(k_pe);
  int64_t q_pe_stride_s = q_pe.stride(0);
  int64_t q_pe_stride_n = q_pe.stride(1);
  int64_t k_pe_stride_s = k_pe.stride(0);
  int64_t out_stride_qs = q_pe_out.stride(0);
  int64_t out_stride_qn = q_pe_out.stride(1);
  int64_t out_stride_ks = k_pe_out.stride(0);

  const auto input_dtype = q_pe.scalar_type();
  TORCH_CHECK(t_pos.scalar_type() == at::kLong, "expect positions to be int64, got ", t_pos.scalar_type());
  TORCH_CHECK(input_dtype == k_pe.scalar_type(), "q_pe and k_pe must have the same data type");
  TORCH_CHECK(input_dtype == t_emb_pos.scalar_type(), "q_pe and t_emb_pos must have the same data type");

  AT_DISPATCH_REDUCED_FLOATING_TYPES(input_dtype, "rotary_position_embedding_cpu", [&] {
    rope_kernel_impl<scalar_t>(
        q_pe_out.data_ptr<scalar_t>(),
        k_pe_out.data_ptr<scalar_t>(),
        t_pos.data_ptr<int64_t>(),
        q_pe.data_ptr<scalar_t>(),
        k_pe.data_ptr<scalar_t>(),
        t_emb_pos.data_ptr<scalar_t>(),
        seq_len,
        num_head,
        rotary_dim,
        HR,
        q_pe_stride_s,
        out_stride_qs,
        out_stride_ks,
        HK,
        k_pe_stride_s,
        q_pe_stride_n,
        out_stride_qn);
  });
  return std::make_tuple(q_pe_out, k_pe_out);
}
