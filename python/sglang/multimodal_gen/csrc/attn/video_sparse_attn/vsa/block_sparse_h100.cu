// # Define TORCH_COMPILE macro

#include "kittens.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <iostream>

using namespace kittens;
namespace cg = cooperative_groups;
constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
template <int D> struct fwd_attend_ker_tile_dims {};
template <> struct fwd_attend_ker_tile_dims<64> {
  constexpr static int tile_width = (64);
  constexpr static int qo_height = (4 * 16);
  constexpr static int kv_height = (4 * 16);
};
template <> struct fwd_attend_ker_tile_dims<128> {
  constexpr static int tile_width = (128);
  constexpr static int qo_height = (4 * 16);
  constexpr static int kv_height = (4 * 16);
};
template <int D> struct fwd_globals {
  using q_tile = st_bf<fwd_attend_ker_tile_dims<D>::qo_height,
                       fwd_attend_ker_tile_dims<D>::tile_width>;
  using k_tile = st_bf<fwd_attend_ker_tile_dims<D>::kv_height,
                       fwd_attend_ker_tile_dims<D>::tile_width>;
  using v_tile = st_bf<fwd_attend_ker_tile_dims<D>::kv_height,
                       fwd_attend_ker_tile_dims<D>::tile_width>;
  using l_col_vec = col_vec<st_fl<fwd_attend_ker_tile_dims<D>::qo_height,
                                  fwd_attend_ker_tile_dims<D>::tile_width>>;
  using o_tile = st_bf<fwd_attend_ker_tile_dims<D>::qo_height,
                       fwd_attend_ker_tile_dims<D>::tile_width>;

  using q_gl = gl<bf16, -1, -1, -1, -1, q_tile>;
  using k_gl = gl<bf16, -1, -1, -1, -1, k_tile>;
  using v_gl = gl<bf16, -1, -1, -1, -1, v_tile>;
  using l_gl = gl<float, -1, -1, -1, -1, l_col_vec>;
  using o_gl = gl<bf16, -1, -1, -1, -1, o_tile>;

  q_gl q;
  k_gl k;
  v_gl v;
  l_gl l;
  o_gl o;

  const int N;
  const int hr;
  const int max_kv_blocks_per_q;

  int32_t *__restrict__ q2k_block_sparse_index;
  int32_t *__restrict__ q2k_block_sparse_num;
  int32_t *__restrict__ block_size;
};

template <int D>
__global__ __launch_bounds__(128, 4) void fwd_attend_ker(
    const __grid_constant__ fwd_globals<D> g) { // use block size of 64
  extern __shared__ int __shm[];
  tma_swizzle_allocator al((int *)&__shm[0]);

  using K = fwd_attend_ker_tile_dims<D>;

  using q_tile = st_bf<64, K::tile_width>;
  using k_tile = st_bf<64, K::tile_width>;
  using v_tile = st_bf<64, K::tile_width>;
  using l_col_vec = col_vec<st_fl<64, K::tile_width>>;
  using o_tile = st_bf<64, K::tile_width>;

  q_tile(&q_smem)[1] = al.allocate<q_tile, 1>();

  k_tile(&k_smem)[1] = al.allocate<k_tile, 1>();

  v_tile(&v_smem)[1] = al.allocate<v_tile, 1>();

  l_col_vec(&l_smem)[1] = al.allocate<l_col_vec, 1>();

  auto(*o_smem) = reinterpret_cast<o_tile(*)>(q_smem);

  int kv_head_idx = blockIdx.y / g.hr;
  int seq_idx = blockIdx.x;

  int32_t *q2k_block_sparse_index_ptr =
      g.q2k_block_sparse_index +
      blockIdx.z * gridDim.y * gridDim.x * g.max_kv_blocks_per_q +
      blockIdx.y * gridDim.x * g.max_kv_blocks_per_q +
      blockIdx.x * g.max_kv_blocks_per_q;
  int32_t *q2k_block_sparse_num_ptr = g.q2k_block_sparse_num +
                                      blockIdx.z * gridDim.y * gridDim.x +
                                      blockIdx.y * gridDim.x + blockIdx.x;
  int32_t kv_blocks = q2k_block_sparse_num_ptr[0];
  __shared__ kittens::semaphore qsmem_semaphore, k_smem_arrived, v_smem_arrived;
  if (threadIdx.x == 0) {
    int32_t kv_block_index = q2k_block_sparse_index_ptr[0];

    init_semaphore(qsmem_semaphore, 0, 1);
    init_semaphore(k_smem_arrived, 0, 1);
    init_semaphore(v_smem_arrived, 0, 1);

    // preload q block
    coord<q_tile> q_tile_idx = {blockIdx.z, blockIdx.y, seq_idx, 0};
    tma::expect_bytes(qsmem_semaphore, sizeof(q_smem));
    tma::load_async(q_smem[0], g.q, q_tile_idx, qsmem_semaphore);

    // preload the zeroth block of kv
    tma::expect_bytes(k_smem_arrived, sizeof(k_tile));
    coord<k_tile> k_tile_idx = {blockIdx.z, kv_head_idx, kv_block_index, 0};
    tma::load_async(k_smem[0], g.k, k_tile_idx, k_smem_arrived);

    tma::expect_bytes(v_smem_arrived, sizeof(v_tile));
    coord<v_tile> v_tile_idx = {blockIdx.z, kv_head_idx, kv_block_index, 0};
    tma::load_async(v_smem[0], g.v, v_tile_idx, v_smem_arrived);
  }
  __syncthreads();

  rt_fl<16, 64> att_block;
  rt_bf<16, 64> att_block_mma;
  rt_fl<16, K::tile_width> o_reg;

  col_vec<rt_fl<16, 64>> max_vec, norm_vec, max_vec_last_scaled, max_vec_scaled;

  neg_infty(max_vec);
  zero(norm_vec);
  zero(o_reg);

  // wait for q block
  wait(qsmem_semaphore, 0);

  for (int kv_idx = 0; kv_idx < kv_blocks - 1; kv_idx++) {
    // preload kv index
    int32_t kv_block_index = q2k_block_sparse_index_ptr[kv_idx + 1];

    // wait k
    wait(k_smem_arrived, kv_idx % 2);

    // compute QK^T
    warpgroup::mm_ABt(att_block, q_smem[0], k_smem[0]);

    copy(max_vec_last_scaled, max_vec);
    if constexpr (D == 64) {
      mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f * 0.125f);
    } else {
      mul(max_vec_last_scaled, max_vec_last_scaled,
          1.44269504089f * 0.08838834764f);
    }

    warpgroup::mma_async_wait();

    // load K
    if (threadIdx.x == 0) {
      tma::expect_bytes(k_smem_arrived, sizeof(k_tile));
      coord<k_tile> k_tile_idx = {blockIdx.z, kv_head_idx, kv_block_index, 0};
      tma::load_async(k_smem[0], g.k, k_tile_idx, k_smem_arrived);
    }

    // exp
    right_fill(att_block, att_block,
               g.block_size[q2k_block_sparse_index_ptr[kv_idx]],
               base_types::constants<float>::neg_infty());
    row_max(max_vec, att_block, max_vec);

    if constexpr (D == 64) {
      mul(att_block, att_block, 1.44269504089f * 0.125f);
      mul(max_vec_scaled, max_vec, 1.44269504089f * 0.125f);
    } else {
      mul(att_block, att_block, 1.44269504089f * 0.08838834764f);
      mul(max_vec_scaled, max_vec, 1.44269504089f * 0.08838834764f);
    }

    sub_row(att_block, att_block, max_vec_scaled);
    exp2(att_block, att_block);
    sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
    exp2(max_vec_last_scaled, max_vec_last_scaled);
    mul(norm_vec, norm_vec, max_vec_last_scaled);
    row_sum(norm_vec, att_block, norm_vec);
    add(att_block, att_block, 0.f);
    copy(att_block_mma, att_block);
    mul_row(o_reg, o_reg, max_vec_last_scaled);

    // wait v
    wait(v_smem_arrived, kv_idx % 2);

    // compute SV
    warpgroup::mma_AB(o_reg, att_block_mma, v_smem[0]);
    warpgroup::mma_async_wait();

    // load V
    if (threadIdx.x == 0) {
      tma::expect_bytes(v_smem_arrived, sizeof(v_tile));
      coord<v_tile> v_tile_idx = {blockIdx.z, kv_head_idx, kv_block_index, 0};
      tma::load_async(v_smem[0], g.v, v_tile_idx, v_smem_arrived);
    }
  }

  // last iter
  {
    int kv_idx = kv_blocks - 1;
    // wait k
    wait(k_smem_arrived, kv_idx % 2);

    // compute QK^T
    warpgroup::mm_ABt(att_block, q_smem[0], k_smem[0]);

    copy(max_vec_last_scaled, max_vec);
    if constexpr (D == 64) {
      mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f * 0.125f);
    } else {
      mul(max_vec_last_scaled, max_vec_last_scaled,
          1.44269504089f * 0.08838834764f);
    }

    warpgroup::mma_async_wait();

    // exp
    right_fill(att_block, att_block,
               g.block_size[q2k_block_sparse_index_ptr[kv_idx]],
               base_types::constants<float>::neg_infty());

    row_max(max_vec, att_block, max_vec);

    if constexpr (D == 64) {
      mul(att_block, att_block, 1.44269504089f * 0.125f);
      mul(max_vec_scaled, max_vec, 1.44269504089f * 0.125f);
    } else {
      mul(att_block, att_block, 1.44269504089f * 0.08838834764f);
      mul(max_vec_scaled, max_vec, 1.44269504089f * 0.08838834764f);
    }

    sub_row(att_block, att_block, max_vec_scaled);
    exp2(att_block, att_block);
    sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
    exp2(max_vec_last_scaled, max_vec_last_scaled);
    mul(norm_vec, norm_vec, max_vec_last_scaled);
    row_sum(norm_vec, att_block, norm_vec);
    add(att_block, att_block, 0.f);
    copy(att_block_mma, att_block);
    mul_row(o_reg, o_reg, max_vec_last_scaled);

    // wait v
    wait(v_smem_arrived, kv_idx % 2);

    // compute SV
    warpgroup::mma_AB(o_reg, att_block_mma, v_smem[0]);
    warpgroup::mma_async_wait();
  }

  div_row(o_reg, o_reg, norm_vec);
  warpgroup::store(o_smem[0], o_reg);
  __syncthreads();

  // TK store_async internally calls syncwarp so we need to route on warp level
  if (threadIdx.x / 32 == 0) {
    coord<o_tile> o_tile_idx = {blockIdx.z, blockIdx.y, seq_idx, 0};
    tma::store_async(g.o, o_smem[0], o_tile_idx);
  }

  mul(max_vec_scaled, max_vec_scaled, 0.69314718056f);
  log(norm_vec, norm_vec);
  add(norm_vec, norm_vec, max_vec_scaled);

  if constexpr (D == 64) {
    mul(norm_vec, norm_vec, -8.0f);
  } else {
    mul(norm_vec, norm_vec, -11.313708499f);
  }

  warpgroup::store(l_smem[0], norm_vec);
  __syncthreads();

  if (threadIdx.x / 32 == 0) {
    coord<l_col_vec> tile_idx = {blockIdx.z, blockIdx.y, 0, seq_idx};
    tma::store_async(g.l, l_smem[0], tile_idx);
  }
  tma::store_async_wait();
}

// ---------------------------------------------------------------------------------------------------
// ----------------------------------- Backward preparation kernel
// -----------------------------------
// ---------------------------------------------------------------------------------------------------

template <int D> struct bwd_prep_globals {
  using og_tile = st_bf<4 * 16, D>;
  using o_tile = st_bf<4 * 16, D>;
  using d_tile = col_vec<st_fl<4 * 16, D>>;

  using og_gl = gl<bf16, -1, -1, -1, -1, og_tile>;
  using o_gl = gl<bf16, -1, -1, -1, -1, o_tile>;
  using d_gl = gl<float, -1, -1, -1, -1, d_tile>;

  og_gl og;
  o_gl o;
  d_gl d;
};

constexpr int PREP_NUM_WARPS = (1);
template <int D>
__global__ __launch_bounds__(
    PREP_NUM_WARPS *kittens::WARP_THREADS,
    (D == 64)
        ? 6 / PREP_NUM_WARPS
        : 3 / PREP_NUM_WARPS) void bwd_attend_prep_ker(const __grid_constant__
                                                           bwd_prep_globals<D>
                                                               g) {
  extern __shared__ int __shm[];
  tma_swizzle_allocator al((int *)&__shm[0]);

  int warpid = kittens::warpid();

  using og_tile = st_bf<4 * 16, D>;
  using o_tile = st_bf<4 * 16, D>;
  using d_tile = col_vec<st_fl<4 * 16, D>>;

  og_tile(&og_smem)[PREP_NUM_WARPS] = al.allocate<og_tile, PREP_NUM_WARPS>();
  o_tile(&o_smem)[PREP_NUM_WARPS] = al.allocate<o_tile, PREP_NUM_WARPS>();
  d_tile(&d_smem)[PREP_NUM_WARPS] = al.allocate<d_tile, PREP_NUM_WARPS>();

  rt_fl<4 * 16, D> og_reg, o_reg;
  col_vec<rt_fl<4 * 16, D>> d_reg;

  __shared__ kittens::semaphore smem_semaphore;

  if (threadIdx.x == 0) {
    init_semaphore(smem_semaphore, 0, 1);
    tma::expect_bytes(smem_semaphore, sizeof(og_smem[0]) * PREP_NUM_WARPS * 2);
  }
  __syncthreads();

  if (warpid == 0) {
    for (int w = 0; w < PREP_NUM_WARPS; w++) {
      coord<o_tile> tile_idx = {blockIdx.z, blockIdx.y,
                                (blockIdx.x * PREP_NUM_WARPS) + w, 0};
      tma::load_async(o_smem[w], g.o, tile_idx, smem_semaphore);
      tma::load_async(og_smem[w], g.og, tile_idx, smem_semaphore);
    }
  }

  wait(smem_semaphore, 0);
  load(o_reg, o_smem[warpid]);
  load(og_reg, og_smem[warpid]);
  mul(og_reg, og_reg, o_reg);
  row_sum(d_reg, og_reg);
  store(d_smem[warpid], d_reg);
  __syncthreads();

  if (warpid == 0) {
    for (int w = 0; w < PREP_NUM_WARPS; w++) {
      coord<d_tile> tile_idx = {blockIdx.z, blockIdx.y, 0,
                                (blockIdx.x * PREP_NUM_WARPS) + w};
      tma::store_async(g.d, d_smem[w], tile_idx);
    }
  }
  tma::store_async_wait();
}

template <int D> struct bwd_attend_ker_tile_dims {};
template <> struct bwd_attend_ker_tile_dims<64> {
  constexpr static int tile_width = (64);
  constexpr static int tile_h = (4 * 16);
  constexpr static int tile_h_qo = (4 * 16);
};
template <> struct bwd_attend_ker_tile_dims<128> {
  constexpr static int tile_width = (128);
  constexpr static int tile_h = (4 * 16);
  constexpr static int tile_h_qo = (4 * 16);
};

template <int D> struct bwd_globals {
  using G = bwd_attend_ker_tile_dims<D>;

  using q_tile = st_bf<G::tile_h_qo, G::tile_width>;
  using k_tile = st_bf<G::tile_h, G::tile_width>;
  using v_tile = st_bf<G::tile_h, G::tile_width>;
  using og_tile = st_bf<G::tile_h_qo, G::tile_width>;
  using qg_tile = st_fl<G::tile_h_qo, G::tile_width>;
  using kg_tile = st_fl<G::tile_h, G::tile_width>;
  using vg_tile = st_fl<G::tile_h, G::tile_width>;
  using l_tile = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
  using d_tile = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;

  using q_gl = gl<bf16, -1, -1, -1, -1, q_tile>;
  using k_gl = gl<bf16, -1, -1, -1, -1, k_tile>;
  using v_gl = gl<bf16, -1, -1, -1, -1, v_tile>;

  using og_gl = gl<bf16, -1, -1, -1, -1, og_tile>;

  using qg_gl = gl<float, -1, -1, -1, -1, qg_tile>;
  using kg_gl = gl<float, -1, -1, -1, -1, kg_tile>;
  using vg_gl = gl<float, -1, -1, -1, -1, vg_tile>;

  using l_gl = gl<float, -1, -1, -1, -1, l_tile>;
  using d_gl = gl<float, -1, -1, -1, -1, d_tile>;

  q_gl q;
  k_gl k;
  v_gl v;
  og_gl og;
  qg_gl qg;
  kg_gl kg;
  vg_gl vg;
  l_gl l;
  d_gl d;

  const int N;
  const int hr;
  const int max_q_blocks_per_kv;

  int32_t *__restrict__ k2q_block_sparse_index;
  int32_t *__restrict__ k2q_block_sparse_num;
  int32_t *__restrict__ block_size;
};

__device__ static inline void stream_tile(auto &reg_tile, auto &smem_vec,
                                          int tic) {
#pragma unroll
  for (int i = 0; i < 4; i++) {
    int base_col = 16 * i + 2 * (kittens::laneid() % 4);
    reg_tile.tiles[0][i].data[0] = *(float2 *)&smem_vec[tic][base_col + 0];
    reg_tile.tiles[0][i].data[1] = *(float2 *)&smem_vec[tic][base_col + 0];
    reg_tile.tiles[0][i].data[2] = *(float2 *)&smem_vec[tic][base_col + 8];
    reg_tile.tiles[0][i].data[3] = *(float2 *)&smem_vec[tic][base_col + 8];
  }
}

__device__ static inline void stream_sub_tile(auto &reg_tile, auto &smem_vec,
                                              int tic) {
#pragma unroll
  for (int i = 0; i < 4; i++) {
    int base_col = 16 * i + 2 * (laneid() % 4);
    reg_tile.tiles[0][i].data[0] = base_ops::sub::template op<float2>(
        reg_tile.tiles[0][i].data[0], *(float2 *)&smem_vec[tic][base_col + 0]);
    reg_tile.tiles[0][i].data[1] = base_ops::sub::template op<float2>(
        reg_tile.tiles[0][i].data[1], *(float2 *)&smem_vec[tic][base_col + 0]);
    reg_tile.tiles[0][i].data[2] = base_ops::sub::template op<float2>(
        reg_tile.tiles[0][i].data[2], *(float2 *)&smem_vec[tic][base_col + 8]);
    reg_tile.tiles[0][i].data[3] = base_ops::sub::template op<float2>(
        reg_tile.tiles[0][i].data[3], *(float2 *)&smem_vec[tic][base_col + 8]);
  }
}

template <int D>
__global__ __launch_bounds__(128, (D == 64) ? 3 : 2) void bwd_attend_ker(
    const __grid_constant__ bwd_globals<D> g) {
  extern __shared__ int __shm[];
  tma_swizzle_allocator al((int *)&__shm[0]);

  const int N = g.N, hr = g.hr;
  using G = bwd_attend_ker_tile_dims<D>;

  using kg_tile = st_fl<G::tile_h, G::tile_width>;
  using vg_tile = st_fl<G::tile_h, G::tile_width>;
  using k_tile = st_bf<G::tile_h, G::tile_width>;
  using v_tile = st_bf<G::tile_h, G::tile_width>;
  using q_tile = st_bf<G::tile_h_qo, G::tile_width>;
  using og_tile = st_bf<G::tile_h_qo, G::tile_width>;
  using qg_tile = st_fl<G::tile_h_qo, G::tile_width>;
  using l_tile = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
  using d_tile = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
  using attn_tile = st_bf<G::tile_h_qo, G::tile_h>;

  k_tile(&k_smem)[1] = al.allocate<k_tile, 1>();
  v_tile(&v_smem)[1] = al.allocate<v_tile, 1>();

  q_tile(&q_smem)[1] = al.allocate<q_tile, 1>();
  og_tile(&og_smem)[1] = al.allocate<og_tile, 1>();
  qg_tile(&qg_smem) = al.allocate<qg_tile>();

  l_tile(&l_smem)[1] = al.allocate<l_tile, 1>();
  d_tile(&d_smem)[1] = al.allocate<d_tile, 1>();
  kg_tile(*kg_smem) = reinterpret_cast<kg_tile *>(&k_smem[0].data[0]);
  vg_tile(*vg_smem) = reinterpret_cast<vg_tile *>(&q_smem[0].data[0]);

  attn_tile(&ds_smem_t)[1] = al.allocate<attn_tile, 1>();

  const int warpid = kittens::warpid();
  const int warpgroupid = warpid / kittens::WARPGROUP_WARPS;
  const int kv_head_idx = (blockIdx.y) / hr;

  int32_t *__restrict__ k2q_block_sparse_index_ptr =
      g.k2q_block_sparse_index +
      blockIdx.z * gridDim.y * gridDim.x * g.max_q_blocks_per_kv +
      blockIdx.y * gridDim.x * g.max_q_blocks_per_kv +
      blockIdx.x * g.max_q_blocks_per_kv;
  int32_t *__restrict__ k2q_block_sparse_num_ptr =
      g.k2q_block_sparse_num + blockIdx.z * gridDim.y * gridDim.x +
      blockIdx.y * gridDim.x + blockIdx.x;
  const int qo_blocks = *k2q_block_sparse_num_ptr;

  if (qo_blocks <= 0) {
    return;
  }

  __shared__ kittens::semaphore kv_b, q_b[1], o_b[1], vec_b[1];

  int32_t store_qg_block_index;
  int32_t load_q_block_index;

  if (threadIdx.x == 0) {
    load_q_block_index = k2q_block_sparse_index_ptr[0];

    init_semaphore(kv_b, 0, 1);

    init_semaphore(q_b[0], 0, 1);
    init_semaphore(o_b[0], 0, 1);
    init_semaphore(vec_b[0], 0, 1);

    // preload KV
    tma::expect_bytes(kv_b, sizeof(k_smem[0]) + sizeof(v_smem[0]));
    coord<k_tile> tile_idx_kv = {blockIdx.z, kv_head_idx, blockIdx.x, 0};
    tma::load_async(k_smem[0], g.k, tile_idx_kv, kv_b);
    tma::load_async(v_smem[0], g.v, tile_idx_kv, kv_b);

    // preload og, vec and q
    coord<q_tile> tile_idx_qo = {blockIdx.z, blockIdx.y, load_q_block_index, 0};
    coord<l_tile> vec_idx = {blockIdx.z, blockIdx.y, 0, load_q_block_index};

    tma::expect_bytes(o_b[0], sizeof(og_smem[0]));
    tma::load_async(og_smem[0], g.og, tile_idx_qo, o_b[0]);

    tma::expect_bytes(vec_b[0], sizeof(l_smem[0]) + sizeof(d_smem[0]));
    tma::load_async(l_smem[0], g.l, vec_idx, vec_b[0]);
    tma::load_async(d_smem[0], g.d, vec_idx, vec_b[0]);

    tma::expect_bytes(q_b[0], sizeof(q_smem[0]));
    tma::load_async(q_smem[0], g.q, tile_idx_qo, q_b[0]);
  }
  __syncthreads();

  rt_fl<16, G::tile_width> kg_reg, vg_reg;

  row_vec<rt_fl<16, 64>> row_reg;

  rt_fl<16, 64> s_block_t, p_block_t;
  rt_fl<16, 64> ds_block_t, dp_block_t;
  rt_bf<16, 64> ds_block_t_mma, p_block_t_mma;

  zero(kg_reg);
  zero(vg_reg);

  // wait for kv
  wait(kv_b, 0);
  int fill_start = g.block_size[blockIdx.x] - 16 * kittens::warpid();
  for (int qo_idx = 0; qo_idx < qo_blocks - 1; qo_idx++) {
    // preload q index
    store_qg_block_index = load_q_block_index;
    load_q_block_index = k2q_block_sparse_index_ptr[qo_idx + 1];

    wait(o_b[0], qo_idx % 2);
    warpgroup::mm_ABt(dp_block_t, v_smem[0], og_smem[0]); // dP^T = VdO^T
    warpgroup::mma_commit_group();                        // ! do not wait

    wait(vec_b[0], qo_idx % 2);
    stream_tile(s_block_t, l_smem, 0);
    wait(q_b[0], qo_idx % 2);
    warpgroup::mma_ABt(s_block_t, k_smem[0], q_smem[0]); // S^T = KQ^T - l
    warpgroup::mma_commit_group();
    warpgroup::mma_async_wait();

    if constexpr (D == 64) {
      mul(s_block_t, s_block_t, 1.44269504089f * 0.125f);
    } else {
      mul(s_block_t, s_block_t, 1.44269504089f * 0.08838834764f);
    }

    lower_fill(s_block_t, s_block_t, fill_start,
               base_types::constants<float>::neg_infty());
    exp2(s_block_t, s_block_t); // P_i
    copy(p_block_t, s_block_t);
    copy(p_block_t_mma, s_block_t);
    stream_sub_tile(dp_block_t, d_smem, 0); // dP - D
    mul(ds_block_t, p_block_t, dp_block_t); // dS = P \odot (dP - D)

    if constexpr (D == 64) {
      mul(ds_block_t, ds_block_t, 0.125f);
    } else {
      mul(ds_block_t, ds_block_t, 0.08838834764f);
    }

    // load vec
    if (threadIdx.x == 0) {
      coord<l_tile> vec_idx = {blockIdx.z, blockIdx.y, 0, load_q_block_index};
      tma::expect_bytes(vec_b[0], sizeof(l_smem[0]) + sizeof(d_smem[0]));
      tma::load_async(l_smem[0], g.l, vec_idx, vec_b[0]);
      tma::load_async(d_smem[0], g.d, vec_idx, vec_b[0]);
    }

    warpgroup::mma_AB(vg_reg, p_block_t_mma, og_smem[0]); // dV += P^TdO
    warpgroup::mma_commit_group();
    copy(ds_block_t_mma, ds_block_t);
    warpgroup::store(ds_smem_t[0], ds_block_t);
    warpgroup::mma_async_wait();

    // load og
    if (threadIdx.x == 0) {
      coord<q_tile> tile_idx = {blockIdx.z, blockIdx.y, load_q_block_index, 0};
      tma::expect_bytes(o_b[0], sizeof(og_smem[0]));
      tma::load_async(og_smem[0], g.og, tile_idx, o_b[0]);
    }

    warpgroup::mma_AB(kg_reg, ds_block_t_mma, q_smem[0]); // dK += dS^TQ
    warpgroup::mma_commit_group();
    warpgroup::mma_async_wait();

    // load q
    if (threadIdx.x == 0) {
      coord<q_tile> q_tile_idx = {blockIdx.z, blockIdx.y, load_q_block_index,
                                  0};
      tma::expect_bytes(q_b[0], sizeof(q_smem[0]));
      tma::load_async(q_smem[0], g.q, q_tile_idx, q_b[0]);
    }

    rt_fl<16, G::tile_width> qg_reg;
    __syncthreads(); // wait for sd_smem shared memory write
    warpgroup::mm_AtB(qg_reg, ds_smem_t[0], k_smem[0]); // delat dQ = dSK
    warpgroup::mma_commit_group();
    warpgroup::mma_async_wait();
    // store qg to shared memory
    warpgroup::store(qg_smem, qg_reg);
    __syncthreads();

    // store and add dQ to global memory
    if (threadIdx.x / 32 == 0) {
      coord<qg_tile> tile_idx = {blockIdx.z, blockIdx.y, store_qg_block_index,
                                 0};
      tma::store_add_async(g.qg, qg_smem, tile_idx);
      tma::store_async_wait();
    }
  }

  // last iter
  {
    int qo_idx = qo_blocks - 1;

    store_qg_block_index = load_q_block_index;

    wait(o_b[0], qo_idx % 2);
    warpgroup::mm_ABt(dp_block_t, v_smem[0], og_smem[0]); // dP = dOV^T
    warpgroup::mma_commit_group();                        // ! do not wait

    wait(vec_b[0], qo_idx % 2);
    stream_tile(s_block_t, l_smem, 0);
    wait(q_b[0], qo_idx % 2);
    warpgroup::mma_ABt(s_block_t, k_smem[0], q_smem[0]); // S = QK^T - l
    warpgroup::mma_commit_group();
    warpgroup::mma_async_wait();

    if constexpr (D == 64) {
      mul(s_block_t, s_block_t, 1.44269504089f * 0.125f);
    } else {
      mul(s_block_t, s_block_t, 1.44269504089f * 0.08838834764f);
    }
    lower_fill(s_block_t, s_block_t, fill_start,
               base_types::constants<float>::neg_infty());
    exp2(s_block_t, s_block_t); // P_i
    copy(p_block_t, s_block_t);
    copy(p_block_t_mma, s_block_t);
    stream_sub_tile(dp_block_t, d_smem, 0); // dP - D
    mul(ds_block_t, p_block_t, dp_block_t); // dS = P \odot (dP - D)

    if constexpr (D == 64) {
      mul(ds_block_t, ds_block_t, 0.125f);
    } else {
      mul(ds_block_t, ds_block_t, 0.08838834764f);
    }

    warpgroup::mma_AB(vg_reg, p_block_t_mma, og_smem[0]); // dV += P^TdO
    warpgroup::mma_commit_group();
    copy(ds_block_t_mma, ds_block_t);
    warpgroup::store(ds_smem_t[0], ds_block_t);
    warpgroup::mma_async_wait();

    warpgroup::mma_AB(kg_reg, ds_block_t_mma, q_smem[0]); // dK += dS^TQ
    warpgroup::mma_commit_group();
    warpgroup::mma_async_wait();

    rt_fl<16, G::tile_width> qg_reg;
    __syncthreads(); // wait for sd_smem shared memory write
    warpgroup::mm_AtB(qg_reg, ds_smem_t[0], k_smem[0]); // delat dQ = dSK
    warpgroup::mma_commit_group();
    warpgroup::mma_async_wait();
    // store qg to shared memory
    warpgroup::store(qg_smem, qg_reg);
    __syncthreads();

    // store and add dQ to global memory
    if (threadIdx.x / 32 == 0) {
      coord<qg_tile> tile_idx = {blockIdx.z, blockIdx.y, store_qg_block_index,
                                 0};
      tma::store_add_async(g.qg, qg_smem, tile_idx);
      tma::store_async_wait();
    }
  }

  // store kq and vq

  // ! the following two line seems unnecessary.
  // tma::store_async_wait(); // ensure qg is finished
  __syncthreads();

  warpgroup::store(kg_smem[0], kg_reg);
  __syncthreads();
  if (threadIdx.x / 32 == 0) {
    coord<kg_tile> tile_idx = {blockIdx.z, kv_head_idx, blockIdx.x, 0};
    tma::store_add_async(g.kg, kg_smem[0], tile_idx);
    tma::store_commit_group();
  }

  warpgroup::store(vg_smem[0], vg_reg);
  __syncthreads();
  if (kittens::warpid() % 4 == 0) {
    coord<vg_tile> tile_idx = {blockIdx.z, kv_head_idx, blockIdx.x, 0};
    tma::store_add_async(g.vg, vg_smem[0], tile_idx);
    tma::store_commit_group();
  }
  tma::store_async_wait();
}

#include "pyutils/torch_helpers.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <iostream>

std::vector<torch::Tensor> block_sparse_attention_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor q2k_block_sparse_index, torch::Tensor q2k_block_sparse_num,
    torch::Tensor block_size) {
  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);

  auto batch = q.size(0);
  auto seq_len = q.size(2);
  auto head_dim = q.size(3);
  auto qo_heads = q.size(1);
  auto kv_heads = k.size(1);
  auto max_kv_blocks_per_q = q2k_block_sparse_index.size(3);
  auto num_q_blocks = block_size.size(0);
  TORCH_CHECK(
      batch == 1,
      "Batch size dim will be removed in the future, please set batch to 1");
  TORCH_CHECK(num_q_blocks * 64 == seq_len,
              "This kernel supports variable block size, but it assumes the "
              "input sequence is properly padded.");
  TORCH_CHECK(num_q_blocks == q2k_block_sparse_index.size(2),
              "Number of Q blocks does not match between "
              "q2k_block_sparse_index and block_size");
  // check to see that these dimensions match for all inputs
  TORCH_CHECK(q.size(0) == batch,
              "Q batch dimension - idx 0 - must match for all inputs");
  TORCH_CHECK(k.size(0) == batch,
              "K batch dimension - idx 0 - must match for all inputs");
  TORCH_CHECK(v.size(0) == batch,
              "V batch dimension - idx 0 - must match for all inputs");
  TORCH_CHECK(q2k_block_sparse_index.size(0) == batch,
              "q2k_block_sparse_index batch dimension - idx 0 - must match for "
              "all inputs");
  TORCH_CHECK(q2k_block_sparse_num.size(0) == batch,
              "q2k_block_sparse_num batch dimension - idx 0 - must match for "
              "all inputs");

  TORCH_CHECK(
      q.size(2) == seq_len,
      "Q sequence length dimension - idx 2 - must match for all inputs");
  TORCH_CHECK(
      k.size(2) == seq_len,
      "K sequence length dimension - idx 2 - must match for all inputs");
  TORCH_CHECK(
      v.size(2) == seq_len,
      "V sequence length dimension - idx 2 - must match for all inputs");
  TORCH_CHECK(q2k_block_sparse_index.size(2) == seq_len / BLOCK_M,
              "q2k_block_sparse_index idx 2 - must match seq_len / BLOCK_M");
  TORCH_CHECK(q2k_block_sparse_num.size(2) == seq_len / BLOCK_M,
              "q2k_block_sparse_num idx 2 - must match seq_len / BLOCK_M");

  TORCH_CHECK(
      q.size(3) == head_dim,
      "Q head dimension - idx 3 - must match for all non-vector inputs");
  TORCH_CHECK(
      k.size(3) == head_dim,
      "K head dimension - idx 3 - must match for all non-vector inputs");
  TORCH_CHECK(
      v.size(3) == head_dim,
      "V head dimension - idx 3 - must match for all non-vector inputs");

  TORCH_CHECK(qo_heads >= kv_heads,
              "QO heads must be greater than or equal to KV heads");
  TORCH_CHECK(qo_heads % kv_heads == 0,
              "QO heads must be divisible by KV heads");
  TORCH_CHECK(q.size(1) == qo_heads,
              "QO head dimension - idx 1 - must match for all inputs");
  TORCH_CHECK(k.size(1) == kv_heads,
              "KV head dimension - idx 1 - must match for all inputs");
  TORCH_CHECK(v.size(1) == kv_heads,
              "KV head dimension - idx 1 - must match for all inputs");
  TORCH_CHECK(q2k_block_sparse_index.size(1) == qo_heads,
              "q2k_block_sparse_index head dimension - idx 1 - must match for "
              "all inputs");
  TORCH_CHECK(q2k_block_sparse_num.size(1) == qo_heads,
              "q2k_block_sparse_num head dimension - idx 1 - must match for "
              "all inputs");
  auto hr = qo_heads / kv_heads;

  c10::BFloat16 *q_ptr = q.data_ptr<c10::BFloat16>();
  c10::BFloat16 *k_ptr = k.data_ptr<c10::BFloat16>();
  c10::BFloat16 *v_ptr = v.data_ptr<c10::BFloat16>();

  bf16 *d_q = reinterpret_cast<bf16 *>(q_ptr);
  bf16 *d_k = reinterpret_cast<bf16 *>(k_ptr);
  bf16 *d_v = reinterpret_cast<bf16 *>(v_ptr);

  // for the returned outputs
  torch::Tensor o = torch::empty(
      {static_cast<const uint>(batch), static_cast<const uint>(qo_heads),
       static_cast<const uint>(seq_len), static_cast<const uint>(head_dim)},
      v.options());

  torch::Tensor l_vec = torch::empty(
      {static_cast<const uint>(batch), static_cast<const uint>(qo_heads),
       static_cast<const uint>(seq_len), static_cast<const uint>(1)},
      torch::TensorOptions()
          .dtype(torch::kFloat)
          .device(q.device())
          .memory_format(at::MemoryFormat::Contiguous));

  bf16 *o_ptr = reinterpret_cast<bf16 *>(o.data_ptr<c10::BFloat16>());
  bf16 *d_o = reinterpret_cast<bf16 *>(o_ptr);

  float *l_ptr = reinterpret_cast<float *>(l_vec.data_ptr<float>());
  float *d_l = reinterpret_cast<float *>(l_ptr);

  // cudadevicesynchronize();
  const c10::cuda::OptionalCUDAGuard device_guard(q.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  if (head_dim == 64) {
    using q_tile = st_bf<fwd_attend_ker_tile_dims<64>::qo_height,
                         fwd_attend_ker_tile_dims<64>::tile_width>;
    using k_tile = st_bf<fwd_attend_ker_tile_dims<64>::kv_height,
                         fwd_attend_ker_tile_dims<64>::tile_width>;
    using v_tile = st_bf<fwd_attend_ker_tile_dims<64>::kv_height,
                         fwd_attend_ker_tile_dims<64>::tile_width>;
    using l_col_vec = col_vec<st_fl<fwd_attend_ker_tile_dims<64>::qo_height,
                                    fwd_attend_ker_tile_dims<64>::tile_width>>;
    using o_tile = st_bf<fwd_attend_ker_tile_dims<64>::qo_height,
                         fwd_attend_ker_tile_dims<64>::tile_width>;

    using q_global = gl<bf16, -1, -1, -1, -1, q_tile>;
    using k_global = gl<bf16, -1, -1, -1, -1, k_tile>;
    using v_global = gl<bf16, -1, -1, -1, -1, v_tile>;
    using l_global = gl<float, -1, -1, -1, -1, l_col_vec>;
    using o_global = gl<bf16, -1, -1, -1, -1, o_tile>;

    using globals = fwd_globals<64>;

    q_global qg_arg{d_q, static_cast<unsigned int>(batch),
                    static_cast<unsigned int>(qo_heads),
                    static_cast<unsigned int>(seq_len), 64U};
    k_global kg_arg{d_k, static_cast<unsigned int>(batch),
                    static_cast<unsigned int>(kv_heads),
                    static_cast<unsigned int>(seq_len), 64U};
    v_global vg_arg{d_v, static_cast<unsigned int>(batch),
                    static_cast<unsigned int>(kv_heads),
                    static_cast<unsigned int>(seq_len), 64U};
    l_global lg_arg{d_l, static_cast<unsigned int>(batch),
                    static_cast<unsigned int>(qo_heads), 1U,
                    static_cast<unsigned int>(seq_len)};
    o_global og_arg{d_o, static_cast<unsigned int>(batch),
                    static_cast<unsigned int>(qo_heads),
                    static_cast<unsigned int>(seq_len), 64U};

    globals g{qg_arg,
              kg_arg,
              vg_arg,
              lg_arg,
              og_arg,
              static_cast<int>(seq_len),
              static_cast<int>(hr),
              static_cast<int>(max_kv_blocks_per_q),
              reinterpret_cast<int32_t *>(q2k_block_sparse_index.data_ptr()),
              reinterpret_cast<int32_t *>(q2k_block_sparse_num.data_ptr()),
              reinterpret_cast<int32_t *>(block_size.data_ptr())};

    constexpr int mem_size = 54000;

    dim3 grid(seq_len / (64), qo_heads, batch);

    cudaFuncSetAttribute(fwd_attend_ker<64>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    fwd_attend_ker<64><<<grid, (128), mem_size, stream>>>(g);

    CHECK_CUDA_ERROR(cudaGetLastError());
    // cudaStreamSynchronize(stream);
  }

  if (head_dim == 128) {
    using q_tile = st_bf<fwd_attend_ker_tile_dims<128>::qo_height,
                         fwd_attend_ker_tile_dims<128>::tile_width>;
    using k_tile = st_bf<fwd_attend_ker_tile_dims<128>::kv_height,
                         fwd_attend_ker_tile_dims<128>::tile_width>;
    using v_tile = st_bf<fwd_attend_ker_tile_dims<128>::kv_height,
                         fwd_attend_ker_tile_dims<128>::tile_width>;
    using l_col_vec = col_vec<st_fl<fwd_attend_ker_tile_dims<128>::qo_height,
                                    fwd_attend_ker_tile_dims<128>::tile_width>>;
    using o_tile = st_bf<fwd_attend_ker_tile_dims<128>::qo_height,
                         fwd_attend_ker_tile_dims<128>::tile_width>;

    using q_global = gl<bf16, -1, -1, -1, -1, q_tile>;
    using k_global = gl<bf16, -1, -1, -1, -1, k_tile>;
    using v_global = gl<bf16, -1, -1, -1, -1, v_tile>;
    using l_global = gl<float, -1, -1, -1, -1, l_col_vec>;
    using o_global = gl<bf16, -1, -1, -1, -1, o_tile>;

    using globals = fwd_globals<128>;

    q_global qg_arg{d_q, static_cast<unsigned int>(batch),
                    static_cast<unsigned int>(qo_heads),
                    static_cast<unsigned int>(seq_len), 128U};
    k_global kg_arg{d_k, static_cast<unsigned int>(batch),
                    static_cast<unsigned int>(kv_heads),
                    static_cast<unsigned int>(seq_len), 128U};
    v_global vg_arg{d_v, static_cast<unsigned int>(batch),
                    static_cast<unsigned int>(kv_heads),
                    static_cast<unsigned int>(seq_len), 128U};
    l_global lg_arg{d_l, static_cast<unsigned int>(batch),
                    static_cast<unsigned int>(qo_heads), 1U,
                    static_cast<unsigned int>(seq_len)};
    o_global og_arg{d_o, static_cast<unsigned int>(batch),
                    static_cast<unsigned int>(qo_heads),
                    static_cast<unsigned int>(seq_len), 128U};

    globals g{qg_arg,
              kg_arg,
              vg_arg,
              lg_arg,
              og_arg,
              static_cast<int>(seq_len),
              static_cast<int>(hr),
              static_cast<int>(max_kv_blocks_per_q),
              reinterpret_cast<int32_t *>(q2k_block_sparse_index.data_ptr()),
              reinterpret_cast<int32_t *>(q2k_block_sparse_num.data_ptr()),
              reinterpret_cast<int32_t *>(block_size.data_ptr())};

    constexpr int mem_size = 54000;

    dim3 grid(seq_len / (64), qo_heads, batch);

    cudaFuncSetAttribute(fwd_attend_ker<128>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    fwd_attend_ker<128><<<grid, (128), mem_size, stream>>>(g);

    CHECK_CUDA_ERROR(cudaGetLastError());
    // cudaStreamSynchronize(stream);
  }

  return {o, l_vec};
  // cudadevicesynchronize();
}

std::vector<torch::Tensor> block_sparse_attention_backward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o,
    torch::Tensor l_vec, torch::Tensor og, torch::Tensor k2q_block_sparse_index,
    torch::Tensor k2q_block_sparse_num, torch::Tensor block_size) {
  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);
  CHECK_INPUT(l_vec);
  CHECK_INPUT(o);
  CHECK_INPUT(og);

  auto batch = q.size(0);
  auto seq_len = q.size(2);
  auto head_dim = q.size(3);
  auto max_q_blocks_per_kv = k2q_block_sparse_index.size(3);
  TORCH_CHECK(k2q_block_sparse_index.size(2) == block_size.size(0),
              "k2q_block_sparse_index.size(2) must match block_size.size(0)");
  // check to see that these dimensions match for all inputs
  TORCH_CHECK(q.size(0) == batch,
              "Q  batch dimension - idx 0 - must match for all inputs");
  TORCH_CHECK(k.size(0) == batch,
              "K  batch dimension - idx 0 - must match for all inputs");
  TORCH_CHECK(v.size(0) == batch,
              "V  batch dimension - idx 0 - must match for all inputs");
  TORCH_CHECK(l_vec.size(0) == batch,
              "L  batch dimension - idx 0 - must match for all inputs");
  TORCH_CHECK(o.size(0) == batch,
              "O  batch dimension - idx 0 - must match for all inputs");
  TORCH_CHECK(og.size(0) == batch,
              "OG batch dimension - idx 0 - must match for all inputs");
  TORCH_CHECK(k2q_block_sparse_index.size(0) == batch,
              "k2q_block_sparse_index batch dimension - idx 0 - must match for "
              "all inputs");
  TORCH_CHECK(k2q_block_sparse_num.size(0) == batch,
              "k2q_block_sparse_num batch dimension - idx 0 - must match for "
              "all inputs");

  TORCH_CHECK(
      q.size(2) == seq_len,
      "Q  sequence length dimension - idx 2 - must match for all inputs");
  TORCH_CHECK(
      k.size(2) == seq_len,
      "K  sequence length dimension - idx 2 - must match for all inputs");
  TORCH_CHECK(
      v.size(2) == seq_len,
      "V  sequence length dimension - idx 2 - must match for all inputs");
  TORCH_CHECK(
      l_vec.size(2) == seq_len,
      "L  sequence length dimension - idx 2 - must match for all inputs");
  TORCH_CHECK(
      o.size(2) == seq_len,
      "O  sequence length dimension - idx 2 - must match for all inputs");
  TORCH_CHECK(
      og.size(2) == seq_len,
      "OG sequence length dimension - idx 2 - must match for all inputs");
  TORCH_CHECK(k2q_block_sparse_index.size(2) == seq_len / BLOCK_N,
              "k2q_block_sparse_index idx 2 - must match seq_len / BLOCK_N");
  TORCH_CHECK(k2q_block_sparse_num.size(2) == seq_len / BLOCK_N,
              "k2q_block_sparse_num idx 2 - must match seq_len / BLOCK_N");

  TORCH_CHECK(
      q.size(3) == head_dim,
      "Q  head dimension - idx 3 - must match for all non-vector inputs");
  TORCH_CHECK(
      k.size(3) == head_dim,
      "K  head dimension - idx 3 - must match for all non-vector inputs");
  TORCH_CHECK(
      v.size(3) == head_dim,
      "V  head dimension - idx 3 - must match for all non-vector inputs");
  TORCH_CHECK(
      o.size(3) == head_dim,
      "O  head dimension - idx 3 - must match for all non-vector inputs");
  TORCH_CHECK(
      og.size(3) == head_dim,
      "OG head dimension - idx 3 - must match for all non-vector inputs");

  auto qo_heads = q.size(1);
  auto kv_heads = k.size(1);

  TORCH_CHECK(qo_heads >= kv_heads,
              "Q heads must be greater than or equal to K and V heads");
  TORCH_CHECK(qo_heads % kv_heads == 0,
              "Q heads must be divisible by KV heads");

  TORCH_CHECK(q.size(1) == qo_heads,
              "Q  heads dimension - idx 1 - must match for all inputs");
  TORCH_CHECK(l_vec.size(1) == qo_heads,
              "L  heads dimension - idx 1 - must match for all inputs");
  TORCH_CHECK(o.size(1) == qo_heads,
              "O  heads dimension - idx 1 - must match for all inputs");
  TORCH_CHECK(og.size(1) == qo_heads,
              "OG heads dimension - idx 1 - must match for all inputs");
  TORCH_CHECK(k.size(1) == kv_heads,
              "K  heads dimension - idx 1 - must match for all inputs");
  TORCH_CHECK(v.size(1) == kv_heads,
              "V  heads dimension - idx 1 - must match for all inputs");
  TORCH_CHECK(k2q_block_sparse_index.size(1) == kv_heads,
              "k2q_block_sparse_index heads dimension - idx 1 - must match for "
              "all inputs");
  TORCH_CHECK(k2q_block_sparse_num.size(1) == kv_heads,
              "k2q_block_sparse_num heads dimension - idx 1 - must match for "
              "all inputs");
  auto hr = qo_heads / kv_heads;

  c10::BFloat16 *q_ptr = q.data_ptr<c10::BFloat16>();
  c10::BFloat16 *k_ptr = k.data_ptr<c10::BFloat16>();
  c10::BFloat16 *v_ptr = v.data_ptr<c10::BFloat16>();
  c10::BFloat16 *o_ptr = o.data_ptr<c10::BFloat16>();
  c10::BFloat16 *og_ptr = og.data_ptr<c10::BFloat16>();
  float *l_ptr = l_vec.data_ptr<float>();

  torch::Tensor qg = torch::zeros(
      {static_cast<const uint>(batch), static_cast<const uint>(qo_heads),
       static_cast<const uint>(seq_len), static_cast<const uint>(head_dim)},
      l_vec.options());
  torch::Tensor kg = torch::zeros(
      {static_cast<const uint>(batch), static_cast<const uint>(kv_heads),
       static_cast<const uint>(seq_len), static_cast<const uint>(head_dim)},
      l_vec.options());
  torch::Tensor vg = torch::zeros(
      {static_cast<const uint>(batch), static_cast<const uint>(kv_heads),
       static_cast<const uint>(seq_len), static_cast<const uint>(head_dim)},
      l_vec.options());

  torch::Tensor d_vec = torch::empty(
      {static_cast<const uint>(batch), static_cast<const uint>(qo_heads),
       static_cast<const uint>(seq_len), static_cast<const uint>(1)},
      l_vec.options());

  float *qg_ptr = qg.data_ptr<float>();
  float *kg_ptr = kg.data_ptr<float>();
  float *vg_ptr = vg.data_ptr<float>();
  float *d_ptr = d_vec.data_ptr<float>();

  bf16 *d_q = reinterpret_cast<bf16 *>(q_ptr);
  bf16 *d_k = reinterpret_cast<bf16 *>(k_ptr);
  bf16 *d_v = reinterpret_cast<bf16 *>(v_ptr);
  bf16 *d_o = reinterpret_cast<bf16 *>(o_ptr);
  bf16 *d_og = reinterpret_cast<bf16 *>(og_ptr);
  float *d_l = reinterpret_cast<float *>(l_ptr);
  float *d_d = reinterpret_cast<float *>(d_ptr);
  float *d_qg = reinterpret_cast<float *>(qg_ptr);
  float *d_kg = reinterpret_cast<float *>(kg_ptr);
  float *d_vg = reinterpret_cast<float *>(vg_ptr);

  constexpr int mem_size = kittens::MAX_SHARED_MEMORY;
  int threads = PREP_NUM_WARPS * kittens::WARP_THREADS;

  // cudadevicesynchronize();
  const c10::cuda::OptionalCUDAGuard device_guard(q.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  //  cudaStreamSynchronize(stream);

  // TORCH_CHECK(seq_len % (4*kittens::TILE_DIM*4) == 0, "sequence length must
  // be divisible by 256");
  dim3 grid_bwd(seq_len / (PREP_NUM_WARPS * kittens::TILE_ROW_DIM<bf16> * 4),
                qo_heads, batch);

  if (head_dim == 64) {
    using og_tile = st_bf<4 * 16, 64>;
    using o_tile = st_bf<4 * 16, 64>;
    using d_tile = col_vec<st_fl<4 * 16, 64>>;

    using og_global = gl<bf16, -1, -1, -1, -1, og_tile>;
    using o_global = gl<bf16, -1, -1, -1, -1, o_tile>;
    using d_global = gl<float, -1, -1, -1, -1, d_tile>;

    using bwd_prep_globals = bwd_prep_globals<64>;

    og_global prep_og_arg{d_og, static_cast<unsigned int>(batch),
                          static_cast<unsigned int>(qo_heads),
                          static_cast<unsigned int>(seq_len), 64U};
    o_global prep_o_arg{d_o, static_cast<unsigned int>(batch),
                        static_cast<unsigned int>(qo_heads),
                        static_cast<unsigned int>(seq_len), 64U};
    d_global prep_d_arg{d_d, static_cast<unsigned int>(batch),
                        static_cast<unsigned int>(qo_heads), 1U,
                        static_cast<unsigned int>(seq_len)};

    bwd_prep_globals bwd_g{prep_og_arg, prep_o_arg, prep_d_arg};

    cudaFuncSetAttribute(bwd_attend_prep_ker<64>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    bwd_attend_prep_ker<64><<<grid_bwd, threads, mem_size, stream>>>(bwd_g);

    using bwd_q_tile = st_bf<bwd_attend_ker_tile_dims<64>::tile_h_qo,
                             bwd_attend_ker_tile_dims<64>::tile_width>;
    using bwd_k_tile = st_bf<bwd_attend_ker_tile_dims<64>::tile_h,
                             bwd_attend_ker_tile_dims<64>::tile_width>;
    using bwd_v_tile = st_bf<bwd_attend_ker_tile_dims<64>::tile_h,
                             bwd_attend_ker_tile_dims<64>::tile_width>;
    using bwd_og_tile = st_bf<bwd_attend_ker_tile_dims<64>::tile_h_qo,
                              bwd_attend_ker_tile_dims<64>::tile_width>;
    using bwd_qg_tile = st_fl<bwd_attend_ker_tile_dims<64>::tile_h_qo,
                              bwd_attend_ker_tile_dims<64>::tile_width>;
    using bwd_kg_tile = st_fl<bwd_attend_ker_tile_dims<64>::tile_h,
                              bwd_attend_ker_tile_dims<64>::tile_width>;
    using bwd_vg_tile = st_fl<bwd_attend_ker_tile_dims<64>::tile_h,
                              bwd_attend_ker_tile_dims<64>::tile_width>;
    using bwd_l_tile = row_vec<st_fl<bwd_attend_ker_tile_dims<64>::tile_h_qo,
                                     bwd_attend_ker_tile_dims<64>::tile_h>>;
    using bwd_d_tile = row_vec<st_fl<bwd_attend_ker_tile_dims<64>::tile_h_qo,
                                     bwd_attend_ker_tile_dims<64>::tile_h>>;

    using bwd_q_global = gl<bf16, -1, -1, -1, -1, bwd_q_tile>;
    using bwd_k_global = gl<bf16, -1, -1, -1, -1, bwd_k_tile>;
    using bwd_v_global = gl<bf16, -1, -1, -1, -1, bwd_v_tile>;

    using bwd_og_global = gl<bf16, -1, -1, -1, -1, bwd_og_tile>;

    using bwd_qg_global = gl<float, -1, -1, -1, -1, bwd_qg_tile>;
    using bwd_kg_global = gl<float, -1, -1, -1, -1, bwd_kg_tile>;
    using bwd_vg_global = gl<float, -1, -1, -1, -1, bwd_vg_tile>;

    using bwd_l_global = gl<float, -1, -1, -1, -1, bwd_l_tile>;
    using bwd_d_global = gl<float, -1, -1, -1, -1, bwd_d_tile>;

    using bwd_global_args = bwd_globals<64>;

    bwd_q_global bwd_q_arg{d_q, static_cast<unsigned int>(batch),
                           static_cast<unsigned int>(qo_heads),
                           static_cast<unsigned int>(seq_len), 64U};
    bwd_k_global bwd_k_arg{d_k, static_cast<unsigned int>(batch),
                           static_cast<unsigned int>(kv_heads),
                           static_cast<unsigned int>(seq_len), 64U};
    bwd_v_global bwd_v_arg{d_v, static_cast<unsigned int>(batch),
                           static_cast<unsigned int>(kv_heads),
                           static_cast<unsigned int>(seq_len), 64U};
    bwd_og_global bwd_og_arg{d_og, static_cast<unsigned int>(batch),
                             static_cast<unsigned int>(qo_heads),
                             static_cast<unsigned int>(seq_len), 64U};
    bwd_qg_global bwd_qg_arg{d_qg, static_cast<unsigned int>(batch),
                             static_cast<unsigned int>(qo_heads),
                             static_cast<unsigned int>(seq_len), 64U};
    bwd_kg_global bwd_kg_arg{d_kg, static_cast<unsigned int>(batch),
                             static_cast<unsigned int>(kv_heads),
                             static_cast<unsigned int>(seq_len), 64U};
    bwd_vg_global bwd_vg_arg{d_vg, static_cast<unsigned int>(batch),
                             static_cast<unsigned int>(kv_heads),
                             static_cast<unsigned int>(seq_len), 64U};
    bwd_l_global bwd_l_arg{d_l, static_cast<unsigned int>(batch),
                           static_cast<unsigned int>(qo_heads), 1U,
                           static_cast<unsigned int>(seq_len)};
    bwd_d_global bwd_d_arg{d_d, static_cast<unsigned int>(batch),
                           static_cast<unsigned int>(qo_heads), 1U,
                           static_cast<unsigned int>(seq_len)};

    bwd_global_args bwd_global{
        bwd_q_arg,
        bwd_k_arg,
        bwd_v_arg,
        bwd_og_arg,
        bwd_qg_arg,
        bwd_kg_arg,
        bwd_vg_arg,
        bwd_l_arg,
        bwd_d_arg,
        static_cast<int>(seq_len),
        static_cast<int>(hr),
        static_cast<int>(max_q_blocks_per_kv),
        reinterpret_cast<int32_t *>(k2q_block_sparse_index.data_ptr()),
        reinterpret_cast<int32_t *>(k2q_block_sparse_num.data_ptr()),
        reinterpret_cast<int32_t *>(block_size.data_ptr())};

    dim3 grid_bwd_2(seq_len / 64, qo_heads, batch);
    threads = 128;

    // cudadevicesynchronize();

    {
      cudaFuncSetAttribute(bwd_attend_ker<64>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, 72000);
      // cudaFuncSetAttribute(
      //     bwd_attend_ker<64>,
      //     cudaFuncAttributePreferredSharedMemoryCarveout,
      //     85
      // );

      bwd_attend_ker<64><<<grid_bwd_2, threads, 72000, stream>>>(bwd_global);
    }

    // CHECK_CUDA_ERROR(cudaGetLastError());
    // cudaStreamSynchronize(stream);
    // cudadevicesynchronize();
    // const auto kernel_end = std::chrono::high_resolution_clock::now();
    // std::cout << "Kernel Time: " <<
    // std::chrono::duration_cast<std::chrono::microseconds>(kernel_end -
    // start).count() << "us" << std::endl; std::cout << "---" << std::endl;
  }

  if (head_dim == 128) {
    using og_tile = st_bf<4 * 16, 128>;
    using o_tile = st_bf<4 * 16, 128>;
    using d_tile = col_vec<st_fl<4 * 16, 128>>;

    using og_global = gl<bf16, -1, -1, -1, -1, og_tile>;
    using o_global = gl<bf16, -1, -1, -1, -1, o_tile>;
    using d_global = gl<float, -1, -1, -1, -1, d_tile>;

    using bwd_prep_globals = bwd_prep_globals<128>;

    og_global prep_og_arg{d_og, static_cast<unsigned int>(batch),
                          static_cast<unsigned int>(qo_heads),
                          static_cast<unsigned int>(seq_len), 128U};
    o_global prep_o_arg{d_o, static_cast<unsigned int>(batch),
                        static_cast<unsigned int>(qo_heads),
                        static_cast<unsigned int>(seq_len), 128U};
    d_global prep_d_arg{d_d, static_cast<unsigned int>(batch),
                        static_cast<unsigned int>(qo_heads), 1U,
                        static_cast<unsigned int>(seq_len)};

    bwd_prep_globals bwd_g{prep_og_arg, prep_o_arg, prep_d_arg};

    cudaFuncSetAttribute(bwd_attend_prep_ker<128>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    bwd_attend_prep_ker<128><<<grid_bwd, threads, mem_size, stream>>>(bwd_g);

    using bwd_q_tile = st_bf<bwd_attend_ker_tile_dims<128>::tile_h_qo,
                             bwd_attend_ker_tile_dims<128>::tile_width>;
    using bwd_k_tile = st_bf<bwd_attend_ker_tile_dims<128>::tile_h,
                             bwd_attend_ker_tile_dims<128>::tile_width>;
    using bwd_v_tile = st_bf<bwd_attend_ker_tile_dims<128>::tile_h,
                             bwd_attend_ker_tile_dims<128>::tile_width>;
    using bwd_og_tile = st_bf<bwd_attend_ker_tile_dims<128>::tile_h_qo,
                              bwd_attend_ker_tile_dims<128>::tile_width>;
    using bwd_qg_tile = st_fl<bwd_attend_ker_tile_dims<128>::tile_h_qo,
                              bwd_attend_ker_tile_dims<128>::tile_width>;
    using bwd_kg_tile = st_fl<bwd_attend_ker_tile_dims<128>::tile_h,
                              bwd_attend_ker_tile_dims<128>::tile_width>;
    using bwd_vg_tile = st_fl<bwd_attend_ker_tile_dims<128>::tile_h,
                              bwd_attend_ker_tile_dims<128>::tile_width>;
    using bwd_l_tile = row_vec<st_fl<bwd_attend_ker_tile_dims<128>::tile_h_qo,
                                     bwd_attend_ker_tile_dims<128>::tile_h>>;
    using bwd_d_tile = row_vec<st_fl<bwd_attend_ker_tile_dims<128>::tile_h_qo,
                                     bwd_attend_ker_tile_dims<128>::tile_h>>;

    using bwd_q_global = gl<bf16, -1, -1, -1, -1, bwd_q_tile>;
    using bwd_k_global = gl<bf16, -1, -1, -1, -1, bwd_k_tile>;
    using bwd_v_global = gl<bf16, -1, -1, -1, -1, bwd_v_tile>;

    using bwd_og_global = gl<bf16, -1, -1, -1, -1, bwd_og_tile>;

    using bwd_qg_global = gl<float, -1, -1, -1, -1, bwd_qg_tile>;
    using bwd_kg_global = gl<float, -1, -1, -1, -1, bwd_kg_tile>;
    using bwd_vg_global = gl<float, -1, -1, -1, -1, bwd_vg_tile>;

    using bwd_l_global = gl<float, -1, -1, -1, -1, bwd_l_tile>;
    using bwd_d_global = gl<float, -1, -1, -1, -1, bwd_d_tile>;

    using bwd_global_args = bwd_globals<128>;

    bwd_q_global bwd_q_arg{d_q, static_cast<unsigned int>(batch),
                           static_cast<unsigned int>(qo_heads),
                           static_cast<unsigned int>(seq_len), 128U};
    bwd_k_global bwd_k_arg{d_k, static_cast<unsigned int>(batch),
                           static_cast<unsigned int>(kv_heads),
                           static_cast<unsigned int>(seq_len), 128U};
    bwd_v_global bwd_v_arg{d_v, static_cast<unsigned int>(batch),
                           static_cast<unsigned int>(kv_heads),
                           static_cast<unsigned int>(seq_len), 128U};
    bwd_og_global bwd_og_arg{d_og, static_cast<unsigned int>(batch),
                             static_cast<unsigned int>(qo_heads),
                             static_cast<unsigned int>(seq_len), 128U};
    bwd_qg_global bwd_qg_arg{d_qg, static_cast<unsigned int>(batch),
                             static_cast<unsigned int>(qo_heads),
                             static_cast<unsigned int>(seq_len), 128U};
    bwd_kg_global bwd_kg_arg{d_kg, static_cast<unsigned int>(batch),
                             static_cast<unsigned int>(kv_heads),
                             static_cast<unsigned int>(seq_len), 128U};
    bwd_vg_global bwd_vg_arg{d_vg, static_cast<unsigned int>(batch),
                             static_cast<unsigned int>(kv_heads),
                             static_cast<unsigned int>(seq_len), 128U};
    bwd_l_global bwd_l_arg{d_l, static_cast<unsigned int>(batch),
                           static_cast<unsigned int>(qo_heads), 1U,
                           static_cast<unsigned int>(seq_len)};
    bwd_d_global bwd_d_arg{d_d, static_cast<unsigned int>(batch),
                           static_cast<unsigned int>(qo_heads), 1U,
                           static_cast<unsigned int>(seq_len)};

    bwd_global_args bwd_global{
        bwd_q_arg,
        bwd_k_arg,
        bwd_v_arg,
        bwd_og_arg,
        bwd_qg_arg,
        bwd_kg_arg,
        bwd_vg_arg,
        bwd_l_arg,
        bwd_d_arg,
        static_cast<int>(seq_len),
        static_cast<int>(hr),
        static_cast<int>(max_q_blocks_per_kv),
        reinterpret_cast<int32_t *>(k2q_block_sparse_index.data_ptr()),
        reinterpret_cast<int32_t *>(k2q_block_sparse_num.data_ptr()),
        reinterpret_cast<int32_t *>(block_size.data_ptr())};

    dim3 grid_bwd_2(seq_len / 64, qo_heads, batch);
    threads = 128;

    // cudadevicesynchronize();

    {
      cudaFuncSetAttribute(bwd_attend_ker<128>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, 113000);

      bwd_attend_ker<128><<<grid_bwd_2, threads, 113000, stream>>>(bwd_global);
    }

    // cudaStreamSynchronize(stream);
    // cudadevicesynchronize();
  }

  return {qg, kg, vg};
  // cudadevicesynchronize();
}
