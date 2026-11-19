#include <metal_stdlib>
using namespace metal;

#ifdef KERNEL_CONV
inline float gdn_silu(float x) { return x / (1.0f + exp(-x)); }

// One thread per (batch row, channel): y = silu(bias + w . (window ++ x)),
// then the slot window shifts left by one and takes x.
kernel void gdn_conv1d_update(
    device const IN_T* x [[buffer(0)]],       // [B, C], row stride x_t
    device ST_T* conv_states [[buffer(1)]],   // [S, C, K-1] strided
    device const W_T* weight [[buffer(2)]],   // [C, K]
    device const W_T* bias [[buffer(3)]],     // [C] (unread without HAS_BIAS)
    device const int* cache_indices [[buffer(4)]],
    device IN_T* y [[buffer(5)]],             // [B, C] contiguous
    constant long& x_t [[buffer(6)]],
    constant long& cs_slot [[buffer(7)]],
    constant long& cs_c [[buffer(8)]],
    constant long& cs_k [[buffer(9)]],
    constant int& C [[buffer(10)]],
    uint tid [[thread_position_in_grid]]) {
  const int bi = tid / C, c = tid % C;
  const int slot = cache_indices[bi];
  if (slot < 0) {
    y[tid] = IN_T(0);
    return;
  }
  device ST_T* row = conv_states + slot * cs_slot + c * cs_c;
  device const W_T* w = weight + c * K;
  const float xv = float(x[bi * x_t + c]);
  float acc = HAS_BIAS ? float(bias[c]) : 0.0f;
  float win[K - 1];
  for (int j = 0; j < K - 1; ++j) {
    win[j] = float(row[j * cs_k]);
    acc += float(w[j]) * win[j];
  }
  acc += float(w[K - 1]) * xv;
  for (int j = 0; j < K - 2; ++j) row[j * cs_k] = ST_T(win[j + 1]);
  row[(K - 2) * cs_k] = ST_T(xv);
  y[tid] = IN_T(gdn_silu(acc));
}

// Sequence owning token t: the last s with qsl[s] <= t, so empty sequences
// (qsl[s] == qsl[s + 1]) never own a token.
inline int gdn_seq_of_token(device const int* qsl, int n, int t) {
  int lo = 0, hi = n - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (qsl[mid] <= t) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}

// One thread per (token, channel). Reads the slot window but never writes it;
// gdn_conv1d_varlen_state runs after it and stores each final window.
kernel void gdn_conv1d_varlen(
    device const IN_T* x [[buffer(0)]],       // [T, C], row stride x_t
    device const ST_T* conv_states [[buffer(1)]],
    device const W_T* weight [[buffer(2)]],
    device const W_T* bias [[buffer(3)]],
    device const int* cache_indices [[buffer(4)]],
    device const int* qsl [[buffer(5)]],      // [N + 1]
    device const uchar* has_initial_state [[buffer(6)]],
    device IN_T* y [[buffer(7)]],             // [T, C] contiguous
    constant long& x_t [[buffer(8)]],
    constant long& cs_slot [[buffer(9)]],
    constant long& cs_c [[buffer(10)]],
    constant long& cs_k [[buffer(11)]],
    constant int& C [[buffer(12)]],
    constant int& N [[buffer(13)]],
    uint tid [[thread_position_in_grid]]) {
  const int t = tid / C, c = tid % C;
  const int s = gdn_seq_of_token(qsl, N, t);
  const int slot = cache_indices[s];
  if (slot < 0) {
    y[tid] = IN_T(0);
    return;
  }
  const int start = qsl[s], pos = t - start;
  const bool from_slot = has_initial_state[s] != 0;
  device const ST_T* row = conv_states + slot * cs_slot + c * cs_c;
  device const W_T* w = weight + c * K;
  float acc = HAS_BIAS ? float(bias[c]) : 0.0f;
  for (int j = 0; j < K; ++j) {
    // Index into (window ++ sequence); the window occupies [0, K-1).
    const int i = pos + j;
    float v;
    if (i >= K - 1) {
      v = float(x[(start + i - (K - 1)) * x_t + c]);
    } else {
      v = from_slot ? float(row[i * cs_k]) : 0.0f;
    }
    acc += float(w[j]) * v;
  }
  y[tid] = IN_T(gdn_silu(acc));
}

// One thread per (sequence, channel): the last K-1 entries of
// (window ++ sequence), the window being the slot or zeros.
kernel void gdn_conv1d_varlen_state(
    device const IN_T* x [[buffer(0)]],
    device ST_T* conv_states [[buffer(1)]],
    device const int* cache_indices [[buffer(2)]],
    device const int* qsl [[buffer(3)]],
    device const uchar* has_initial_state [[buffer(4)]],
    constant long& x_t [[buffer(5)]],
    constant long& cs_slot [[buffer(6)]],
    constant long& cs_c [[buffer(7)]],
    constant long& cs_k [[buffer(8)]],
    constant int& C [[buffer(9)]],
    uint tid [[thread_position_in_grid]]) {
  const int s = tid / C, c = tid % C;
  const int slot = cache_indices[s];
  if (slot < 0) return;
  const int start = qsl[s], len = qsl[s + 1] - start;
  const bool from_slot = has_initial_state[s] != 0;
  device ST_T* row = conv_states + slot * cs_slot + c * cs_c;
  float win[K - 1];
  for (int j = 0; j < K - 1; ++j) {
    const int i = len + j;
    if (i >= K - 1) {
      win[j] = float(x[(start + i - (K - 1)) * x_t + c]);
    } else {
      win[j] = from_slot ? float(row[i * cs_k]) : 0.0f;
    }
  }
  for (int j = 0; j < K - 1; ++j) row[j * cs_k] = ST_T(win[j]);
}
#endif

#ifdef KERNEL_RECURRENT
// LPR lanes own one value row; each lane keeps the 32 contiguous fp32 state
// columns at lane_in_row * 32 in registers, so a simdgroup covers ROWS rows
// and every reduction stays inside its LPR-wide row group.
constant constexpr int LPR = DK / 32;
constant constexpr int SG_PER_TG = 4;
constant constexpr int ROWS = 32 / LPR;
constant constexpr int NCH = 8;  // 32 columns per lane, as float4 chunks

inline float gdn_row_sum(float v) {
  for (ushort m = 1; m < LPR; m <<= 1) v += simd_shuffle_xor(v, m);
  return v;
}

// One thread per (token, key head) for the inverse q/k L2 norms and per
// (token, value head) for the fp32 decay and delta rate, so the recurrence
// loads them instead of re-running the reductions and exp/log in every lane.
// l2norm as fla / HF: x * rsqrt(sum(x^2) + 1e-6); q also gets DK^-0.5.
kernel void gdn_prep(
    device const IN_T* q [[buffer(0)]],         // [T, Hk, DK] strided
    device const IN_T* k [[buffer(1)]],         // [T, Hk, DK] strided
    device const GATE_T* a [[buffer(2)]],       // [T, Hv], row stride a_t
    device const GATE_T* b [[buffer(3)]],       // [T, Hv], row stride b_t
    device const AL_T* A_log [[buffer(4)]],     // [Hv]
    device const DT_T* dt_bias [[buffer(5)]],
    device float* g [[buffer(6)]],              // [T, Hv] contiguous
    device float* beta [[buffer(7)]],           // [T, Hv] contiguous
    device float* norms [[buffer(8)]],          // [T, Hk, 2] contiguous
    constant long& q_t [[buffer(9)]],
    constant long& q_h [[buffer(10)]],
    constant long& k_t [[buffer(11)]],
    constant long& k_h [[buffer(12)]],
    constant long& a_t [[buffer(13)]],
    constant long& b_t [[buffer(14)]],
    constant int& Hk [[buffer(15)]],
    constant int& Hv [[buffer(16)]],
    constant long& norm_count [[buffer(17)]],   // T * Hk
    uint tid [[thread_position_in_grid]]) {
  if (tid < norm_count) {
    const long t = tid / Hk;
    const int hk = tid % Hk;
    device const IN_T* qp = q + t * q_t + hk * q_h;
    device const IN_T* kp = k + t * k_t + hk * k_h;
    float sq = 0.0f, sk = 0.0f;
    for (int i = 0; i < DK; ++i) {
      const float qv = float(qp[i]), kv = float(kp[i]);
      sq += qv * qv;
      sk += kv * kv;
    }
    norms[tid * 2] = rsqrt(sq + 1e-6f) * rsqrt(float(DK));
    norms[tid * 2 + 1] = rsqrt(sk + 1e-6f);
    return;
  }
  const long j = tid - norm_count;
  const int hv = j % Hv;
  const long t = j / Hv;
  const float ax = float(a[t * a_t + hv]) + float(dt_bias[hv]);
  const float softplus = ax > 20.0f ? ax : log(1.0f + exp(ax));
  g[j] = exp(-exp(float(A_log[hv])) * softplus);
  beta[j] = 1.0f / (1.0f + exp(-float(b[t * b_t + hv])));
}

// One simdgroup per (sequence, value head, ROWS value rows); the fp32 state
// of those rows stays in registers across the sequence's tokens.
kernel void gdn_recurrent(
    device const IN_T* q [[buffer(0)]],         // [T, Hk, DK] strided
    device const IN_T* k [[buffer(1)]],         // [T, Hk, DK] strided
    device const IN_T* v [[buffer(2)]],         // [T, Hv, Dv] strided
    device const float* g [[buffer(3)]],        // [T, Hv], row stride g_t
    device const float* beta [[buffer(4)]],     // [T, Hv], row stride g_t
    device const float* norms [[buffer(5)]],    // [T, Hk, 2] contiguous
    device float* ssm_states [[buffer(6)]],     // [S, Hv, Dv, DK] strided
    device const int* cache_indices [[buffer(7)]],
    device const int* qsl [[buffer(8)]],
    device const uchar* has_initial_state [[buffer(9)]],
    device IN_T* o [[buffer(10)]],              // [T, Hv, Dv] contiguous
    constant long& q_t [[buffer(11)]],
    constant long& q_h [[buffer(12)]],
    constant long& k_t [[buffer(13)]],
    constant long& k_h [[buffer(14)]],
    constant long& v_t [[buffer(15)]],
    constant long& v_h [[buffer(16)]],
    constant long& g_t [[buffer(17)]],
    constant long& s_slot [[buffer(18)]],
    constant long& s_h [[buffer(19)]],
    constant long& s_row [[buffer(20)]],
    constant int& Hv [[buffer(21)]],
    constant int& Hk [[buffer(22)]],
    constant int& Dv [[buffer(23)]],
    constant int& N [[buffer(24)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const int blocks = (Dv + ROWS - 1) / ROWS;
  const int gid = tg * SG_PER_TG + sg;
  if (gid >= N * Hv * blocks) return;
  const int n = gid / (Hv * blocks);
  const int hv = (gid / blocks) % Hv;
  const int hk = hv / (Hv / Hk);
  const int row = (gid % blocks) * ROWS + lane / LPR;
  const int col = (lane % LPR) * 32;
  const int start = qsl[n], end = qsl[n + 1];
  const int slot = cache_indices[n];
  const bool live = row < Dv;
  device IN_T* op = o + ((long)start * Hv + hv) * Dv + row;
  if (slot < 0) {
    if (live && lane % LPR == 0) {
      for (int t = start; t < end; ++t, op += Hv * Dv) *op = IN_T(0);
    }
    return;
  }

  device float4* state4 = (device float4*)(
      ssm_states + slot * s_slot + hv * s_h + row * s_row + col);
  float4 st[NCH];
  const bool init = has_initial_state[n] != 0;
  for (int i = 0; i < NCH; ++i) st[i] = (init && live) ? state4[i] : float4(0.0f);

  device const float* np = norms + (start * Hk + hk) * 2;
  device const IN_T* qp = q + start * q_t + hk * q_h + col;
  device const IN_T* kp = k + start * k_t + hk * k_h + col;
  device const IN_T* vp = v + start * v_t + hv * v_h + row;
  device const float* gp = g + start * g_t + hv;
  device const float* bp = beta + start * g_t + hv;

  for (int t = start; t < end; ++t) {
    device const vec<IN_T, 4>* q4 = (device const vec<IN_T, 4>*)qp;
    device const vec<IN_T, 4>* k4 = (device const vec<IN_T, 4>*)kp;
    float4 qc[NCH], kc[NCH];
    for (int i = 0; i < NCH; ++i) {
      qc[i] = float4(q4[i]);
      kc[i] = float4(k4[i]);
    }
    const float q_inv = np[0], k_inv = np[1];

    const float decay = *gp;
    const float delta_rate = *bp;

    float kv = 0.0f;
    for (int i = 0; i < NCH; ++i) {
      kc[i] *= k_inv;
      st[i] *= decay;
      kv += dot(st[i], kc[i]);
    }
    const float delta =
        ((live ? float(*vp) : 0.0f) - gdn_row_sum(kv)) * delta_rate;
    float out = 0.0f;
    for (int i = 0; i < NCH; ++i) {
      st[i] += kc[i] * delta;
      out += dot(st[i], qc[i]);
    }
    out = gdn_row_sum(out) * q_inv;
    if (live && lane % LPR == 0) *op = IN_T(out);

    qp += q_t;
    kp += k_t;
    np += Hk * 2;
    vp += v_t;
    gp += g_t;
    bp += g_t;
    op += Hv * Dv;
  }

  if (live) {
    for (int i = 0; i < NCH; ++i) state4[i] = st[i];
  }
}
#endif
