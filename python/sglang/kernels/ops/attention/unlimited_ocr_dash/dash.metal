#include <metal_stdlib>
using namespace metal;
constant float SCORE_NEG = -1.0e9f;
constant float ENTMAX_NEG = -1.0e4f;
constant float ACT_CLAMP = 1.0e4f;

// Request and absolute position of query row ``row``: decode rows are
// ``dql`` per request at the sequence end; prefill rows map through row2req.
struct RowPos { int req; int pos; };
RowPos row_pos(int row, bool prefill, long dql, device const long* seq_lens,
               device const int* row2req, device const long* cu, device const long* ctx) {
  RowPos r;
  if (prefill) {
    r.req = row2req[row];
    r.pos = int(ctx[r.req] + row - cu[r.req]);
  } else {
    r.req = row / int(dql);
    r.pos = int(seq_lens[r.req]) - int(dql) + row % int(dql);
  }
  return r;
}

#ifdef NSG
// Threadgroup reductions over NSG simdgroups in a fixed order (deterministic).
template <typename T> T tg_sum(T v, threadgroup T* red, uint sg, uint lane) {
  v = simd_sum(v);
  if (lane == 0) red[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  T r = red[0];
  for (int i = 1; i < NSG; ++i) r += red[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return r;
}
float tg_max(float v, threadgroup float* red, uint sg, uint lane) {
  v = simd_max(v);
  if (lane == 0) red[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float r = red[0];
  for (int i = 1; i < NSG; ++i) r = max(r, red[i]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return r;
}
float tg_min(float v, threadgroup float* red, uint sg, uint lane) {
  return -tg_max(-v, red, sg, lane);
}
#endif

#ifdef KERNEL_ATTEND
// One threadgroup per (query row, query head); simdgroups split the keys and
// merge their online-softmax states at the end. Lane l holds dims l*PER..+PER.
// Serves decode and prefill: both attend the selected remote blocks below the
// local window plus the window itself.
kernel void dash_sparse_attend(
    device const bfloat* q [[buffer(0)]],        // [nq, H, D]
    device const bfloat* k_cache [[buffer(1)]],  // [P, BS, Hkv, D]
    device const bfloat* v_cache [[buffer(2)]],
    device const int* idx [[buffer(3)]],         // [nq, T] row stride s_idx
    device const float* bias [[buffer(4)]],      // [nq, T] row stride s_bias
    device const int* block_table [[buffer(5)]], // [R, *] row stride s_bt
    device const long* seq_lens [[buffer(6)]],
    device bfloat* out [[buffer(7)]],            // [nq, H, D]
    constant long& num_heads [[buffer(8)]],
    constant long& num_kv_heads [[buffer(9)]],
    constant long& topk [[buffer(10)]],
    constant long& s_idx [[buffer(11)]],
    constant long& s_bias [[buffer(12)]],
    constant long& s_bt [[buffer(13)]],
    constant long& dql [[buffer(14)]],
    constant long& local_blocks [[buffer(15)]],
    constant long& has_bias [[buffer(16)]],
    constant float& scale [[buffer(17)]],
    device const int* row2req [[buffer(18)]],
    device const long* cu [[buffer(19)]],
    device const long* ctx [[buffer(20)]],
    constant long& prefill [[buffer(21)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const int H = int(num_heads), HKV = int(num_kv_heads);
  const int row = int(tg) / H, head = int(tg) % H, kvh = head / (H / HKV);
  const RowPos rp = row_pos(row, prefill != 0, dql, seq_lens, row2req, cu, ctx);
  const int req = rp.req, q_pos = rp.pos;
  const int lw = int(local_blocks) * BS;
  const int win_start = max(0, q_pos - lw + 1);
  const int T = int(topk);
  device const int* bt = block_table + long(req) * s_bt;

  float qv[PER];
  for (int i = 0; i < PER; ++i)
    qv[i] = float(q[(long(row) * H + head) * D + lane * PER + i]) * scale;
  float m = -INFINITY, l = 0.0f, acc[PER];
  for (int i = 0; i < PER; ++i) acc[i] = 0.0f;

  // Keys: T remote slots of BS, then local blocks q_blk - 0 .. q_blk - L.
  const int n_remote = T * BS;
  const int n_keys = n_remote + (int(local_blocks) + 1) * BS;
  for (int j = int(sg); j < n_keys; j += SG) {
    int blk, pos;
    float b = 0.0f;
    bool vis;
    if (j < n_remote) {
      const int slot = j / BS;
      blk = idx[long(row) * s_idx + slot];
      pos = blk * BS + j % BS;
      vis = blk >= 0 && pos < win_start;
      if (vis && has_bias) b = bias[long(row) * s_bias + slot];
    } else {
      const int lj = j - n_remote;
      blk = q_pos / BS - lj / BS;
      pos = blk * BS + lj % BS;
      vis = blk >= 0 && pos >= win_start && pos <= q_pos;
    }
    if (!vis) continue;
    const long base = ((long(bt[blk]) * BS + pos % BS) * HKV + kvh) * D + lane * PER;
    float s = 0.0f;
    for (int i = 0; i < PER; ++i) s += qv[i] * float(k_cache[base + i]);
    s = simd_sum(s) + b;
    const float m_new = max(m, s);
    const float corr = exp(m - m_new), p = exp(s - m_new);
    l = l * corr + p;
    for (int i = 0; i < PER; ++i) acc[i] = acc[i] * corr + p * float(v_cache[base + i]);
    m = m_new;
  }

  threadgroup float sm[SG], sl[SG], sacc[SG][D];
  if (lane == 0) { sm[sg] = m; sl[sg] = l; }
  for (int i = 0; i < PER; ++i) sacc[sg][lane * PER + i] = acc[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  float gm = -INFINITY;
  for (int g = 0; g < SG; ++g) gm = max(gm, sm[g]);
  float gl = 0.0f, o[PER];
  for (int i = 0; i < PER; ++i) o[i] = 0.0f;
  for (int g = 0; g < SG; ++g) {
    const float w = sl[g] > 0.0f ? exp(sm[g] - gm) : 0.0f;
    gl += sl[g] * w;
    for (int i = 0; i < PER; ++i) o[i] += sacc[g][lane * PER + i] * w;
  }
  const float inv = gl > 0.0f ? 1.0f / gl : 0.0f;
  for (int i = 0; i < PER; ++i)
    out[(long(row) * H + head) * D + lane * PER + i] = bfloat(o[i] * inv);
}
#endif

#ifdef KERNEL_ROUTE
// One threadgroup per decode row, mirroring _score_kernel + _entmax_kernel over
// the full block-table width. Thread t owns the contiguous columns
// [t * per, (t + 1) * per) so the packing pass is a prefix sum over threads.
kernel void dash_route(
    device const bfloat* idx_q [[buffer(0)]],    // [nd, DQ]
    device const float* k_bar [[buffer(1)]],     // [pages, DQ]
    device const int* block_table [[buffer(2)]],
    device const long* seq_lens [[buffer(3)]],
    device const char* img [[buffer(4)]],        // [nd, *] int8, row stride s_img
    device int* out_idx [[buffer(5)]],
    device float* out_bias [[buffer(6)]],
    constant long& W [[buffer(7)]],
    constant long& s_bt [[buffer(8)]],
    constant long& s_img [[buffer(9)]],
    constant long& s_oi [[buffer(10)]],
    constant long& s_ob [[buffer(11)]],
    constant long& dql [[buffer(12)]],
    constant long& local_blocks [[buffer(13)]],
    constant long& has_img [[buffer(14)]],
    constant long& budget [[buffer(15)]],
    constant long& M [[buffer(16)]],
    constant float& idx_scale [[buffer(17)]],
    constant float& prior_floor [[buffer(18)]],
    constant float& inv_sigma [[buffer(19)]],
    device const int* row2req [[buffer(20)]],
    device const long* cu [[buffer(21)]],
    device const long* ctx [[buffer(22)]],
    constant long& prefill [[buffer(23)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float zs[MAXW];
  threadgroup char cand[MAXW];
  threadgroup float qs[DQ];
  threadgroup float redf[NSG];
  threadgroup int redi[NSG];
  const int w_all = int(W);
  const RowPos rp = row_pos(int(row), prefill != 0, dql, seq_lens, row2req, cu, ctx);
  const int req = rp.req;
  const int seq = int(seq_lens[req]);
  const int q_blk = rp.pos / BS;
  device const int* bt = block_table + long(req) * s_bt;
  device const char* im = img + long(row) * s_img;
  for (int d = int(tid); d < DQ; d += NT) qs[d] = float(idx_q[long(row) * DQ + d]);
  if (int(tid) < int(M)) {
    out_idx[long(row) * s_oi + tid] = -1;
    out_bias[long(row) * s_ob + tid] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
  const int per = (w_all + NT - 1) / NT;
  const int c0 = min(int(tid) * per, w_all), c1 = min(c0 + per, w_all);
  const float inv_sqrt_d = rsqrt(float(DQ));

  int n_img = 0;
  for (int w = c0; w < c1; ++w) {
    float s = SCORE_NEG;
    if (w <= q_blk - int(local_blocks) && w * BS < seq) {
      device const float* kb = k_bar + long(bt[w]) * DQ;
      float z = 0.0f;
      for (int d = 0; d < DQ; ++d) z += kb[d] * qs[d];
      s = clamp(z * inv_sqrt_d, -ACT_CLAMP, ACT_CLAMP);
    }
    zs[w] = s;
    n_img += (has_img && s > SCORE_NEG * 0.5f && im[w] != 0) ? 1 : 0;
  }
  const bool use_img = has_img && tg_sum(n_img, redi, sg, lane) > 0;
  int n_c = 0;
  float xm = -INFINITY;
  for (int w = c0; w < c1; ++w) {
    const bool c = zs[w] > SCORE_NEG * 0.5f && (!use_img || im[w] != 0);
    cand[w] = c;
    n_c += c ? 1 : 0;
    float x = c ? zs[w] * idx_scale : ENTMAX_NEG;
    x = isnan(x) ? ENTMAX_NEG : x;
    zs[w] = x;
    if (c) xm = max(xm, x);
  }
  const int n_cand = tg_sum(n_c, redi, sg, lane);
  xm = n_cand > 0 ? tg_max(xm, redf, sg, lane) : 0.0f;
  float zmin = INFINITY;
  for (int w = c0; w < c1; ++w) {
    zs[w] = (zs[w] - xm) * 0.5f;
    if (cand[w]) zmin = min(zmin, zs[w]);
  }

  float lo = -1.0f, hi = 0.0f;
  for (int it = 0; it < 24; ++it) {
    const float mid = (lo + hi) * 0.5f;
    float s = 0.0f;
    for (int w = c0; w < c1; ++w) {
      const float p = max(zs[w] - mid, 0.0f);
      s += cand[w] ? p * p : 0.0f;
    }
    const bool over = tg_sum(s, redf, sg, lane) > 1.0f;
    lo = over ? mid : lo;
    hi = over ? hi : mid;
  }
  const float tau = (lo + hi) * 0.5f;

  float thr = -INFINITY;  // active = cand & (z > thr)
  if (n_cand > int(budget)) {
    float lo2 = tg_min(zmin, redf, sg, lane) - 1.0f, hi2 = 0.0f;
    for (int it = 0; it < 40; ++it) {
      const float mid = (lo2 + hi2) * 0.5f;
      int c = 0;
      for (int w = c0; w < c1; ++w) c += (cand[w] && zs[w] > mid) ? 1 : 0;
      const bool over = tg_sum(c, redi, sg, lane) > int(budget);
      lo2 = over ? mid : lo2;
      hi2 = over ? hi2 : mid;
    }
    thr = hi2;
  }
  int n_a = 0;
  float s_logp = 0.0f;
  for (int w = c0; w < c1; ++w) {
    if (cand[w] && zs[w] > thr) {
      const float p = max(zs[w] - tau, 0.0f);
      n_a += 1;
      s_logp += log(max(p * p, prior_floor));
    }
  }
  const int cnt = max(tg_sum(n_a, redi, sg, lane), 1);
  const float mu = tg_sum(s_logp, redf, sg, lane) / float(cnt);
  // Exclusive prefix of the per-thread active counts gives each thread's slot.
  const int inc = simd_prefix_exclusive_sum(n_a);
  if (lane == 31) redi[sg] = inc + n_a;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int pos = inc;
  for (uint g = 0; g < sg; ++g) pos += redi[g];
  for (int w = c0; w < c1 && pos < int(M); ++w) {
    if (cand[w] && zs[w] > thr) {
      const float p = max(zs[w] - tau, 0.0f);
      out_idx[long(row) * s_oi + pos] = w;
      out_bias[long(row) * s_ob + pos] = (log(max(p * p, prior_floor)) - mu) * inv_sigma;
      ++pos;
    }
  }
}
#endif

#ifdef KERNEL_KBAR
// One simdgroup per plan entry: head_cls masked-softmax summary of one page.
kernel void dash_kbar_refresh(
    device const bfloat* kv [[buffer(0)]],        // [pages, BS, DQ]
    device const bfloat* head_cls [[buffer(1)]],  // [DQ]
    device const long* pages [[buffer(2)]],
    device const int* tok [[buffer(3)]],
    device float* k_bar [[buffer(4)]],            // [pages, DQ]
    uint e [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const int ntok = tok[e];
  // Entries without tokens are not pages of a request.
  if (ntok <= 0) return;
  const long page = pages[e];
  device const bfloat* keys = kv + page * BS * DQ;
  threadgroup float ws[BS];
  const float inv_sqrt_d = rsqrt(float(DQ));
  float aw[TPL];
  float mx = -INFINITY;
  for (int i = 0; i < TPL; ++i) {
    const int t = int(lane) + 32 * i;
    float a = -INFINITY;
    if (t < BS && t < ntok) {
      float s = 0.0f;
      for (int d = 0; d < DQ; ++d) s += float(keys[t * DQ + d]) * float(head_cls[d]);
      a = clamp(s * inv_sqrt_d, -ACT_CLAMP, ACT_CLAMP);
    }
    aw[i] = a;
    mx = max(mx, a);
  }
  mx = simd_max(mx);
  float den = 0.0f;
  for (int i = 0; i < TPL; ++i) {
    aw[i] = aw[i] > -INFINITY ? exp(aw[i] - mx) : 0.0f;
    den += aw[i];
  }
  den = max(simd_sum(den), 1e-6f);
  for (int i = 0; i < TPL; ++i) {
    const int t = int(lane) + 32 * i;
    if (t < BS) ws[t] = aw[i] / den;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int d = int(lane); d < DQ; d += 32) {
    float acc = 0.0f;
    for (int t = 0; t < ntok && t < BS; ++t) acc += ws[t] * float(keys[t * DQ + d]);
    k_bar[page * DQ + d] = isnan(acc) ? 0.0f : acc;
  }
}
#endif

#ifdef KERNEL_ROPE
// One thread per (token, head, i < HALF) of x then y: NeoX rotation with the
// PyTorch bf16 rounding (each product rounds to bf16, then the sum does).
kernel void page2d_rope_pair(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* y [[buffer(1)]],
    device const bfloat* cos_t [[buffer(2)]],   // [t, HD]
    device const bfloat* sin_t [[buffer(3)]],
    device bfloat* ox [[buffer(4)]],
    device bfloat* oy [[buffer(5)]],
    constant long& nx [[buffer(6)]],
    constant long& ny [[buffer(7)]],
    constant long& total [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
  if (long(gid) >= total) return;
  const long i = gid % HALF;
  const long th = gid / HALF;
  const long tok = th / (nx + ny), h = th % (nx + ny);
  const bool is_x = h < nx;
  device const bfloat* src = is_x ? x + (tok * nx + h) * HD : y + (tok * ny + h - nx) * HD;
  device bfloat* dst = is_x ? ox + (tok * nx + h) * HD : oy + (tok * ny + h - nx) * HD;
  device const bfloat* c = cos_t + tok * HD;
  device const bfloat* s = sin_t + tok * HD;
  const float x1 = float(src[i]), x2 = float(src[i + HALF]);
  const float a1 = float(bfloat(x1 * float(c[i]))), b1 = float(bfloat(-x2 * float(s[i])));
  const float a2 = float(bfloat(x2 * float(c[i + HALF])));
  const float b2 = float(bfloat(x1 * float(s[i + HALF])));
  dst[i] = bfloat(a1 + b1);
  dst[i + HALF] = bfloat(a2 + b2);
}
#endif

#ifdef KERNEL_DECODE_META
// Port of decode_meta._decode_meta_kernel, one thread per request.
kernel void dash_decode_meta(
    device const long* ids [[buffer(0)]],
    device const int* bt [[buffer(1)]],
    device const long* seq_lens [[buffer(2)]],
    device const long* slots [[buffer(3)]],
    device const long* rows [[buffer(4)]],
    device int* s_page [[buffer(5)]],
    device int* s_mark [[buffer(6)]],
    device int* s_run [[buffer(7)]],
    device int* s_pp [[buffer(8)]],
    device int* c_mark [[buffer(9)]],
    device int* c_pp [[buffer(10)]],
    device int* c_cnt [[buffer(11)]],
    device int* q_page_out [[buffer(12)]],
    device long* tok_out [[buffer(13)]],
    device long* page_out [[buffer(14)]],
    constant long& R [[buffer(15)]],
    constant long& s_bt [[buffer(16)]],
    constant long& s_cnt [[buffer(17)]],
    constant long& image_id [[buffer(18)]],
    constant long& page_id [[buffer(19)]],
    constant long& per_page [[buffer(20)]],
    constant long& n_slots [[buffer(21)]],
    constant long& max_pages [[buffer(22)]],
    constant long& bt_w [[buffer(23)]],
    constant long& has_marker [[buffer(24)]],
    uint r [[thread_position_in_grid]]) {
  if (long(r) >= R) return;
  const long tok_id = ids[r];
  const bool is_img = tok_id >= 1000000 || tok_id == image_id;
  const bool is_mark = has_marker && tok_id == page_id;
  const int seq = int(seq_lens[r]);
  device const int* btr = bt + long(r) * s_bt;

  const int prev_pos = seq - 2;
  const bool valid = prev_pos >= 0;
  const int pp0 = max(prev_pos, 0);
  const long blk = min(long(pp0 / BS), bt_w - 1);
  long prev_slot = clamp(long(btr[blk]) * BS + pp0 % BS, 0L, n_slots - 1);
  const int prev_m = valid ? s_mark[prev_slot] : 0;
  const int prev_i = valid ? s_run[prev_slot] : 0;
  const int prev_pp = valid ? s_pp[prev_slot] : 0;
  const int marker_cum = prev_m + (is_mark ? 1 : 0);
  const int img_run = prev_i + (is_img ? 1 : 0);

  int img_page = marker_cum > 0 ? marker_cum - 1 : (img_run - 1) / int(per_page);
  img_page = is_img ? img_page : -1;
  const int pp = max(is_img ? marker_cum : 0, prev_pp);
  int resp = marker_cum - 1 - pp;
  resp = resp >= 0 ? resp : -1;
  q_page_out[r] = is_img ? img_page : resp;

  const long slot = clamp(slots[r], 0L, n_slots - 1);
  s_page[slot] = img_page;
  s_mark[slot] = marker_cum;
  s_run[slot] = img_run;
  s_pp[slot] = pp;

  // Clear unused slots of the last partial block (stale image-page state).
  const long last_blk = clamp(long((seq - 1) / BS), 0L, bt_w - 1);
  const long tail_phys = long(btr[last_blk]);
  const int rem = seq % BS;
  if (rem > 0) {
    for (int b = rem; b < BS; ++b) s_page[clamp(tail_phys * BS + b, 0L, n_slots - 1)] = -1;
  }

  const long row = rows[r];
  const int c_m = c_mark[row] + (is_mark ? 1 : 0);
  const int c_p = max(is_img ? c_m : 0, c_pp[row]);
  const int c_resp = min(max(c_m - c_p, 0), c_p);
  const long cpage = clamp(long(c_resp), 0L, max_pages - 1);
  const int cnt_at = c_cnt[row * s_cnt + cpage];
  tok_out[r] = cnt_at;
  page_out[r] = cpage;
  c_cnt[row * s_cnt + cpage] = cnt_at + 1;
  c_mark[row] = c_m;
  c_pp[row] = c_p;
}

// Port of decode_meta._decode_allow_kernel, one threadgroup per request.
kernel void dash_decode_allow(
    device const int* bt [[buffer(0)]],
    device const int* s_page [[buffer(1)]],
    device const int* q_page [[buffer(2)]],
    device const long* seq_lens [[buffer(3)]],
    device char* out [[buffer(4)]],
    constant long& s_bt [[buffer(5)]],
    constant long& s_out [[buffer(6)]],
    constant long& W [[buffer(7)]],
    constant long& num_blocks [[buffer(8)]],
    constant long& max_block [[buffer(9)]],
    uint r [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup int redi[NSG];
  const int qp = q_page[r];
  const int seq = int(seq_lens[r]);
  device const int* btr = bt + long(r) * s_bt;
  int n_same = 0;
  for (long w = tid; w < max_block; w += NT) {
    bool same = false;
    if (w < W && w * BS < seq) {
      const long page = clamp(long(btr[w]), 0L, num_blocks - 1);
      int mx = -1, mn = 32767;
      for (int b = 0; b < BS; ++b) {
        const int ip = s_page[page * BS + b];
        if (ip >= 0) { mx = max(mx, ip); mn = min(mn, ip); }
      }
      mn = mx >= 0 ? mn : -1;
      same = mn >= 0 && mn <= qp && qp <= mx && qp >= 0;
    }
    n_same += same ? 1 : 0;
    out[long(r) * s_out + w] = same ? 1 : 0;
  }
  const bool open_row = qp < 0 || tg_sum(n_same, redi, sg, lane) == 0;
  if (!open_row) return;
  for (long w = tid; w < max_block; w += NT) out[long(r) * s_out + w] = 1;
}
#endif
