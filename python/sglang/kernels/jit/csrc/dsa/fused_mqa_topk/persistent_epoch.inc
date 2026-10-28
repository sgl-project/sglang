// Body of one persistent epoch, included twice by persistent_kernel.cuh: the
// first epoch is straight-line code so the hot loops are not nested in an outer
// loop.
if (warp_idx == 9 && cute::elect_one_sync()) {
  if (epoch)
    for (uint32_t i = 0; i < B::total; ++i)
      mbar_inval(bar_words + i);
  for (uint32_t i = 0; i < B::total; ++i)
    reinterpret_cast<volatile uint64_t*>(bar_words)[i] = 0;
  full_q.init(1);
  empty_q.init(kMathThreads + 32);
  for (uint32_t i = 0; i < kKVStages; ++i) {
    full_kv[i].init(kFullKvArrivals);
    empty_kv[i].init(kMathThreads);
  }
  for (uint32_t i = 0; i < B::kTmemSlots; ++i) {
    full_tmem[i].init(1);
    empty_tmem[i].init(kMathWgThreads);
  }
  full_lin->init(32);
  for (uint32_t i = 0; i < 2; ++i) {
    claim_full[i].init(32);
    claim_empty[i].init(kClaimConsumers * 32);
  }  // every reader lane arrives
  sel_done.init(kMathWarps);
  for (uint32_t i = 0; i < 2 * kMathWarps; ++i) {
    h_full[i].init(32);
    h_empty[i].init(32);
  }
  cutlass::arch::fence_barrier_init();
  unsafe_s = 0;
  // Shared addresses and lane constants the math warps read back with a
  // volatile load (opaque_ld_u32).
  opaque_bases[0] = static_cast<uint32_t>(__cvta_generic_to_shared(bar_words));
  opaque_bases[1] = static_cast<uint32_t>(__cvta_generic_to_shared(&sm.sf_kv[0][0]));
  opaque_bases[2] = static_cast<uint32_t>(__cvta_generic_to_shared(&sdata.recs[0][0][0][0]));
  for (uint32_t l = 0; l < 32; ++l) {
    lane_tab[0][l] = l;
    lane_tab[1][l] = (1u << l) - 1u;
  }
}
__syncwarp();
if (warp_idx == 10) cute::TMEM::Allocator1Sm().allocate(512, &tmem_ptr);
__syncthreads();
if (epoch == 0) cudaGridDependencySynchronize();

if (warp_idx == 8) {
  // ---- warp 8: claimer, Q / weights TMA, drain of math WG1's rows ----
  cutlass::arch::warpgroup_reg_dealloc<kSpecRegs>();
  uint32_t n_q = 0, n_blk = 0, drain_epoch = 0;
  for (uint32_t k = 0;; ++k) {
    if (lane == 0) {
      empty_q.wait((n_q & 1) ^ 1);
    }
    __syncwarp();
    claim_empty[k & 1].wait(((k >> 1) & 1) ^ 1);  // every lane: lanes 0-7 write the entry
    uint32_t b = 0;
    if (lane == 0) b = atomicOr(&pause_req, 0u) ? kPause : atomicAdd(counter, 1u);
    b = __shfl_sync(kFullMask, b, 0);
    ClaimEntry& e = entries[k & 1];
    if (b >= nblocks) {
      if (lane == 0) {
        const uint32_t kind = b == kPause ? kPause : kTerm;
        e.vblock = kind;
        epoch_kind = kind;
      }
      __syncwarp();
      claim_full[k & 1].arrive();  // every writer lane arrives
      break;
    }
    const uint32_t vblock = nblocks - 1 - b, row_base = vblock * kBlockRows;
    uint32_t st = 0, en = 0;
    if (lane < kBlockRows) {
      const uint32_t r = row_base + lane;
      if (r < rows) {
        st = min(starts_g[r], keys);
        en = max(min(ends_g[r], keys), st);
      }
      e.starts[lane] = st;
      e.ends[lane] = en;
      e.out_base[lane] = !relative ? int32_t(st) : out_offsets ? (r < rows ? out_offsets[r] : 0) : 0;
    }
    const bool all_short = __ballot_sync(kFullMask, lane < kBlockRows && en - st > topk) == 0;
    const bool gated = __ballot_sync(kFullMask, lane < kBlockRows && en - st > sample_target) != 0;
    // KV range: rows past the end are clamped to the last valid row (DeepGEMM
    // scheduler).
    const uint32_t src = min(row_base + (lane & 7), rows - 1) - row_base;
    uint32_t lo = __shfl_sync(kFullMask, st, src), hi = __shfl_sync(kFullMask, en, src);
#pragma unroll
    for (int32_t d = 4; d >= 1; d >>= 1) {
      lo = min(lo, __shfl_xor_sync(kFullMask, lo, d));
      hi = max(hi, __shfl_xor_sync(kFullMask, hi, d));
    }
    const uint32_t kv_base = lo / 4 * 4;
    if (!all_short && lane < kBlockRows) counts[row_base + lane] = 0;
    if (lane == 0) {
      // CTA-private slot pair: block n of this CTA uses slot 2 * cta + (n & 1);
      // at most two blocks (one in selection, one claimed ahead) are in flight,
      // so no claim atomics.
      const uint32_t slot = 2 * blockIdx.x + (n_blk & 1);
      e.vblock = vblock;
      e.all_short = all_short;
      e.kv_base = kv_base;
      e.splits = (hi - kv_base + kTileKeys - 1) / kTileKeys;
      e.samples = gated ? sample_tiles : 0u;
      e.slot = slot;
    }
    __syncwarp();
    claim_full[k & 1].arrive();  // every writer lane arrives
    if (all_short) continue;
    if (lane == 0) {
      tma_load(&tq, &full_q, sm.q, 0, row_base * kNumHeads);
      tma_load(&tw, &full_q, sm.weights, 0, row_base);
      full_q.arrive_and_expect_tx(kQBytes + kWBytes);
    }
    __syncwarp();
    ++n_q;
    ++n_blk;
    const uint32_t splits = (hi - kv_base + kTileKeys - 1) / kTileKeys;
    const uint32_t slot = __shfl_sync(kFullMask, e.slot, 0);
    const int64_t row_delta = int64_t(slot) * kBlockRows - int64_t(row_base);
    for (uint32_t base = 0; base <= splits; base += Interval, ++drain_epoch)
      drain_interval(
          sdata,
          h_full,
          h_empty,
          4,
          drain_epoch,
          row_base + 4,
          row_count + 4,
          kCandGlobalCap,
          base + Interval > splits,
          vals,
          indices,
          row_delta,
          kv_base);
    cutlass::arch::NamedBarrier(kDrainThreads, kBarDrain).arrive();
  }
} else if (warp_idx == 9) {
  // ---- warp 9: KV TMA, the sample tiles then the scan tiles in one ring ----
  cutlass::arch::warpgroup_reg_dealloc<kSpecRegs>();
  RingPipeline<kKVStages> kv_pipe;
  uint32_t n_blk = 0;
  for (uint32_t k = 0;; ++k) {
    ClaimHeader e;
    if (!consume_claim(k, claim_full, claim_empty, entries, e)) break;
    if (e.all_short) continue;
    const uint32_t kv_base = e.kv_base, splits = e.splits, samples = e.samples;
    // The whole warp stages the strided sample scales into sf_kv[stage]
    // (prefetched one tile ahead in registers), lane 0 issues the KV TMA.
    const uint32_t center = sample_stride / 2;
    float pf[8];
#pragma unroll
    for (uint32_t j = 0; j < 8; ++j) {
      const uint32_t g = j * 32 + lane;
      pf[j] = g < sample_groups ? __ldg(scales + g * sample_stride + center) : 0.0f;
    }
    // the first sample scales load while the previous block still selects
    if (lane == 0 && n_blk > 0) sel_done.wait((n_blk - 1) & 1);
    __syncwarp();
#pragma unroll 1
    for (uint32_t t = 0; t < samples; ++t) {
      // Only lane 0's ring state is maintained across blocks (the scan loop
      // runs on lane 0).
      CUTE_TIE_DECL(kv_pipe.advance(), s0, ph);
      const uint32_t s = __shfl_sync(kFullMask, s0, 0);
      // Every writing lane acquires and releases the stage itself.
      empty_kv[s].wait(__shfl_sync(kFullMask, ph, 0) ^ 1);
#pragma unroll
      for (uint32_t j = 0; j < 8; ++j)
        sm.sf_kv[s][j * 32 + lane] = pf[j];
      // The generic scale writes before the stage's later TMA writes.
      cutlass::arch::fence_view_async_shared();
      if (lane == 0) {
        full_kv[s].expect_transaction(kKVBytes);
        tma_load(&tks, &full_kv[s], sm.kv[s], 0, t * kTileKeys, center);
      }
      full_kv[s].arrive();
#pragma unroll
      for (uint32_t j = 0; j < 8; ++j) {
        const uint32_t g = (t + 1) * kTileKeys + j * 32 + lane;
        pf[j] = g < sample_groups ? __ldg(scales + g * sample_stride + center) : 0.0f;
      }
    }
    if (lane == 0) {
#pragma unroll 1
      for (uint32_t t = 0; t < splits; ++t) {
        CUTE_TIE_DECL(kv_pipe.advance(), s, ph);
        empty_kv[s].wait(ph ^ 1);
        const uint32_t off = kv_base + t * kTileKeys;
        tma_load(&tk, &full_kv[s], sm.kv[s], 0, off);
        tma_load(&ts, &full_kv[s], sm.sf_kv[s], off, 0);
        mbar_arrive_expect_tx(&full_kv[s], kKVBytes + kScaleBytes, kFullKvArrivals);
      }
    }
    __syncwarp();
    ++n_blk;
  }
} else if (warp_idx == 10) {
  // ---- warp 10: UMMA issuer ----
  cutlass::arch::warpgroup_reg_dealloc<kSpecRegs>();
  RingPipeline<1> q_pipe;
  RingPipeline<kKVStages> kv_pipe;
  RingPipeline<B::kTmemSlots> tmem_pipe;
  auto instr_desc = cute::UMMA::make_instr_desc<
      cutlass::float_e4m3_t,
      cutlass::float_e4m3_t,
      float,
      128,
      kUmmaN,
      cute::UMMA::Major::K,
      cute::UMMA::Major::K>();
  auto runtime_instr_desc = cute::UMMA::make_runtime_instr_desc(instr_desc);
  for (uint32_t k = 0;; ++k) {
    ClaimHeader e;
    if (!consume_claim(k, claim_full, claim_empty, entries, e)) break;
    if (e.all_short) continue;
    const uint32_t splits = e.splits, samples = e.samples;
    CUTE_TIE_DECL(q_pipe.advance(), q_stage, q_phase);
    (void)q_stage;
    full_q.wait(q_phase);
    full_lin->wait(q_phase);  // this block's linear-term rows (warp 11)
    ptx::tcgen05_after_thread_sync();
    const uint32_t total = samples + splits;
    // One elected lane issues the whole block (elect.sync picks the same lane
    // every block, so that lane alone carries the ring state), and the next
    // tile's KV stage is awaited before the tile's last MMA position, so a
    // tile's first MMA follows the previous tile's last directly. A per-tile
    // elect or stage wait leaves a ~200-cycle tensor-core bubble per tile with
    // the 3-slot TMEM ring (MMA-only ~1350 vs ~1155 cycles per tile; the
    // 4 x M128N144K128 bound is 1152).
    if (cute::elect_one_sync() && total != 0) {
      // A descriptors = stage-0 descriptor + (stage, half, k) start-address
      // offsets (start field = smem address >> 4, 14 bits; the stages lie
      // below 256 KB, so the adds never carry out of the field)
      const uint64_t a_desc0 =
          deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, 128>(sm.kv[0], 0, 0);
      const uint32_t a_lo0 = static_cast<uint32_t>(a_desc0), a_hi = static_cast<uint32_t>(a_desc0 >> 32);
      CUTE_TIE_DECL(kv_pipe.advance(), s0, ph0);
      uint32_t s_next = s0;
      full_kv[s0].wait(ph0);
#pragma unroll 1
      for (uint32_t t = 0; t < total; ++t) {
        const uint32_t s = s_next;
        // Per KV half, one N=144 MMA per math warpgroup into its own TMEM
        // slot (ring of 3; WG0 takes even, WG1 odd positions of the issue
        // sequence).
#pragma unroll
        for (uint32_t i = 0; i < 2; ++i) {
#pragma unroll
          for (uint32_t g = 0; g < 2; ++g) {
            if (i == 1 && g == 1 && t + 1 < total) {
              CUTE_TIE_DECL(kv_pipe.advance(), s1, ph1);
              s_next = s1;
              full_kv[s1].wait(ph1);
            }
            CUTE_TIE_DECL(tmem_pipe.advance(), ts_idx, ts_ph);
            const uint32_t tmem_addr = ts_idx * kSlotCols;
            empty_tmem[ts_idx].wait(ts_ph ^ 1);
            ptx::tcgen05_after_thread_sync();
            cutlass::float_e4m3_t* b_base =
                g == 0 ? reinterpret_cast<cutlass::float_e4m3_t*>(sm.lin_lo) : sm.q + kQBytes / 2;
            const uint32_t a_lo = a_lo0 + s * (kKVBytes >> 4) + i * ((kKVBytes / 2) >> 4);
#pragma unroll
            for (uint32_t kk = 0; kk < 4; ++kk) {
              const uint64_t a_desc = (uint64_t(a_hi) << 32) | (a_lo + kk * (32 >> 4));
              auto b_desc =
                  deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, kHeadDim, 128>(b_base, 0, kk * 32);
              ptx::SM100_MMA_F8F6F4_SS::fma(a_desc, b_desc, tmem_addr, kk, runtime_instr_desc);
            }
            ptx::umma_arrive_no_elect(full_tmem[ts_idx]);
          }
        }
      }
    }
    __syncwarp();
    empty_q.arrive();
  }
} else if (warp_idx == 11) {
  // ---- warp 11: linear-term rows, drain of math WG0's rows ----
  cutlass::arch::warpgroup_reg_dealloc<kSpecRegs>();
  uint32_t drain_epoch = 0, n_lin = 0;
  for (uint32_t k = 0;; ++k) {
    ClaimHeader e;
    if (!consume_claim(k, claim_full, claim_empty, entries, e)) break;
    if (e.all_short) continue;
    const uint32_t vblock = e.vblock, kv_base = e.kv_base, splits = e.splits, slot = e.slot;
    // Q of block k lands while the math warps still select block k-1, so
    // this is off the critical path except for a CTA's first block.
    full_q.wait(n_lin & 1);
    ++n_lin;
    compute_lin_rows(sm, lane);
    // The generic shared-memory writes before the UMMA (async proxy) reads.
    cutlass::arch::fence_view_async_shared();
    full_lin->arrive();  // every lane arrives (count 32)
    const uint32_t row_base = vblock * kBlockRows;
    const int64_t row_delta = int64_t(slot) * kBlockRows - int64_t(row_base);
    for (uint32_t base = 0; base <= splits; base += Interval, ++drain_epoch)
      drain_interval(
          sdata,
          h_full,
          h_empty,
          0,
          drain_epoch,
          row_base,
          row_count,
          kCandGlobalCap,
          base + Interval > splits,
          vals,
          indices,
          row_delta,
          kv_base);
    cutlass::arch::NamedBarrier(kDrainThreads, kBarDrain).arrive();
  }
} else {
  // ---- warps 0-7: math ----
  cutlass::arch::warpgroup_reg_alloc<kMathRegs>();
  const uint32_t math_tid = threadIdx.x;
  const uint32_t query_base = (warp_idx / 4) * 4;
  RingPipeline<1> q_pipe;
  RingPipeline<kKVStages> kv_pipe;
  RingPipeline<B::kTmemSlots> tmem_pipe;
  // Each warpgroup consumes every second TMEM slot position (WG1 starts at 1).
  constexpr uint32_t kTmemStep = 2;
  if (warp_idx >= 4) tmem_pipe.advance(1);
  const uint32_t heads_col = warp_idx < 4 ? 16u : 0u;
  const uint32_t lin_col = warp_idx < 4 ? 0u : 128u;
  // Shared addresses of the barriers and of the scale stages, held in
  // registers (see opaque_ld_u32).
  const uint32_t bar_a = opaque_ld_u32(&opaque_bases[0]);
  const uint32_t sfkv_a = opaque_ld_u32(&opaque_bases[1]);
  // Lane id and lanemask_lt of the emission, held in registers (ptxas
  // otherwise re-reads SR_LANEID / SR_LTMASK in every emission).
  const uint32_t lane_o = opaque_ld_u32(&lane_tab[0][lane]);
  const uint32_t lt_o = opaque_ld_u32(&lane_tab[1][lane]);
  // The staging area base (records / list ends / bases).
  const uint32_t stage_a = opaque_ld_u32(&opaque_bases[2]);
  uint32_t produce_epoch = 0;
  for (uint32_t k = 0;; ++k) {
    claim_full[k & 1].wait((k >> 1) & 1);
    const ClaimEntry& e = entries[k & 1];
    const uint32_t vblock = e.vblock;
    if (vblock >= kPause) break;
    const uint32_t row_base = vblock * kBlockRows;
    if (e.all_short) {
      for (uint32_t r = 0; r < kBlockRows && row_base + r < rows; ++r) {
        const uint32_t vis = e.ends[r] - e.starts[r];
        const int32_t first = e.out_base[r];
        for (uint32_t j = math_tid; j < topk; j += kMathThreads)
          fallback.output[(uint64_t)(row_base + r) * topk + j] = j < vis ? first + int32_t(j) : -1;
      }
      if (paged.table) {
        cutlass::arch::NamedBarrier(kMathThreads, kBarSelect).sync();
        if (row_base + warp_idx < rows) finish_row(paged, fallback.output, row_base + warp_idx, topk, 0, lane);
      }
      __syncwarp();
      claim_empty[k & 1].arrive();
      continue;
    }
    // ---- block setup: Q, weights, sample histogram ----
    {
      CUTE_TIE_DECL(q_pipe.advance(), q_stage, q_phase);
      (void)q_stage;
      full_q.wait(q_phase);
      full_lin->wait(q_phase);
    }
    // Zero the histograms and references (the stage needs no init).
    static_assert(offsetof(SampleShared, stage) % 16 == 0);
    for (uint32_t j = math_tid; j < offsetof(SampleShared, stage) / 16; j += kMathThreads)
      reinterpret_cast<uint4*>(&stream)[j] = make_uint4(0, 0, 0, 0);
    // This block's per-row candidate counters (shared memory; the drains and
    // self-drains of the warpgroup's rows add to them). Zeroed by the
    // warpgroup's first warp, whose h_full arrivals order it before the
    // drain's first atomic.
    if (warp_idx % 4 == 0 && lane < 4) row_count[query_base + lane] = 0;
    float weights[4][32];
    float row_s[4];  // pow2 row scale of the linear term
    uint32_t seq_k_start[4], seq_k_end[4];
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      seq_k_start[i] = e.starts[query_base + i];
      seq_k_end[i] = e.ends[query_base + i];
      // Weights in units of 1 / s_r (exact: s_r is a power of two).
      row_s[i] = sm.lin_scale[query_base + i];
      const float inv_s = 1.0f / row_s[i];
#pragma unroll
      for (uint32_t j = 0; j < 32; j += 4) {
        const float4 w4 = *reinterpret_cast<const float4*>(sm.weights + (query_base + i) * kNumHeads + j);
        weights[i][j] = w4.x * 0.5f * inv_s;
        weights[i][j + 1] = w4.y * 0.5f * inv_s;
        weights[i][j + 2] = w4.z * 0.5f * inv_s;
        weights[i][j + 3] = w4.w * 0.5f * inv_s;
      }
    }
    {
      const float w = sm.weights[warp_idx * 32 + lane];
      bool bad_w = w != 0.0f && fabsf(w) < 0x1p-125f;
      // Scaled weights that become subnormal or infinite take the exact
      // recovery path.
      const float wr = w * 0.5f * (1.0f / sm.lin_scale[warp_idx]);
      bad_w |= (wr != 0.0f && fabsf(wr) < 0x1p-125f) || isinf(wr);
      const uint32_t bad = __ballot_sync(kFullMask, bad_w);
      if (lane == 0 && bad) atomicOr(&unsafe_s, 1u << warp_idx);
    }
    // Sample reference of each gated row: the maximum of its valid samples in
    // the first sample tile holding any of its keys (byte i of ref_tiles: that
    // tile of row query_base + i; 255: no sample in view or the row is not
    // gated, reference 0). The samples of that tile are staged in shared memory
    // and binned once the warpgroup's maximum is known; later tiles bin
    // directly.
    uint32_t ref_tiles = 0;
    float ref[4];
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      const uint32_t st = seq_k_start[i], en = seq_k_end[i], c0 = sample_stride / 2;
      const uint32_t g0 = st > c0 ? (st - c0 + sample_stride - 1) / sample_stride : 0u;
      const bool has = en - st > sample_target && g0 < sample_groups && g0 * sample_stride + c0 < en;
      ref_tiles |= (has ? min(g0 / kTileKeys, 254u) : 255u) << (8 * i);
      ref[i] = 0.0f;
    }
    math_sync();
    // ---- sample tiles ----
    // One sample tile. A tile that sets the reference of the rows in ref_rows
    // stages its samples and takes their maximum instead of binning them,
    // decided at run time (one loop body: the scan loop keeps its register
    // allocation).
    uint32_t ref_max[4];  // ordered codes
    const auto sample_tile = [&](const uint32_t t, const uint32_t ref_rows) {
      // The math warps read only the scales of the KV stage: it is awaited
      // after half 0's first TMEM loads, as in scan_tile.
      CUTE_TIE_DECL(kv_pipe.advance(), kv_stage, kv_phase);
#pragma unroll
      for (uint32_t kv_half = 0; kv_half < 2; ++kv_half) {
        const uint32_t kv_lane = (math_tid % 128) + kv_half * 128;
        const uint32_t group = t * kTileKeys + kv_lane;
        const uint32_t kv_offset = group * sample_stride + sample_stride / 2;
        const bool in_view = group < sample_groups;
        CUTE_TIE_DECL(tmem_pipe.advance(kTmemStep), tmem_stage, tmem_phase);
        full_tmem[tmem_stage].wait(tmem_phase);
        ptx::tcgen05_after_thread_sync();
        // One query row (32 TMEM columns) at a time keeps the sample loop's
        // live registers low enough to carry the scan weights through it.
        // Row i + 1 is loaded (into the same registers) as soon as row i's
        // sums are formed, so its TMEM latency overlaps row i's binning.
        float acc_n[32], lin_n[4];
        ptx::tmem_load_32dp32b<32>(tmem_stage * kSlotCols + heads_col, reinterpret_cast<uint32_t*>(acc_n));
        ptx::tmem_load_32dp32b<4>(tmem_stage * kSlotCols + lin_col, reinterpret_cast<uint32_t*>(lin_n));
        // Half 0 awaits the stage, each half reads its scales (0 outside the
        // view), half 1 then releases the stage.
        if (kv_half == 0) full_kv[kv_stage].wait(kv_phase);
        const float scale_kv = ptx::ld_shared(sm.sf_kv[kv_stage] + kv_lane);
        if (kv_half == 1) {
          cutlass::arch::fence_view_async_shared();
          empty_kv[kv_stage].arrive();
        }
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
          float acc[32];
          float lin[4];
          cutlass::arch::fence_view_async_tmem_load();
#pragma unroll
          for (uint32_t c = 0; c < 32; ++c)
            acc[c] = acc_n[c];
#pragma unroll
          for (uint32_t c = 0; c < 4; ++c)
            lin[c] = lin_n[c];
          if (i == 3) {
            ptx::tcgen05_before_thread_sync();
            empty_tmem[tmem_stage].arrive();
          }
          // terms 2, 3 are in units of s * 2^-8
          float2 sum_0 = make_float2(lin[0], lin[1]),
                 sum_1 = __fmul2_rn(make_float2(lin[2], lin[3]), make_float2(0x1p-8f, 0x1p-8f));
#pragma unroll
          for (uint32_t j = 0; j < 32; j += 4) {
            auto transform = [&](uint32_t at, float2 sum) {
              const auto abs_x = make_float2(fabsf(acc[at]), fabsf(acc[at + 1]));
              const auto w = make_float2(weights[i][at], weights[i][at + 1]);
              // relu = (x + |x|) / 2, linear part on the TC
              return __ffma2_rn(abs_x, w, sum);
            };
            sum_0 = transform(j, sum_0);
            sum_1 = transform(j + 2, sum_1);
          }
          const auto sum = __fadd2_rn(sum_0, sum_1);
          if (i < 3) {
            ptx::tmem_load_32dp32b<32>(
                tmem_stage * kSlotCols + heads_col + (i + 1) * 32, reinterpret_cast<uint32_t*>(acc_n));
            ptx::tmem_load_32dp32b<4>(
                tmem_stage * kSlotCols + lin_col + (i + 1) * 4, reinterpret_cast<uint32_t*>(lin_n));
          }
          const float bq = (sum.x + sum.y) * (row_s[i] * scale_kv);
          const bool valid = in_view && kv_offset >= seq_k_start[i] && kv_offset < seq_k_end[i] && isfinite(bq);
          if (ref_rows) {
            stream.stage[query_base + i][kv_lane] = valid ? bq : CUDART_NAN_F;
            if (valid && (ref_rows >> (8 * i) & 1u)) ref_max[i] = max(ref_max[i], ordered_code(bq));
          } else if (valid) {
            sample_add(stream.hist[query_base + i], half_bin<kSampleBinShift>(bq - ref[i]));
          }
        }
      }
    };
#pragma unroll 1
    for (uint32_t t = 0; t < e.samples; ++t) {
      // Warpgroup-uniform: this tile sets the reference of some of its rows.
      const uint32_t ref_rows = __vcmpeq4(ref_tiles, t * 0x01010101u);
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i)
        ref_max[i] = 0u;
      sample_tile(t, ref_rows);
      if (!ref_rows) continue;
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        if (!(ref_rows >> (8 * i) & 1u)) continue;
        const uint32_t m = __reduce_max_sync(kFullMask, ref_max[i]);
        if (lane == 0 && m) atomicMax(stream.ref + query_base + i, m);
      }
      // Warpgroup barrier (the static form: no runtime id assertion).
      cutlass::arch::NamedBarrier::sync(kMathWgThreads, kBarWarpgroup + warp_idx / 4);
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        if (ref_rows >> (8 * i) & 1u) ref[i] = sample_ref_value(stream.ref[query_base + i]);
#pragma unroll
        for (uint32_t h = 0; h < 2; ++h) {
          const float x = stream.stage[query_base + i][math_tid % 128 + h * 128];
          if (!isnan(x)) sample_add(stream.hist[query_base + i], half_bin<kSampleBinShift>(x - ref[i]));
        }
      }
    }
    math_sync();
    finish_sample_histogram(
        stream.hist[warp_idx],
        gate_s + warp_idx,
        e.ends[warp_idx] - e.starts[warp_idx],
        sample_target,
        sample_ref_value(stream.ref[warp_idx]),
        lane);
    math_sync();
    const uint32_t row_base2 = volatile_load(e.vblock) * kBlockRows;

    // ---- scan tiles ----
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      const float inv_s = 1.0f / row_s[i];
#pragma unroll
      for (uint32_t j = 0; j < 32; j += 4) {
        const float4 w4 = *reinterpret_cast<const float4*>(sm.weights + (query_base + i) * kNumHeads + j);
        weights[i][j] = w4.x * 0.5f * inv_s;
        weights[i][j + 1] = w4.y * 0.5f * inv_s;
        weights[i][j + 2] = w4.z * 0.5f * inv_s;
        weights[i][j + 3] = w4.w * 0.5f * inv_s;
      }
    }
    const uint32_t kv_base = e.kv_base, splits = e.splits;
    const int64_t cand_row_delta = int64_t(e.slot) * kBlockRows - int64_t(row_base2);
    float gate_score[4];
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      gate_score[i] = gate_s[query_base + i];
      if (volatile_load(unsafe_s) >> (query_base + i) & 1u) {
        gate_score[i] = __int_as_float(0x7f800000);
        if (math_tid % 128 == 0) atomicAdd(&row_count[query_base + i], kCandGlobalCap + 1u);
      }
      // Compare S * scale_kv against gate / row_s (row_s is a power of two ->
      // exact).
      gate_score[i] *= 1.0f / row_s[i];
    }
    // Tiles [vis_lo, vis_lo + vis_span] are fully visible to all 4 rows of the
    // warpgroup, so the emission tests the per-key bounds only on the other
    // (edge) tiles. vis_lo = ~0u: no such tile.
    uint32_t vis_lo = ~0u, vis_span = 0;
    {
      uint32_t lo = 0, hi = ~0u;
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        const uint32_t st = e.starts[query_base + i], en = e.ends[query_base + i];
        // Short rows (padding rows past the end included) never pass their
        // +inf gate, so only the other rows bound the fully visible tiles.
        const bool live = en - st > topk;
        lo = max(lo, !live ? 0u : st < kv_base ? ~0u : (st - kv_base + kTileKeys - 1) / kTileKeys);
        hi = min(hi, !live ? ~0u : (en - kv_base) / kTileKeys);
      }
      if (hi > lo) {
        vis_lo = lo;
        vis_span = hi - 1u - lo;
      }
    }
    // Warp-uniform: per row the shared address of the next free record of
    // this warp's staging list in the current bank.
    uint32_t stage_ptr[4];
    const auto stage_row = [&](uint32_t bank, uint32_t i) {
      return stage_a + ((bank * kMathWarps + warp_idx) * 4u + i) * (kStageCap * 4u);
    };
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i)
      stage_ptr[i] = stage_row(produce_epoch & 1, i);
    uint32_t flush_base = 0;
    // Half 1's TMEM slot was already full when half 0 tested it.
    bool slot_ready = false;
    // One scan tile; edge: per-key bounds in the emission.
    const auto scan_tile = [&](const uint32_t kv_split_idx, auto edge) {
      if ((kv_split_idx & (Interval - 1)) == 0) {
        mbar_wait_addr(
            bar_a + (B::h_empty + (produce_epoch & 1) * kMathWarps + warp_idx) * 8u, ((produce_epoch >> 1) & 1) ^ 1);
        __syncwarp();
      }
      // The KV stage holds this tile's scales; the math warps await it after
      // half 0's first TMEM loads (the tile's MMAs already waited for it).
      CUTE_TIE_DECL(kv_pipe.advance(), kv_stage, kv_phase);
#pragma unroll
      for (uint32_t kv_half = 0; kv_half < 2; ++kv_half) {
        const uint32_t kv_lane = (math_tid % 128) + kv_half * 128;
        const uint32_t kv_offset = kv_base + kv_split_idx * kTileKeys + kv_lane;
        float scale_kv = 0.0f;  // loaded after the first TMEM loads
        CUTE_TIE_DECL(tmem_pipe.advance(kTmemStep), tmem_stage, tmem_phase);
        // The slot's TMEM addresses are formed before the probe: the empty asm
        // consumes them first.
        const uint32_t heads_a = tmem_stage * kSlotCols + heads_col, lin_a = tmem_stage * kSlotCols + lin_col;
        asm volatile("" ::"r"(heads_a), "r"(lin_a));
        if (!(kv_half == 1 && slot_ready)) mbar_wait_addr(bar_a + (B::full_tmem + tmem_stage) * 8u, tmem_phase);
        ptx::tcgen05_after_thread_sync();
        float accum_pair[2][32];
        float lin_pair[2][4];
        float bp4[4];  // row scores of the slot
        // Rows 0-1 are voted as soon as their scores are formed (the vote at
        // the end of the half then waits only on rows 2-3).
        bool vote01 = false;
#pragma unroll
        for (uint32_t pair = 0; pair < 4; pair += 2) {
          // Half 1 releases the KV stage (KV + scales) after its scale load.
          if (kv_half == 1 && pair == 2) {
            cutlass::arch::fence_view_async_shared();
            mbar_arrive_addr(bar_a + (B::empty_kv + kv_stage) * 8u);
          }
          ptx::tmem_load_32dp32b<64>(heads_a + pair * 32, reinterpret_cast<uint32_t*>(accum_pair));
          ptx::tmem_load_32dp32b<8>(lin_a + pair * 4, reinterpret_cast<uint32_t*>(lin_pair));
          // Half 0 tests (non-blocking, acquire on success) whether half 1's
          // slot is already full; half 1 then skips its full_tmem probe.
          if (kv_half == 0 && pair == 2) {
            RingPipeline<B::kTmemSlots> next = tmem_pipe;
            CUTE_TIE_DECL(next.advance(kTmemStep), ns, nph);
            slot_ready = mbar_test_addr(bar_a + (B::full_tmem + ns) * 8u, nph);
          }
          // The scale load (half 0: after the full_kv wait) follows the first
          // TMEM loads; a shared load issued ahead of the full_tmem probe
          // delays the probe.
          if (pair == 0) {
            if (kv_half == 0) mbar_wait_addr(bar_a + (B::full_kv + kv_stage) * 8u, kv_phase);
            scale_kv = ld_shared_f32_addr(sfkv_a + kv_stage * kScaleBytes + kv_lane * 4u);
          }
          cutlass::arch::fence_view_async_tmem_load();
          if (pair + 2 >= 4) {
            ptx::tcgen05_before_thread_sync();
            mbar_arrive_addr(bar_a + (B::empty_tmem + tmem_stage) * 8u);
          }
          // The linear term (sum_t term_t . k, in units of the row scale) seeds
          // the sums.
          float2 sum_0[2], sum_1[2];
#pragma unroll
          for (uint32_t r = 0; r < 2; ++r) {
            sum_0[r] = make_float2(lin_pair[r][0], lin_pair[r][1]);
            // terms 2, 3 are in units of s * 2^-8
            sum_1[r] = __fmul2_rn(make_float2(lin_pair[r][2], lin_pair[r][3]), make_float2(0x1p-8f, 0x1p-8f));
          }
#pragma unroll
          for (uint32_t j = 0; j < 32; j += 4) {
#pragma unroll
            for (uint32_t r = 0; r < 2; ++r) {
              const uint32_t i = pair + r;
              auto* a = accum_pair[r];
              auto transform = [&](uint32_t at, float2 sum) {
                const auto abs_x = make_float2(fabsf(a[at]), fabsf(a[at + 1]));
                const auto w = make_float2(weights[i][at], weights[i][at + 1]);
                return __ffma2_rn(abs_x, w, sum);
              };
              sum_0[r] = transform(j, sum_0[r]);
              sum_1[r] = transform(j + 2, sum_1[r]);
            }
          }
#pragma unroll
          for (uint32_t r = 0; r < 2; ++r) {
            const auto sum = __fadd2_rn(sum_0[r], sum_1[r]);
            bp4[pair + r] = (sum.x + sum.y) * scale_kv;
          }
          if (pair == 0) vote01 = __any_sync(kFullMask, (bp4[0] > gate_score[0]) | (bp4[1] > gate_score[1]));
        }
        // One warp vote per row pair: most keys fail every gate, so the common
        // path is 4 compares + 2 votes + 1 uniform branch.
        const bool any23 = (bp4[2] > gate_score[2]) | (bp4[3] > gate_score[3]);
        if (vote01 | __any_sync(kFullMask, any23)) [[unlikely]] {
          // Ballot-compacted emission: per row the passing lanes append
          // their records to this warp's staging list of the row with one
          // predicated STS each (per-key bounds only on edge tiles).
          const uint32_t rel = ((kv_split_idx - flush_base) * 2 + kv_half) * 32 + lane_o;
          const uint32_t lt = lt_o;
          const auto emit = [&](auto bounded) {
#pragma unroll
            for (uint32_t i = 0; i < 4; ++i) {
              const float bq = bp4[i] * row_s[i];
              bool p = bp4[i] > gate_score[i] && isfinite(bq);
              if constexpr (decltype(bounded)::value)
                p = p && kv_offset >= seq_k_start[i] && kv_offset < seq_k_end[i];
              const uint32_t m = __ballot_sync(kFullMask, p);
              // Record: raw score bits 8-31 | rel; the drains form the
              // ordered code (staged_code).
              st_shared_if(p, stage_ptr[i] + __popc(m & lt) * 4u, (__float_as_uint(bq) & 0xffffff00u) | rel);
              stage_ptr[i] += __popc(m) * 4u;
            }
          };
          emit(edge);
        }
      }
      if (((kv_split_idx + 1) & (Interval - 1)) == 0) {
        if (lane_o == 0) {
          // The lists' end addresses (the drain subtracts the list bases).
          constexpr uint32_t kCntOff = offsetof(StageLists, cnt), kBasesOff = offsetof(StageLists, bases);
          const uint32_t bw = (produce_epoch & 1) * kMathWarps + warp_idx;
          asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(stage_a + kCntOff + bw * 16u),
                       "r"(stage_ptr[0]),
                       "r"(stage_ptr[1]),
                       "r"(stage_ptr[2]),
                       "r"(stage_ptr[3])
                       : "memory");
          asm volatile("st.shared.u32 [%0], %1;" ::"r"(stage_a + kBasesOff + bw * 4u), "r"(flush_base) : "memory");
        }
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i)
          stage_ptr[i] = stage_row((produce_epoch + 1) & 1, i);
        __syncwarp();
        mbar_arrive_addr(bar_a + (B::h_full + (produce_epoch & 1) * kMathWarps + warp_idx) * 8u);
        ++produce_epoch;
        flush_base = kv_split_idx + 1;
      }
    };
    {
      // The tiles [full_beg, full_end) are fully visible for all 4 rows of this
      // warpgroup and run without per-key bounds.
      const uint32_t full_beg = min(vis_lo, splits),
                     full_end = vis_lo == ~0u ? splits : min(vis_lo + vis_span + 1u, splits);
      uint32_t t = 0;
#pragma unroll 1
      for (; t < full_beg; ++t)
        scan_tile(t, std::true_type{});
#pragma unroll 1
      for (uint32_t n = full_end - t; n != 0; --n, ++t)
        scan_tile(t, std::false_type{});
#pragma unroll 1
      for (; t < splits; ++t)
        scan_tile(t, std::true_type{});
    }
    {
      // Final interval: every math warp stores its own staged records (8 warps
      // in parallel instead of 2 drain warps after the scan); the drain warps
      // only cycle the barriers.
      const uint32_t bank = produce_epoch & 1, rbase = row_base2 + query_base;
      if ((splits & (Interval - 1)) == 0) {  // this interval has no tile yet
        h_empty[bank * kMathWarps + warp_idx].wait(((produce_epoch >> 1) & 1) ^ 1);
        __syncwarp();
      }
      self_drain_last(
          sdata.recs[bank][warp_idx][0],
          (stage_ptr[0] - stage_row(bank, 0)) / 4u,
          (stage_ptr[1] - stage_row(bank, 1)) / 4u,
          (stage_ptr[2] - stage_row(bank, 2)) / 4u,
          (stage_ptr[3] - stage_row(bank, 3)) / 4u,
          row_count + query_base,
          rbase,
          kv_base + (warp_idx % 4) * 32,
          flush_base,
          cand_row_delta,
          vals,
          indices);
      __syncwarp();
      h_full[bank * kMathWarps + warp_idx].arrive();
      ++produce_epoch;
    }
    cutlass::arch::fence_view_async_shared();
    empty_q.arrive();
    cutlass::arch::NamedBarrier(kDrainThreads, kBarDrain).sync();  // drains of this block done
    // ---- selection ----
    const uint32_t vblock3 = volatile_load(e.vblock), row_base3 = vblock3 * kBlockRows;
    __threadfence_block();
    cutlass::arch::NamedBarrier(kMathThreads, kBarSelect).sync();
    const ClaimEntry& ce = entries[k & 1];
    if (math_tid == 0) {
      uint32_t recover = 0, short_rows = 0;
      for (uint32_t r = 0; r < kBlockRows; ++r) {
        const bool is_short = ce.ends[r] - ce.starts[r] <= topk;
        short_rows |= uint32_t(is_short) << r;
        const int32_t c = int32_t(row_count[r]);
        // Publish the row's candidate count (the kernel itself reads the
        // shared counter).
        counts[row_base3 + r] = c;
        recover |= !is_short && (c < int32_t(topk) || c > int32_t(kCandidatesPerRow));
      }
      if (recover) {
        defer_next[vblock3] = int32_t(defer_head);
        defer_head = vblock3;
        atomicExch(&pause_req, 1u);
      }
      sel_recover = recover;
      sel_short = short_rows;
      unsafe_s = 0;
    }
    cutlass::arch::NamedBarrier(kMathThreads, kBarSelect).sync();
    const uint32_t recover = sel_recover, short_rows = sel_short;
    const auto write_short = [&](uint32_t r, uint32_t t, uint32_t stride) {
      if (r >= rows) return;
      const uint32_t visible = ce.ends[r - row_base3] - ce.starts[r - row_base3];
      const int32_t first = ce.out_base[r - row_base3];
      int32_t* o = fallback.output + (uint64_t)r * topk;
      for (uint32_t j = t; j < topk; j += stride)
        o[j] = j < visible ? first + int32_t(j) : -1;
    };
    if (!recover) {
      const int64_t row_delta = int64_t(ce.slot) * kBlockRows - int64_t(row_base3);
      // One row per warp, read straight from the global slab (L2).
      const uint32_t local = warp_idx;
      uint32_t* scratch = reinterpret_cast<uint32_t*>(smem_buffer + kSelBase) + warp_idx * row_select::kStreamWords;
      const uint32_t r = row_base3 + local;
      const uint64_t base = uint64_t(int64_t(r) + row_delta) * kCandidatesPerRow;
      if (short_rows >> local & 1u)
        write_short(r, lane, 32);
      else {
        const uint32_t cnt = row_count[local];
        int32_t* o = fallback.output + (uint64_t)r * topk;
        // bracket miss: once more with a wide margin, then the exact
        // levels straight from global memory
        if (!row_select::select_stream_row(scratch, cnt, vals + base, indices + base, o, topk, false)) {
          if (!row_select::select_stream_row(scratch, cnt, vals + base, indices + base, o, topk, true)) {
            row_select::select_global_row(cnt, vals + base, indices + base, o, topk, scratch);
          }
        }
      }
      cutlass::arch::NamedBarrier(kMathThreads, kBarSelect).sync();
    }
    cutlass::arch::NamedBarrier(kMathThreads, kBarSelect).sync();
    // The selection scratch (KV stages) is free: the next block's pipeline
    // starts while the rows are rewritten to relative indices (selected rows
    // hold absolute key indices; short rows are final unless paged).
    __syncwarp();
    if (lane == 0) sel_done.arrive();
    if (relative && !recover && row_base3 + warp_idx < rows) {
      const bool is_short = short_rows >> warp_idx & 1u;
      finish_row(
          paged,
          fallback.output,
          row_base3 + warp_idx,
          topk,
          is_short ? 0 : ce.out_base[warp_idx] - int32_t(ce.starts[warp_idx]),
          lane);
    }
    __syncwarp();
    claim_empty[k & 1].arrive();
  }
}

// ---- exact rescan of the blocks the selection queued ----
__syncthreads();
if (warp_idx == 0) cute::TMEM::Allocator1Sm().free(0, 512);
if (defer_head != kTerm) {
  __syncthreads();
  for (uint32_t cur = defer_head; cur != kTerm;) {
    const uint32_t vblock = cur, row_base = vblock * kBlockRows;
    if (threadIdx.x < kBlockRows) {
      const uint32_t r = row_base + threadIdx.x;
      row_bounds[0][threadIdx.x] = r < rows ? min(starts_g[r], keys) : 0u;
      row_bounds[1][threadIdx.x] = r < rows ? max(min(ends_g[r], keys), row_bounds[0][threadIdx.x]) : 0u;
    }
    if (threadIdx.x == 0) fslot = 2 * blockIdx.x;  // no block is in flight during the rescan
    __syncthreads();
    const uint32_t* const starts = row_bounds[0] - row_base;
    const uint32_t* const ends = row_bounds[1] - row_base;
    const int64_t row_delta = int64_t(fslot) * kBlockRows - int64_t(row_base);
    RescanBuffers fb = fallback;
    fb.retained += row_delta * kRetainedPerRow;
    fb.indices += row_delta * kRetainedPerRow;
    rescan::rescan_block(dense::block_scheduler(vblock, rows, keys, starts, ends), fb, topk, tq, tk, ts, tw);
    __syncthreads();
    if (threadIdx.x == 0) {
      auto& fst = *reinterpret_cast<dense::SharedStorage*>(smem_buffer);
      for (auto& bb : fst.full_q_barriers)
        mbar_inval(&bb);
      for (auto& bb : fst.empty_q_barriers)
        mbar_inval(&bb);
      for (auto& bb : fst.full_kv_barriers)
        mbar_inval(&bb);
      for (auto& bb : fst.empty_kv_barriers)
        mbar_inval(&bb);
      for (auto& bb : fst.full_tmem_barriers)
        mbar_inval(&bb);
      for (auto& bb : fst.empty_tmem_barriers)
        mbar_inval(&bb);
    }
    __syncthreads();
    // The rescan wrote absolute key indices for every row, short ones too.
    if (relative && warp_idx < kMathWarps && row_base + warp_idx < rows) {
      const uint32_t r = row_base + warp_idx;
      finish_row(
          paged,
          fallback.output,
          r,
          topk,
          (out_offsets ? out_offsets[r] : 0) - int32_t(row_bounds[0][warp_idx]),
          lane);
    }
    cur = uint32_t(__ldcg(defer_next + vblock));
    __syncthreads();
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    defer_head = kTerm;
    pause_req = 0;
  }
}  // deferred blocks
more = epoch_kind != kTerm;
__syncthreads();
