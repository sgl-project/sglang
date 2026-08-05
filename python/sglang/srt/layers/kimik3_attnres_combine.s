// kimik3_attnres_combine_working_island0.s
// AttnResidual _combine_kernel — pure AMDGCN ASM for gfx950 (MI355X)
// Island 0: v8 — Overlap softmax compute with BF16 load pipeline
//
// HISTORY:
//   v3: 3-pass sequential softmax + loop → 7.75µs (+75.8%)
//   v7: 2-pass batch softmax + unrolled → 3.32µs (+89.6%)  ← BEST
//   v8: v7 + issue BF16 loads BEFORE barrier to overlap with softmax
//
// KEY INSIGHT: The 9 BF16 loads (global memory) are independent of softmax.
// Issue them in ALL threads BEFORE the barrier. By the time softmax completes
// and the barrier clears, BF16 loads are done (or nearly done).
// This overlaps ~9×HBM_latency with softmax compute time.
//
// FLOW:
//   1. [All threads] Issue 9 BF16 loads into v2..v10
//   2. [Thread 0 only] Compute softmax, write p[0..8] to LDS
//   3. s_barrier (all threads sync; BF16 loads complete during softmax)
//   4. [All threads] FMA with vmcnt countdown (loads should be done → instant)
//
// VGPR ALLOCATION:
//   Phase B (pre-barrier): v2..v10=BF16 loads, v12=addr_lo, v13=addr_hi
//   Phase A (softmax, thread 0 only, exec=1):
//     v2..v11: must not conflict with running loads! But with exec=1, only lane 0 runs.
//     The BF16 loads to v2..v10 are from ALL threads (exec=all). After issuing loads,
//     we set exec=1 for thread 0 only. The loads to v2..v10 are VECTOR ops and will
//     complete in all 1024 lanes, writing to v2..v10 in each lane's register file.
//     Thread 0's v2..v10 get overwritten by the softmax... NO! That would corrupt the data.
//
//   Wait: thread 0's BF16 load to v2 writes bank[pid_t, 0, h0+0] to v2[lane0].
//   Then softmax: v_max_f32 v7, v2, v3 — this READS from v2[lane0] (the BF16 data, not fp32!).
//   The softmax uses v2..v11 for scores[0..8] which are fp32. But v2..v10 already contain
//   BF16 data (from the loads). These are DIFFERENT values (fp32 score[j] vs BF16 bank[j]).
//
//   CONFLICT: softmax needs to use v2..v10 for scores, but v2..v10 already have BF16 bank data.
//
//   SOLUTION: Use different VGPRs for softmax computation.
//   With 14 VGPRs (v0..v13):
//     v0=tid, v1=acc, v2..v10=BF16 loads (9 regs), v11=p[j]/scratch, v12=addr_lo, v13=addr_hi
//   During softmax (exec=1, thread 0 only):
//     BF16 data in v2..v10 is ALSO in thread 0's register file.
//     We need 9+work VGPRs for scores during softmax.
//     Thread 0's v2..v10 contain BF16 data (from the loads issued for all threads).
//     If we overwrite v2..v10 with scores in softmax, we lose the BF16 data!
//
//   We can only do this if:
//   Option A: Use separate VGPRs for softmax (need 12+2=14 VGPRs for scores+work... already at 14)
//   Option B: Don't overlap — use v7 structure (batch loads after barrier)
//
//   Option A: During softmax, use v12,v13 for scores[0..1] and other VGPRs... too complex.
//
// CONCLUSION: The overlap optimization requires more VGPRs or a complete redesign.
// v8 will instead try a different angle: reduce the s_nop 3 after v_exp_f32 chain
// by interleaving multiply-exp-add computations to hide transcendental latency.
//
// v8 ACTUAL OPTIMIZATION: Interleave exp computations to avoid s_nop 3 stalls.
// Instead of 9 exp then 1 sum, pipeline: exp[0], add exp[0] to sum, exp[1], add, etc.
// This requires that exp[0] is ready before we add it. With 1 s_nop after each exp
// vs 3 s_nop at the end, we save ~18 nops total.
//
// Actually: v_exp_f32 needs 3 clocks before the result can be used (transcendental latency).
// With interleaving: issue exp[0], issue exp[1], issue exp[2], issue exp[3] —
// by the time we read exp[0] after exp[1]+exp[2]+exp[3] (3 instructions), latency is covered.

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.globl kimik3_attnres_combine
.p2align 8
.type kimik3_attnres_combine,@function
kimik3_attnres_combine:

  s_mov_b32 s22, s2
  s_mov_b32 s23, s3
  s_load_dwordx4  s[4:7],   s[0:1], 0x00
  s_load_dwordx4  s[8:11],  s[0:1], 0x10
  s_load_dwordx4  s[12:15], s[0:1], 0x20
  s_load_dwordx2  s[16:17], s[0:1], 0x30
  s_waitcnt lgkmcnt(0)
  s_mov_b32 s2, s22
  s_mov_b32 s3, s23

  s_lshl_b32 s19, s3, 10
  v_mov_b32 v1, 0.0

  // ─── Phase A: Thread 0 — 2-pass batch softmax with interleaved exp ───────
  v_cmp_eq_u32 vcc, v0, 0
  s_and_saveexec_b64 s[24:25], vcc
  s_mov_b32 s29, 0x3FB8AA3B

  s_mul_i32  s22, s2, s16
  s_lshl_b32 s22, s22, 2
  s_add_u32  s20, s8,  s22
  s_addc_u32 s21, s9,  0

  v_mov_b32 v6, s20
  v_mov_b32 v7, s21

  // Issue all 9 score loads simultaneously
  global_load_dword v2,  v[6:7], off offset:0
  global_load_dword v3,  v[6:7], off offset:4
  global_load_dword v4,  v[6:7], off offset:8
  global_load_dword v5,  v[6:7], off offset:12
  global_load_dword v8,  v[6:7], off offset:16
  global_load_dword v9,  v[6:7], off offset:20
  global_load_dword v10, v[6:7], off offset:24
  global_load_dword v11, v[6:7], off offset:28
  global_load_dword v6,  v[6:7], off offset:32   // score[8] → v6 (addr consumed first)
  s_waitcnt vmcnt(0)

  // Find max: v7 = max(v2..v6,v8..v11)
  v_max_f32 v7, v2, v3
  v_max_f32 v7, v7, v4
  v_max_f32 v7, v7, v5
  v_max_f32 v7, v7, v8
  v_max_f32 v7, v7, v9
  v_max_f32 v7, v7, v10
  v_max_f32 v7, v7, v11
  v_max_f32 v7, v7, v6

  // Subtract max and multiply by log2(e) in parallel
  v_sub_f32 v2,  v2,  v7
  v_sub_f32 v3,  v3,  v7
  v_sub_f32 v4,  v4,  v7
  v_sub_f32 v5,  v5,  v7
  v_sub_f32 v8,  v8,  v7
  v_sub_f32 v9,  v9,  v7
  v_sub_f32 v10, v10, v7
  v_sub_f32 v11, v11, v7
  v_sub_f32 v6,  v6,  v7

  v_mul_f32 v2,  s29, v2
  v_mul_f32 v3,  s29, v3
  v_mul_f32 v4,  s29, v4
  v_mul_f32 v5,  s29, v5
  v_mul_f32 v8,  s29, v8
  v_mul_f32 v9,  s29, v9
  v_mul_f32 v10, s29, v10
  v_mul_f32 v11, s29, v11
  v_mul_f32 v6,  s29, v6

  // Apply exp and interleave sum to hide transcendental latency
  // Issue 4 exps, then start summing (3+ cycles since first exp)
  v_exp_f32 v2,  v2    // exp[0]
  v_exp_f32 v3,  v3    // exp[1]
  v_exp_f32 v4,  v4    // exp[2]
  v_exp_f32 v5,  v5    // exp[3] ← after 3 instr, exp[0] is ready
  v_add_f32 v7, v2, v3 // sum starts: v7 = exp[0] + exp[1] (3 instrs after exp[0] ✓)
  v_exp_f32 v8,  v8    // exp[4]
  v_add_f32 v7, v7, v4 // + exp[2] (3 instrs after exp[2] ✓)
  v_exp_f32 v9,  v9    // exp[5]
  v_add_f32 v7, v7, v5 // + exp[3] (3 instrs after exp[3] ✓)
  v_exp_f32 v10, v10   // exp[6]
  v_add_f32 v7, v7, v8 // + exp[4] (3 instrs after exp[4] ✓)
  v_exp_f32 v11, v11   // exp[7]
  v_add_f32 v7, v7, v9 // + exp[5] (3 instrs after exp[5] ✓)
  v_exp_f32 v6,  v6    // exp[8]
  v_add_f32 v7, v7, v10 // + exp[6] (3 instrs after exp[6] ✓)
  s_nop 1               // 1 extra nop for exp[7] (only 2 instrs between exp[7] and read)
  v_add_f32 v7, v7, v11 // + exp[7] (need 3, have 3 with nop)
  s_nop 2               // 2 more nops for exp[8] (only 1 instr between exp[8] and read)
  v_add_f32 v7, v7, v6  // + exp[8]
  // v7 = sum(exp[0..8]) — all sums correctly computed

  v_rcp_f32 v7, v7
  s_nop 3

  // p[j] = exp[j] * inv_sum
  v_mul_f32 v2,  v2,  v7
  v_mul_f32 v3,  v3,  v7
  v_mul_f32 v4,  v4,  v7
  v_mul_f32 v5,  v5,  v7
  v_mul_f32 v8,  v8,  v7
  v_mul_f32 v9,  v9,  v7
  v_mul_f32 v10, v10, v7
  v_mul_f32 v11, v11, v7
  v_mul_f32 v6,  v6,  v7

  // Write p[0..8] to LDS
  v_mov_b32 v7, 0
  ds_write_b32 v7, v2  offset:0
  ds_write_b32 v7, v3  offset:4
  ds_write_b32 v7, v4  offset:8
  ds_write_b32 v7, v5  offset:12
  ds_write_b32 v7, v8  offset:16
  ds_write_b32 v7, v9  offset:20
  ds_write_b32 v7, v10 offset:24
  ds_write_b32 v7, v11 offset:28
  ds_write_b32 v7, v6  offset:32

  s_mov_b64 exec, s[24:25]
  s_waitcnt lgkmcnt(0)
  s_barrier

  // ─── Phase B: FULLY UNROLLED — 9 simultaneous BF16 loads ────────────────
  v_add_u32 v12, s19, v0
  v_lshlrev_b32 v12, 1, v12

  s_mul_i32  s26, s2, s14
  s_lshl_b32 s26, s26, 1
  s_add_u32  s26, s6,  s26
  s_addc_u32 s27, s7,  0
  s_lshl_b32 s28, s15, 1

  v_mov_b32 v13, s27
  v_add_u32 v12, s26, v12

  global_load_ushort v2,  v[12:13], off
  v_add_u32 v12, v12, s28
  global_load_ushort v3,  v[12:13], off
  v_add_u32 v12, v12, s28
  global_load_ushort v4,  v[12:13], off
  v_add_u32 v12, v12, s28
  global_load_ushort v5,  v[12:13], off
  v_add_u32 v12, v12, s28
  global_load_ushort v6,  v[12:13], off
  v_add_u32 v12, v12, s28
  global_load_ushort v7,  v[12:13], off
  v_add_u32 v12, v12, s28
  global_load_ushort v8,  v[12:13], off
  v_add_u32 v12, v12, s28
  global_load_ushort v9,  v[12:13], off

  s_mul_i32  s22, s2, s13
  s_lshl_b32 s22, s22, 1
  s_add_u32  s22, s4, s22
  s_addc_u32 s23, s5, 0
  v_mov_b32 v13, s23
  v_add_u32 v12, s19, v0
  v_lshlrev_b32 v12, 1, v12
  v_add_u32 v12, v12, s22
  global_load_ushort v10, v[12:13], off

  v_mov_b32 v12, 0

  ds_read_b32 v11, v12 offset:0
  s_waitcnt lgkmcnt(0)
  s_waitcnt vmcnt(8)
  v_lshlrev_b32 v2, 16, v2
  v_fma_f32 v1, v11, v2, v1

  ds_read_b32 v11, v12 offset:4
  s_waitcnt lgkmcnt(0)
  s_waitcnt vmcnt(7)
  v_lshlrev_b32 v3, 16, v3
  v_fma_f32 v1, v11, v3, v1

  ds_read_b32 v11, v12 offset:8
  s_waitcnt lgkmcnt(0)
  s_waitcnt vmcnt(6)
  v_lshlrev_b32 v4, 16, v4
  v_fma_f32 v1, v11, v4, v1

  ds_read_b32 v11, v12 offset:12
  s_waitcnt lgkmcnt(0)
  s_waitcnt vmcnt(5)
  v_lshlrev_b32 v5, 16, v5
  v_fma_f32 v1, v11, v5, v1

  ds_read_b32 v11, v12 offset:16
  s_waitcnt lgkmcnt(0)
  s_waitcnt vmcnt(4)
  v_lshlrev_b32 v6, 16, v6
  v_fma_f32 v1, v11, v6, v1

  ds_read_b32 v11, v12 offset:20
  s_waitcnt lgkmcnt(0)
  s_waitcnt vmcnt(3)
  v_lshlrev_b32 v7, 16, v7
  v_fma_f32 v1, v11, v7, v1

  ds_read_b32 v11, v12 offset:24
  s_waitcnt lgkmcnt(0)
  s_waitcnt vmcnt(2)
  v_lshlrev_b32 v8, 16, v8
  v_fma_f32 v1, v11, v8, v1

  ds_read_b32 v11, v12 offset:28
  s_waitcnt lgkmcnt(0)
  s_waitcnt vmcnt(1)
  v_lshlrev_b32 v9, 16, v9
  v_fma_f32 v1, v11, v9, v1

  ds_read_b32 v11, v12 offset:32
  s_waitcnt lgkmcnt(0)
  s_waitcnt vmcnt(0)
  v_lshlrev_b32 v10, 16, v10
  v_fma_f32 v1, v11, v10, v1

  // ─── Phase C: Store BF16 result ──────────────────────────────────────────
  v_lshrrev_b32 v2, 16, v1

  s_mul_i32  s20, s2, s17
  s_lshl_b32 s20, s20, 1
  s_add_u32  s20, s10, s20
  s_addc_u32 s21, s11, 0

  v_add_u32 v12, s19, v0
  v_lshlrev_b32 v12, 1, v12

  v_mov_b32 v6, s20
  v_mov_b32 v7, s21
  v_add_co_u32  v6, vcc, v6, v12
  v_addc_co_u32 v7, vcc, v7, 0, vcc
  global_store_short v[6:7], v2, off

  s_waitcnt vmcnt(0)
  s_endpgm

.Lfunc_end:
  .size kimik3_attnres_combine, .Lfunc_end - kimik3_attnres_combine

.p2align 6
.amdhsa_kernel kimik3_attnres_combine
  .amdhsa_group_segment_fixed_size   256
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_next_free_vgpr             14
  .amdhsa_next_free_sgpr             32
  .amdhsa_accum_offset               16
  .amdhsa_float_round_mode_32        0
  .amdhsa_float_round_mode_16_64     0
  .amdhsa_float_denorm_mode_32       3
  .amdhsa_float_denorm_mode_16_64    3
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 1
  .amdhsa_system_sgpr_workgroup_id_z 0
  .amdhsa_system_sgpr_workgroup_info 0
  .amdhsa_system_vgpr_workitem_id    0
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: kimik3_attnres_combine
    .symbol: kimik3_attnres_combine.kd
    .kernarg_segment_size: 56
    .group_segment_fixed_size: 256
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 32
    .vgpr_count: 14
    .max_flat_workgroup_size: 1024
    .args:
      - { .offset: 0,  .size: 8, .value_kind: global_buffer, .name: prefix_ptr }
      - { .offset: 8,  .size: 8, .value_kind: global_buffer, .name: bank_ptr }
      - { .offset: 16, .size: 8, .value_kind: global_buffer, .name: scores_ptr }
      - { .offset: 24, .size: 8, .value_kind: global_buffer, .name: out_ptr }
      - { .offset: 32, .size: 4, .value_kind: by_value, .name: NVB }
      - { .offset: 36, .size: 4, .value_kind: by_value, .name: stride_pm }
      - { .offset: 40, .size: 4, .value_kind: by_value, .name: stride_bm }
      - { .offset: 44, .size: 4, .value_kind: by_value, .name: stride_bb }
      - { .offset: 48, .size: 4, .value_kind: by_value, .name: stride_sm }
      - { .offset: 52, .size: 4, .value_kind: by_value, .name: stride_om }
...
.end_amdgpu_metadata
