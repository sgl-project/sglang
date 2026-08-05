// kimik3_attnres_score_working_island0.s
// AttnResidual _score_kernel — pure AMDGCN ASM for gfx950 (MI355X)
// Island 0: v8d — DPP interleaved + 8×ds_read_b128 + 16-slot LDS
// STATUS: CORRECT snr=135.7dB, 5.54µs (+84.0% vs Triton 34.7µs baseline)
//
// ALGORITHM:
//   1. 7-iteration main loop: each thread loads 1 BF16 + 1 FP32 cw per H-chunk
//   2. Interleaved DPP prefix scan (sumsq+dotv simultaneous): wave-reduce 64→1
//      s_nop 1 between same-VGPR DPP passes; interleaving hides most NOPs
//   3. Lane 63 of each wave writes wave sums to LDS (16 slots × 4B × 2 = 128B)
//   4. Barrier + thread 0 reads 16 wave sums via 4×ds_read_b128 per accumulator
//   5. Thread 0 computes rrms (with s_nop 3 after transcendentals) + score + store
//
// KEY HARDWARE BUGS FIXED:
//   A. v0 = HW work-item ID X (0..1023). v_mbcnt gives per-wave lane (0..63).
//   B. v_rcp_f32/v_rsq_f32 need s_nop 3 before reading result.
//   C. DPP row_shr/row_bcast need s_nop 1 (or interleaving) between same-VGPR ops.
//
// SGPR: s[0:1]=kernarg, s2=pid_t, s3=j, s[4:5]=prefix_ptr, s[6:7]=bank_ptr,
//       s[8:9]=cw_ptr, s[10:11]=scores_ptr, s12=NVB, s14=stride_pm,
//       s15=stride_bm, s16=stride_bb, s17=stride_sm, s18-s21=scratch
// VGPR: v0=workitem_id, v1=sumsq_acc, v2=dotv_acc, v3=BF16→FP32, v4=cw_FP32,
//       v5=tid*2, v6=addr_lo, v7=addr_hi, v8=DPP_sumsq, v9=DPP_dotv,
//       v10=wave_id*4/LDS_addr, v11=lane_in_wave, v4..v7=ds_read_b128_dst
// LDS:  [0..63]=sumsq_wave[0..15], [64..127]=dotv_wave[0..15]

.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
.text
.globl kimik3_attnres_score
.p2align 8
.type kimik3_attnres_score,@function
kimik3_attnres_score:

  s_mov_b32 s18, s2
  s_mov_b32 s19, s3
  s_load_dwordx4  s[4:7],   s[0:1], 0x00
  s_load_dwordx4  s[8:11],  s[0:1], 0x10
  s_load_dwordx4  s[12:15], s[0:1], 0x20
  s_load_dwordx2  s[16:17], s[0:1], 0x30
  s_waitcnt lgkmcnt(0)
  s_mov_b32 s2, s18
  s_mov_b32 s3, s19

  s_cmp_gt_u32 s3, s12
  s_cbranch_scc1 .Lexit

  v_mov_b32 v1, 0
  v_mov_b32 v2, 0

  s_cmp_lt_u32 s3, s12
  s_cbranch_scc0 .Luse_prefix

.Luse_bank:
  s_mul_i32  s20, s2, s15
  s_mul_i32  s19, s3, s16
  s_add_u32  s20, s20, s19
  s_lshl_b32 s20, s20, 1
  s_add_u32  s20, s6, s20
  s_addc_u32 s21, s7, 0
  s_branch .Lptr_ready

.Luse_prefix:
  s_mul_i32  s20, s2, s14
  s_lshl_b32 s20, s20, 1
  s_add_u32  s20, s4, s20
  s_addc_u32 s21, s5, 0

.Lptr_ready:
  v_lshlrev_b32 v5, 1, v0
  s_mov_b32 s18, 0

.Lmain_loop:
  s_lshl_b32 s19, s18, 1
  v_mov_b32 v6, s20
  v_mov_b32 v7, s21
  v_add_co_u32  v6, vcc, v6, v5
  v_addc_co_u32 v7, vcc, v7, 0, vcc
  v_add_co_u32  v6, vcc, v6, s19
  v_addc_co_u32 v7, vcc, v7, 0, vcc
  global_load_ushort v3, v[6:7], off

  s_lshl_b32 s19, s18, 2
  v_lshlrev_b32 v4, 2, v0
  v_mov_b32 v6, s8
  v_mov_b32 v7, s9
  v_add_co_u32  v6, vcc, v6, v4
  v_addc_co_u32 v7, vcc, v7, 0, vcc
  v_add_co_u32  v6, vcc, v6, s19
  v_addc_co_u32 v7, vcc, v7, 0, vcc
  global_load_dword v4, v[6:7], off
  s_waitcnt vmcnt(0)

  v_lshlrev_b32 v3, 16, v3
  v_fma_f32 v1, v3, v3, v1
  v_fma_f32 v2, v3, v4, v2

  s_add_u32 s18, s18, 1024
  s_cmp_lt_u32 s18, 7168
  s_cbranch_scc1 .Lmain_loop

  // ── Interleaved DPP prefix scan (sumsq v8, dotv v9) ───────────────────────
  // Interleaving eliminates most s_nop 1 requirements between DPP passes
  v_add_f32 v8, v1, v1 row_shr:1    row_mask:0xf bank_mask:0xf bound_ctrl:1
  v_add_f32 v9, v2, v2 row_shr:1    row_mask:0xf bank_mask:0xf bound_ctrl:1
  v_add_f32 v8, v8, v8 row_shr:2    row_mask:0xf bank_mask:0xf bound_ctrl:1
  v_add_f32 v9, v9, v9 row_shr:2    row_mask:0xf bank_mask:0xf bound_ctrl:1
  v_add_f32 v8, v8, v8 row_shr:4    row_mask:0xf bank_mask:0xf bound_ctrl:1
  v_add_f32 v9, v9, v9 row_shr:4    row_mask:0xf bank_mask:0xf bound_ctrl:1
  v_add_f32 v8, v8, v8 row_shr:8    row_mask:0xf bank_mask:0xf bound_ctrl:1
  v_add_f32 v9, v9, v9 row_shr:8    row_mask:0xf bank_mask:0xf bound_ctrl:1
  v_add_f32 v8, v8, v8 row_bcast:15 row_mask:0xa bank_mask:0xf bound_ctrl:0
  v_add_f32 v9, v9, v9 row_bcast:15 row_mask:0xa bank_mask:0xf bound_ctrl:0
  v_add_f32 v8, v8, v8 row_bcast:31 row_mask:0xc bank_mask:0xf bound_ctrl:0
  s_nop 1                              // one NOP needed for final v8 before v9 reads v9
  v_add_f32 v9, v9, v9 row_bcast:31 row_mask:0xc bank_mask:0xf bound_ctrl:0
  // v8[63] = wave sumsq total, v9[63] = wave dotv total (per wave)

  // ── LDS write: lane 63 of each wave writes wave sums ─────────────────────
  s_nop 1                              // settle DPP chain
  v_lshrrev_b32 v10, 6, v0
  v_and_b32     v11, 63, v0

  v_cmp_eq_u32 vcc, v11, 63
  s_and_saveexec_b64 s[18:19], vcc

  v_lshlrev_b32 v10, 2, v10
  ds_write_b32 v10, v8                 // sumsq[wave_id]
  ds_write_b32 v10, v9 offset:64      // dotv[wave_id]

  s_mov_b64 exec, s[18:19]
  s_waitcnt lgkmcnt(0)
  s_barrier

  // ── Thread 0 reads 16 wave sums via ds_read_b128 ─────────────────────────
  v_cmp_eq_u32 vcc, v0, 0
  s_and_saveexec_b64 s[18:19], vcc

  v_mov_b32 v10, 0
  v_mov_b32 v1, 0

  ds_read_b128 v[4:7], v10 offset:0   // sumsq[0..3]
  s_waitcnt lgkmcnt(0)
  v_add_f32 v1, v1, v4
  v_add_f32 v1, v1, v5
  v_add_f32 v1, v1, v6
  v_add_f32 v1, v1, v7

  ds_read_b128 v[4:7], v10 offset:16  // sumsq[4..7]
  s_waitcnt lgkmcnt(0)
  v_add_f32 v1, v1, v4
  v_add_f32 v1, v1, v5
  v_add_f32 v1, v1, v6
  v_add_f32 v1, v1, v7

  ds_read_b128 v[4:7], v10 offset:32  // sumsq[8..11]
  s_waitcnt lgkmcnt(0)
  v_add_f32 v1, v1, v4
  v_add_f32 v1, v1, v5
  v_add_f32 v1, v1, v6
  v_add_f32 v1, v1, v7

  ds_read_b128 v[4:7], v10 offset:48  // sumsq[12..15]
  s_waitcnt lgkmcnt(0)
  v_add_f32 v1, v1, v4
  v_add_f32 v1, v1, v5
  v_add_f32 v1, v1, v6
  v_add_f32 v1, v1, v7

  v_mov_b32 v2, 0

  ds_read_b128 v[4:7], v10 offset:64  // dotv[0..3]
  s_waitcnt lgkmcnt(0)
  v_add_f32 v2, v2, v4
  v_add_f32 v2, v2, v5
  v_add_f32 v2, v2, v6
  v_add_f32 v2, v2, v7

  ds_read_b128 v[4:7], v10 offset:80  // dotv[4..7]
  s_waitcnt lgkmcnt(0)
  v_add_f32 v2, v2, v4
  v_add_f32 v2, v2, v5
  v_add_f32 v2, v2, v6
  v_add_f32 v2, v2, v7

  ds_read_b128 v[4:7], v10 offset:96  // dotv[8..11]
  s_waitcnt lgkmcnt(0)
  v_add_f32 v2, v2, v4
  v_add_f32 v2, v2, v5
  v_add_f32 v2, v2, v6
  v_add_f32 v2, v2, v7

  ds_read_b128 v[4:7], v10 offset:112 // dotv[12..15]
  s_waitcnt lgkmcnt(0)
  v_add_f32 v2, v2, v4
  v_add_f32 v2, v2, v5
  v_add_f32 v2, v2, v6
  v_add_f32 v2, v2, v7

  // Compute rrms = rsqrt(sumsq/H + eps)
  v_mov_b32 v4, 0x45E00000             // 7168.0
  v_rcp_f32 v4, v4
  s_nop 3                              // transcendental hazard
  v_mul_f32 v1, v1, v4

  v_mov_b32 v4, 0x358637BD             // 1e-6
  v_add_f32 v1, v1, v4

  v_rsq_f32 v1, v1
  s_nop 3                              // transcendental hazard

  v_mul_f32 v2, v2, v1               // score = dotv * rrms

  // Store scores[pid_t, j]
  s_mul_i32  s20, s2, s17
  s_add_u32  s20, s20, s3
  s_lshl_b32 s20, s20, 2

  v_mov_b32 v6, s10
  v_mov_b32 v7, s11
  v_add_co_u32  v6, vcc, v6, s20
  v_addc_co_u32 v7, vcc, v7, 0, vcc
  global_store_dword v[6:7], v2, off

  s_mov_b64 exec, s[18:19]

.Lexit:
  s_waitcnt vmcnt(0)
  s_endpgm

.Lfunc_end:
  .size kimik3_attnres_score, .Lfunc_end - kimik3_attnres_score

.p2align 6
.amdhsa_kernel kimik3_attnres_score
  .amdhsa_group_segment_fixed_size   128
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_next_free_vgpr             16
  .amdhsa_next_free_sgpr             24
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
  - .name: kimik3_attnres_score
    .symbol: kimik3_attnres_score.kd
    .kernarg_segment_size: 64
    .group_segment_fixed_size: 128
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 64
    .sgpr_count: 24
    .vgpr_count: 16
    .max_flat_workgroup_size: 1024
    .args:
      - { .offset: 0,  .size: 8, .value_kind: global_buffer, .name: prefix_ptr }
      - { .offset: 8,  .size: 8, .value_kind: global_buffer, .name: bank_ptr }
      - { .offset: 16, .size: 8, .value_kind: global_buffer, .name: cw_ptr }
      - { .offset: 24, .size: 8, .value_kind: global_buffer, .name: scores_ptr }
      - { .offset: 32, .size: 4, .value_kind: by_value, .name: NVB }
      - { .offset: 36, .size: 4, .value_kind: by_value, .name: eps }
      - { .offset: 40, .size: 4, .value_kind: by_value, .name: stride_pm }
      - { .offset: 44, .size: 4, .value_kind: by_value, .name: stride_bm }
      - { .offset: 48, .size: 4, .value_kind: by_value, .name: stride_bb }
      - { .offset: 52, .size: 4, .value_kind: by_value, .name: stride_sm }
...
.end_amdgpu_metadata
