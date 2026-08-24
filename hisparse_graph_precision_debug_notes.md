# NPU Selective HiSparse 图模式精度问题 — 定位结论与 dump 方法

> P/D 分离 + Selective HiSparse（19 个 selected 层 KV 驻留 Host DRAM）部署 GLM-5.2，D 节点 decode 开启 NPU graph 后精度随 selected 层数线性劣化（7 层 0.90-0.96 → 19 层 0.74-0.82；eager 锚点 0.94）。
>
> **状态（2026-08-29）**：精度问题结案——根因为 RoPE cos/sin 图捕获冻结（根因 3），修复后 19 层 50 题 0.90；根因 1/2（队列类型、ACL 订阅）为早期 0.7→0.9 的真实修复；completion-flag 协议三步退役完成，每步复验 0.90。二次战役发现独立于 hi-sparse 的 draft 链图内 NaN：张量别名链式污染已修，其余定案为 `npu_kv_quant_sparse_flash_attention` 图内 replay 异常，证据包齐、升级 CANN/算子侧。两模式精度无系统差（0.88~0.96 噪声带）。全套 diff-dump 探针与比对工具已删除（eager24/graph24 dump 文件仍在机器上），本文只保留定位结论与 dump 方法论；完整过程记录见 git 历史。

## 1. 背景与运行时 KV 数据流

```
anchor 层(L-3) topk → build_loc_plan: topk 位置 → Host pool 逻辑 loc
→ H2D: sparse_copy 预取 topk KV 到 HBM packed_staging                     [MIX DMA]
→ selected 层(L) set_kv_buffer → publish_new_packed_kv:
    scratch.copy_(packed_kv) + D2H: sparse_copy 备份新 KV 到 Host DRAM    [MIX DMA]
→ run_selected_attention: current-patch 覆盖 staging current 行
    → unpack 656B→BF16 → SFA 稀疏注意力                                    [计算算子]
```

关键文件（`python/sglang/srt/hardware_backend/npu/` 下）：`selective_hisparse.py`（coordinator）、`modules/deepseek_v2_attention_mla_npu.py`（**根因 3 所在，已修**）、`graph_runner/npu_graph_runner.py`、`memory_pool_npu.py`；memfabric `src/acc_offload/csrc/operators/acc_offload_sparse_copy.*`。

硬件/调度事实：`KERNEL_TYPE_AIV_ONLY` 走独立向量核队列，图 replay 与计算队列**并行推进、无完成排序**；`KERNEL_TYPE_MIX_AIV_1_0` 走计算核队列，与计算算子 **FIFO 串行**。AIV kernel 内部 MTE 异步完成依赖 ACL host 回调（`acl.rt.subscribe_report` 按流订阅）。**NPU capture 流程**（`npu_cudagraph_backend.capture_one`）：对**同一个 forward_batch** 先跑 2 次 warmup 再正式 capture——根因 3 的触发条件。

## 2. 定位结论

### 2.1 原始症状的三个独立根因（均已修复，0.74-0.82 → 0.90）

| # | 根因 | 修复 | 效果 |
|---|---|---|---|
| 1 | `sparse_copy` 为 AIV_ONLY：图 replay 时 copy 在向量队列、patch/SFA 在计算队列并行推进 → SFA 读到未落地的 DMA 数据 | copy/wait_flag 改 `MIX_AIV_1_0`（与消费者同队列 FIFO） | 0.70→0.78 |
| 2 | ACL 回调只订阅 capture 流，replay 在主流执行从未订阅 → 图内 sparse_copy 的 MTE 完成报告无人处理 | `prepare_graph_replay` 无条件订阅当前流（幂等；flag 协议退役后保留，plain copy 的 DMA 完成仍依赖它） | 0.78→0.90 |
| 3 | **最终根因：RoPE cos/sin 图捕获冻结**。`_apply_dsa_interleave_half_rope`（MLAPO/DSA 路径）把 cos/sin latch 到 `forward_batch.npu_dsa_interleave_half_rope_cache` 供层间复用；capture 前 2 次 warmup 共用同一 forward_batch → 首次 warmup 用**假 positions** 算出 cos/sin 并 latch → capture 走缓存分支，`index_select(positions)` 链**从未入图** → 每次 replay positions 正确但 cos/sin 永远是 warmup 角度 → q_pe/k_pe 错角度旋转，每个 selected 层每步把错角度 k_rope 写入 KV/host 池 → **毒按层数线性累积**（~1.25pt/层，7-9 层阈值） | 捕获期 `torch.npu.is_current_stream_capturing()` 为真则强制重算 cos/sin（计算链入图；eager latch 语义不变），`deepseek_v2_attention_mla_npu.py` 一处改动覆盖所有图 | 19 层 **0.90** |

根因 3 定位指纹（与机制一一对应）：kin（k_nope 不旋转）位级一致 vs kinv（k_rope 过 RoPE）差 65%、pos buffer 一致（毒在 RoPE 读 positions 的方式而非 buffer 本身）、eager 完全干净（每步新 ForwardBatch 重算）、所有同步机制 no-op。修复后验证：warmup dump 全字段位级一致（out max rel diff 2.1e-07）；19 层 50 题 0.90，剩余 4pt 为 graph/eager 常规数值差（7 层时代即有同款 gap）。

**通用教训**：任何「首次计算后 latch 到 forward_batch 供后续层复用」的 positions 依赖缓存，在 NPU capture（warmup×2 + capture 共用同一 forward_batch）下都会把计算链挡在图外。排查清单：`npu_dsa_interleave_half_rope_cache`（已修）、`npu_mlaprolog_runtime_cache`（LongCat，layer0 无条件刷新，语义正确）、`rotary_emb.sin_cos_cache`（start_layer 条件刷新，同型风险，建议复核）。

### 2.2 精度无系统性差距

原始症状"graph 0.90 vs eager 0.94"不是系统差：50 题单次散布 ±3pt，全部跑法落在 0.88~0.96 交叠带；且已证 target 前向对相同输入逐位一致、accept_len=1 时提交 token 由 target argmax 决定 → 两模式文本序列应一致，分数波动为运行间噪声（含 eager 自身核级非确定性）。**精度复核需 200+ 题**。

### 2.3 二次战役定案：draft 链图内 NaN（与 hi-sparse 无关，性能 bug 非精度 bug）

效果 = 垃圾提案 → 接受率塌 1 → spec 加速失效；贪心精度由 target argmax 决定，不受害。

- **已修复的真根因（张量别名链式污染）**：`_init_cuda_graph_metadata` 的 `metadata.seq_lens = seq_lens` 别名 draft runner 静态 buffer，四个 per-step 后端共享同一 forward_batch → replay 前逐后端刷新（读 buffer、加本步偏移、写回）变成链式累加，末步 15 被全体继承 → 图内 attention 全读 15（正确 6/7/8/9）→ 越界读未初始化 KV → NaN。**修复**：`seq_lens.clone()`（每后端独立 buffer），附带消除对 runner 输入 buffer 的原地污染。**证据**：`am_kvlen` 修复后两侧逐位一致 [6,7,8,9]。
- **定案（Round-24，升级 CANN/算子侧）**：`npu_kv_quant_sparse_flash_attention`（fp8 DSA，`quant_scale_repo_mode=1`；调用点 `ascend_backend.py::forward_sparse`，NEXTN draft 层 self-attention，被 eagle draft NPU graph 捕获）在**链步 0 全部显式输入位级全同下图内 replay 输出 NaN**：

  | 输入（captured-op 探针） | eager | graph |
  |---|---|---|
  | am_q（q_nope） | 587.8189697265625 | 同 |
  | am_qpe（q_rope） | 459.941650390625 | 同 |
  | am_tik（sparse_indices） | -2033 | 同 |
  | am_kvlen（actual_seq_lengths_kv） | 6 | 同 |
  | am_bt（block_table） | 1 | 同 |
  | am_pgsum（kernel 经页表实际读到的 KV 页字节和） | 628294 | 同 |

  输出：eager `attn_raw` = -136.17（正常量级），graph = **NaN@[0,1,2,3] 全部四链步**，跨 round-23/24 确定性复现。
  **证据边界（引用本结论必须附带）**：①「输入一致」只在链步 0 严格成立；链步 1-3 的 am_pgsum 发散是 step0 NaN 的下游级联（NaN hidden → quant scale 异常 → 量化 KV 写 0），非独立证据。②graph 链步 1-3 的 am_q/am_qpe 读出精确 0 而其输入为 NaN——图内 per-step 探针存储伪影，不得作为输入证据。③am_seqlens graph[6,7,8,9] vs eager[5,5,5,5] 是元数据语义差（graph 把 per-step offset 写进 seq_lens 设备张量，eager 只进 cpu_int）；kernel 实际消费 kvlen（两侧一致），无涉本结论。
- **归属**：NaN 路径（quant kernel 调用、`quant_scale_repo_mode=1`、`draft_replay_pack_npu`、`eagle_draft_npu_graph_runner`）全部由 08-15 `94daad003b` / 08-19 `1aad5706b9`（A5 优化）引入；hi-sparse 主提交仅 36 行 selected 层路由。hi-sparse 的 diff-dump 第一次深入 draft 链才让这个潜伏 bug 显形。可选运行时铁证：`SELECTIVE_LAYER_IDS=""` 关 hisparse 后 NaN 仍在即终证（开关保留在启动脚本）。
- **临时缓解方向**：该 kernel 仅 DSA 草稿链使用 → 实验 draft 链 eager attention 或 non-quant kernel 路径，先兑现 spec 加速。

### 2.4 定位步骤合并（早期 + 二次战役全部轮次浓缩）

**早期（0.7→0.9→结案）**：eager 基线正常确认问题图模式专属 → staging checksum 两侧吻合排除 DMA 落地、指向时序 → 根因 1/2 修复（0.70→0.78→0.90）→ 层数扫描确认 7-9 层阈值 + 线性 1.25pt/层指纹 → R5-R13 及 B/C/E/C8 共 15 轮排除法（per-layer staging/scratch/loc-plan、ping-pong、删 D2H/H2D wait、MTE3_S 回退、notify tail、plain copy、spec 维度）全部波段内 no-op，证明 python 可及杠杆均非毒源 → **D1 内容级 diff 逐字段二分**：q/locs/stg_pre 排除上游 → pub==pkg 排除 stale scratch → eager 跨 run 位级一致排除量化非确定性 → kin 一致 vs kinv 差 65% 锁定 RoPE → pos 匹配锁定"毒在 RoPE 读 positions 的方式" → latch 机制实锤 → 捕获期强制重算修复、19 层 0.90 结案 → 收尾三步（D2H plain / H2D plain+删 wait / memfabric 删 notify+wait_flag 全链 14 处）每步复验 0.90，最终形态 = 纯 MIX plain `sparse_copy` + 计算队列 FIFO + ACL 流订阅。

**二次战役（draft 链 NaN，R1-R24）**：diff-dump 显示 selected 层全字段吻合但 verify round 3 起发散 → logits 指纹锁定首发散 target 前向、输入 token id 偏 → root/draft in_ids 对比：root 吻合（eagle_sample 无辜）、draft 提案退化为 0 → NaN 归属表：graph 侧 draft 全 NaN、eager 全净 → clone 实验洗净 hidden 链但 NaN 仍在 → 毒在 draft 前向内部 → dloc/dkvh/dseql 净（写址、KV 历史、seq_lens 无辜）→ 子块探针经 mark_step 显式索引修正后首次可信：attn@0 NaN、输入全净 → 毒在 draft self-attention → graph kvlen=常数 15 → 修 actual_seq_lengths_kv 刷新未命中 → DBG-META 证明 input==written → 别名实锤 → R14 clone 修复，am_kvlen 修正为 6/7/8/9 但 attn 仍 NaN → R15 attn_raw 直捕确认 NaN 诞生于 kernel 内部 → R16 q_pe/sparse_indices 一致 → R17/R23/R24 补齐最后盲区 am_pgsum → 六项全同仍 NaN，定案升级 CANN。探针插曲（教训见 §3.2）：0-dim 探针被 NPU auto-dispatch 捕获拒绝（改 `[:1,:1]` 全程 ≥1-D）；round-22 参照 dm 异常为一次性坏运行伪影，R24 干净参照复现 -136 后排除；一次误将 eager24 与自身比对得出的「NaN 非确定」结论已撤回。

## 3. dump 方法与判读（工具链已删除，方法论存档）

> 2026-08-29 全套 diff-dump 探针（`SGLANG_SELECTIVE_DIFF_DUMP/DUMP_DIR/DUMP_MAX_STEPS/DUMP_READBACK`、coordinator `_dbg_*` 缓冲、am_*/attn_raw 探针、`DBG-META`、`D_EAGER`、`hisparse_diff_compare.py`）已随定案删除；以下内容供未来任何精度回归重建同类工具时参考。

### 3.1 开关与产物

- 总开关 `SGLANG_SELECTIVE_DIFF_DUMP=1`；`DUMP_DIR`；`DUMP_MAX_STEPS`（默认 6，首发散在 step2-3 够用）；`DUMP_READBACK`（Host pool 回读，回读+同步是采集时间大头，默认关）。
- 产物三类：selected 层逐字段快照（locs/valid/stg_pre/stg_post/q/qrope/out/pkg/crow/allv/pub/kin/kinv/pos/d2h_locs/unpk/unpkr + 标量）；target 前向二分 + draft 链全套（in_ids/in_pos/hidden/din_*/dout_*/dstep_*/dext_out/dloc/dkvh/dseql/dm_*）；accept 指纹（accept_lens + logit_argmax/rowsum）。
- 比对工具输出段顺序：verify-round fingerprint（NaN 归属表、handoff 链）→ target-forward bisect → per-pair divergence → first divergence。

### 3.2 采集与判读规则（方法论）

1. **warmup 即出 dump**：不发请求、不跑精度——起两个服务，dummy 请求驱动的 verify 轮自动产生全部 dump（采集成本由崩塌式下降确立）。
2. `MAX_STEPS=6` 足够：首个发散轮总在 step2-3，其后全是级联。
3. 两侧必须同一 prompt/请求流，否则 round 1 即假性发散。
4. dump 开关对精度无损（隔离矩阵实证）。
5. **captured op 探针是图内取证唯一可信手段**：host 侧读数可能读到 replay 后已自愈/已改写的值，只有图内 captured op 冻结消费时刻的值。
6. 探针切片索引必须用业务真值（链步循环变量），不要用 host 计数器——基线在 eager 与 capture 间会错位。
7. 行宽/桶填充差异是聚合探针头号伪影源：跨模式比对只比"真实行"；整行页表求和会把 padding 计入。
8. NaN 类问题先做**归属表**（哪侧、哪些 step、哪个字段先 NaN），再做输入二分。
9. 直捕字段以显式链步索引为准，module hook 切片有错位前科；rowsum 相等是强证据非逐位相等，整型字段才是精确比对。
10. 垃圾提案不影响贪心精度：draft 侧 NaN/垃圾是性能问题；精度问题必须先证 target 前向对相同输入一致。
11. 判读必须用波段：gsm8k 50 题单次散布 ±3pt，关键结论需复测（前 15 轮 no-op 有部分是在主导噪声下测的，修复后需复验）。

### 3.3 判读决策树

selected 层快照（graph vs eager 逐字段，自 accept 首个发散 step 起）：

```
pos/kin/kinv 偏   → 上游 KV 投影/RoPE 输入已错（图基础设施级）
kin/kinv 吻合+pub 偏 → fp8 quant 图内产出不同（captured-replay bug）
pub 吻合+pkg 偏   → patch 读到旧 scratch（跨队列竞态）
locs/d2h_locs 偏  → topk 或 req_to_token 映射图内错位
d2h_locs 吻合+d2h_rb 偏 → D2H 读到被覆盖 scratch（Host pool 污染）
q/qrope 偏        → Q 投影 / RoPE 分支
stg_pre 偏        → H2D 未落地即被消费 或 Host 读址内容已错
stg_pre/post 吻合+unpk 偏 → unpack 反量化链图内行为不同
全吻合但 out 偏    → SFA kernel 自身
```

draft 链 dm 表按管线序 `ids/pos → prevraw/prev → emb → eh → am_q/am_qpe/am_tik/am_kvlen/am_seqlens/am_bt/am_pgsum → attn → mlp → out → lmin → lmw → lmout`，**graph 侧首个 NaN/发散键 = 毒点**。

## 4. 启动与验证配置速查

```bash
# D 节点启动（19 层标准规格，repo 根目录）
D_MEM_FRACTION=0.915 \
SELECTIVE_LAYER_IDS="5 9 13 17 21 25 29 33 37 41 45 49 53 57 61 65 69 73 77" \
bash pd_disaggregation_dram_offload.sh

# bench（路由节点）
python -m sglang.test.few_shot_gsm8k --host http://141.61.49.198 --port 6688 \
    --num-questions 50 --num-shots 5 --data-path /home/r00648901/GSM8K.jsonl

# 已知配置坑
# 1) --disable-cuda-graph 时保留 --cuda-graph-bs，否则 fixed bias 按默认 bs 列表 max=512 算到 14GB
# 2) eager 满载 token 数超 SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128 复发 161002
# 3) eager 19 层需 MAX_RUNNING_REQ=24 压内存
```

精度矩阵：eager 19 层 0.94（锚点）｜graph 19 层修复前 0.74-0.82｜修复后 0.90（flag 退役 step-1/2/3 复验均 0.90，最终形态 plain MIX `sparse_copy`）。启动命令维护规则：每次试验后刷新 `pd_disaggregation_dram_offload.sh` 尾部「启动命令速查」节（当前尾部 = Round-26 CANN 升级证据包定稿）。

## 5. 遗留事项

1. **CANN/算子侧升级**（Round-24 定案，证据包：dm 表 + step-0 链值 eager24/graph24 + §2.3 证据边界；dump 文件在机器上，不依赖已删代码）。
2. **临时缓解**：draft 链 eager attention 或 non-quant kernel 路径实验，先兑现 spec 加速。
3. **gsm8k 200 题**精度对齐复核。
4. **accept_len 恢复观察**：修复前恒 [1]，恢复 >1 轮次占比 = spec 加速兑现程度（NaN 缓解前不会恢复）。
5. **可选**：`SELECTIVE_LAYER_IDS=""` 关 hisparse 的 E3 铁证（NaN 仍在即 hi-sparse 无关运行时终证）。
