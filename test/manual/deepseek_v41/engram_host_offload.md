# DeepSeek V4.1 Engram 模块分析与 Host Offload 设计实现

> 适用仓库：JamesBond（sglang NPU 分支）+ memfabric_hybrid（acc_offload）
> 状态：代码已落地，运行时验证需在 A3 NPU 机器执行（本文档第 7 节）

---

## 1. Engram 模块背景

### 1.1 是什么、如何构建加载

Engram 是门控 n-gram 哈希记忆：在少数几个 Transformer 层把哈希检索到的 embedding 写回 hc（hyper-connection）残差流。核心实现在 `python/sglang/srt/layers/engram.py`。

- **结构（模型构造期静态构建，不读 checkpoint）**：`build_engram_layout()` 按 HF config 的 `engram_layer_ids / engram_num_embeddings / engram_max_ngram_size / engram_n_heads / engram_head_dim / engram_vocab_size` 生成 `EngramLayout`。素数表 primes 从一条**共享升序素数序列**（起点 `vocab_size-1`，全局去重）按 (layer, n-gram size, head) 顺序切分——因此**各层表大小 E_l 严格递增**（后层素数更大），`engram_num_embeddings` 是 per-layer 列表。
- **token 压缩表**：`EngramHasher.from_config()` 加载 HF tokenizer，用 NFKC→NFD→去音标→小写→空白归一→Strip 把词表归一化去重，得到 `token_map: token_id → compressed_id`（Host 侧一次性构建，`register_buffer` 随模型上 device）。压缩词表大小与 config 的 `engram_compressed_vocab_size` 强一致断言（所有哈希乘子依赖它）。
- **权重加载**：`engram.embed.weight`（fp8_e4m3fn `[E_l, head_dim]`）与 `.scale`（e8m0 `[E_l, head_dim/128]`）经 `_load_rows` 按 TP 行分片加载；`engram.wkv` 是独立 ReplicatedLinear（block-fp8），被排除在 wq_a/wkv 融合之外；`q_weight/k_weight` 缺省全 1。

### 1.2 推理时如何计算

前置：per-request 的 n-gram token 表（device 上 `[max_requests, context_len]` int32，由 `NgramEmbeddingManager` 维护：prefill 前 `prepare_for_forward` 批量填、decode 采样后 `update_after_decode` 增量写，chunked prefill 用 `skip_token_table_update` 防伪 token 污染）。

层循环（`_forward_layers_hc_pre_from_prev`，deepseek_v4.py:3192）——**prefill / decode / mixed 共用，无 mode 分支**：

1. `EngramHasher.forward` 一次性算出所有层 hash ids；
2. 每个 engram 层：查表 →（offload 时 entry_gather）→ 反量化 → wkv 投影 → `engram_gate`（per-hc-copy 归一化点积、signed-sqrt、sigmoid 门控）把 value 加到残差流各副本；V4.1 多模态下 image token 位置的 engram 输出回滚；
3. 紧接着该层 `forward_hc_pre_from_prev`（attention/hc-pre）。

### 1.3 entry ids 在 Host 还是 Device

**完全在 Device 侧**：hasher 的输入（token 表、req_pool_indices、positions、input_ids）全是设备张量，token_map/乘子/素数/偏移是随模型驻留设备的 buffer，输出 `hash_ids [T, n_engram_layers, (n-1)·n_heads]` 直接喂同设备查表；decode 时整段在 NPU graph capture 内。Host 只做两件事：加载期构建压缩 token map；prefill 前把 Python 列表切片上传填表。

---

## 2. Host Offload 设计（acc_offload SHARED 模式）

### 2.1 目标与总体形态

把 EngramEmbedding 的 fp8 weight 大表卸载到 memfabric_hybrid acc_offload **SHARED 模式**的 GVA 池；scale 表比 weight 小 128 倍，**留在 device 常驻全量**。forward 用 `entry_gather` 把选中的行拉到 device staging，反量化后走原 wkv/gate 路径。

- **拉取模式（已确认）**：全量拉取、去掉 embedding 后的 TP all_reduce——每 rank 经 GVA 直接拉本 forward 全部选中行（含 peer slot），本地反量化，图内少一次集合通信。
- **切分不跨机**：offload 分组 = 单节点内 TP rank 组（`local_world = min(tp_size, LOCAL_WORLD_SIZE)`，`rank = tp_rank % local_world`）；多机 TP 时每节点各持全表副本，地址不出本节点池。A3 单机多卡 ≤16。
- **不整除处理**：每层 chunk 行数 `R_l = ceil(E_l / local_world)`，末 chunk 尾部空置（hash id 恒 < E_l，不会触达）。

### 2.2 池布局（最终形态：slot 分段连续排布）

每个 rank 的 slot 内，各层 chunk **从 slot 起始地址开始严格连续排布**：

```
层 l 块偏移 blockOffset_l = Σ_{k<l} R_k × head_dim
行距（entry_bytes）= head_dim，零 padding
slot 尾部空白 = GB 对齐富余（memfabric 强制 reserve/alloc GB 对齐，不可避免）
```

行 id 的地址映射（kernel 内完成）：

```
src = poolGva + blockOffset + (id ÷ rowsPerSlot) × slotStride + (id mod rowsPerSlot) × entryBytes
```

- id = 该层表内全局行号（hash id 本身，**调用方零地址算术**）；
- `× slotStride` 一步跨过目标 slot 的全部内容（含尾部空白）；
- div/mod 用 32 位除法（约定 id < 2³²，行号现实上远小）；
- 当 `rowsPerSlot × entryBytes == slotStride` 时数学上退化为全池均匀网格（benchmark 示例即此用法）。

> 演进记录：初版用全池均匀网格 `poolGva + id × entryBytes`，要求 entry 大小整除 GB 对齐的 slot（2³⁰），因此做过 pow2 对齐 padding——被否决（非 2 幂 head_dim 浪费最多一半空间）；最终改为上述分段映射，行级零浪费。

### 2.3 表注册机制（tableId）

布局在 init 阶段（权重加载前）就从 config 全部可知，但每层不同（E_l 递增 → R_l、blockOffset 逐层不同），不能塞进池级一次性的 `offload_init`。采用**注册**：

```c
/* init 后每层一次（本地账本，无跨 rank 通信；各 rank 同序注册 → tableId 一致） */
int32_t offload_register_entry_table(uint32_t entryBytes, uint32_t rowsPerSlot,
                                     uint64_t blockOffset, uint32_t *tableId);
/* forward 调用只带表句柄 */
int32_t offload_entry_gather(uint64_t dstPtr, uint64_t idsPtr, uint64_t countPtr,
                             uint32_t tableId, uint16_t deviceId);
```

参数归属原则——**"池不理解行宽"**：

| 属性 | 持有方 | 时机 |
|---|---|---|
| poolGva、slotStride | native（池属性，不越接口） | init |
| entryBytes、rowsPerSlot、blockOffset | 调用方算出、注册上报 | init 后一次 |
| tableId | native 发放、调用方保存 | 注册时 |
| id（行号） | device 张量，每 forward 变 | 调用时 |

注册时一次性校验（行宽 ∈ (0, 120KB]（kernel UB ping-pong slot 上限）、chunk 装进一个 slot、不越 slot 尾）；**无 batch cap**（任意 uint32 count 可执行，0 为 no-op）；registry 在 uninit 清空。

### 2.4 forward 数据流（JamesBond 侧）

```
模型构造期（每 rank）：
  offload_init(SHARED, reserve=alloc=GB_align(Σ R_l×head_dim), world=local_world, rank)
  slot_base = offload.malloc(slot)            # 整 slot 一次分配 → 返回 slot 起始地址
  for 每层: tableId_l = register_entry_table(head_dim, R_l, blockOff_l)

权重加载：
  weight → meta 参数路由 → pinned host staging 切片 [rank·R, ...) →
  post_load_weights: finalize_offload() 整块刷入池（host→host copy）
  scale → device 全量参数（不走池）

forward（每层每步）：
  indices = hash_ids 的本层切片（[T, L, H] 在 dim 1 上切，跨层 stride、非连续）
  → 单次 copy_ 进固定地址 buffers.ids（kernel 按扁平数组读 ids；
    copy_ 一步完成密集化 + dtype 归一，拷贝量仅为 gather 数据量的 8/head_dim ≈ 3%）
  count.fill_(T×H)
  entry_gather(staging, ids, count, tableId_l, device)   # 异步，当前 NPU stream
  反量化：staging u8 → view(fp8) → reshape(T,H,head_dim) → float
          × self.scale[indices] → bf16        # 与 device 常驻路径同数学
  → wkv → engram_gate（不变）
```

**NPU graph capture 兼容**：ids/count/staging 在 `finalize_offload` 时**一次性预分配**（容量 = `max(chunked_prefill_size, decode graph max_bs) × n_hash_cols`），此后**冻结、禁止再长**——多 bs 依次 capture 的场景下，若允许扩容，小图固化的旧地址会被 caching allocator 回收复用，小图 replay 即野指针（静默损坏）；launch 参数（注册布局）capture 时烘焙固定；count 从 device 读、launch 不带 host 标量。容量不足时：capture 中直接 fail-fast（预分配过小 = 配置错误）；eager 超界走一次性临时 buffer（地址稳定性只在 replay 需要），固定 buffer 地址永不变。

---

## 3. 关键正确性核查结论（已静态验证）

1. **chunk 从 slot 起始地址开始**：allocator 初始唯一空闲块 `{0, slot}`，best-fit 整池分配必返回 `base_+0`；各 rank 布局算术相同 → slot 内偏移一致 → 跨 chunk 寻址成立。
2. **entry 不写满 slot**：尾部"已分配未写"，`hostGva_`（全池 GVA 起点 = rank 0 slot 起点）由 native 注入 kernel；kernel 对 id 无越界校验，界内性由调用方保证（hash id < E_l → id < 注册范围）。
3. **E % 卡数有余数的分布**：ceil 切分下前 `floor(E/R)` 个 chunk 满额等大，余数集中落在下一个 chunk，**其后 chunk 可能为空**（不止最后一个偏少，如 E=13/N=8 → 6 满 + 1 短 + 1 空）；实现不依赖具体形态（统一 R 预留、有效行加载、空 chunk 无 id 可达）。
4. **数值等价**：每 rank 独立算全量值，与原 partial+all_reduce 等价（仅浮点求和顺序差异）。

---

## 4. entry ids 产生与预取分析

### 4.1 产生链路（EngramHasher，device，每 forward 一次）

```
token_table[req, positions−s] → tokens[T,n]（列0=当前，列s=第s前驱；
  lookback<0 → blocked；image token 及更老前驱 → PAD，cummax 传播）
→ token_map 压缩（blocked → pad_id）
→ 每 (token, 回看位) × 奇数乘子 → 沿回看方向逐步 XOR（第 i 步后 = (i+1)-gram 哈希）
→ 对每层每 n-gram 每 head 素数取模 + 层内桶偏移
→ hash_ids [T, n_engram_layers, (n−1)·n_heads]
```

### 4.2 预取边界

- hash 与 entry_gather **均零 hidden_states 依赖**——输入只有 input_ids / token 表 / 调度元数据；
- 最早起点 = **本步 forward 输入就绪**（上一步 sample 完成 + token 表更新落流之后），即可与 embed_tokens 段并行发出，重叠窗口覆盖整个层循环前段；
- **跨步预取不可行**（下一步 input_ids 依赖本步采样输出）——预取深度上限是"本步之内提前"。

### 4.3 双流并行的最优对象

entry_gather 是 AIV 的 MTE 搬数；影响最小的重叠对象是**紧邻前一层的 attention/hc-pre（AIC 段）**：

| 维度 | entry_gather | 前一层 attention/hc-pre | 冲突 |
|---|---|---|---|
| 算力 | AIV MTE（几乎不占 vector 管线） | AIC（Cube）矩阵计算 | 异构核不抢算力 |
| 访存 | host 互连（PCIe/DDR）读 host 池 | HBM（KV cache/权重） | 链路不同 |
| 依赖 | 写 staging、读 ids | 只产出 hidden_states | 零依赖 |

```
主流：  layer i-1 attention/hc-pre（AIC） │ wait(event) │ layer i: 反量化+wkv+gate → ...
侧流：  event=record; entry_gather(layer i) ────────────┘
```

- staging 需双缓冲（decode 规模小，可每 engram 层各配一份；prefill 大 T 滚动）；
- hash 本身是 AIV 小核，留主流开头串行开销可忽略，值得搬侧流的是 gather 的 host 链路传输；
- **避开**同时段其它 host 方向传输（hicache KV 加载等）——那才与 gather 抢同一条链路；
- decode graph 内双流是待攻坚项：NPU graph 多流 capture 本仓无先例（`SGLANG_OPT_USE_MULTI_STREAM_OVERLAP` 目前 CUDA-only），建议先在 eager / TC_PIECELESS 分片路径验证。

### 4.4 prefill 与 decode 都有 engram

hc-pre 层循环无 mode 分支，两者共用；hasher 内部按 `forward_mode.is_decode()` 吸收差异（decode 每 req 一 token；prefill/extend/mixed 按 `extend_seq_lens` 展开逐 token；chunked prefill 前 n−1 上下文 token 已预填进表）。执行环境差异：prefill eager/分片、token 表 chunk 前批量填、T 大；decode graph 内重放、采样后增量写表、T 小。**数值完全相同**；双流预取对两种模式都适用，prefill 是更安全的先行验证场。

### 4.5 DP-attention idle rank 兼容

DP attention 的 idle rank 跑 dummy forward：`ForwardBatch.init_new` 对 IDLE batch 在 `_init_ngram_embedding_info` 之前提前返回，`ngram_embedding_info` 为 None；`prepare_mlp_sync_batch` 还可能把模式转成 EXTEND/TARGET_VERIFY 并用 padded dummy token。hasher 对此做两处兼容（offload/device 两条路径共用）：

- `info is None` → 直接返回全零 row id（每层表内的合法行），dummy token 正常流过层循环，其输出在 `post_forward_mlp_sync_batch` 中被切掉；
- token 级张量（positions/input_ids）按 attn_tp_size 对齐、而 request 级 `req_pool_indices` 保持真实 batch 时，把 req 对齐到 token 数（不足补零、超出截断），padded 行同样哈希自表行 0 并被丢弃。

---

## 5. 配置与开关

- `SGLANG_OPT_ENGRAM_HOST_OFFLOAD`（environ.py，`EnvBool(False)`，默认关）：开启条件还需 NPU 平台（`is_npu()`）。
- 依赖：memfabric-hybrid wheel（acc_offload 子模块）+ `MEMFABRIC_HYBRID_EXTEND_LIB_PATH` 指向已构建的 `libmf_hybm_accoffload.so`；导入失败给友好报错。
- 机内分组坐标优先读 `SGLANG_NODE_LOCAL_WORLD/RANK`（DP controller 为每节点 scheduler 盖章），其次 `LOCAL_WORLD_SIZE/RANK`（torchrun），退化 TP 组（须单机）。

### 5.1 日志观测（grep 前缀 `[engram_offload]`）

所有 engram 卸载/读取日志共用 `[engram_offload]` 前缀，`grep engram_offload` 可过滤全链路。加载路径一次性 INFO，推理热路径节流（首次必打 + 每 `_OFFLOAD_LOG_EVERY=1000` 次心跳）：

| 阶段 | 日志（rank 前缀） | 含义 |
|---|---|---|
| 池初始化 | `pool init: slot=.. tables=.. world=.. rank=..` | 池/slot 规格与利用率 |
| 表注册 | `registered table N (E=.. rows_per_slot=.. block_off=.. table_id=..)` | 每层布局注册成功 |
| offload 启用 | `layer N OFFLOAD mode engaged` | EngramEmbedding 走卸载分支 |
| 权重刷入 | `layer N weight chunk flushed to GVA pool` | finalize：pinned staging → 池 |
| buffer 冻结 | `forward buffers pre-sized and frozen (capacity=..)` | 预分配成功（野指针防护） |
| 查表首次/心跳 | `layer N entry_gather call #K (.. ids, table_id=..)` | 推理读取链路存活 |
| 超容量 eager | `WARN: exceeds frozen capacity; throwaway buffer`（WARN） | 预分配偏小（非致命） |
| 容量不足 capture | `RuntimeError: cannot serve a capture` | 预分配过小且命中图（致命，配置错误） |
| DP idle | `hasher: ngram_embedding_info is None (DP-attention idle rank)`（节流） | idle rank dummy forward 走零 id |
| 拆除 | `pool uninitialize (collective teardown)` | atexit 集体退出 |

观察一次正常启动应看到：每 rank 1 条 pool init + L 条 registered + L 条 OFFLOAD engaged + L 条 flushed + L 条 frozen；首个请求起每层 1 条 entry_gather #1，之后每 1000 次一条。

## 6. 文件清单

**JamesBond**：

| 文件 | 内容 |
|---|---|
| `python/sglang/srt/environ.py` | 开关 `SGLANG_OPT_ENGRAM_HOST_OFFLOAD` |
| `python/sglang/srt/layers/engram_offload.py` | `EngramOffloadManager`：机内分组、ceil chunk、连续布局、注册、转发、atexit 集体 uninit |
| `python/sglang/srt/layers/engram.py` | `EngramEmbedding` 双模式（device 常驻 / offload：meta weight + staging 加载 + finalize + entry_gather forward + 固定地址 buffers） |
| `python/sglang/srt/models/deepseek_v4.py` | `post_load_weights` 挂 `finalize_offload()` |
| `test/manual/deepseek_v41/test_engram_offload.py` | 多 rank 对拍（含 E=37 不整除）+ graph capture spike + bench |

**memfabric_hybrid**：

| 文件 | 内容 |
|---|---|
| `src/acc_offload/include/host/acc_offload.h` | `offload_register_entry_table` / `offload_entry_gather`（tableId 版）+ `OFFLOAD_ENTRY_GATHER_MAX_ENTRY_BYTES` |
| `src/acc_offload/csrc/acc_offload.cpp` | API 入口（注册透传 / gather 仅校验地址与 tableId） |
| `src/acc_offload/csrc/acc_offload_entry.h/.cpp`、`acc_offload_entry_manager.h/.cpp` | 基类默认实现 + manager 透传（注册加锁） |
| `src/acc_offload/csrc/acc_offload_shared_dram_entry.h/.cpp` | registry（校验/查表/launch/uninit 清空） |
| `src/acc_offload/csrc/launch/acc_offload_launch.h`、`acc_offload_operators_launch.cpp` | `AccOffloadEntryGatherLayout` struct + launch 透传 |
| `src/acc_offload/csrc/operators/acc_offload_entry_gather.h/.cpp` | kernel：分段寻址 + 32 位 div/mod + ping-pong MTE |
| `src/acc_offload/csrc/python_wrapper/pymf_acc_offload.cpp`、`src/smem/.../mf_acc_offload.py`、`__init__.py` | pybind 与 Python wrapper（`register_entry_table` / `entry_gather`） |
| `examples/kv_offload/shared_dram_offload/share_engram_offload.py` | 示例（退化满排布用法） |

## 7. 验证方法（NPU 机器执行）

1. **重新构建 memfabric kernel lib**（`libmf_hybm_accoffload.so`，签名已多次变更，旧 .so 会加载失败——兼作版本错配检查）。
2. 对拍脚本：`python test_engram_offload.py --npu-ids 0,1 --capture --bench`
   - 多 rank SHARED 池初始化 → 合成表走真实 weight-loader 路径 → offload 查表与 device 全表 gather oracle 对拍 allclose（E=37 用例覆盖末 chunk 尾部空白 + 跨空白寻址）；
   - `--capture`：最小 NPU graph 包住 ids/count 填充 + entry_gather + 反量化，replay 换 id 验证（decode 路径集成形态）；
   - `--bench`：decode 规模延迟（可加大 token 数验证大 count，无 batch cap）。
3. e2e：`SGLANG_OPT_ENGRAM_HOST_OFFLOAD=1` 起 DSv4.1——单机 TP 与双机 TP（验证表副本/地址均在机内）；对比基线 logits、HBM 下降（≈ Σ E_l × head_dim，无 padding 放大）、decode 吞吐。

## 8. 已知风险与待验证项

- `entry_gather`（RunOpApiV2 自定义算子）在 `torch.npu.graph` capture 下的可重放性未在本仓验证——固定地址设计满足前提，`--capture` spike 专测；失败则 fail-fast（提示关闭 offload 或 graph），不做运行时自动回退。
- NPU graph 多流 capture 无先例——双流预取先在 eager / 分片路径落地（见 4.3）。
- 未做 SoC 型号检查（A2 不支持 DRAM offload、跨机 URMA 为 A5-only）；目标环境 A3 之外开启需自行确认，必要时在 `_open_engram_offload` 加 `acl.get_soc_name()` 检测。
- id 约定 < 2³²（kernel 32 位除法）；表行数现实上远小，如超出需升级 64 位除法。
