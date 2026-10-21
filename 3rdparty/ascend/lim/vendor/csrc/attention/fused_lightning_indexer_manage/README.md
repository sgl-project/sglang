# FusedLightningIndexerManage

## 产品支持情况

| 产品 | 是否支持 |
|:---|:---:|
| <term>Ascend 950PR/Ascend 950DT</term> | × |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：`fused_lightning_indexer_manage`面向大序列稀疏KV Cache卸载场景，在一次NPU调用中完成Lightning Indexer TopK检索、HBM稀疏缓存的初始化或稳态淘汰、MTP多路Query的TopK Union管理，以及Cache Miss搬运计划生成。一个Batch可以混合非卸载、首次卸载和稳态卸载请求，每个请求支持1～14路Query。

- Lightning Indexer计算公式：

  $$
  Indices=\operatorname{TopK}\left\{[1]_{1\times N}@\left[(W@[1]_{1\times S_k})\odot\operatorname{ReLU}\left(Q_{index}@K_{index}^{T}\right)\right]\right\}
  $$

  其中，$Q_{index}\in\mathbb{R}^{N\times128}$为某一路Index Query，$K_{index}\in\mathbb{R}^{S_k\times128}$为该路Query因果可见的Index Key，$W\in\mathbb{R}^{N\times1}$为Index Head聚合权重。算子为每一路Query输出Top-2048 Source Token ID及其HBM Logical Slot。

- 稀疏缓存管理：算子通过`request_state`区分非卸载、首次卸载和稳态卸载状态，并通过`cache_slots_pool`维护Source Token到HBM Logical Slot的持久化映射。算子输出本轮需要从CPU侧搬入HBM的Source Token ID、目标Slot及有效搬运数量，实际的数据搬运由上层框架执行。

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
|:---|:---:|:---|:---:|:---:|
| index_weights | 输入 | 每条Query、每个Index Head的聚合权重，shape为(T,N)，N支持32或64。 | FLOAT16、BFLOAT16 | ND |
| query_dequant_scale | 输入 | Query反量化Scale，shape为(T,N)。当前版本为后续量化能力预留的必选参数，仅校验shape和数据类型，不读取其数值，不影响计算结果；后续将支持使用该Scale对量化Query进行反量化。 | FLOAT | ND |
| query | 输入 | TND排布的Index Query，shape为(T,N,128)。同一请求的多路Query在T维连续。 | FLOAT16、BFLOAT16 | ND |
| index_key_dequant_scale | 输入 | Index Key反量化Scale，shape为(index_block_num,128,1)。当前版本为后续量化能力预留的必选参数，仅校验shape和数据类型，不读取其数值，不影响计算结果；后续将支持使用该Scale对量化Index Key进行反量化。 | FLOAT | ND |
| index_key_cache | 输入 | PageAttention排布的Index Key Cache，shape为(index_block_num,128,1,128)，数据类型必须与`query`、`index_weights`一致。 | FLOAT16、BFLOAT16 | ND |
| index_block_table | 输入 | 每个请求的Index Key逻辑块到物理块映射表，shape为(B,index_max_blocks)，其中index_max_blocks不超过16384。 | INT32 | ND |
| actual_seq_lengths_query | 输入 | 各请求Query在T维的累计结束位置，shape为(B)。元素严格递增，最后一个元素必须等于T。 | INT32 | ND |
| actual_seq_lengths_key | 输入 | 每个请求最后一路Query对应的实际KV长度，shape为(B)。其他路Query的因果可见长度由该值和Query位置推导。 | INT32 | ND |
| offload_seq_lengths_key | 输入 | 每个请求可检索、可卸载的稳定Source Prefix长度，shape为(B)，仅`request_state`为-2或-1时使用。 | INT32 | ND |
| num_cache_tokens | 输入 | 每个请求的HBM稀疏缓存容量，shape为(B)，仅`request_state`为-2或-1时使用。 | INT32 | ND |
| request_state | 输入 | 请求状态，shape为(B)。仅支持-3（非卸载）、-2（首次卸载）和-1（稳态卸载）。 | INT32 | ND |
| req_pool_entries | 输入 | 每个请求占用的`cache_slots_pool`行号，shape为(B)。同一次调用中的有效请求必须使用不同的行号。 | INT32 | ND |
| cache_slots_pool | 输入/输出（原地更新） | 跨Decode Step持久化的Source Token到HBM Logical Slot映射，shape为(pool_size,source_capacity)，其中source_capacity=index_max_blocks*128。 | INT32 | ND |
| topk_src_ids | 输出 | 每一路Query的TopK Source Token ID，shape为(T,1,2048)。可见长度不足2048时使用-1填充。 | INT32 | ND |
| topk_dst_slots | 输出 | 与`topk_src_ids`逐项对应的HBM Logical Slot，shape为(T,1,2048)；无有效Slot时为-1。 | INT32 | ND |
| topk_miss_counts | 输出 | 每一路Query在`topk_src_ids`中的Miss前缀长度，shape为(T)；非卸载状态固定为0。 | INT32 | ND |
| miss_src_ids | 输出 | 每个请求本轮需要搬入HBM的Source Token ID，shape为(B,32768)，仅前`miss_counts[i]`项有效。 | INT32 | ND |
| miss_dst_slots | 输出 | 与`miss_src_ids`逐项对应的目标HBM Logical Slot，shape为(B,32768)，仅前`miss_counts[i]`项有效。 | INT32 | ND |
| miss_counts | 输出 | 每个请求的有效搬运项数量，shape为(B)。 | INT32 | ND |

其中，B表示Batch中的请求数量，T表示所有请求的Query总数，N表示Index Head数量。所有Tensor均不支持空Tensor或非连续Tensor。

## MTP Query排布

对于请求`i`：

```text
query_start = i == 0 ? 0 : actual_seq_lengths_query[i - 1]
query_end   = actual_seq_lengths_query[i]
Q           = query_end - query_start
```

Q为该请求的Query路数，取值范围为1～14。`actual_seq_lengths_key[i]`表示最后一路Query的KV长度；区间内第`query_row`路Query的因果可见长度为：

```text
visible_length = actual_seq_lengths_key[i] - (query_end - 1 - query_row)
```

非卸载状态下，每一路Query在各自的`visible_length`范围内检索；卸载状态下，多路Query共同在稳定Prefix `[0,L)`内检索，其中`L=offload_seq_lengths_key[i]`，该Prefix必须对请求的全部Query因果可见。

## 请求状态与输出

### 非卸载状态（`request_state=-3`）

- 每一路Query在`[0,visible_length)`范围内执行普通Lightning Indexer。
- `topk_src_ids`按Score排序；`topk_dst_slots`与`topk_src_ids`逐项相同，表示Identity Slot。可见长度不足2048时，两个输出均使用-1填充。
- `topk_miss_counts`和`miss_counts`均为0，`miss_src_ids`和`miss_dst_slots`中的内容无效。
- 每次调用都会将对应的完整`cache_slots_pool`行恢复为Identity Mapping：`pool[row,source]=source`。

### 首次卸载状态（`request_state=-2`）

- 每一路Query在稳定Prefix `[0,L)`内检索，不使用对应Pool行的旧映射。
- 算子先计算多路Top-2048的去重Union，再按Source ID递增补充不在Union中的Token，直至Resident Token数量达到C，其中`C=num_cache_tokens[i]`。
- 算子清空并重建对应Pool行，将C个Resident Token分配到Logical Slot `[0,C)`。
- `miss_src_ids[i,:C]`保存本次需要填充的Resident Source Token，`miss_dst_slots[i,:C]`为`[0,1,...,C-1]`，`miss_counts[i]=C`。
- 每一路TopK均视为Miss，`topk_miss_counts`为2048；TopK输出按Source ID递增排列。

### 稳态卸载状态（`request_state=-1`）

- 正常Pool行在`[0,L)`中包含C个Resident Source Token，且对应Slot构成`[0,C)`的双射；非Resident Source使用`INT32_MIN`表示。
- 算子完成Hit/Miss判断、多路Miss Union、受TopK保护的Victim选择、Slot回收及Pool映射更新。
- 每一路TopK输出由Miss前缀和Hit后缀组成，两个区间分别按Source ID递增；`topk_miss_counts`表示Miss前缀长度。
- `miss_src_ids[i,:miss_counts[i]]`是所有路TopK Miss Source的有序去重Union，`miss_dst_slots`保存这些Source获得的回收Slot。
- 当输入Pool行来自非卸载状态的Identity Mapping时，算子会保留`[0,C)`并将后缀归一化为`INT32_MIN`，再执行稳态更新。正常稳态调用不会重复扫描完整Pool行。

除-3、-2和-1之外的`request_state`均不支持。

## 约束说明

- 该接口仅支持推理场景，并支持图模式。
- 当前版本仅计算FLOAT16或BFLOAT16数据；`query`、`index_key_cache`和`index_weights`的数据类型必须一致。`query_dequant_scale`和`index_key_dequant_scale`必须为FLOAT，但当前不参与计算，后续将用于支持量化Query和Index Key。
- N支持32或64，Index Head Dim固定为128，Index Key Cache中的KV Head数量固定为1，Block Size固定为128。
- 静态shape必须满足：`B>0`、`B<=T<=14B`、`pool_size>0`、`index_block_num>0`、`1<=index_max_blocks<=16384`，并且`cache_slots_pool`宽度必须等于`index_max_blocks*128`。
- `topk_src_ids`和`topk_dst_slots`的shape必须为(T,1,2048)，`topk_miss_counts`为(T)，`miss_src_ids`和`miss_dst_slots`为(B,32768)，`miss_counts`为(B)。这些输出均由调用方预先分配。
- `actual_seq_lengths_query`必须严格递增，最后一个元素必须等于T；每个请求的Q必须在[1,14]范围内。
- 每个请求必须满足`Q<=actual_seq_lengths_key[i]<=source_capacity`。`index_key_cache`和`index_block_table`必须覆盖实际访问的物理块。
- 对-2和-1状态，令`L=offload_seq_lengths_key[i]`、`C=num_cache_tokens[i]`，必须满足：
    - `L>=2048`，`C<=L<=actual_seq_lengths_key[i]`；
    - L和C均为128的倍数；
    - 当`L<=Q*2048`时，必须满足`C=L`；
    - 当`L>Q*2048`时，必须满足`Q*2048<=C<=32640`；
    - `L<=floor((actual_seq_lengths_key[i]-Q)/128)*128`，保证稳定Prefix对全部Query因果可见。
- `req_pool_entries[i]`必须位于`[0,pool_size)`，同一次调用的有效请求不能写入同一Pool行。同一请求生命周期内，Pool行号必须保持不变。
- `source_capacity`最大为`2^21 = 2097152`。Source长度超过`2^17 = 131072`时，内部使用21-bit Source ID编码；所有对外输出始终为完整INT32 Source ID。
- HBM Logical Slot使用15 bit编码，32767为内部无效值，因此C最大为32640。`INT32_MIN`仅用于`cache_slots_pool`中的非Resident标记，不会作为有效Slot输出。
- 所有输入、输出必须位于同一NPU设备并保持连续。
- 对逐请求动态非法值，Kernel执行安全保护：不破坏对应Pool行，将TopK输出填充为-1，并将相关计数置0。该保护不能替代上层框架的参数校验。

## 生命周期

| 场景 | 状态序列 | 行为 |
|:---|:---|:---|
| 请求从一开始就卸载 | `-2 -> -1 -> ...` | 首次调用重建Resident集合，后续稳态维护同一Pool行。 |
| 非卸载后直接进入稳态卸载 | `-3 -> -1 -> ...` | 算子将Identity Pool行归一化为Sparse Mapping后执行稳态更新。 |
| 非卸载后重新选择Resident Token | `-3 -> -2 -> -1 -> ...` | `-2`忽略旧Identity Mapping并重新建立Resident集合。 |

## 调用示例

该算子通过vLLM Ascend的Torch扩展调用。调用方负责预先分配全部输出Tensor，算子返回值为`None`，计算结果写入`cache_slots_pool`及6个输出Tensor。

```python
import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

# 此处省略输入Tensor的构造，shape和dtype要求见“参数说明”。
topk_src_ids = torch.full((T, 1, 2048), -1, dtype=torch.int32, device="npu")
topk_dst_slots = torch.full_like(topk_src_ids, -1)
topk_miss_counts = torch.zeros(T, dtype=torch.int32, device="npu")
miss_src_ids = torch.full((B, 32768), -1, dtype=torch.int32, device="npu")
miss_dst_slots = torch.full_like(miss_src_ids, -1)
miss_counts = torch.zeros(B, dtype=torch.int32, device="npu")

torch.ops._C_ascend.npu_fused_lightning_indexer_manage(
    index_weights=index_weights,
    query_dequant_scale=query_dequant_scale,
    query=query,
    index_key_dequant_scale=index_key_dequant_scale,
    index_key_cache=index_key_cache,
    index_block_table=index_block_table,
    actual_seq_lengths_query=actual_seq_lengths_query,
    actual_seq_lengths_key=actual_seq_lengths_key,
    offload_seq_lengths_key=offload_seq_lengths_key,
    num_cache_tokens=num_cache_tokens,
    request_state=request_state,
    req_pool_entries=req_pool_entries,
    cache_slots_pool=cache_slots_pool,
    topk_src_ids=topk_src_ids,
    topk_dst_slots=topk_dst_slots,
    topk_miss_counts=topk_miss_counts,
    miss_src_ids=miss_src_ids,
    miss_dst_slots=miss_dst_slots,
    miss_counts=miss_counts,
)
```

Python接口内部通过`EXEC_NPU_CMD`调用`aclnnFusedLightningIndexerManageGetWorkspaceSize`和`aclnnFusedLightningIndexerManage`两段式接口，调用方无需自行申请Workspace或管理`aclOpExecutor`。

## 测试说明

算子正确性测试位于`tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_lightning_indexer_manage.py`，需要在已编译安装当前代码的Ascend NPU环境执行：

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_lightning_indexer_manage.py
```

测试覆盖以下场景：

- FP16、BF16以及32、64个Index Head的代表组合；
- 非卸载、首次卸载、稳态卸载及混合状态Batch；
- MTP多路Query、Q=14边界、非连续Pool行和非Identity Block Table；
- 首次建Cache、稳态淘汰、Miss搬运计划、Source到Slot映射及重复调用All-Hit；
- `-2 -> -1 -> -1`、`-3 -> -1`、`-3 -> -2 -> -1`生命周期；
- 跨越`2^17`、`2^18`、`2^19`、`2^20`及接近`2^21`上限的长Source ID编码路径，包括长序列多路Query混合状态；
- 2048/2049和4095/4096/4097 Miss Occurrence排序边界；
- 代表性的shape、dtype、连续性和最大Query路数校验。

测试共收集107个pytest case：Q=1～14覆盖非卸载、首次卸载和稳态卸载状态，代表性Q值覆盖FLOAT16/BFLOAT16与32/64个Index Head的交叉组合，并覆盖非卸载短序列的TopK无效后缀、窄路由、宽路由、混合状态Batch及多档长序列。每个pytest case结束后会输出`[当前case/总case] [完成百分比]`，便于观察长测试的执行进度。

测试使用`torch_npu.npu_lightning_indexer`作为TopK参考实现，仅验证功能正确性，不包含性能测试、耗时阈值或性能基线。

## 典型时延

下表为64K MTP5（Q=6）稳态卸载场景的实测结果。测试设备为Ascend 910_93（`SOC_VERSION=ascend910_9391`），数据类型为BF16，Index Head数量为32，Source长度为65536，HBM Cache容量为12288，`request_state=-1`。每一路Query约有200个Cache Miss，多路Query去重后的Union Miss为400；测试包含10次预热和300次计时迭代，时延单位为μs。

其中，Official表示仅调用`torch_npu.npu_lightning_indexer`完成TopK检索的时延；Fused表示调用`fused_lightning_indexer_manage`完成TopK检索和稀疏缓存管理的总时延；Overhead为两者之差。

| Batch Size | Official | Fused | Overhead |
| ---: | ---: | ---: | ---: |
| 1 | 134.030 | 288.410 | +154.380 |
| 2 | 149.060 | 294.270 | +145.210 |
| 5 | 220.570 | 375.950 | +155.380 |
| 16 | 578.230 | 776.090 | +197.860 |
| 24 | 838.920 | 1044.890 | +205.970 |

时延会随硬件、CANN版本、编译选项和运行负载变化，表中结果仅用于说明该测试配置下的典型表现，不作为性能承诺或测试通过阈值。
