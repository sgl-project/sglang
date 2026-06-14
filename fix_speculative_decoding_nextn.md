# SGLang NEXTN 投机解码修复文档

## 问题描述

在使用 SGLang 运行 `--speculative-algorithm NEXTN` 参数启动 BailingMoE-v3 (KDA) 模型时，出现启动失败和服务卡住两个阶段的问题。

### 复现命令

```bash
python -m sglang.launch_server \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --model-path /home/admin/tiny3.0/ \
  --tp-size 1 \
  --speculative-algorithm NEXTN
```

### 错误现象

#### 阶段一：启动失败

##### 错误 1: `extend_prefix_lens` 为 `None`

```
File "python/sglang/srt/layers/attention/linear/kda_backend.py", line 195
  has_initial_state = forward_batch.extend_prefix_lens > 0
TypeError: '>' not supported between instances of 'NoneType' and 'int'
```

##### 错误 2: `BailingMoeV3ForCausalLM` 缺少 `get_embed_and_head` 方法

```
File "python/sglang/srt/speculative/eagle_worker.py", line 158
  embed, head = self.target_worker.model_runner.model.get_embed_and_head()
AttributeError: 'BailingMoeV3ForCausalLM' object has no attribute 'get_embed_and_head'
```

#### 阶段二：服务卡住

修复阶段一后服务可以启动，但运行时完全卡住无响应。不使用 `--speculative-algorithm NEXTN` 时正常。

---

## 修复方案

### 修复 1: kda_backend.py - 支持 TARGET_VERIFY 模式

**文件**: `python/sglang/srt/layers/attention/linear/kda_backend.py`

**问题分析**:
- 在投机解码的 `TARGET_VERIFY` 模式下，`forward_batch.extend_prefix_lens` 为 `None`
- `kda_backend.py` 的 `forward_extend` 方法直接访问该属性导致类型错误
- 参考 `triton_backend.py` 的处理方式，需要从 `spec_info.draft_token_num` 推断

**修复内容** (第 195-217 行):

```python
# 修改前:
has_initial_state = forward_batch.extend_prefix_lens > 0

# 修改后:
# Handle TARGET_VERIFY mode where extend_prefix_lens might not be set
if forward_batch.extend_prefix_lens is not None:
    has_initial_state = forward_batch.extend_prefix_lens > 0
    extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
else:
    # TARGET_VERIFY mode: infer from spec_info
    # In speculative decoding, prefix_len = seq_len - draft_token_num
    if forward_batch.spec_info is not None and hasattr(
        forward_batch.spec_info, "draft_token_num"
    ):
        bs = forward_batch.batch_size
        draft_token_num = forward_batch.spec_info.draft_token_num
        # All sequences have initial state in TARGET_VERIFY mode
        has_initial_state = torch.ones(bs, dtype=torch.bool, device=mixed_qkv.device)
        extend_seq_lens_cpu = [draft_token_num] * bs
    else:
        raise RuntimeError(
            "extend_prefix_lens is None but cannot infer from spec_info. "
            "This should not happen in TARGET_VERIFY mode."
        )
```

同时将后续使用 `forward_batch.extend_seq_lens_cpu` 的地方改为使用新的变量 `extend_seq_lens_cpu`。

---

### 修复 2: bailing_moe_v3.py - 添加 EAGLE 所需方法

**文件**: `python/sglang/srt/models/bailing_moe_v3.py`

**问题分析**:
- EAGLE/NEXTN 投机解码需要访问和设置模型的 embedding 和 lm_head 权重
- `BailingMoeV3ForCausalLM` 类缺少 `get_embed_and_head` 和 `set_embed_and_head` 方法
- 参考其他模型（如 `qwen2.py`, `llama.py` 等）的实现

**修复内容** (在 `post_process_weights_if_quant` 方法后添加):

```python
def get_embed_and_head(self):
    """Get the embedding and lm_head weights for EAGLE speculative decoding."""
    return self.model.word_embeddings.weight, self.lm_head.weight

def set_embed_and_head(self, embed, head):
    """Set the embedding and lm_head weights for EAGLE speculative decoding."""
    del self.model.word_embeddings.weight
    del self.lm_head.weight
    self.model.word_embeddings.weight = embed
    self.lm_head.weight = head
    torch.cuda.empty_cache()
```

---

### 修复 3: KDA TARGET_VERIFY 中间状态保存缺失（服务卡住问题）

#### 根因分析

在 EAGLE/NEXTN 投机解码的 TARGET_VERIFY 模式中，Target 模型需要验证 Draft 模型生成的候选 token。验证完成后，需要将 SSM 状态和 conv 状态回滚到被接受的 token 位置，然后继续 DECODE 推理。

KDA 后端存在三个问题导致状态无法正确回滚：

1. **Conv 状态被腐蚀**：`causal_conv1d_update` 在 TARGET_VERIFY 期间原地修改 conv 状态（前进了所有 draft token 步），但没有保存中间 conv 窗口快照，验证后无法回滚到正确的步数
2. **SSM 中间状态未保存**：`target_verify` 内核虽然用 `disable_state_update=True` 不更新主 SSM 状态，但也没保存每步的中间 SSM 状态，导致验证后无法将 SSM 状态推进到正确位置
3. **状态回滚未触发**：`eagle_worker.py` 中 `_mamba_verify_update` 的条件缺少 `kimi_linear_config`，KDA 模型验证后根本不执行状态回滚

三者叠加导致后续 DECODE 使用腐蚀的 conv 状态和过时的 SSM 状态，输出错误或卡住。

#### 状态流转示意

```
TARGET_VERIFY 前:
  conv_state: [t0]  (正确的当前状态)
  ssm_state:  [t0]  (正确的当前状态)

TARGET_VERIFY 期间 (draft_token_num=4, 假设接受 2 个):
  conv_state: [t0, t1, t2, t3, t4]  (被腐蚀，前进了 4 步)
  ssm_state:  [t0]                   (未更新，停在验证前)
  intermediate_ssm:  [t0, t1, t2, t3, t4]  (每步快照，用于回滚)
  intermediate_conv: [t0, t1, t2, t3, t4]  (每步快照，用于回滚)

回滚后 (accepted_steps=2):
  conv_state: [t2]  (从 intermediate_conv 恢复到第 2 步)
  ssm_state:  [t2]  (从 intermediate_ssm 恢复到第 2 步)

修复前 (没有保存中间状态 + 没有触发回滚):
  conv_state: [t0, t1, t2, t3, t4]  (腐蚀状态)
  ssm_state:  [t0]                   (过时状态)
  -> 后续 DECODE 使用错误状态 -> 卡住或输出错误
```

#### 文件变更

**文件 1**: `python/sglang/srt/layers/attention/linear/kda_backend.py`

**修复内容**:

重构 `forward_extend` 的 TARGET_VERIFY 路径，对齐 GDN 后端实现：

- 将整个 mixed_qkv 一起通过 `causal_conv1d_update` 处理（而非分别处理 Q/K/V），与 GDN 后端和 decode 路径保持一致
- 添加 `intermediate_conv_window` 传递给 `causal_conv1d_update`，保存每步的 conv 窗口状态
- 添加 `intermediate_states_buffer` 和 `intermediate_state_indices` 传递给 `target_verify` 内核，保存每步的 SSM 状态
- 添加 `retrieve_parent_token` 支持树状投机解码（topk > 1）
- 注意：KDA 的 conv 状态存储格式为 `(K-1, dim)`，需转置为 `(dim, K-1)` 以匹配 `causal_conv1d_update` 的期望格式

```python
# 修改前: 分别处理 Q/K/V，不保存中间状态
q = causal_conv1d_update(q, q_conv_state, q_conv_weight, q_bias, ...)
k = causal_conv1d_update(k, k_conv_state, k_conv_weight, k_bias, ...)
v = causal_conv1d_update(v, v_conv_state, v_conv_weight, v_bias, ...)

# 修改后: 整体处理 mixed_qkv，保存中间状态
assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)
intermediate_conv_window_cache = mamba_cache_params.intermediate_conv_window[0]
intermediate_state_cache = mamba_cache_params.intermediate_ssm

mixed_qkv_reshaped = mixed_qkv.view(batch_size, draft_token_num, -1).transpose(1, 2)
intermediate_conv_window_transposed = intermediate_conv_window_cache.transpose(-1, -2)

mixed_qkv_processed = causal_conv1d_update(
    mixed_qkv_reshaped,
    conv_states,
    layer.conv_weights,
    layer.bias,
    activation="silu",
    conv_state_indices=conv_state_indices,
    intermediate_conv_window=intermediate_conv_window_transposed,
    intermediate_state_indices=intermediate_state_indices[:batch_size],
    retrieve_next_token=retrieve_next_token,
    retrieve_next_sibling=retrieve_next_sibling,
    retrieve_parent_token=retrieve_parent_token,
)
```

```python
# target_verify 内核调用也需传递中间状态
# 修改前:
core_attn_out = self.kernel_dispatcher.target_verify(
    ..., cache_steps=draft_token_num
)

# 修改后:
core_attn_out = self.kernel_dispatcher.target_verify(
    ...,
    intermediate_states_buffer=intermediate_state_cache,
    intermediate_state_indices=intermediate_state_indices[:batch_size],
    cache_steps=draft_token_num,
    retrieve_parent_token=retrieve_parent_token,
)
```

**文件 2**: `python/sglang/srt/layers/attention/linear/kernels/kda_triton.py`

**修复内容**:

`target_verify` 方法添加中间状态参数，传递给底层 `fused_sigmoid_gating_delta_rule_update` 内核：

```python
# 修改前:
def target_verify(self, ..., ssm_states, cache_indices, query_start_loc, **kwargs):
    return fused_sigmoid_gating_delta_rule_update(
        ..., is_kda=True, disable_state_update=True
    )

# 修改后:
def target_verify(self, ..., ssm_states, cache_indices, query_start_loc,
                  intermediate_states_buffer=None, intermediate_state_indices=None,
                  cache_steps=None, retrieve_parent_token=None, **kwargs):
    return fused_sigmoid_gating_delta_rule_update(
        ..., is_kda=True, disable_state_update=True,
        intermediate_states_buffer=intermediate_states_buffer,
        intermediate_state_indices=intermediate_state_indices,
        cache_steps=cache_steps,
        retrieve_parent_token=retrieve_parent_token,
    )
```

**文件 3**: `python/sglang/srt/speculative/eagle_worker.py`

**修复内容**:

`_mamba_verify_update` 的调用条件添加 `kimi_linear_config`：

```python
# 修改前:
if (
    self.target_worker.model_runner.hybrid_gdn_config is not None
    or self.target_worker.model_runner.mamba2_config is not None
    or self.target_worker.model_runner.hybrid_lightning_config is not None
):

# 修改后:
if (
    self.target_worker.model_runner.hybrid_gdn_config is not None
    or self.target_worker.model_runner.mamba2_config is not None
    or self.target_worker.model_runner.hybrid_lightning_config is not None
    or self.target_worker.model_runner.kimi_linear_config is not None
):
```

---

## 技术背景

### 什么是 TARGET_VERIFY 模式？

在 EAGLE/NEXTN 投机解码中：
1. **Draft 模型** 生成候选 token
2. **Target 模型** (主模型) 验证这些候选 token

`TARGET_VERIFY` 是用于验证阶段的特殊 forward 模式。在这种模式下：
- 输入序列由原始序列 + draft token 组成
- `extend_prefix_lens` 需要从 `seq_len - draft_token_num` 计算得出
- 验证完成后需要将 SSM/conv 状态回滚到被接受的 token 位置

### 为什么需要保存中间状态？

在 TARGET_VERIFY 期间，Target 模型一次性处理所有 draft token：
- `causal_conv1d_update` 会原地修改 conv 状态（每处理一个 draft token，conv 窗口前进一步）
- `fused_sigmoid_gating_delta_rule_update` 用 `disable_state_update=True` 不更新主 SSM 状态

验证完成后，只接受部分 draft token（例如 4 个 draft token 中只接受 2 个），需要将状态回滚到第 2 步：
- 从 `intermediate_ssm[request_index, accepted_step]` 恢复 SSM 状态
- 从 `intermediate_conv_window[request_index, accepted_step]` 恢复 conv 状态

如果不保存中间状态，验证后状态无法正确回滚，后续 DECODE 推理会使用错误的状态。

### 为什么需要 `get_embed_and_head`？

EAGLE 投机解码中，draft 模型通常共享主模型的：
- **Embedding 层** (输入嵌入)
- **LM Head** (输出投影)

这样可以减少内存占用，并确保 draft 模型和 target 模型的 token 表示一致。

---

## 相关文件参考

- `python/sglang/srt/layers/attention/linear/kda_backend.py` - KDA 注意力后端（主要修复点）
- `python/sglang/srt/layers/attention/linear/gdn_backend.py` - GDN 后端（参考实现）
- `python/sglang/srt/layers/attention/linear/kernels/kda_triton.py` - KDA Triton 内核
- `python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py` - GDN Triton 内核（参考实现）
- `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py` - causal_conv1d 内核
- `python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py` - SSM 内核
- `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` - 混合线性注意力调度层
- `python/sglang/srt/speculative/eagle_worker.py` - EAGLE 工作器
- `python/sglang/srt/models/bailing_moe_v3.py` - BailingMoE-v3 模型
- `python/sglang/srt/mem_cache/memory_pool.py` - MambaPool / SpeculativeState
- `python/sglang/srt/model_executor/forward_batch_info.py` - ForwardMode 定义

---

### 修复 4: kimi_linear.py - TARGET_VERIFY 模式下 gate/beta 双重计算导致精度严重下降

#### 问题现象

修复 1-3 后服务可以正常启动和运行，但 gsm8k 评测得分从 95（不开投机解码）暴降至 23（开 NEXTN），输出大量复读且无法正常结束。

#### 根因分析

在 `KimiDeltaAttention.forward()` 中，gate 和 beta 的预处理条件存在遗漏：

```python
# 修改前:
if not forward_batch.forward_mode.is_decode():
    forget_gate = fused_kda_gate(forget_gate, self.A_log, self.head_dim, g_bias=self.dt_bias)
    beta = beta.float().sigmoid()
    forget_gate = forget_gate.unsqueeze(0)
```

这段代码的意图是：
- **decode 模式**：跳过预处理，因为 decode 内核 (`fused_sigmoid_gating_delta_rule_update`) 在内部计算 gating（从原始 `a`, `A_log`, `dt_bias` 计算 `-exp(A_log) * softplus(a + dt_bias)`，并对 beta 计算 sigmoid）
- **extend 模式**：预处理 gate 和 beta，因为 extend 内核 (`chunk_kda`) 期望接收已计算好的 gate 值

问题在于 `TARGET_VERIFY` 不是 `is_decode()`，所以走了预处理路径。但 `target_verify` 内核使用的是和 decode **相同的** `fused_sigmoid_gating_delta_rule_update`，该内核在内部会再次计算 gating。这导致了**双重计算**：

```
第一次 (fused_kda_gate 预处理):
  forget_gate = -exp(A_log) * softplus(raw_a + dt_bias)
  beta = sigmoid(raw_beta)

第二次 (fused_sigmoid_gating_delta_rule_update 内核内部):
  g = -exp(A_log) * softplus(forget_gate + dt_bias)   ← 用已处理的值再算一次！
  beta_internal = sigmoid(beta)                         ← sigmoid(sigmoid(raw_beta))！
```

双重 gating 导致 SSM 状态更新完全错误，验证阶段产生错误的 logits，错误的 token 被接受/拒绝，后续 decode 输出退化为复读。

#### 修复内容

**文件**: `python/sglang/srt/models/kimi_linear.py`（第 431-440 行）

```python
# 修改前:
if not forward_batch.forward_mode.is_decode():
    forget_gate = fused_kda_gate(...)
    beta = beta.float().sigmoid()
    forget_gate = forget_gate.unsqueeze(0)

# 修改后:
if not forward_batch.forward_mode.is_decode() and not forward_batch.forward_mode.is_target_verify():
    forget_gate = fused_kda_gate(...)
    beta = beta.float().sigmoid()
    forget_gate = forget_gate.unsqueeze(0)
```

将 `target_verify` 与 `decode` 同样排除在预处理之外，因为两者使用相同的内核，内核会从原始值计算 gating。

#### 三种模式的 gate 处理流程对比

| 模式 | 预处理 | 内核 | gate 来源 |
|------|--------|------|-----------|
| **extend** (prefill) | `fused_kda_gate` + `sigmoid` | `chunk_kda` | 预处理后的值 |
| **decode** | 无 | `fused_sigmoid_gating_delta_rule_update` | 内核内部计算 |
| **target_verify** | ~~预处理~~ → 无 (修复后) | `fused_sigmoid_gating_delta_rule_update` | 内核内部计算 |

---

### 修复 5: ngram_worker.py - NGRAM 投机解码缺少 SSM 状态回滚

#### 问题现象

使用 `--speculative-algorithm NGRAM` 启动 Kimi-Linear 模型后，模型输出乱码/重复：

```
正常输出: "我是Kimi，一个由月之暗面科技有限公司（Moonshot AI）训练的大语言模型..."
NGRAM输出: "我是K7 我是KimiChatGPT-Kimi，由Kimi，由月，我是Kimi，由月之Kimi，由月之暗号..."
```

不使用 `--speculative-algorithm NGRAM` 时输出正常。

#### 根因分析

`ngram_worker.py` 的 TARGET_VERIFY 流程中，验证完成后没有调用 `update_mamba_state_after_mtp_verify` 来回滚 SSM/conv 状态。

对比 `eagle_worker_v2.py`，该文件在验证后有如下调用：

```python
if (
    self.target_worker.model_runner.hybrid_gdn_config is not None
    or self.target_worker.model_runner.mamba2_config is not None
    or self.target_worker.model_runner.hybrid_lightning_config is not None
    or self.target_worker.model_runner.kimi_linear_config is not None
):
    self._mamba_verify_update(batch, accept_lens, accept_index, bs)
```

而 `ngram_worker.py` 完全没有这段逻辑。TARGET_VERIFY 期间：
- `causal_conv1d_update` 原地修改了 conv 状态（前进了所有 draft token 步）
- `fused_sigmoid_gating_delta_rule_update` 保存了中间 SSM 状态到 `intermediate_ssm`

但验证后，由于没有回滚，conv 状态停留在"处理完所有 draft token"的位置，而 SSM 状态停留在验证前的位置。两者都不在被接受的 token 位置上，导致后续 decode 输出错误。

#### 修复内容

**文件 1**: `python/sglang/srt/speculative/ngram_info.py`

在 `verify()` 方法中，`_fill_requests()` 之前计算 `last_correct_step_indices`（每个 request 最后被接受的 draft token 在其窗口内的步数索引）：

```python
# 修改前: 直接调用 _fill_requests
self._fill_requests(batch, logits_output)

# 修改后: 先计算 last_correct_step_indices，再调用 _fill_requests
bs = self.accept_indices.shape[0]
req_idx = torch.arange(bs, dtype=torch.int64, device=self.accept_indices.device)
accept_indices_offset = (req_idx * self.draft_token_num).to(self.accept_indices.dtype)
self.last_correct_step_indices = (
    self.accept_indices[req_idx, self.num_correct_drafts.to(torch.int64)]
    - accept_indices_offset
)

self._fill_requests(batch, logits_output)
```

**为什么必须在 `_fill_requests` 之前**：`_fill_requests` 会将 `accept_indices` 从 2D `[bs, draft_token_num]` 展平为 1D（过滤掉 -1），之后无法再用 `[req_idx, num_correct_drafts]` 索引。

**文件 2**: `python/sglang/srt/speculative/ngram_worker.py`

在 `verify_input.verify()` 调用后，添加 `_mamba_verify_update` 方法及其调用：

```python
# 在 verify 后添加:
if self.target_worker.model_runner.mambaish_config is not None:
    self._mamba_verify_update(batch, verify_input)
```

```python
def _mamba_verify_update(self, batch, verify_input):
    """Update mamba/SSM states after NGRAM verify for hybrid linear attention models."""
    last_correct_step_indices = verify_input.last_correct_step_indices
    # ... 计算 mamba_steps_to_track（用于 prefix cache 的 mamba 状态追踪）
    self.target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify(
        last_correct_step_indices=last_correct_step_indices,
        mamba_track_indices=batch.mamba_track_indices,
        mamba_steps_to_track=mamba_steps_to_track,
        model=self.target_worker.model_runner.model,
    )
```

#### NGRAM vs EAGLE 的 `last_correct_step_indices` 计算对比

| 投机解码算法 | accept_index 形状 | 计算方式 |
|---|---|---|
| **EAGLE** | `[bs, spec_steps+1]` | `accept_index[req_idx, accept_lens-1] - req_idx * spec_steps` |
| **NGRAM** | `[bs, draft_token_num]` | `accept_indices[req_idx, num_correct_drafts] - req_idx * draft_token_num` |

两者本质相同：找到最后一个被接受的 token 在该 request 的 draft 窗口内的位置索引，用于从 `intermediate_ssm` 和 `intermediate_conv_window` 中恢复正确的状态。

---

## 修复日期

- 2026-04-21: 修复 1 和修复 2（启动失败问题）
- 2026-04-23: 修复 3（服务卡住问题）
- 2026-04-24: 修复 4（精度严重下降问题，gsm8k 95→23）
- 2026-06-12: 修复 5（NGRAM 投机解码输出乱码，缺少 SSM 状态回滚）
