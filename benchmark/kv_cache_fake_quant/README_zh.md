# SGLang 中 MXFP4 KV Cache 假量化的实现说明

## 1. 目的

评估 KV cache 量化成 MXFP4 后在 AMD GPU（MI355X）上的精度，实验设置对齐 LMSYS 的 NVFP4 KV cache 博客（2026-09-16）。我们只关心精度，所以不实现真正的 FP4 attention kernel，而是用**假量化（fake quantization，QDQ）**：

> K/V 先量化到 MXFP4，再立即反量化回 BF16，存进 BF16 的 KV cache。attention 仍然用原来的 BF16 kernel。

具体到一次 QDQ（对 K 和 V 各做一次，输入是 `[tokens, kv_heads, head_dim]` 的 BF16 张量），步骤如下。以下都是对**每个 token、每个 head、沿 `head_dim` 每 32 个连续元素为一组**独立进行的：

1. **分组**：把 `head_dim` 切成每 32 个一组（head_dim=256 就是 8 组）。
2. **求 scale**（E8M0，即 2 的整数次幂）：
   - 取组内 `amax = max(|x|)`。
   - 按 Quark 的 `even` 模式调整 `amax`：尾数 ≥ 1.75 就进位到下一个 2 的幂，否则取 2^⌊log2 amax⌋。
   - `scale = 2^(⌊log2(调整后的 amax)⌋ − 2)`，其中 2 是 E2M1 最大指数。
3. **量化**（BF16 → FP4）：
   - 计算 `x / scale`。因为 scale 是 2 的幂，这一步是精确的。
   - 把结果舍入到最近的 E2M1 值，可取的绝对值是 {0, 0.5, 1, 1.5, 2, 3, 4, 6}，符号单独保留。
   - 恰好落在两个相邻值正中间时按就近取偶（RNE）：0.25→0，0.75→1，1.25→1，1.75→2，2.5→2，3.5→4，5→4。
   - 绝对值超过 6 的截断到 ±6。
   - 真实的 MXFP4 在这里存的是每个元素 4 bit 加每组 8 bit 的 scale。假量化不会真的存这些，只是在内存里把它们算出来，马上用于下一步。
4. **反量化**（FP4 → BF16）：用步骤 3 得到的 E2M1 值乘回这一组的 `scale`，得到 BF16 张量，形状和输入完全一样。
5. **写入 KV cache**：把反量化后的 BF16 张量当成普通的 K/V 写入 BF16 的 KV cache。之后 attention 读到的就是这些"被 MXFP4 网格取整过"的值。

**一个具体的例子**：某组的 32 个值里，最大绝对值是 5.3，另外有 0.7、−2.4、0.2。

- 求 `amax = 5.3` 的尾数：
  - 先求指数：`⌊log2(5.3)⌋ = ⌊2.406⌋ = 2`，也就是 4 ≤ 5.3 < 8。
  - 把 5.3 写成 `尾数 × 2^指数`，尾数 = `5.3 / 2^2 = 5.3 / 4 = 1.325`，所以 `5.3 = 1.325 × 2^2`，尾数总是落在 [1, 2) 之间。
- 尾数 1.325 < 1.75，不进位，调整后的 amax = 2^2 = 4，所以 `scale = 2^(⌊log2 4⌋ − 2) = 2^(2 − 2) = 1`。
- 量化：5.3 落在 4 和 6 之间，离 6 更近，取 6。0.7 在 0.5 和 1 之间，小于中点 0.75，取 0.5。−2.4 在 2 和 3 之间，小于中点 2.5，取 −2。0.2 小于 0.25，取 0。
- 反量化：乘上 scale=1，得到 6、0.5、−2、0。这四个值写进 cache，原来的 5.3、0.7、−2.4、0.2 不再保留。

再看一个 scale 不为 1 的组：`amax = 7.2`，尾数 1.8 ≥ 1.75，进位到 8，`scale = 2^(3 − 2) = 2`。此时 7.2 / 2 = 3.6，取 4，反量化回 8。如果 `amax = 6.8`，尾数 1.7 < 1.75，取 4，`scale = 1`，6.8 被截断到 6。

这种做法能精确还原 MXFP4 的数值误差，但显存和速度不会有任何收益。另外，MXFP4 的每个值都是 E2M1 乘上一个 2 的整数次幂，用 BF16 存放不会再产生额外的舍入误差。所以 cache 里存的就是 MXFP4 能表示的值，没有偏差。

## 2. 量化格式：MXFP4（OCP MX）

| 项 | 取值 |
|---|---|
| 元素格式 | FP4 E2M1，可表示的绝对值是 {0, 0.5, 1, 1.5, 2, 3, 4, 6} |
| 分组 | 沿 `head_dim` 每 32 个连续元素一组，也就是每个 token、每个 head 内部分组。head_dim=256 就是 8 组 |
| Scale | E8M0，只能取 2 的整数次幂，每组一个 |
| Scale 的计算 | Quark 的 `"even"` 模式，见下文 |
| 元素舍入 | 就近取偶（RNE），超过 ±6 的截断到 ±6 |
| 量化位置 | K 在做完 RoPE 之后，V 在投影之后，也就是写进 cache 之前 |

**`even` 模式的 scale 计算**（`quark.torch.quantization.utils.even_round`）：

1. 取组内绝对值的最大值 `amax`。它的尾数如果 ≥ 1.75，就向上进位到下一个 2 的幂，否则向下取到 2^⌊log2 amax⌋。
2. `scale = 2^(⌊log2(上一步结果)⌋ − 2)`。这里的 2 是 E2M1 的最大指数，6 = 1.5 × 2²。

按这个规则，`amax/scale` 落在 [3.5, 7) 之间，只有尾数在 1.5 到 1.75 之间的最大值才会被截断到 6。相比 OCP 规范里朴素的 `floor(log2(amax)) − 2`，截断误差更小。

QDQ 直接调用 Quark 的 `quark.torch.kernel.mx.qdq_mxfp4(x, "even")`，实际走的是 HIP 实现 `qdq_mxfp4_hip`。这样 MXFP4 的定义和 Quark 完全一致。

## 3. 语义：哪些 K/V 被量化

对齐博客中真量化的行为：

| 阶段 | 当前 chunk 自己的 K/V | 从 cache 读的历史 K/V |
|---|---|---|
| prefill（一次做完） | **原始精度** | 无 |
| chunked prefill | **原始精度** | MXFP4 |
| decode | MXFP4（新 token 先写 cache 再读） | MXFP4 |

一句话：**当前 chunk 用原始 K/V 计算 attention，凡是从 cache 读出来的都是 MXFP4。**

初始预填充和分块预填充在代码里都是 extend 模式，`k_qdq` 在各阶段的作用如下（代码见 4.2 节）：

| 阶段 | 当前 chunk 自己的 K/V | 前缀（之前的 token）的 K/V | `k_qdq` 的作用 |
|---|---|---|---|
| 初始预填充（一次处理完整个 prompt） | 原始 BF16 | 没有前缀 | 不参与本次计算，只是算完后覆盖进 cache，给后面的 decode 用 |
| 分块预填充（第 2 个及之后的 chunk） | 原始 BF16 | 从 cache 读，读到的是前面 chunk 写进去的 `k_qdq`，也就是 MXFP4 的值 | 本 chunk 算完后同样覆盖进 cache，给下一个 chunk 和 decode 用 |
| decode | 新 token 直接用 `k_qdq`：先写进 cache 再从 cache 读 | 从 cache 读，是 MXFP4 的值 | 全部是 MXFP4 的值 |

## 4. 代码改动

一共改了两个文件，用环境变量 `SGLANG_KV_CACHE_FAKE_QUANT` 控制。不设置时走原来的代码路径，行为不变。

### 4.1 `python/sglang/srt/environ.py`

新增 `SGLANG_KV_CACHE_FAKE_QUANT = EnvStr("")`，目前只接受 `mxfp4`，其他值在导入时直接报错。

### 4.2 `python/sglang/srt/layers/radix_attention.py`

原来的 `RadixAttention.forward` 改名为 `_forward`。新的 `forward` 在下面几种情况下保持原有路径不变：开关没打开；`k` 为 None（跨层共享 KV）；不写 cache；cross attention；idle batch。其余情况走 `_forward_kv_fake_quant`：

```python
k_qdq = qdq_mxfp4(k)   # [tokens, kv_heads, head_dim]，沿 head_dim 每 32 个一组
v_qdq = qdq_mxfp4(v)

if decode:
    # decode 时 attention 直接从 cache 读，新 token 也在其中。
    # 把 QDQ 后的 K/V 交给 backend 写入 cache，读到的就全是 MXFP4。
    return _forward(q, k_qdq, v_qdq, save_kv_cache=True)

# extend / prefill:
out = _forward(q, k, v, save_kv_cache=True)       # ① 当前 chunk 用原始 K/V 计算并写 cache
pool.set_kv_buffer(layer, out_cache_loc, k_qdq, v_qdq)  # ② 用 QDQ 值覆盖本 chunk 在 cache 中的槽位
return out
```

**为什么要算 `k_qdq`**

`k_qdq` 是 K 经过 MXFP4 量化再反量化后的结果，dtype 和形状都和输入一样（BF16，`[tokens, kv_heads, head_dim]`），只是每个值都被取整到了 MXFP4 能表示的网格上。`v_qdq` 同理。

目的是在不改 attention kernel、不改 cache 格式的前提下，让“从 cache 读出来的 K/V”带上和真实 MXFP4 存储完全相同的数值误差。

真实的 MXFP4 KV cache 是这样工作的：

```text
写 cache：BF16 K ──量化──> 4 bit E2M1 + 每 32 个元素一个 E8M0 scale（存进显存）
读 cache：4 bit + scale ──反量化──> 参与 attention 计算
```

attention 真正用到的，是“反量化之后的值”。假量化把量化和反量化一步做完（QDQ），直接得到这个值，用 BF16 存起来：

```text
BF16 K ──QDQ──> k_qdq（BF16，值在 MXFP4 网格上）──> 写进 BF16 cache
```

两种方式下，attention 读到的数值是一样的。之所以能做到完全一致，是因为 MXFP4 的每个值都是“E2M1 × 2 的整数次幂”，BF16 能精确表示，存进 BF16 cache 不会再产生额外的舍入误差。

extend 分两步做，原因如下：

- ① 让 backend 按原有流程写 cache 并计算 attention。有些 backend（比如 aiter 的部分 prefill kernel）会把当前 chunk 从 cache 里**回读**，所以必须先写原始值，当前 chunk 才能看到原始精度。
- ② attention 算完以后，把这个 chunk 在 cache 里的位置覆盖成 QDQ 值。之后的 chunk 和 decode 读到的都是 MXFP4。

### 4.3 为什么放在 attention 层，而不是写 cache 的函数里

aiter backend 写 KV cache 有五六条路径：普通写入、fused FP8 写入、5D shuffle 布局、unified attention 等。如果在 `set_kv_buffer` 或各个 backend 里加 QDQ，就得逐条路径修改，容易漏掉，也很难保证"当前 chunk 不量化"。放在 `RadixAttention` 这一层统一处理，与具体 backend 无关。

### 4.4 约束

- 必须使用 BF16/FP16 的 KV cache，即 `--kv-cache-dtype auto`，代码里有 assert 检查。如果 cache 本身是 FP8，就会变成 MXFP4 再叠加 FP8 两次量化，结果不是要测的东西。
- 需要能 `import quark`（AMD Quark），而且 Quark 的 HIP 扩展要能编译或加载。
- QDQ 能被 CUDA graph 捕获，decode 时可以照常用 CUDA graph。
- extend 阶段多了一次 `set_kv_buffer`，会有少量额外开销，用于精度实验可以接受。
- 线性注意力或 Mamba 层（比如 Qwen3.5/3.8 中的 GDN 层）本来就没有 KV cache，不受影响，只有 full attention 层会被量化。

## 5. 正确性验证

### 5.1 QDQ kernel 逐 bit 对齐

用纯 PyTorch 写了一个参考实现：even scale 加 RNE 取整到 E2M1。用 8192×2×256 的随机 BF16 张量对比：

- `qdq_mxfp4_hip`：和参考实现**逐 bit 一致**，0 个元素不同。
- `qdq_mxfp4_triton`：约 **0.7%** 的元素不同。原因是这个实现在恰好落在两个值中间时，是远离 0 取整，而不是就近取偶。所以我们固定使用 HIP 实现，并建议 Quark 团队看一下 Triton 版本。

### 5.2 端到端语义验证

模型是 Qwen3.8-27B-FP8，prompt 长 1067 token，greedy 解码，关闭 radix cache。比较前 3 个输出 token 的 logprob：

| 配置 | chunked_prefill_size | token 1 | token 2 | token 3 |
|---|---|---|---|---|
| 关（BF16） | 8192（一次 prefill） | −0.02455 | −1.00951 | −0.25823 |
| 关（BF16） | 128 | −0.02455 | −1.00951 | −0.25823 |
| MXFP4 | 8192（一次 prefill） | **−0.02455** | −1.18503 | −0.37080 |
| MXFP4 | 128 | −0.02232 | −1.10742 | −0.32116 |

- 一次 prefill 时，MXFP4 的第 1 个 token 和 BF16 **完全相同**，说明当前 chunk 用的是原始 K/V；从第 2 个 token（decode）开始不同，说明 decode 读到的是量化后的 cache。
- chunk=128 时，第 1 个 token 也不同了，说明 chunked prefill 中前缀从 cache 读出来的是 MXFP4。
- BF16 下 128 和 8192 两种分块的结果完全相同，排除了分块本身带来差异的可能。

## 6. 使用方法

```bash
SGLANG_KV_CACHE_FAKE_QUANT=mxfp4 python -m sglang.launch_server \
  --model-path <MODEL> --kv-cache-dtype auto --disable-radix-cache ...
```

第 7 节的评测：GSM8K 8-shot 用 `sglang.test.run_eval`，GPQA-Diamond 和 AIME25 用 `sgl-eval`，采样参数是温度 0.6、top_p 0.95、top_k 20。除 KV 配置外，三组的服务参数完全相同。

## 7. 当前结果（Qwen3.8-27B-FP8，MI355X，TP1）

| KV | GSM8K | GPQA-D pass@1（4 次） | AIME25 pass@1（8 次，xhigh） | 平均输出长度 GPQA / AIME |
|---|---|---|---|---|
| BF16 | 97.33 | 89.77 ± 0.73 | 95.00 ± 1.41 | 13.7k / 12.9k |
| FP8 | 98.09 | 89.39 ± 0.29 | 95.00 ± 1.26 | 14.2k / 13.0k |
| MXFP4（假量化） | 97.79 | 90.53 ± 0.13 | 94.58 ± 0.88 | 14.7k / 14.3k |

（± 为标准误。）

**初步结论**：三项分数上，MXFP4 和 BF16、FP8 的差距都在噪声范围内。但 MXFP4 下的思考输出变长了：相对 BF16，GPQA 长约 7%，AIME 长约 11%；相对 FP8，GPQA 长约 4%，AIME 长约 10%。被截断的比例也略高。这一点值得在更大的模型（Qwen3.5-397B）上进一步确认。

## 8. 局限

- 这是假量化，只模拟 K/V 存储上的数值误差。真实的 MXFP4 attention kernel 如果还量化了 Q 或 P，或者反量化的精度、累加顺序不同，这部分误差都模拟不到。
- 博客中描述 NVFP4 是从 FP8 KV 再量化得到的，我们的 MXFP4 是从 BF16 直接量化，起点精度略高。
- MXFP4 是动态量化，scale 按组实时计算，不需要校准。NVFP4 那样的 per-tensor global scale 在这里没有对应物。
