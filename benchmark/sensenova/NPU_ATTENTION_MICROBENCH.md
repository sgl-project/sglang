# SenseNova NPU Attention 微基准

这个测试绕过服务和完整模型，直接调用 SenseNova 当前 NPU 路径使用的
`torch.nn.functional.scaled_dot_product_attention`。它回答三个问题：

1. B2 Attention 相比 B1 是否获得吞吐提升；
2. 显式 mask 本身有多少开销；
3. 变长样本合批是否比两个紧凑的单样本 Attention 更快。

## 1. 准备代码

进入已经安装好 `torch_npu` 的环境：

```bash
cd /workspace/sglang
export PYTHONPATH=/workspace/sglang/python:${PYTHONPATH:-}
mkdir -p /workspace/sensenova-attention-results
```

运行测试时不要同时启动 SGLang 服务，避免模型占用显存和干扰计时。

## 2. 先运行小尺寸冒烟测试

```bash
python3 benchmark/sensenova/bench_npu_attention.py \
  --query-length 1024 \
  --short-prefix-length 260 \
  --long-prefix-length 286 \
  --heads 32 \
  --kv-heads 8 \
  --head-dim 128 \
  --warmup 3 \
  --iterations 10 \
  --output /workspace/sensenova-attention-results/attention-q1024.json
```

脚本末尾三个 `allclose` 都应为 `true`。否则先停止，不执行下一步。

## 3. 运行 2048 图像分辨率对应的测试

SenseNova 的 2048×2048 图像在 Attention 中对应 4096 个 image token。之前
的服务日志显示最大 prefix 为 286，因此使用 `K=4096+286=4382`：

```bash
python3 benchmark/sensenova/bench_npu_attention.py \
  --query-length 4096 \
  --short-prefix-length 260 \
  --long-prefix-length 286 \
  --heads 32 \
  --kv-heads 8 \
  --head-dim 128 \
  --warmup 5 \
  --iterations 20 \
  --native-fia \
  --output /workspace/sensenova-attention-results/attention-q4096.json
```

这个测试只执行单层 Attention，不会加载 SenseNova 模型，也不会生成 profiler
trace。结果文件通常只有几 KB。

## 4. 查看结果

```bash
python3 - <<'PY'
import json

path = "/workspace/sensenova-attention-results/attention-q4096.json"
data = json.load(open(path, encoding="utf-8"))
print("correctness:", data["correctness"])
for name, value in data["measurements"].items():
    print(f'{name:28s} {value["median_ms"]:10.3f} ms')
print("derived:")
for name, value in data["derived"].items():
    print(f"  {name}: {value:.4f}")
PY
```

各项含义：

- `b1_long_no_mask`：一个长 prompt 样本；
- `b2_no_mask`：两个等长样本直接合批，不传 mask；
- `b2_all_true_mask`：相同 B2 输入，传入全部为 True 的显式 mask；
- `b2_mixed_padding_mask`：一个短 prefix 和一个长 prefix 合批，屏蔽 padding；
- `two_compact_singletons`：将两个样本拆开，短样本移除 padding，顺序执行两次；
- `native_fia_b2_right_padded`：把短样本排成“有效 prefix + image token + 尾部
  padding”，再通过 `npu_fused_infer_attention_score` 的
  `actual_seq_lengths_kv` 跳过无效 KV；
- `native_fia_b2_left_padded`：把 prefix 右对齐并保持 image token 的统一写入
  位置，传入单元素 `kv_padding_size=[0]` 和每样本有效 KV 长度；这个布局若
  通过验证，正式接入时只需在去噪前重排一次 prefix cache；
- `b2_throughput_speedup_vs_b1`：B2 相对连续两个 B1 的理论吞吐倍数；
- `all_true_mask_overhead_pct`：仅进入显式 mask 路径产生的开销；
- `mixed_mask_overhead_vs_all_true_pct`：mask 中存在 False 后的额外开销；
- `batched_mixed_speedup_vs_compact_singletons`：当前变长合批相对两个紧凑
  单样本调用的加速倍数。
- `native_fia_right_*` 和 `native_fia_left_*`：两种原生 FIA 布局相对当前
  SDPA 的加速倍数，以及 B2 吞吐相对当前 B1 SDPA 的倍数。

判断标准：

- `b2_throughput_speedup_vs_b1` 接近 1，说明单层 Attention 在 B2 下也几乎
  没有吞吐收益；明显大于 1 才说明端到端损失发生在其他算子或调度阶段；
- `all_true_mask_overhead_pct` 较大，说明 NPU 的显式 mask 路径本身较慢；
- `mixed_mask_overhead_vs_all_true_pct` 较小，说明性能主要取决于 mask 的形状和
  算子路径，而不是 False 元素的数量；
- `batched_mixed_speedup_vs_compact_singletons` 小于或接近 1，说明当前 padding
  合批没有优于顺序执行紧凑 KV，下一步应验证支持有效序列长度的 NPU 原生接口。
