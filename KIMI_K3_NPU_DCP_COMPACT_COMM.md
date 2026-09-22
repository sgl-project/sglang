# Kimi-K3 NPU DCP 紧凑通信实现

## 结论

NPU 不需要新增 AscendC 算子。SGLang GPU DCP 已有的两个 Triton 算子可以由
Triton-NPU 直接编译并复用：

- `dcp_pack_a2a_send`：把 BF16/FP16 partial output 与 FP32 LSE 写入同一连续发送
  buffer；
- `dcp_lse_combine_triton`：直接从接收 buffer 的 output/LSE view 读取并完成 online
  softmax merge，不生成独立 unpack tensor。

NPU backend 因此直接复用公共 `dcp_a2a_lse_reduce`，通信仍然是每层一次 HCCL
AllToAll。

## 数据路径变化

修改前：

```text
BF16 partial output -> FP32 cast
FP32 output + FP32 LSE -> HCCL AllToAll
unpack/reshape -> torch_npu.npu_attention_update -> BF16 output
```

修改后：

```text
BF16 partial output + FP32 LSE
        -> Triton compact pack
        -> one HCCL AllToAll over raw bytes
        -> Triton fused packed-view read + LSE merge
        -> BF16 output
```

K3 的 `head_dim=512`。每个 partial state 的通信量由
`(512 + 1) * 4 = 2052` 字节降为 `512 * 2 + 4 = 1028` 字节，约减少
49.9%。collective 次数没有变化。

## 现有算子复用验证

在 A3 NPU 上使用 K3 实际 shape（DCP=2、每个目的 rank 6 heads、head_dim=512）验证：

- compact pack 的 BF16 output 和 FP32 LSE 均逐位一致；
- fused combine 与 PyTorch reference 的单算子探针 `max_abs=0`；
- 双 NPU、真实 HCCL AllToAll 端到端结果 `max_abs=0.00774`；
- raw `uint8` HCCL AllToAll 逐位一致；
- 双 NPU NPUGraph capture/replay 输出有限且执行成功。

GPU kernel 原本未处理“所有 DCP shard 都为空”的 graph padding 行，会产生 `0/0`
NaN。本分支把该行定义为 online-softmax identity：output 为 0、LSE 为 `-inf`。这既满足
NPU Target Verify graph 语义，也使公共 kernel 的边界行为完整。

## 后续验收

四机服务验收需要对比修改前后：

- GPQA 精度；
- 单请求及 BS=2/4/8 TPOT；
- Target Verify graph HBM；
- profiler 中 DCP AllToAll 字节量和 merge 周边算子耗时；
- 长稳、空 DP graph row 和 128K 输入。
