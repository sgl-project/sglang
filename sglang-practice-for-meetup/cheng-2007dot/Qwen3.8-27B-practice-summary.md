# Qwen3.8-27B 昇腾 SGLang MeetUp 实操总结

> 活动：昇腾开源软件极客营 · 上海 SGLang 场（2026-11-14）
> 任务：Qwen3.8-27B MeetUp 众测（推理部署调优）
> 环境：Atlas 800T A2（910B3）× 4 卡，TP4；镜像 `sglang:cann9.1.0-910b-20261112`（CANN 9.1.0，HDK 26.1.1）

## 1. 实操过程

1. **环境检查**：`npu-smi info` 确认 4 卡 910B3 可用；`pip show sglang` 确认 SGLang `0.0.0.dev22121+g0fa9ff55f.d20261112` 已安装，源码位于 `/sgl-workspace/sglang`；权重位于 `/workspace/shared_assets/OssSGLangMeetup/model/`（BF16 52GB / W8A8 29GB）。
2. **部署 BF16 服务**：参考 Qwen3.6-27B 部署文档，按 A2 四卡将 `--tp-size` 调整为 4，服务在 127.0.0.1:6688 拉起，`/health` 返回 200，chat completions 冒烟通过。
3. **精度测试**：`evalscope eval --datasets gsm8k --limit 200`，BF16 精度 **99.0%（198/200）**。
4. **性能基线**：128K-1K-90%cache 用例（`generated-shared-prefix`，system 115200 + question 12800 + output 1000）。基线配置并发 10 时 Mean TPOT 40.3ms；并发 16 时 TPOT 60.0ms，超出 50ms 阈值。
5. **参数调优**：
   - 开启 NEXTN 投机解码（`--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`）+ `SGLANG_ENABLE_SPEC_V2=1`，并发 16 的 TPOT 从 60.0ms 降至 28.2ms；
   - `--mem-fraction-static 0.80→0.85`、`--max-mamba-cache-size 128→192`、`--max-running-requests 32`、`--cuda-graph-bs-decode` 扩到 32；
   - 最终并发 32：**Mean TPOT 39.4ms，总吞吐 55,879 tok/s**，满足 TPOT≤50ms。
6. **部署 W8A8 服务**：增加 `--quantization modelslim`，其余同上。GSM8K 精度 **98.0%（196/200）**。
7. **W8A8 调优**：基线并发 16 时 TPOT 52.9ms 超标；应用同一套调优参数后，并发 16 TPOT 19.4ms（−63%），并发 32 **TPOT 29.2ms、总吞吐 65,487 tok/s**。

## 2. 结果汇总

| 项目 | Qwen3.8-27B (BF16) | Qwen3.8-27B-w8a8 |
|---|---|---|
| GSM8K 精度（200 条） | 99.0% | 98.0% |
| 最佳 Mean TPOT（128K-1K-90%cache） | 39.4 ms @cc32 | 29.2 ms @cc32 |
| 最佳总吞吐（TPOT≤50ms） | 55,879 tok/s | 65,487 tok/s |
| 权重体积 | 52 GB | 29 GB |

**结论**：W8A8 量化精度仅损失约 1pp，TPOT 降低约 26%、吞吐提升约 17%，长上下文高并发场景推荐 W8A8；NEXTN 投机解码是 128K 场景下 TPOT 达标的关键调优项。

## 3. 问题与建议（拟提交 Issue）

1. 部署/最佳实践文档以 A3（TP2）为例，A2 四卡用户需自行改 `--tp-size 4`，建议补充 A2 章节。
2. 最佳实践缺少 128K-1K-90%cache 用例，需按 0.9 比例手工推算输入长度，建议补充完整命令。
3. `--max-mamba-cache-size` 缺少与 `--max-running-requests`、`--mem-fraction-static` 的配比说明，建议给出经验公式。
4. 容器环境若被注入外部 `LD_LIBRARY_PATH`，会出现 `torch_npu` 导入失败，建议在 FAQ 增加排查条目。

## 4. 运行记录

- 启动/压测/精度原始日志与复现脚本见随附 `run/` 目录（`01~05_*.log`、`bench_128k_*.log`、`eval_*.log`、evalscope 报告）。
- 详细数据与完整启动命令见《Qwen3.8-27B部署调优报告》。
