# 昇腾单卡复现 AdaFlash Thinking

## 环境

实测为 Ascend 910B2 64GB 单卡，镜像：

```text
swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.1.0-910b-20261111
```

配套版本：Python 3.12.13、CANN 9.1.0、torch 2.10.0、torch-npu
2.10.0.post6、Transformers 5.19.0，使用镜像自带的 sgl-kernel-npu。
在已加载 CANN/ATB 环境、`python3` 能导入 `torch_npu` 的终端执行下列命令。
使用该镜像时，配套 Python 位于 `/usr/local/python3.12.13/bin`，必要时将它加入 PATH。

## 四组一键运行

在本分支仓库根目录执行：

```bash
export PYTHONPATH="$PWD/python:${PYTHONPATH:-}"
export HF_ENDPOINT=https://hf-mirror.com  # 可选，直接访问 HF 时省略
export HF_HUB_DISABLE_XET=1

python3 test/manual/ascend/benchmark_adaflash_aime25.py \
  --output-dir "results/aime25-$(date +%Y%m%d-%H%M%S)"
```

脚本自动下载模型和 AIME 2025 I、II 共 30 题，先短测确认 Graph 生效，再顺序运行
AR、DFlash、AdaFlash Thinking 开动态头、关动态头。没有节点路径或外部辅助脚本依赖。
模型下载使用 Hugging Face 标准缓存，可通过 `HF_HOME` 更改位置。

默认参数与实测一致：thinking=True、temperature=0、并发8、最大输出32768、
context_length40960、BF16、TP1、mem_fraction_static0.90、NPU decode Graph full、
capture bs1..8；prefill Graph 和 overlap 关闭。

已有本地模型时可传入自己的路径，或替换为其他 Hugging Face ID：

```bash
python3 test/manual/ascend/benchmark_adaflash_aime25.py \
  --model /your/models/Qwen3-8B \
  --dflash-model /your/models/Qwen3-8B-DFlash-b16 \
  --adaflash-model /your/models/Qwen3-8B-AdaFlash-Thinking \
  --output-dir results/aime25-local
```

仅运行一组：

```bash
python3 test/manual/ascend/benchmark_adaflash_aime25.py \
  --methods adaflash-dynamic --output-dir results/aime25-dynamic
```

`--methods` 支持 `ar`、`dflash`、`adaflash-dynamic`、`adaflash-fixed`。
`--help` 可查看并发、输出长度、端口等参数。每次使用新的结果目录；脚本拒绝覆盖已有结果。
建议在 tmux 中运行上述命令，脱离会话时按 Ctrl-b、d。

## 输出

- `comparison.md`：四列汇总表；`comparison.json` 保留完整指标。
- `status.json`：当前阶段或失败原因。
- `*-responses.jsonl`、`*-raw/`：逐题原始响应，不评分正确率。
- `*-server.log`、`*-graph-proof.json`：服务日志及 Graph 回放记录。
- `config.json`、`resolved-models.json`：配置和实际模型缓存路径。

吞吐为整轮实际输出 token / 总耗时，包含 prefill 和 decode，排除模型加载、Graph
capture 和预热。预热为并发数个请求、每个32 token，之后清空前缀缓存再开始计时。
默认固定 Thinking 权重和数据集 revision；target 的缓存路径会记录实际下载版本。

## 已有实测

以下为同配置的已有完整30题结果；新增脚本保留相同提示词、预热及计时方式。

| 方法 | 吞吐（token/s） | 相对 AR 加速 | 平均接受长度 |
|---|---:|---:|---:|
| AR | 202.90 | 1.00× | — |
| DFlash | 315.35 | 1.55× | 2.92 |
| AdaFlash Thinking，动态头开 | 431.75 | 2.13× | 4.42 |
| AdaFlash Thinking，动态头关 | 675.50 | 3.33× | 6.00 |

各组实际生成的 token 总量不同，吞吐按各自实际输出计算。
