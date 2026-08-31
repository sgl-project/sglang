# 复现手册:Agent KV 压测与包络实验(创新点 2)

> 全部结果(包络主图、三 seed 矩阵、敏感性)由以下链条产生。
> 硬件:单卡 RTX 3090(sm86);软件:sglang v0.5.2(分支 `paper/mxfp4-kv-sm86`,
> 含 `--enable-tool-retention`)+ 容器 `lmsysorg/sglang:v0.5.2-cu126`。

## 0. 环境

```bash
# 容器(bind-mount,仓库在宿主 ~/mxfp4/sglang)
docker run -d --name sglang-qwen3 --gpus '"device=3"' -p 30000:30000 \
  -v /data/models:/data/models \
  -v /home/xubowen/.ssh:/root/.ssh:ro \
  -v /home/xubowen/mxfp4/sglang:/sgl-workspace/sglang \
  lmsysorg/sglang:v0.5.2-cu126 sleep infinity

# 宿主客户端(任一有 openai 包的 python;我们用 sgl conda 环境)
# openai>=2.x,matplotlib 仅绘图需要
```

工作目录约定:实验原始输出在 `/data/xbw/kvbench/`(repo 内 `results/` 是其拷贝)。
模型:`/data/models/Qwen3-4B-Instruct-2507`。

## 1. 启动服务(四种配置)

所有实验共用同一基座命令,仅差两个 flag:

```bash
# 通用启动函数(与 scripts/*.sh 内一致)
restart() {   # $1 = 额外 flag, $2 = 日志路径
  docker exec sglang-qwen3 bash -c "pkill -9 -f 'sglang' || true"
  sleep 6
  docker exec sglang-qwen3 bash -c "cd /sgl-workspace/sglang && \
    nohup python3 -m sglang.launch_server \
      --model-path /data/models/Qwen3-4B-Instruct-2507 \
      --port 30000 --host 0.0.0.0 \
      --tool-call-parser qwen25 --context-length 32768 $1 > $2 2>&1 & sleep 1; echo started"
  for i in $(seq 1 36); do sleep 5; \
    docker exec sglang-qwen3 grep -q "fired up" $2 2>/dev/null && return 0; done
  echo TIMEOUT; exit 1
}
restart ""                                          /tmp/s_bf16.log        # BF16
restart "--kv-cache-dtype fp4_mx_block32"           /tmp/s_fp4.log         # MXFP4
restart "--enable-tool-retention"                   /tmp/s_score.log       # BF16+Score
restart "--kv-cache-dtype fp4_mx_block32 --enable-tool-retention" /tmp/s_fp4s.log  # MXFP4+Score
# no-radix 基线:BF16 命令 + --disable-radix-cache
```

> 坑:切换配置前必须 `pkill -9 -f sglang` 并等 6s,否则残留进程 OOM。
> 就绪标志:日志出现 `fired up`;池大小参照 bf16=92,495 tok / fp4=279,685 tok
> (日志 `max_total_num_tokens=`)。

## 2. 跑实验

单次运行(harness 一个参数组合 = 一份 jsonl):

```bash
B=/data/xbw/conda_envs/sglang/bin/python   # 任一带 openai 的 python
H=<repo>/paper_docs/agent_kv_bench/harness.py
$B $H --concurrency 32 --rounds 6 --seed 7 --out run.jsonl
# 关键参数:--concurrency N(会话数) --seed S(工具混合/延迟采样序列)
#          --latency-scale 0.5|2(延迟敏感性) --base-chars/--growth-chars(上下文长度)
```

论文全部矩阵用 `scripts/` 里的批处理脚本(依次重启 server + 跑各点):

| 脚本 | 内容 | 产出(→ results/) |
|---|---|---|
| `run_matrix_v3.sh` | 主矩阵:{bf16,fp4}×{LRU,Score}×N∈{8,16,32},seed7 | `v3_lru_*`,`v5_score_*`(v5=最终策略实现) |
| `run_fp4.sh` | fp4 包络加密:N∈{32,48,64}×{LRU,Score} | `fp4_lru_*`,`fp4_score_*` |
| `run_d4.sh` | 包络细扫(bf16 N∈{20,24,28}、fp4 N∈{40,44,52,56})+ 延迟敏感性(lat 0.5/2)+ seed9 | `d4_*`,`s9_*` |
| `run_seed8.sh` | seed8 头条点重复(方差) | `s8_*` |
| `run_matrix.sh`/`run_v4.sh` | 迭代中间版(策略 v1-v4,论文不用,留档) | `lru2_*`,`v4_score_*` |

复现主图只需:`run_matrix_v3.sh` + `run_fp4.sh` + `run_d4.sh`(共 ~3.5h)。

## 3. 出图与汇总

```bash
<py> <repo>/paper_docs/agent_kv_bench/plot_envelope.py \
     <repo>/paper_docs/agent_kv_bench/results  \
     <repo>/paper_docs/agent_kv_bench/envelope
# 产出 envelope.png/.pdf;数据点打印到 stdout 可直接核对
```

聚合统计(miss%/浪费 token/fast-slow 拆分)逻辑见 `harness.py:summarize`
与 `plot_envelope.py:stats`;每份 run 的即时摘要存 `<out>_summary.txt`。

## 4. 结果核对基准(seed 7,LRU,续轮 miss% / TTFT p50)

| N | bf16 | fp4 |
|---|---|---|
| 16 | 13.5% / 0.23s | — |
| 20 | 41.7% / 0.44s | — |
| 32 | 81.3% / 15.6s | 17.7% / 0.30s |
| 48 | — | 61.8% / 0.96s |
| 64 | — | 75.3% / 13.6s |

允许波动:同 seed 不同机器/批次 ±3pt(调度竞争非确定性);跨 seed 见
PROGRESS-2026-08-30.md §6-7(s7/s8/s9 三 seed 全表)。

## 5. 方法要点(写进论文的 Experimental Setup)

- 会话:每路 6 轮,每轮 = 指定工具的 tool-call 轮(非流式)→ sleep(该工具
  采样的执行延迟)→ 续轮(流式,**TTFT/prompt_tokens 为指标**)
- 工具混合(工作负载定义,指令模型执行,实测 100% 依从):
  quick_fs 0.5 / web_search 0.2 / run_tests 0.2 / code_edit 0.1;
  延迟 ~ lognormal(ln0.3,0.5) / (ln2,0.6) / (ln10,0.8) / (ln4,0.6) 秒
- 上下文:每会话唯一文档 fabricated(~2.2k tok 起步 + 每轮 ~1.2k 增长),
  续轮前缀与上轮输出精确连续(radix 命中可判定)
- miss 判定:sglang usage 不含 cached_tokens,用 TTFT 归一化
  `miss ⇔ ttft_ms > 0.1×prompt_tokens + 200`(命中/未命中实测差 100×)
- 指标:续轮 miss 率(主)、浪费重算 token(miss 轮 prompt 之和)、TTFT p50
