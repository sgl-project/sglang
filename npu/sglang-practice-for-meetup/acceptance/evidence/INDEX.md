# Evidence index — what proves what

Screenshots live in `evidence/`. Raw artifacts were packed on the box into `/workspace/v4run-evidence.tgz`
(logs + JSONL + accuracy JSON). Both are needed: screenshots show the actions, artifacts carry the numbers.

| # | Evidence | File | Proves | Strength |
| --- | --- | --- | --- | --- |
| 1 | MMLU 5-shot run | `evidence/01-mmlu-accuracy.webp` | accuracy test executed: command + `questions=1140`, progress, **acc=89.04% (1015/1140 answered)**, per-subject worst rows, `saved …/mmlu_shot.json` | **strong** — command and result in one frame |
| 2 | Graph-mode TPOT | `evidence/02-graph-tpot.webp` | benchmark commands + **8k: TPOT p50 20.66 ms**; **c1: TPOT p50 19.58 ms** | **strong** — TPOT block visible for both cells |
| 3 | Serving + hardware | `evidence/03-smoke-npusmi.webp` | `/generate` returns correct text (`Paris…`), **4 × Ascend950PR**, HBM 123625/131072 per die, `sglangschedul` process | **strong** — functional + hardware in one frame |

## Covered by artifacts only (not visible in the three screenshots)

| Claim | Where it lives | Action needed |
| --- | --- | --- |
| Graph capture succeeded (13.25 s, 5.06 GB) | `server-v8graph.log` inside `/workspace/v4run-evidence.tgz` | either submit the tarball, or screenshot `grep -E "Capture target decode NPU graph" <log>` |
| Saturation sweep (c=16/32/64 → 803 tok/s) | `dsv4-g-sat*.jsonl` + earlier terminal output | screenshot `python3 -c` summary of `dsv4-g-sat*.jsonl`, or quote the pasted output |
| `health=200` | server log / any benchmark implies it | restart briefly and screenshot `curl /health`, or cite the benchmark runs |
| Launch env / exact command that produced the numbers | `logs/server-v8graph.log` first line `server_args={...}` | screenshot that line, or submit the tarball |

## Minimum acceptable submission

1. `ACCEPTANCE.md` (the report, with limitations)
2. `EVIDENCE_INDEX.md` + the three screenshots
3. `/workspace/v4run-evidence.tgz` downloaded from the box (raw logs/JSONL/accuracy JSON) — **this is the piece that
   turns screenshots into reproducible evidence**
4. PR link once pushed

## Statement that ties it together (paste in the submission)

> 评测均在本机（4×昇腾950PR，512GB）完成：MMLU 5-shot 1,140 题/作答 1,015 题，准确率 89.04%（保守口径 79.3%），
> 原始 JSON 见证据包；图模式下单流 TPOT 19.58 ms（c=1，in128/out512）、20.66 ms（in8192/out1024），
> 并发 c=8 输出 178.9 tok/s，饱和 c=64 达 803 tok/s；`/generate` 输出正确，`npu-smi` 显示 4 卡健康、HBM 约 94%。
> 完整日志与压测 JSONL 打包于 `v4run-evidence.tgz`。
