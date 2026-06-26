# KV Transfer / Mooncake ratelimit-empty baseline 空载 rate-limit 基线实验报告

**日期:** 2026-06-25

**数据目录:** `kv_empty/`

**实验名称:** `ratelimit-empty baseline / 空载 rate-limit 基线实验`

## 技术摘要

- **这是一组空载基线，不是背景流实验。** `ratelimit-empty.log` 中共有 8 个 `RUN`，没有 `capfill`、`background`、`moonbg` 或 `ib_write_bw` 记录；前景观测流只使用应用层 `--rate-limit-gbps`。因此这组数据应该作为后续 `bg=1/10/50/90` 背景流实验的无背景干扰基线。
- **8 组实验均完成，且没有错误。** 每组 `aggregated-summary.csv` 都有 21 个 size、每个 size `repeat_count=20`、`error_count=0`；raw samples 中也没有 `ret != 0`。
- **split 在多流配置下吞吐更高。** 2GiB 下，`800 / 4x200` split 为 62.548 GiB/s，multi-HCA 为 24.066 GiB/s；`400 / 4x100` split 为 46.502 GiB/s，multi-HCA 为 25.536 GiB/s；`400 / 2x200` split 为 37.473 GiB/s，multi-HCA 为 24.208 GiB/s。`200 / 1x200` 两者等价，均为 23.267 GiB/s。
- **multi-HCA 单逻辑流更像一个稳定但较低的上限，而不是自动拆成多条 shard。** 在 400G/800G 目标下，multi-HCA 大 size 带宽稳定落在约 24-25.5 GiB/s，明显低于目标；这延续了此前“single logical multi-HCA 不等价于 manual split”的观察。
- **split 的稳定性取决于 per-shard 目标。** `400 / 4x100` split 和 `200 / 1x200` 基本贴住目标，尾延迟也极窄；但 `400 / 2x200` 和 `800 / 4x200` 没有达到理论目标，说明不能简单把 `mlx5_bond_i` 当作一个裸物理 200G 口来解释。结合此前事实，`mlx5_bond_i` 实际是 LACP bond，200G/shard 的结果需要按 bond、hash、单流调度和 Mooncake transfer 行为谨慎解释。

## 实验目的和配置

本次实验补齐无背景干扰下的对照：只对观测流设置应用层 `--rate-limit-gbps`，比较两种多 HCA 使用方式在空载时的延迟和吞吐。

1. **split:** 把一个逻辑传输拆成多条 shard 流，每条流绑定一个 `mlx5_bond_i`，并给每条流设置 rate-limit。
2. **multihca:** 不拆 shard，只启动一条逻辑流，把多个 HCA 作为逗号分隔的 `--ib-device` 传入，例如 `mlx5_bond_0,mlx5_bond_1,...`，并给这条逻辑流设置总 rate-limit。

所有 run 都使用 21 个逻辑总 size，从 1MiB 到 2GiB；split 组的单 shard size 会按 shard 数缩小。例如 `800 / 4x200` split 的逻辑 2GiB 对应每 shard 512MiB，而 `800 / 4x200` multihca 是一条逻辑流传 2GiB。

| Profile | Mode | Logical execution | HCA binding | Rate-limit setting | 2GiB logical size maps to |
| --- | --- | --- | --- | --- | --- |
| 800 / 4x200 | split | 4 shard | `mlx5_bond_0..3` | 4 x 200G | 512MiB per shard |
| 800 / 4x200 | multihca | 1 logical flow | `mlx5_bond_0..3` | 1 x 800G | 2GiB single flow |
| 400 / 4x100 | split | 4 shard | `mlx5_bond_0..3` | 4 x 100G | 512MiB per shard |
| 400 / 4x100 | multihca | 1 logical flow | `mlx5_bond_0..3` | 1 x 400G | 2GiB single flow |
| 400 / 2x200 | split | 2 shard | `mlx5_bond_0..1` | 2 x 200G | 1GiB per shard |
| 400 / 2x200 | multihca | 1 logical flow | `mlx5_bond_0..1` | 1 x 400G | 2GiB single flow |
| 200 / 1x200 | split | 1 shard | `mlx5_bond_0` | 1 x 200G | 2GiB single flow |
| 200 / 1x200 | multihca | 1 logical flow | `mlx5_bond_0` | 1 x 200G | 2GiB single flow |

单位说明：CSV 列名为 `bandwidth_GBps_p50`，但其数值与二进制 size 口径一致，本文按 GiB/s 解读。换算目标达成率时使用 `1 GiB/s = 8.589934592 Gbps`，也就是 `target_GiB/s = target_Gbps / 8 / 1.073741824`。

## 数据完整性

8 组数据均存在 `aggregated-summary.csv`，每组 21 行。raw samples 的行数也符合 `size_count * repeat_count * shard_count`。本地解压包没有落地每组的 `rdma-rcv-monitor.csv`，但本实验关注的是前景观测流的应用层 rate-limit 空载基线；完整性判断主要依赖 aggregated summary、raw samples 和总 log。

| Run | Mode | Target | Rows | Repeat | Raw samples | CSV errors | Raw `ret!=0` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `800_4x200_bg0_ratelimit_split` | split | 800G | 21 | 20 | 1680 | 0 | 0 |
| `800_4x200_bg0_ratelimit_multihca` | multihca | 800G | 21 | 20 | 420 | 0 | 0 |
| `400_4x100_bg0_ratelimit_split` | split | 400G | 21 | 20 | 1680 | 0 | 0 |
| `400_4x100_bg0_ratelimit_multihca` | multihca | 400G | 21 | 20 | 420 | 0 | 0 |
| `400_2x200_bg0_ratelimit_split` | split | 400G | 21 | 20 | 840 | 0 | 0 |
| `400_2x200_bg0_ratelimit_multihca` | multihca | 400G | 21 | 20 | 420 | 0 | 0 |
| `200_1x200_bg0_ratelimit_split` | split | 200G | 21 | 20 | 420 | 0 | 0 |
| `200_1x200_bg0_ratelimit_multihca` | multihca | 200G | 21 | 20 | 420 | 0 | 0 |

## 重点 size 对比

下表单元格格式为 `p50/p90/p99 latency ms, p50 bandwidth GiB/s`。`multi-HCA bw vs split` 是同一 profile 和 size 下 multi-HCA 相对 split 的 p50 带宽差异；负数表示 multi-HCA 更低。

### 512MiB

| Profile | split p50/p90/p99, bw | multihca p50/p90/p99, bw | multi-HCA bw vs split | multi-HCA p50 latency vs split |
| --- | --- | --- | ---: | ---: |
| 800 / 4x200 | 8.138/8.168/8.172 ms, 61.440 GiB/s | 20.770/20.820/20.863 ms, 24.073 GiB/s | -60.8% | +155.2% |
| 400 / 4x100 | 10.795/10.801/10.856 ms, 46.318 GiB/s | 19.587/19.615/20.927 ms, 25.527 GiB/s | -44.9% | +81.4% |
| 400 / 2x200 | 13.346/13.365/13.383 ms, 37.464 GiB/s | 20.590/20.631/20.646 ms, 24.284 GiB/s | -35.2% | +54.3% |
| 200 / 1x200 | 21.533/21.556/21.556 ms, 23.220 GiB/s | 21.533/21.553/21.562 ms, 23.221 GiB/s | +0.0% | -0.0% |

### 1GiB

| Profile | split p50/p90/p99, bw | multihca p50/p90/p99, bw | multi-HCA bw vs split | multi-HCA p50 latency vs split |
| --- | --- | --- | ---: | ---: |
| 800 / 4x200 | 16.285/16.531/17.213 ms, 61.407 GiB/s | 41.557/41.615/41.939 ms, 24.063 GiB/s | -60.8% | +155.2% |
| 400 / 4x100 | 21.534/21.554/21.648 ms, 46.438 GiB/s | 39.164/39.194/39.284 ms, 25.534 GiB/s | -45.0% | +81.9% |
| 400 / 2x200 | 26.680/26.715/26.731 ms, 37.481 GiB/s | 41.198/41.265/41.316 ms, 24.273 GiB/s | -35.2% | +54.4% |
| 200 / 1x200 | 43.007/43.015/43.070 ms, 23.252 GiB/s | 43.007/43.027/43.030 ms, 23.252 GiB/s | -0.0% | +0.0% |

### 2GiB

| Profile | split p50/p90/p99, bw | multihca p50/p90/p99, bw | multi-HCA bw vs split | multi-HCA p50 latency vs split |
| --- | --- | --- | ---: | ---: |
| 800 / 4x200 | 31.975/32.319/32.498 ms, 62.548 GiB/s | 83.104/83.292/83.651 ms, 24.066 GiB/s | -61.5% | +159.9% |
| 400 / 4x100 | 43.009/43.013/43.031 ms, 46.502 GiB/s | 78.322/78.362/78.430 ms, 25.536 GiB/s | -45.1% | +82.1% |
| 400 / 2x200 | 53.372/53.497/54.526 ms, 37.473 GiB/s | 82.619/82.734/82.900 ms, 24.208 GiB/s | -35.4% | +54.8% |
| 200 / 1x200 | 85.958/85.980/86.026 ms, 23.267 GiB/s | 85.958/85.979/85.981 ms, 23.267 GiB/s | +0.0% | -0.0% |

## 目标达成率和稳定性

2GiB 是最接近稳态大块 KV transfer 的点。按 `bandwidth_GBps_p50` 的 GiB/s 口径换算：

| Run | 2GiB p50 bw | Target equiv | Utilization | p99-p50 latency | p99/p50 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| `800_4x200_bg0_ratelimit_split` | 62.548 GiB/s | 93.132 GiB/s | 67.2% | 0.523 ms | 1.0163 |
| `800_4x200_bg0_ratelimit_multihca` | 24.066 GiB/s | 93.132 GiB/s | 25.8% | 0.547 ms | 1.0066 |
| `400_4x100_bg0_ratelimit_split` | 46.502 GiB/s | 46.566 GiB/s | 99.9% | 0.022 ms | 1.0005 |
| `400_4x100_bg0_ratelimit_multihca` | 25.536 GiB/s | 46.566 GiB/s | 54.8% | 0.108 ms | 1.0014 |
| `400_2x200_bg0_ratelimit_split` | 37.473 GiB/s | 46.566 GiB/s | 80.5% | 1.154 ms | 1.0216 |
| `400_2x200_bg0_ratelimit_multihca` | 24.208 GiB/s | 46.566 GiB/s | 52.0% | 0.281 ms | 1.0034 |
| `200_1x200_bg0_ratelimit_split` | 23.267 GiB/s | 23.283 GiB/s | 99.9% | 0.067 ms | 1.0008 |
| `200_1x200_bg0_ratelimit_multihca` | 23.267 GiB/s | 23.283 GiB/s | 99.9% | 0.022 ms | 1.0003 |

**达成率结论。**

- `200 / 1x200` 是单流 200G 对照，split 和 multihca 完全重合，2GiB 约 23.267 GiB/s，等效约 199.9Gbps，说明应用层 `--rate-limit-gbps 200` 本身能在空载下稳定生效。
- `400 / 4x100` split 几乎贴住 400G 目标，2GiB 为 46.502 GiB/s，利用率 99.9%。这说明把 400G 拆成 4 条 100G shard 是本组里最符合目标的多流方式。
- `400 / 2x200` split 只有 37.473 GiB/s，约 80.5% 目标；`800 / 4x200` split 只有 62.548 GiB/s，约 67.2% 目标。它们都使用 200G per-shard rate-limit，但没有线性叠加到理论值。
- `400G` 和 `800G` 的 multihca 单逻辑流均明显低于目标，2GiB 约 24-25.5 GiB/s。增加 HCA 名称没有自动把一条逻辑流扩展到 400G 或 800G。

**稳定性结论。**

- 从大 size 的带宽曲线看，所有组在 512MiB 到 2GiB 区间都很平，说明结果不是由单个 size 异常驱动。
- 从尾延迟看，multi-HCA 单逻辑流的 p99/p50 很小，尤其 400G 组只有约 1.001-1.003；但这是在较低吞吐平台上稳定，并不代表更好地利用了目标带宽。
- split 的 `400 / 4x100` 和 `200 / 1x200` 同时满足“高吞吐”和“低尾延迟”；`400 / 2x200` 和 `800 / 4x200` 吞吐更高于 multihca，但 p99/p50 略大，并且未达到理论目标。

## 大 size 趋势

下表列出 512MiB 到 2GiB 的 p50 bandwidth，单位为 GiB/s。

| Run | 512MiB | 768MiB | 1GiB | 1.25GiB | 1.5GiB | 1.75GiB | 2GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `800_4x200_bg0_ratelimit_split` | 61.440 | 61.257 | 61.407 | 61.271 | 61.291 | 61.461 | 62.548 |
| `800_4x200_bg0_ratelimit_multihca` | 24.073 | 24.072 | 24.063 | 24.068 | 24.070 | 24.073 | 24.066 |
| `400_4x100_bg0_ratelimit_split` | 46.318 | 46.387 | 46.438 | 46.463 | 46.482 | 46.493 | 46.502 |
| `400_4x100_bg0_ratelimit_multihca` | 25.527 | 25.526 | 25.534 | 25.535 | 25.535 | 25.533 | 25.536 |
| `400_2x200_bg0_ratelimit_split` | 37.464 | 37.477 | 37.481 | 37.466 | 37.465 | 37.434 | 37.473 |
| `400_2x200_bg0_ratelimit_multihca` | 24.284 | 24.303 | 24.273 | 24.277 | 24.295 | 24.257 | 24.208 |
| `200_1x200_bg0_ratelimit_split` | 23.220 | 23.242 | 23.252 | 23.258 | 23.261 | 23.264 | 23.267 |
| `200_1x200_bg0_ratelimit_multihca` | 23.221 | 23.241 | 23.252 | 23.258 | 23.262 | 23.265 | 23.267 |

趋势可以分成三类：

1. **贴住目标的组:** `200 / 1x200` split/multihca 和 `400 / 4x100` split。它们从 512MiB 到 2GiB 基本线性扩展，p50 bandwidth 贴近 rate-limit 目标。
2. **稳定但低于目标的 multi-HCA 组:** `400 / 4x100`、`400 / 2x200`、`800 / 4x200` 的 multihca 都很平，但主要落在约 24-25.5 GiB/s。这说明 single logical flow 在空载下也没有等价拆成多条 shard。
3. **高于 multi-HCA 但未满目标的 200G/shard split 组:** `400 / 2x200` split 和 `800 / 4x200` split 明显快于对应 multihca，但未达到理论 400G/800G。这个现象不能简单归因于“物理口不够”，因为 `mlx5_bond_i` 是 LACP bond；更合理的表述是：在当前 Mooncake、应用层限速、bond 和路径选择组合下，200G/shard 没有像 100G/shard 那样稳定线性叠加。

## 对后续背景流实验的意义

这组结果提供了 `bg=0` 的空载参照：

- 后续 `bg=1/10/50/90` 如果使用 split，应该分别与同 profile 的 split 空载基线比较，而不是只和理论 cap 比较。尤其 `400 / 2x200` 和 `800 / 4x200` 的空载基线本来就没有满理论值。
- 后续如果使用 multihca 单逻辑流，在 400G/800G 目标下应先假设其空载上限约为 24-25.5 GiB/s，再观察背景流带来的额外下降；不要把 400G/800G 理论值当作该执行模型已经能达到的无背景上界。
- `400 / 4x100` split 是本批最干净的多流基线：空载下既贴近 400G，又有极窄尾延迟，适合用来评估背景流是否真正造成吞吐下降或尾延迟上升。

## Caveats

- `bandwidth_GBps_p50` 沿用原始 CSV 列名；本文按 GiB/s 解读，并用 `bandwidth_GBps_p50 * 8.589934592` 换算等效 Gbps。
- 本地结果包只有 aggregated summary 和 raw foreground samples，没有落地每组 `raw/rdma-rcv-monitor.csv`；因此本报告不做端口级接收 counter 结论。
- `mlx5_bond_i` 需要按 LACP bond 谨慎解释。即使名字里有 `bond_0`、`bond_1`，也不能把它简单等同为“单个物理 200G 口”，更不能据此直接推导 2x200 或 4x200 应当线性达到 400G/800G。
- 本报告是描述性基线分析，不证明 single logical multi-HCA 低于目标的根因；可能因素包括 Mooncake 单逻辑 transfer 调度、连接/队列路径选择、LACP hashing、应用层 rate-limit 和 GPU/host 侧执行细节。

## 结论

空载基线显示：**如果目标是稳定吃满 400G，当前最可靠的是 `400 / 4x100` split；如果目标是简化为 single logical multi-HCA，它在 200G 单流下表现正常，但在 400G/800G 下不会自动变成多 shard 聚合。** 后续背景流实验应把这份 bg0 结果作为基线，分别评估 split 和 multihca 在已有空载上限之上的退化，而不是直接用理论总带宽判断成败。
