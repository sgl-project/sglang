# 整合实现与最终验收

实现分支：`rfc-kvc`。执行代码快照：`f65f0393c4c916184c6c9017fafb9fa389ec2f4a`；Rust Router 代码与 `35a18bf008` 一致。后续提交补充文档与证据。固定上游基线已合入，原分支历史保留。

已交付首轮 P0–P8 范围：独立 PREFETCH、区分保存与回收的 DEMOTE、实际 L2 eviction-exclusion PIN、Mooncake 单写者共享 soft-pin owner、全局 session/branch 关联与冻结选择器、Router 后台准备和正式选路、原生缓存水位、独立反馈与缺口恢复、Indexer 分片/epoch/CAS 围栏。每个功能在测试通过后单独提交；提交证据见仓库根目录 PLAN-REVIEW-LOG.md。

验证配置为双 Engine、各 TP=1、Qwen3-8B BF16、FULL MHA、Python radix、persistent host、page64、page_first_direct、direct I/O、write_through、Mooncake TCP v0.3.14、wait_complete、no-overlap；CUDA decode/prefill graphs 均禁用。GPU A 为 A100 80GB PCIe，B 为 A100 SXM4 80GB。两个 Engine 的模型、容量及随机种子一致。文件后端用于正确性测试。配置、模型/tokenizer 哈希、实际能力和原始响应均封存。

两处实测问题已修复：匹配的正式请求可共享在途 native 读取，受控 query/GET 从两次降为一次；host 容量不足时，显式预取在目标与传输预算通过后调用原生驱逐，回收当前缺口，保留 PIN 与在途引用。空间不足的扫描共享 100ms 退避。驱逐以 radix 节点为单位，可能多释放部分未保护缓存；后端 miss 也可能发生在此类置换之后。

## 最终快照的性能

下表使用最终代码，等待 1s、提前 0.5s，每组 N=3，配对顺序随机。所有 18 条记录均通过缓存路径和输出一致性校验。首 token 延迟与完整前台周期独立统计，单位 ms。

| 前缀 | 按需 TTFT | 预取 TTFT | TTFT 下降 | 按需周期 | 预取周期 | 周期下降 | 恢复前 ready |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 297.07 | 170.74 | 42.5% | 2529.57 | 2568.02 | -1.5% | 3/3 |
| 8192 | 807.69 | 422.53 | 47.7% | 4287.55 | 3723.41 | 13.2% | 0/3 |
| 16384 | 998.78 | 468.90 | 53.1% | 5299.63 | 4628.76 | 12.7% | 0/3 |

2K 的完整周期增加约 1.5%；TTFT 降低不能代替周期收益。8K/16K 在恢复前均未 ready，随后复用的是在途共享读取，实际反馈记为 storage，不能把它们写成已提前完成的 host 命中。配对均值 bootstrap 95% CI：8K TTFT −573～−151ms、周期 −781～−405ms；16K TTFT −933～−182ms、周期 −1234～−300ms。2K 周期 CI 跨 0。原始记录及全部 p50/p95/p99 保存在 [final-short-window](final-short-window/summary.md)。

host 压力实验预置四条 16K 无关前缀，每次通过真实生成、DEMOTE 和 scoped host 查询核验；等待 3s、提前 2.5s，每组 N=3。最终六条全部有效：按需/预取 TTFT 为 **591.20/96.41ms，下降 83.7%**；完整前台周期为 **5782.69/4874.56ms，下降 15.7%**。预取 3/3 提前 ready，恢复均复用 8192 host tokens；重复 prefill 为 0。配对 ΔTTFT 95% CI 为 −791～−310ms，周期为 −1272～−368ms。见 [final-host-pressure](final-host-pressure/summary.md)。

容量修复前的相同压力配置为 0/3 提前 ready、TTFT 759.44/819.89ms，预取较慢；该记录仍保留在初始报告。跨快照还包含 P7 反馈、重启和采集程序差异，不能把前后数值差全部归给一处修改。新回归用例在旧生产代码上失败，以及最终同轮按需/预取配对，分别提供机制与实际收益证据。

完整前台周期包括 A 源计算、确认共享存储发布、工具等待和全部 8 个输出 token；压力构造、flush 和响应后的 drain 另计，不代表整个应用或后台服务的总成本。所有相同条件采用相同输入、容量和等待；正式请求按 deadline 发出，不额外等待预取。小样本 CI 未覆盖长期时序相关、共享主机抖动或生产负载分布，p95/p99 只作插值描述。

初始六次运行的 202 条记录、原始 16K 退化、五组 PIN/DEMOTE 对照、后台吞吐下降均保留在 [原始报告](README.md)。最终新增 36 条场景，共封存 **238 条有效场景记录**；功能保护验收单独计数。并发 16K 的前台收益伴随后台完成吞吐下降，尚不能确认总体服务吞吐提升。

## Router 与保护验收

P6 Router 的 8K 三次配对与路由改变负例共 7 条有效：B3 三次均自由选择准备好的 B，并实际复用全部 host 前缀。B2 直达 B 的均值 184.83ms，B3 为 159.45ms；配对 Δ −25.39ms、95% CI −83.99～30.34ms，不能认定额外选路加速。两臂使用相同 Router 准备信号，EMA 历史和直连流量的学习屏障差异有记录。见 [P6 B3 记录](router-p6-b3/router-summary.md)。

最终代码另运行 legacy 8K 的 3 条场景及 reorg 2K 的 2 条场景，全部有效，B3 均选 B 并复用 host；它们用于整合功能验收。两台真实 Engine 的反馈均被 Router 按 source/epoch/native watermark 轮询，未退回 legacy 模式；query_known_scopes 与 global_snapshot_complete=false 均核验。见 [legacy](final-router-legacy/router-summary.md)、[reorg](final-router-reorg/router-summary.md)。首次 reorg preflight 早于服务监听，连接拒绝日志保留；等待 capabilities 就绪后的两条完整场景独立存储。

真实保护验收中，两个 task 对同一 Mooncake K/V 对象获得 active soft-pin；释放一个 task 后 sibling 保持 active，两个都结束后 backend_quiescent=true。最终代码还验证了实际 L2 PIN active、eviction_exclusion，以及 task cleanup 后释放。见 [保护回执](final-protection/proof.json)。本地 HTTP 故障测试另覆盖 delayed Removed/completion、gap、epoch/restart、共享取消、未知后端结果和 late cleanup。

## 测试、限制与复现

最终独立验证：CPU 控制/策略/反馈 154 passed、96 subtests；GPU host/file/真实 Mooncake 2 passed、2 subtests；Router unit 1114 passed；完整 serial proxy 241 passed、1 个既有长耗时用例跳过；Indexer 156 passed；orchestrator 44 passed、1 个可选后端 live 用例跳过（实际 Mooncake 另有现场证明）；benchmark harness 16 passed；scheduler 11 passed、4 subtests；构建和 Rustfmt 通过。原始输出在 checks。并行 Router 单测曾出现三个 ZMQ 订阅超时，保留失败输出；完整 serial 重跑通过。Python 端口 fixture 的异步 ZMQ probe 关闭竞争已改为同步 TCP probe，随后联合测试通过。

首轮支持单内存 Router owner 与单后端保护写者。Router/indexer 无持久化全局快照或 HA；重启保护域保持 unknown，FreshDomain 需明确外部恢复依据。准备观察只覆盖所查 scope，有新水位、gap、失联或失效时重新查询；历史完成结果不证明当前驻留。Engine 普通 caller=request 的使用反馈不自动进入 Router 配置 caller 的轮询；请求关联仍由精确渲染输入或明确 request_observed 提供。

多 rank、Hybrid、PD、其他 cache mode 和多 Router 属于 P9，尚未声明支持。真实 RL/工具轨迹、共享比例扫描、物理 I/O 去重字节、host 字节峰值、服务端 CPU 与长期吞吐仍缺少测量。file 测试不代替跨 Engine Mooncake 证据。

启动配置与 API 见仓库 `sgl-router/KV_ORCHESTRATION.md`；实验入口见 `benchmark/session_kv_control/README.md` 与 `ROUTER_BENCH.md`。环境变量保存认证值，归档不包含 Authorization。离线验证不访问任何服务：

```bash
python3 verify_artifacts.py
python3 verify_acceptance.py
sha256sum -c SHA256SUMS
```
