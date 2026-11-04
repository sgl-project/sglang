# Router B2/B3 配对

B3 free normal Router selection minus B2 direct B; same Router preparation targeting B in both arms; neither arm is B0

| group | attempts | completed | valid | mean TTFT ms | ready before resume | changed route | fallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| b2_direct_b | 1 | 1 | 1 | 151.690 | 1 | 0 | 0 |
| b3_router | 1 | 1 | 1 | 141.167 | 1 | 0 | 0 |
| route_change | 0 | 0 | 0 | — | 0 | 0 | 0 |

完整配对差值、N 与 bootstrap CI 见 router-summary.json。所有完成响应进入主统计；错误归属、控制失败和未完成 cleanup 的条目保留为 invalid。A 在发布后清本地缓存，Router 自由选择 A 时需证明普通 L3 命中。两臂使用相同准备信号，EMA 的历史状态可能不同；直接 B2 请求不会触发 Router 的正式调度学习屏障。
