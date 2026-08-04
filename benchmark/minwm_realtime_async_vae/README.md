# MinWM Realtime Async VAE Benchmark

该目录提供同步 VAE 与异步 VAE 的同合同 A/B 压测。每个并发档位会建立独立 WebSocket Session，先运行 warmup chunk，再交替发送完整 Action 状态并记录 action 到首批帧到达、chunk 总耗时、阶段 Trace、FPS 和错误率。

生产验收不允许直连 Denoiser。完整入口必须是
`NLB -> Gateway -> Coordinator -> H100 Denoiser -> L4/L40S TAEHV -> Gateway -> Browser`。
AWS 控制面、不可变镜像、版本化模型制品、Kubernetes 拓扑、真实浏览器探针和显式清理
脚本都在本目录中，详细发布顺序见 `k8s/README.md`。

```bash
python benchmark/minwm_realtime_async_vae/load_test.py \
  --ws-url ws://HOST/v1/realtime_video/generate \
  --profile async \
  --concurrency 1,2,4,8 \
  --output artifacts/async.json

python benchmark/minwm_realtime_async_vae/summarize.py \
  --baseline artifacts/sync.json \
  --async-profile artifacts/async.json \
  --output-json artifacts/report.json \
  --output-md artifacts/report.zh-CN.md
```

最高稳定并发的默认门槛为：P95 action 到首批帧 `< 1000 ms`、每会话生成速度 `>= 16 FPS`、错误率 `0`。最终人工浏览器验证另行记录 action 到 canvas 首帧的真实耗时。

生产链路门禁命令：

```bash
node benchmark/minwm_realtime_async_vae/browser_probe.cjs \
  --url http://NLB_HOST/?mode=t2v \
  --output artifacts/browser.json \
  --screenshot artifacts/browser.png

python benchmark/minwm_realtime_async_vae/e2e_production_chain.py \
  --ws-url ws://NLB_HOST/v1/realtime_video/generate \
  --hardware-json artifacts/hardware.json \
  --browser-metrics-json artifacts/browser.json \
  --output-dir artifacts/production-run
```
