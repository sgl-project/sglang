# 天鹏 gap12 MinWM：单机 B200 部署

这个目录记录 2026-07-31 的可重复部署配置。服务运行在一台
`p6-b200.48xlarge` 上，但只使用一张 B200；Nginx 把 API 和 Realtime Studio
统一暴露到 80 端口。

## 运行合同

- checkpoint：`global_step_010000/generator/model.pt`，SHA-256
  `18a48a2709d74b93ce26f0b808f381d191553853aae81dd72d2438430251d379`
- pipeline：`MinWMCausalDMDPipeline`
- action type：`primitive_token_residual`
- 分辨率/FPS：832×480 / 24
- local/window/sink：32 / 32 / 8
- RoPE：`block_relative`，gap 12，prompt first-frame pin 开启
- DMD：4 steps，guidance 0，CFG parallel 关闭

## 性能档

```text
MINWM_ATTENTION_IMPL=packed
MINWM_PACKED_ATTENTION_DETERMINISTIC=false
MINWM_NATIVE_COMPONENTS=
MINWM_SEGMENT_COMPILE=true
MINWM_ENABLE_TORCH_COMPILE=false
```

B200 上 packed 路径实际调用 FA4。whole-DiT compile 在这个 T2V/window 32
工作负载上是负优化；即使把 FA4 设为 graph break，chunk 也比 eager/segment
compile 慢约一个数量级。因此这里显式关闭 whole-DiT compile，而不是沿用旧
checkpoint 的开关。

## 文件和端口

- `run_server.sh`：安装当前 SGLang、校验模型、跑 138 个 realtime 单测、启动
  WebUI 和 API。
- `minwm-tianpeng.service`：systemd 单元，GPU 3、API 30060、WebUI 18060。
- `nginx-minwm.conf`：80 → API 30060 / WebUI 18060，支持 WebSocket。
- `containerd.toml`、`docker-daemon.json`：把容器数据放到本地 NVMe。

部署：

```bash
sudo install -m 0755 run_server.sh /opt/dlami/nvme/minwm-tianpeng/run_server.sh
sudo install -m 0644 minwm-tianpeng.service /etc/systemd/system/
sudo install -m 0644 nginx-minwm.conf /etc/nginx/conf.d/minwm.conf
sudo systemctl daemon-reload
sudo systemctl enable --now minwm-tianpeng.service
sudo nginx -t && sudo systemctl reload nginx
```

验收：

```bash
systemctl is-enabled minwm-tianpeng.service
systemctl is-active minwm-tianpeng.service
curl http://127.0.0.1:30060/health
curl http://127.0.0.1:18060/
curl http://127.0.0.1/health
```

不要只看编码 FPS。当前同一 5 秒 case 的 server scheduler 为 13.21 FPS，
window=32 饱和后为 11.50 FPS；WebUI 写 24 FPS 是播放合同，不代表模型能以
24 FPS 生产。
