# Set up self hosted runner for GitHub Action

## Config Runner

```bash
# https://github.com/sgl-project/sglang/settings/actions/runners/new?arch=x64&os=linux
# Involves some TOKEN and other private information, click the link to view specific steps.
```

## Start Runner

add `/lib/systemd/system/runner.service`
```
[Unit]
StartLimitIntervalSec=0
[Service]
Environment="CUDA_VISIBLE_DEVICES=7"
Environment="XDG_CACHE_HOME=/data/.cache"
Environment="HF_TOKEN=hf_**"
Environment="OPENAI_API_KEY=sk-**"
Environment="HOME=/data/zhyncs"
Restart=always
RestartSec=1
ExecStart=/data/zhyncs/actions-runner/run.sh
[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl start runner
sudo systemctl enable runner
sudo systemctl status runner
```
