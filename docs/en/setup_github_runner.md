# Set up self hosted runner for GitHub Action

## Config Runner

```bash
# https://github.com/sgl-project/sglang/settings/actions/runners/new?arch=x64&os=linux
# Involves some TOKEN and other private information, click the link to view specific steps.
```

## Start Runner

add `/lib/systemd/system/e2e.service`
```
[Unit]
StartLimitIntervalSec=0
[Service]
Environment="CUDA_VISIBLE_DEVICES=7"
Environment="XDG_CACHE_HOME=/data/.cache"
Environment="HF_TOKEN=hf_xx"
Environment="OPENAI_API_KEY=sk-xx"
Environment="HOME=/data/zhyncs/runner-v1"
Environment="SGLANG_IS_IN_CI=true"
Restart=always
RestartSec=1
ExecStart=/data/zhyncs/runner-v1/actions-runner/run.sh
[Install]
WantedBy=multi-user.target
```

add `/lib/systemd/system/unit.service`
```
[Unit]
StartLimitIntervalSec=0
[Service]
Environment="CUDA_VISIBLE_DEVICES=6"
Environment="XDG_CACHE_HOME=/data/.cache"
Environment="HF_TOKEN=hf_xx"
Environment="OPENAI_API_KEY=sk-xx"
Environment="HOME=/data/zhyncs/runner-v2"
Environment="SGLANG_IS_IN_CI=true"
Restart=always
RestartSec=1
ExecStart=/data/zhyncs/runner-v2/actions-runner/run.sh
[Install]
WantedBy=multi-user.target
```

add `/lib/systemd/system/accuracy.service`
```
[Unit]
StartLimitIntervalSec=0
[Service]
Environment="CUDA_VISIBLE_DEVICES=5"
Environment="XDG_CACHE_HOME=/data/.cache"
Environment="HF_TOKEN=hf_xx"
Environment="OPENAI_API_KEY=sk-xx"
Environment="HOME=/data/zhyncs/runner-v3"
Environment="SGLANG_IS_IN_CI=true"
Restart=always
RestartSec=1
ExecStart=/data/zhyncs/runner-v3/actions-runner/run.sh
[Install]
WantedBy=multi-user.target
```

```bash
cd /data/zhyncs/runner-v1
python3 -m venv venv

cd /data/zhyncs/runner-v2
python3 -m venv venv

cd /data/zhyncs/runner-v3
python3 -m venv venv

sudo systemctl daemon-reload

sudo systemctl start e2e
sudo systemctl enable e2e
sudo systemctl status e2e

sudo systemctl start unit
sudo systemctl enable unit
sudo systemctl status unit

sudo systemctl start accuracy
sudo systemctl enable accuracy
sudo systemctl status accuracy
```
