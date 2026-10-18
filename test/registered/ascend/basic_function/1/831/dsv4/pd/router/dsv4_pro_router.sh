#dsv4_pro_router.sh
#!/bin/bash
### 路由服务 在P节点启动
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
clear
unset no_proxy
cd /home/w00889861/sglang_dsv4/sglang
#export PYTHONPATH=/home/w00889861/sglang_dsv4/sglang/python:$PYTHONPATH
export MODEL_PATH=/home/weights/DeepSeek-V4-Pro-0813-w4a8
python -m sglang_router.launch_router \
    --pd-disaggregation --policy cache_aware \
    --prefill http://192.168.25.209:30000 8998 \
    --decode http://192.168.25.216:30000 \
    --host 0.0.0.0 --port 6699 \
    --prometheus-port 9192

