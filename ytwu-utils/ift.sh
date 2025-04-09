#!/bin/bash

while true; do
    # 重新执行命令
    echo "Starting ft syncd sgl-workspace"
    ft syncd /sgl-workspace &
    # wait after the background process died
    wait
done