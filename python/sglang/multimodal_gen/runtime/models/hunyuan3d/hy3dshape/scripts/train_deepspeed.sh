# If: ImportError: /usr/lib64/libstdc++.so.6: version `GLIBCXX_3.4.20' not found 
# Do: ln /usr/local/gcc-8.3/lib64/libstdc++.so.6 -sf /usr/lib64/libstdc++.so.6

# Try to auto-detect a suitable network interface if NCCL_SOCKET_IFNAME is not already set.
if [ -z "$NCCL_SOCKET_IFNAME" ]; then
    # Find the first physical-like interface by excluding common virtual/loopback names.
    DETECTED_IFACE=$(ls /sys/class/net | grep -vE '^(lo|docker|veth|cali|tunl|kube|ib|usb)' | head -n 1)
    if [ -n "$DETECTED_IFACE" ]; then
        echo "NCCL_SOCKET_IFNAME is not set. Auto-detected and exporting: $DETECTED_IFACE"
        export NCCL_SOCKET_IFNAME=$DETECTED_IFACE
    else
        echo "Warning: Could not auto-detect a network interface. You may need to set NCCL_SOCKET_IFNAME manually if NCCL fails."
    fi
fi

export NCCL_IB_TIMEOUT=24
export NCCL_NVLS_ENABLE=0
NET_TYPE="high" 
if [[ "${NET_TYPE}" = "low" ]]; then
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_HCA=mlx5_2:1,mlx5_2:1
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
else
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
    export NCCL_COLLNET_ENABLE=0
    export SHARP_COLL_ENABLE_SAT=0
    export NCCL_NET_GDR_LEVEL=2
    export NCCL_IB_QPS_PER_CONNECTION=4
    export NCCL_IB_TC=160
    export NCCL_PXN_DISABLE=0
fi
export NCCL_DEBUG=WARN

node_num=$1
node_rank=$2
num_gpu_per_node=$3
master_ip=$4
config=$5
output_dir=$6

echo "--- Script Arguments ---"
echo "node_num: $node_num"
echo "node_rank: $node_rank"
echo "master_ip: $master_ip"
echo "config: $config"
echo "output_dir: $output_dir"
echo "----------------------"

if test -d "$output_dir"; then
    cp $config $output_dir
else
    mkdir -p "$output_dir"
    cp $config $output_dir
fi

NODE_RANK=$node_rank \
HF_HUB_OFFLINE=0 \
MASTER_PORT=12348 \
MASTER_ADDR=$master_ip \
NCCL_IB_GID_INDEX=3 \
NCCL_NVLS_ENABLE=0 \
python3 main.py \
    --num_nodes $node_num \
    --num_gpus $num_gpu_per_node \
    --config $config \
    --output_dir $output_dir \
    --deepspeed
