#!/bin/bash -l

#SBATCH -o SLURM_Logs/%x_%j_master.out
#SBATCH -e SLURM_Logs/%x_%j_master.out
#SBATCH -D ./
#SBATCH -J deepseek_docker_optimized

#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --partition="amd-rccl"
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00

# Define parameters
DOCKER_IMAGE="rocmshared/mlse-dev:v0.5.3.post1-rocm700-mi30x-20251021"
MODEL_PATH="/home/yajizhan/models/deepseek-ai/DeepSeek-V3"
IBDEVICES="mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9"

echo "[INFO] Starting DeepSeek-V3 with Docker (optimized version)"
echo "[INFO] Docker Image: $DOCKER_IMAGE"
echo "[INFO] Model: $MODEL_PATH"

# Get node information
NODES=$(scontrol show hostname "$SLURM_NODELIST")
NODE_ARRAY=($NODES)

echo "[INFO] Node 0: ${NODE_ARRAY[0]} (Prefill + Router)"
echo "[INFO] Node 1: ${NODE_ARRAY[1]} (Decode)"

# Create log directory
mkdir -p SLURM_Logs

# Docker parameters
DOCKER_RUN_ARGS="--rm --network=host --ipc=host --privileged --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband --security-opt seccomp=unconfined --cap-add=SYS_PTRACE --cap-add=IPC_LOCK --cap-add=NET_RAW --shm-size=16g -v /home/yajizhan:/home/yajizhan -v /mnt:/mnt --group-add video"

# Global variables for PIDs
PREFILL_PID=""
DECODE_PID=""
ROUTER_PID=""

# Signal handler function
cleanup() {
    echo "[INFO] Received termination signal, cleaning up..."
    
    # Kill background processes
    if [ ! -z "$ROUTER_PID" ]; then
        echo "[INFO] Killing router service (PID: $ROUTER_PID)"
        kill $ROUTER_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$PREFILL_PID" ]; then
        echo "[INFO] Killing prefill service (PID: $PREFILL_PID)"
        kill $PREFILL_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$DECODE_PID" ]; then
        echo "[INFO] Killing decode service (PID: $DECODE_PID)"
        kill $DECODE_PID 2>/dev/null || true
    fi
    
    # Wait for processes to end
    wait $ROUTER_PID 2>/dev/null || true
    wait $PREFILL_PID 2>/dev/null || true
    wait $DECODE_PID 2>/dev/null || true
    
    # Clean up Docker containers
    cleanup_existing_containers $SLURM_NODEID
    
    echo "[INFO] Cleanup completed on Node $SLURM_NODEID"
    exit 0
}

# Cleanup function for Docker containers
cleanup_existing_containers() {
    local node_id=$1
    if [ $node_id -eq 0 ]; then
        echo "[INFO] Cleaning up existing containers on Node 0 (prefill, router)"
        docker rm -f deepseek-prefill 2>/dev/null || true
        docker rm -f deepseek-router 2>/dev/null || true
    else
        echo "[INFO] Cleaning up existing containers on Node 1 (decode)"
        docker rm -f deepseek-decode 2>/dev/null || true
    fi
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT SIGUSR1 SIGUSR2

# Display current node ID
echo "[INFO] Current SLURM_NODEID: $SLURM_NODEID"

# Assign tasks based on node ID
if [ $SLURM_NODEID -eq 0 ]; then
    # Node 0: Start prefill service, then start router service
    
    cleanup_existing_containers 0
    
    echo "[INFO] Node 0: Starting Prefill Service with Docker"
    
    # Start prefill service (run in background)
    docker run $DOCKER_RUN_ARGS --name deepseek-prefill $DOCKER_IMAGE \
        bash -c "cd /home/yajizhan/slurm_work_docker && bash prefill_launch.sh '$MODEL_PATH' '$IBDEVICES' '${NODE_ARRAY[0]}' 30000" &
    
    PREFILL_PID=$!
    echo "[INFO] Prefill service started with PID: $PREFILL_PID"
    
    # Use srun to start decode service on node 1 with separate log file
    echo "[INFO] Launching Decode service on Node 1 using srun"
    srun --nodes=1 --ntasks=1 -w ${NODE_ARRAY[1]} \
         --output="SLURM_Logs/%x_%j_node1_decode.out" \
         --error="SLURM_Logs/%x_%j_node1_decode.out" \
         bash -c "
        # Set up signal handler for node 1
        cleanup_node1() {
            echo '[INFO] Node 1: Received termination signal, cleaning up...'
            docker rm -f deepseek-decode 2>/dev/null || true
            echo '[INFO] Node 1 cleanup completed'
            exit 0
        }
        trap cleanup_node1 SIGTERM SIGINT SIGUSR1 SIGUSR2
        
        echo '[INFO] Node 1: Starting Decode Service with Docker'
        echo '[INFO] Current SLURM_NODEID on Node 1: \$SLURM_NODEID'
        docker rm -f deepseek-decode 2>/dev/null || true
        docker run $DOCKER_RUN_ARGS --name deepseek-decode $DOCKER_IMAGE \
            bash -c 'cd /home/yajizhan/slurm_work_docker && bash decode_launch.sh \"$MODEL_PATH\" \"$IBDEVICES\" \"${NODE_ARRAY[1]}\" 30001'
    " &
    
    DECODE_PID=$!
    echo "[INFO] Decode service started on Node 1 with PID: $DECODE_PID"
    
    # Create separate log for prefill service
    echo "[INFO] Prefill service logs will be in: SLURM_Logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_node0_prefill.out"
    
    # Wait for prefill service to be ready
    echo "[INFO] Waiting for Prefill service to be ready..."
    MAX_WAIT=12000
    WAIT_COUNT=0
    while ! nc -z ${NODE_ARRAY[0]} 30000; do
        sleep 5
        WAIT_COUNT=$((WAIT_COUNT + 5))
        echo "[INFO] Waited ${WAIT_COUNT}s for Prefill service on ${NODE_ARRAY[0]}:30000"
        
        if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
            echo "[ERROR] Prefill service failed to start within ${MAX_WAIT}s"
            cleanup
            exit 1
        fi
    done
    echo "[INFO] Prefill service is ready!"
    
    # Wait for decode service to be ready
    echo "[INFO] Waiting for Decode service to be ready..."
    WAIT_COUNT=0
    while ! nc -z ${NODE_ARRAY[1]} 30001; do
        sleep 5
        WAIT_COUNT=$((WAIT_COUNT + 5))
        echo "[INFO] Waited ${WAIT_COUNT}s for Decode service on ${NODE_ARRAY[1]}:30001"
        
        if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
            echo "[ERROR] Decode service failed to start within ${MAX_WAIT}s"
            cleanup
            exit 1
        fi
    done
    echo "[INFO] Decode service is ready!"
    
    # After both services are ready, start router service
    echo "[INFO] Both Prefill and Decode services are ready. Starting Router Service..."
    
    # Start router service (run in background) and redirect logs
    docker run $DOCKER_RUN_ARGS --name deepseek-router $DOCKER_IMAGE \
        bash -c "cd /home/yajizhan/slurm_work_docker && bash router_launch.sh '${NODE_ARRAY[0]}' '${NODE_ARRAY[1]}' 8000" > "SLURM_Logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_node0_router.out" 2>&1 &
    
    ROUTER_PID=$!
    echo "[INFO] Router service started with PID: $ROUTER_PID"
    echo "[INFO] Router service logs will be in: SLURM_Logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_node0_router.out"
    
    # Wait for the router process to complete or be interrupted
    echo "[INFO] All services started. Waiting for completion or interruption..."
    wait $ROUTER_PID
    ROUTER_EXIT_CODE=$?
    
    echo "[INFO] Router service exited with code: $ROUTER_EXIT_CODE"
    
    # Perform cleanup
    cleanup
    
else
    # If script runs on node 1 (which shouldn't happen), exit directly
    echo "[INFO] Node 1: This node should be managed by srun from Node 0"
    exit 0
fi

echo "[INFO] Node $SLURM_NODEID: Script completed"