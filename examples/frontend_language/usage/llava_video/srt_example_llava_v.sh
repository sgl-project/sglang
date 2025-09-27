#!/bin/bash

##### USAGE #####
#    - First node:
#      ```sh
#      bash examples/usage/llava_video/srt_example_llava_v.sh K 0 YOUR_VIDEO_PATH YOUR_MODEL_PATH FRAMES_PER_VIDEO
#      ```
#    - Second node:
#      ```sh
#      bash examples/usage/llava_video/srt_example_llava_v.sh K 1 YOUR_VIDEO_PATH YOUR_MODEL_PATH FRAMES_PER_VIDEO
#      ```
#    - The K node:
#      ```sh
#      bash examples/usage/llava_video/srt_example_llava_v.sh K K-1 YOUR_VIDEO_PATH YOUR_MODEL_PATH FRAMES_PER_VIDEO
#      ```


# Replace `K`, `YOUR_VIDEO_PATH`, `YOUR_MODEL_PATH`, and `FRAMES_PER_VIDEO` with your specific details.
# CURRENT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CURRENT_ROOT=$(dirname "$0")

echo ${CURRENT_ROOT}

cd ${CURRENT_ROOT}

export PYTHONWARNINGS=ignore

START_TIME=$(date +%s)  # Capture start time

NUM_NODES=$1

CUR_NODES_IDX=$2

VIDEO_DIR=$3

MODEL_PATH=$4

NUM_FRAMES=$5


# FRAME_FORMAT=$6

# FRAME_FORMAT=$(echo $FRAME_FORMAT | tr '[:lower:]' '[:upper:]')

# # Check if FRAME_FORMAT is either JPEG or PNG
# if [[ "$FRAME_FORMAT" != "JPEG" && "$FRAME_FORMAT" != "PNG" ]]; then
#     echo "Error: FRAME_FORMAT must be either JPEG or PNG."
#     exit 1
# fi

# export TARGET_FRAMES=$TARGET_FRAMES

echo "Each video you will sample $NUM_FRAMES frames"

# export FRAME_FORMAT=$FRAME_FORMAT

# echo "The frame format is $FRAME_FORMAT"

# Assuming GPULIST is a bash array containing your GPUs
GPULIST=(0 1 2 3 4 5 6 7)
LOCAL_CHUNKS=${#GPULIST[@]}

echo "Number of GPUs in GPULIST: $LOCAL_CHUNKS"

ALL_CHUNKS=$((NUM_NODES * LOCAL_CHUNKS))

# Calculate GPUs per chunk
GPUS_PER_CHUNK=1

echo $GPUS_PER_CHUNK

for IDX in $(seq 1 $LOCAL_CHUNKS); do
    (
        START=$(((IDX-1) * GPUS_PER_CHUNK))
        LENGTH=$GPUS_PER_CHUNK # Length for slicing, not the end index

        CHUNK_GPUS=(${GPULIST[@]:$START:$LENGTH})

        # Convert the chunk GPUs array to a comma-separated string
        CHUNK_GPUS_STR=$(IFS=,; echo "${CHUNK_GPUS[*]}")

        LOCAL_IDX=$((CUR_NODES_IDX * LOCAL_CHUNKS + IDX))

        echo "Chunk $(($LOCAL_IDX - 1)) will run on GPUs $CHUNK_GPUS_STR"

        # Calculate the port for this chunk. Ensure it's incremented by 5 for each chunk.
        PORT=$((10000 + RANDOM % 55536))

        MAX_RETRIES=10
        RETRY_COUNT=0
        COMMAND_STATUS=1  # Initialize as failed

        while [ $RETRY_COUNT -lt $MAX_RETRIES ] && [ $COMMAND_STATUS -ne 0 ]; do
            echo "Running chunk $(($LOCAL_IDX - 1)) on GPUs $CHUNK_GPUS_STR with port $PORT. Attempt $(($RETRY_COUNT + 1))"

#!/bin/bash
            CUDA_VISIBLE_DEVICES=$CHUNK_GPUS_STR python3 srt_example_llava_v.py \
            --port $PORT \
            --num-chunks $ALL_CHUNKS \
            --chunk-idx $(($LOCAL_IDX - 1)) \
            --save-dir work_dirs/llava_next_video_inference_results \
            --video-dir $VIDEO_DIR \
            --model-path $MODEL_PATH \
            --num-frames $NUM_FRAMES #&

            wait $!  # Wait for the process to finish and capture its exit status
            COMMAND_STATUS=$?

            if [ $COMMAND_STATUS -ne 0 ]; then
                echo "Execution failed for chunk $(($LOCAL_IDX - 1)), attempt $(($RETRY_COUNT + 1)). Retrying..."
                RETRY_COUNT=$(($RETRY_COUNT + 1))
                sleep 180  # Wait a bit before retrying
            else
                echo "Execution succeeded for chunk $(($LOCAL_IDX - 1))."
            fi
        done

        if [ $COMMAND_STATUS -ne 0 ]; then
            echo "Execution failed for chunk $(($LOCAL_IDX - 1)) after $MAX_RETRIES attempts."
        fi
    ) #&
    sleep 2  # Slight delay to stagger the start times
done

wait

cat work_dirs/llava_next_video_inference_results/final_results_chunk_*.csv > work_dirs/llava_next_video_inference_results/final_results_node_${CUR_NODES_IDX}.csv

END_TIME=$(date +%s)  # Capture end time
ELAPSED_TIME=$(($END_TIME - $START_TIME))
echo "Total execution time: $ELAPSED_TIME seconds."
