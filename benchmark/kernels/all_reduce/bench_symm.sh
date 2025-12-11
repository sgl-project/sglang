: <<'result_in_h800'
| msg_size   |   [AllGather] torch eager time |   [AllGather] pynccl symm graph time |   [ReduceScatter] pynccl eager time |   [ReduceScatter] pynccl symm graph time |
|------------|--------------------------------|--------------------------------------|-------------------------------------|------------------------------------------|
| 2.0 KiB    |                        33.7312 |                              2.88258 |                             34.1088 |                                  2.8155  |
| 4.0 KiB    |                        27.4848 |                              2.97794 |                             29.6736 |                                  2.92332 |
| 8.0 KiB    |                        25.6896 |                              3.43691 |                             25.8368 |                                  3.08607 |
| 16.0 KiB   |                        60.4384 |                              4.94141 |                             25.28   |                                  3.14567 |
| 32.0 KiB   |                        25.5264 |                              5.3824  |                             31.664  |                                  3.36464 |
| 64.0 KiB   |                        26.8    |                              7.09412 |                             29.5456 |                                  3.61114 |
result_in_h800

export PER_NODE_GPU=8
export WORLD_SIZE=1
export RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12345
export NCCL_GRAPH_MIXING_SUPPORT=0
export NCCL_CUMEM_ENABLE=1
export NCCL_NVLS_ENABLE=2

torchrun  --nproc_per_node $PER_NODE_GPU \
          --nnodes $WORLD_SIZE \
          --node_rank $RANK \
          --master_addr $MASTER_ADDR \
          --master_port $MASTER_PORT benchmark_symm_mem.py
