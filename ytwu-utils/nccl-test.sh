#!/bin/bash
# all_gather_perf  alltoall_perf   gather_perf     reduce_perf          scatter_perf 
# all_reduce_perf  broadcast_perf  hypercube_perf  reduce_scatter_perf  sendrecv_perf

benchmarks=(all_gather_perf alltoall_perf gather_perf reduce_perf scatter_perf all_reduce_perf broadcast_perf hypercube_perf reduce_scatter_perf sendrecv_perf)

for benchmark in ${benchmarks[@]}; do
  echo "Running ${benchmark}"
  # `which mpirun` --mca btl_tcp_if_include bond1 --allow-run-as-root -hostfile /sgl-workspace/hosts.txt -np 16 -N 8  bash -c "PS1=[] source ~/.bashrc && /sgl-workspace/nccl-tests/build/${benchmark} -b 1K -e 16M -f 2 -g 1 --warmup_iters 1" > /sgl-workspace/nccl-tests/${benchmark}.log 2>&1


  # `which mpirun` --mca btl_tcp_if_include bond1 --allow-run-as-root -hostfile /sgl-workspace/hosts.txt -np 16 -N 8  bash -c "PS1=[] source ~/.bashrc && /sgl-workspace/nccl-tests/build/${benchmark} -b 1G -e 16G -f 2 -g 1 --warmup_iters 1" >> /sgl-workspace/nccl-tests/${benchmark}.log 2>&1

  `which mpirun` --mca btl_tcp_if_include bond1 --allow-run-as-root -hostfile /sgl-workspace/hosts.txt -np 8 -N 8  bash -c "PS1=[] source ~/.bashrc && /sgl-workspace/nccl-tests/build/${benchmark} -b 1K -e 16M -f 2 -g 1 --warmup_iters 1" > /sgl-workspace/nccl-tests/8gpu-intra-node-${benchmark}.log 2>&1


  `which mpirun` --mca btl_tcp_if_include bond1 --allow-run-as-root -hostfile /sgl-workspace/hosts.txt -np 8 -N 8  bash -c "PS1=[] source ~/.bashrc && /sgl-workspace/nccl-tests/build/${benchmark} -b 1G -e 16G -f 2 -g 1 --warmup_iters 1" >> /sgl-workspace/nccl-tests/8gpu-intra-node-${benchmark}.log 2>&1
done
