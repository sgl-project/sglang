we conduct environments on the dgx box of A100 (80GBx8)
you should be able to use nvtop to see the current status of gpu. other co-workers are running experiments on that node as well.

for most of the experiments like development and testing, conduct on single gpu (gpu4)
for compute intensive tasks like profiling and data collection, conduct on all available gpus (check if memory and compute are both empty)

large data should be stored in /data folder, feel free create sub directory /data/heter-moe/ and store results there