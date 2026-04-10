we conduct environments on the dgx box of A100 (80GBx8)
for most of the experiments like development and testing, conduct on single gpu (gpu3) when possible
for compute intensive tasks like profiling and data collection, conduct on all available gpus (check if memory and compute are both empty)

data should be stored in /data folder, feel free create sub directory /data/heter-moe/ and store results there