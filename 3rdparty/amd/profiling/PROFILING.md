## Profiling SGLang Infer System with AMD GPUs
This AppNote describes the SGLang profiling technical, code augment and running steps for systems with AMD Instinct GPUs, nevertheless the same procedure may work with Nvidia GPUs too.
Examples and steps are provided in detail, to facilitate easy reproduce and use to localize performance problem towards optimizations.
Two primary methods are covered:
- [RPD](https://github.com/ROCm/rocmProfileData.git)
- [Torch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

### Profiling SGLang Infer System with RPD Profiler
RPD profiler is a low-overhead cross-platform profiler. To use RPD profiler on this repository, please use scripts and patch files included in this directory and follow the steps below:
1. Install RPD with rpd.patch applied during installation using install_rpd.sh, both files are in this directory.
![alt text](rpd_patch.png)

```bash
# download and install RPD
apt update && apt install -y sqlite3 libsqlite3-dev libfmt-dev
   
# install rpd module
git clone https://github.com/ROCmSoftwarePlatform/rocmProfileData
cd rocmProfileData
git apply rpd.patch 
make && make install
cd rocpd_python && python setup.py install && cd ..
cd rpd_tracer && make clean;make install && python setup.py install && cd ..
```
2. Add loadTracer.sh file included in this directory to /sglang/python/sglang.
3. Apply patch (provided in this directory) with "git apply rpd_profile_server_enable.patch". 
![alt text](rpd_profile_server_enable_patch.png)
4. Untar dummy_grok1.tar in this directory and copy it to the right path for "--model-path" if you want to use the server.sh file provided.
5. Launch server with rpd enabled script ./server.sh in one terminal inside the docker container. 
#### Common Notes
- Remember to change model-path to the correct path
- loadTracer.sh is needed to conduct profiling
- SGLANG_TORCH_PROFILER_DIR is used for default torch profiler
- Do not use loadTracer.sh if you are using the torch profiler, simply use python3 -m sglang.launch_server. It may interrupt and cause the GPU info to be unavailable (though CPU data will still be there).  

```bash
#!/bin/bash
 
# export SGLANG_TORCH_PROFILER_DIR=/data/sglang/
export SGLANG_TORCH_PROFILER_DIR=/sgl-workspace/sglang/profile/
 
# Get the current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
 
# Define the log file with a timestamp
LOGFILE="sglang_server_log_$TIMESTAMP.json"
 
# Run the Python command and save the output to the log file
loadTracer.sh python3 -m sglang.launch_server \
    --model-path /sgl-workspace/sglang/dummy_grok1 \
    --tokenizer-path Xenova/grok-1-tokenizer \
    --load-format dummy \
    --quant fp8 \
    --tp 8 \
    --port 30000 \
    --disable-radix-cache 2>&1 | tee "$LOGFILE"
```
6. Open another terminal for the same docker container, and run the rpd enabled ./client.sh after you see "The server is fired up and is ready to roll!" message from server side terminal.
#### Common Notes
- Use curl http://localhost:30000/start_profile & curl http://localhost:30000/stop_profile to control the start and end of profiling. Check sglang/python/sglang/srt/managers/scheduler.py for more details.
- Please don't use RPD profiler together with together profier to avoid interference.
```bash
#!/bin/bash
 
# Start profiling via API
curl http://localhost:30000/start_profile -H "Content-Type: application/json"
 
# Benchmark serving using sglang with random dataset and tokenizer
# Define the log file with a timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="sglang_client_log_$TIMESTAMP.json"
 
# Run the benchmark with specified parameters and save logs
python3 -m sglang.bench_serving \
    --backend sglang \
    --tokenizer Xenova/grok-1-tokenizer \
    --dataset-name random \
    --random-input 1024\
    --random-output 1024 \
    --num-prompts 240 \
    --request-rate 8 \
    --output-file online.jsonl 2>&1 | tee "$LOGFILE"
 
# Stop profiling via API
curl http://localhost:30000/stop_profile -H "Content-Type: application/json"
 
# Convert tracing file to csv & json
sqlite3 trace.rpd ".mode csv" ".header on" ".output trace.csv" "select * from top;" ".output stdout"
python3 /sgl-workspace/rocmProfileData/tools/rpd2tracing.py trace.rpd trace.json
```
7. Follow [Perfetto docs](https://perfetto.dev/docs/visualization/large-traces) to visualize large json files. Try to adjust parameters so that the trace.json file size is less than 9GB.



