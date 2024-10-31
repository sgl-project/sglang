## Profiling SGLang Infer System with AMD GPUs
This AppNote describes the SGLang profiling technical, code augment and running steps for systems with AMD Instinct GPUs, nevertheless the same procedure may work with Nvidia GPUs too.
Examples and steps are provided in detail, to facilitate easy reproduce and use to localize performance problem towards optimizations.
Two primary methods are covered:
- [RPD](https://github.com/ROCm/rocmProfileData.git)
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

### Profiling SGLang Infer System with RPD Profiler
RPD profiler is a low-overhead cross-platform profiler. To use RPD profiler on this repository, please use scripts and patch files included in this directory and follow the steps below:
1. Install RPD with rpd.patch applied during installation using install_rpd.sh, both files are in this directory.

install_rpd.sh

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

rpd.patch

```bash
diff --git a/rpd_tracer/Makefile b/rpd_tracer/Makefile
index e9d9feb..b2e9e1a 100644
--- a/rpd_tracer/Makefile
+++ b/rpd_tracer/Makefile
@@ -16,7 +16,7 @@ ifneq (,$(HIP_PATH))
         $(info Building with roctracer)
         RPD_LIBS += -L/opt/rocm/lib -lroctracer64 -lroctx64 -lamdhip64 -lrocm_smi64
         RPD_INCLUDES += -I/opt/rocm/include -I/opt/rocm/include/roctracer -I/opt/rocm/include/hsa
-        RPD_SRCS += RoctracerDataSource.cpp RocmSmiDataSource.cpp
+        RPD_SRCS += RoctracerDataSource.cpp
         RPD_INCLUDES += -D__HIP_PLATFORM_AMD__
 endif
```
2. Add loadTracer.sh file included in this directory to /sglang/python/sglang.

loadTracer.sh

```bash
#!/bin/bash
################################################################################
# Copyright (c) 2021 - 2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
################################################################################
OUTPUT_FILE="trace.rpd"
 
if [ "$1" = "-o" ] ; then
  OUTPUT_FILE=$2
  shift
  shift
fi
  
if [ -e ${OUTPUT_FILE} ] ; then
  rm ${OUTPUT_FILE}
fi
 
python3 -m rocpd.schema --create ${OUTPUT_FILE}
if [ $? != 0 ] ; then
  echo "Error: Could not create rpd file. Please run 'python setup.py install' from the rocpd_python dir"
  exit
fi
 
export RPDT_FILENAME=${OUTPUT_FILE}
export RPDT_AUTOSTART=0
LD_PRELOAD=librocm-smi_64:librpd_tracer.so "$@"
```
3. Apply patch (provided in this directory) with "git apply rpd_profile_server_enable.patch". 

```bash
diff --git a/python/sglang/srt/managers/scheduler.py b/python/sglang/srt/managers/scheduler.py
index 62d1ff9..9021c01 100644
--- a/python/sglang/srt/managers/scheduler.py
+++ b/python/sglang/srt/managers/scheduler.py
@@ -71,6 +71,8 @@ from sglang.srt.utils import (
     suppress_other_loggers,
 )
 from sglang.utils import get_exception_traceback
+from rpdTracerControl import rpdTracerControl
+rpdTracerControl.skipCreate()
 
 logger = logging.getLogger(__name__)
 
@@ -245,6 +247,7 @@ class Scheduler:
                 ],
                 with_stack=True,
             )
+            self.rpd = rpdTracerControl()
 
     @torch.inference_mode()
     def event_loop(self):
@@ -1027,15 +1030,24 @@ class Scheduler:
     def start_profile(self) -> None:
         if self.profiler is None:
             raise RuntimeError("Profiler is not enabled.")
-        self.profiler.start()
+        #self.profiler.start() #block pytorch profiler for rpd profiler enabling
+        if self.tp_rank == 0 or self.tp_rank == 1:
+            self.rpd.start()
+            self.rpd.rangePush("", "rpd profile range", "")
+            logger.info("rpd is enabled")
 
     def stop_profile(self) -> None:
         if self.profiler is None:
             raise RuntimeError("Profiler is not enabled.")
-        self.profiler.stop()
-        self.profiler.export_chrome_trace(
-            self.torch_profiler_trace_dir + "/" + str(time.time()) + ".trace.json.gz"
-        )
+        #self.profiler.stop()
+        #self.profiler.export_chrome_trace(
+        #    self.torch_profiler_trace_dir + "/" + str(time.time()) + ".trace.json.gz"
+        #)
+        if self.tp_rank ==0 or self.tp_rank ==1:
+            self.rpd.rangePop()
+            self.rpd.stop()
+            self.rpd.flush()
+            logger.info("rpd is done")
         logger.info("Profiler is done")
```

4. Untar dummy_grok1.tar in this directory and copy it to the right path for "--model-path" if you want to use the server.sh file provided.
5. Launch server with rpd enabled script ./server.sh in one terminal inside the docker container. 
#### Common Notes
- Remember to change model-path to the correct path
- loadTracer.sh is needed to conduct profiling
- SGLANG_TORCH_PROFILER_DIR is used for default torch profiler
- Do not use loadTracer.sh if you are using the torch profiler, simply use python3 -m sglang.launch_server. It may interrupt and cause the GPU info to be unavailable (though CPU data will still be there).  


server.sh

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

client.sh

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

### Profiling SGLang Infer System with PyTorch Profiler

Please use the steps as follows:

1. Apply the patch torch_profiler.patch. Note that you can modify "if self.tp_rank == 0" in the patch to allow more ranks be recorded in profiling.

torch_profiler.patch
```bash
diff --git a/python/sglang/srt/managers/scheduler.py b/python/sglang/srt/managers/scheduler.py
index 62d1ff9..6ecd78c 100644
--- a/python/sglang/srt/managers/scheduler.py
+++ b/python/sglang/srt/managers/scheduler.py
@@ -240,7 +240,6 @@ class Scheduler:
             )
             self.profiler = torch.profiler.profile(
                 activities=[
-                    torch.profiler.ProfilerActivity.CPU,
                     torch.profiler.ProfilerActivity.CUDA,
                 ],
                 with_stack=True,
@@ -1033,9 +1032,11 @@ class Scheduler:
         if self.profiler is None:
             raise RuntimeError("Profiler is not enabled.")
         self.profiler.stop()
-        self.profiler.export_chrome_trace(
-            self.torch_profiler_trace_dir + "/" + str(time.time()) + ".trace.json.gz"
-        )
+        if self.tp_rank == 0:
+            with open(f"stats_repro_{int(time.time())}.txt", "w") as f:
+                print(self.profiler.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=-1), file=f)
+                print("Profiling stats done.")
+
         logger.info("Profiler is done")
```

2. Untar dummy_grok1.tar in this directory and copy it to the right path for "--model-path" if you want to use the server.sh file provided.

3. Modify the included server.sh by removing "loadTracer.sh" before python command and launch script ./server.sh in one terminal inside the docker container. 

4. Similar to step 6 in RPD profiling section, but remove the last 2 lines, which converted rpd file into csv and json files, from in client.sh. Run modified client.sh for PyTorch profiling.