## Profiling SGLang Infer System with AMD GPUs
This AppNote describes the SGLang profiling technical, code augment and running steps for systems with AMD Instinct GPUs, nevertheless the same procedure may work with Nvidia GPUs too.
Examples and steps are provided in detail, to facilitate easy reproduce and use to localize performance problem towards optimizations.
Two primary methods are covered:
- [RPD](https://github.com/ROCm/rocmProfileData.git)
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

### Profiling SGLang Infer System with RPD Profiler
RPD profiler is a low-overhead cross-platform profiler. Therefore, the same RPD code augment not only works for profiling on ROCm/AMD GPUs, but also works for profiling on CUDA/Nvidia GPUs as well. To do RPD profiling on SGLang repository, please follow the code examples and similar techniques:

1. Install RPD with the instructions found in the repo: https://github.com/ROCm/rocmProfileData/tree/master. You can directly use install_rpd.sh.
Summarized install instructions in install_rpd.sh are as below:

```bash
#!/bin/bash

# install other tools
apt-get update -y
apt-get install -y sqlite3 libsqlite3-dev libfmt-dev

# download and install RPD
git clone https://github.com/ROCm/rocmProfileData.git

# install rpd module
cd rocmProfileData/
make && make install

```

2. Use following code snippets in the server code if the main profiling purpose is to get info on gpu kernels as well as limited cpu activity info.

#### Common Notes 1
Please note that although we are doing TP=8 in the example, we purposely only log RPD profiling on 0/1 rank in the patch file (i.e.tp_rank=0 and 1) for profiling/visualization convenience, as even Perfetto streaming mode can only load maximal 8GB json file for visualization. With 0 and 1 rank logged in RPD profiling, we could still check whether there are issues among ranks (e.g. load imbalance issue, nccl issue), and at the same time, we could log relatively longer time duration before the json file generated from RPD file hits 8GB size.

`sglang/python/sglang/srt/managers/scheduler.py` has an option for passing or setting on "RPD" profiler option and use the start and stop methods accordingly.

```bash
diff --git a/python/sglang/srt/managers/scheduler.py b/python/sglang/srt/managers/scheduler.py
index a44515ab..1e28fd7c 100644
--- a/python/sglang/srt/managers/scheduler.py
+++ b/python/sglang/srt/managers/scheduler.py
@@ -2193,7 +2193,7 @@ class Scheduler(
         if output_dir is None:
             output_dir = os.getenv("SGLANG_TORCH_PROFILER_DIR", "/tmp")
         if activities is None:
-            activities = ["CPU", "GPU"]
+            activities = ["CPU", "GPU", "RPD"]

         self.torch_profiler_output_dir = output_dir
         self.torch_profiler_with_stack = with_stack
@@ -2262,9 +2262,12 @@ class Scheduler(

             self.rpd_profiler = rpdTracerControl()
             self.rpd_profiler.setPythonTrace(True)
-            self.rpd_profiler.start()
-            self.rpd_profiler.rangePush("", "rpd profile range", "")
-            self.profile_in_progress = True
+
+            if self.tp_rank == 0 or self.tp_rank == 1:
+               self.rpd_profiler.start()
+               self.rpd_profiler.rangePush("", "rpd profile range", "")
+               self.profile_in_progress = True
+
```

You can also set `--rpd` option via starting a client app for profiling i.e python/sglang/profiler.py.

#### Advanced Debugging with RPD Profiler
Sometimes, we want to use rpd profiler to capture more CPU and python activities in order to debug some challenging issues (e.g. root cause of load imbalance across gpu processes, root cause of bubbles, etc). Only in such cases, we need to apply patch "git apply rpd_profile_server_enable_wCPU_activities.patch", where 3 files are modified.

rpd_profile_server_enable_wCPU_activities.patch

```bash
diff --git a/python/sglang/srt/managers/scheduler.py b/python/sglang/srt/managers/scheduler.py
index 62d1ff9..2edb427 100644
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
@@ -1027,15 +1030,26 @@ class Scheduler:
     def start_profile(self) -> None:
         if self.profiler is None:
             raise RuntimeError("Profiler is not enabled.")
-        self.profiler.start()
+        #self.profiler.start()
+        logger.info("torch profiler is disabled")
+        if self.tp_rank == 0 or self.tp_rank == 1:
+            self.rpd.setPythonTrace(True)
+            self.rpd.start()
+            self.rpd.rangePush("", "scheduler", "")
+        logger.info("rpd is enabled inside scheduler profiling")

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
+            logger.info("rpd is done inside scheduler")
         logger.info("Profiler is done")


diff --git a/python/sglang/srt/managers/tokenizer_manager.py b/python/sglang/srt/managers/tokenizer_manager.py
index 2621ccd..181df85 100644
--- a/python/sglang/srt/managers/tokenizer_manager.py
+++ b/python/sglang/srt/managers/tokenizer_manager.py
@@ -58,6 +58,10 @@ from sglang.srt.sampling.sampling_params import SamplingParams
 from sglang.srt.server_args import PortArgs, ServerArgs
 from sglang.srt.utils import is_generation_model, is_multimodal_model

+from rpdTracerControl import rpdTracerControl
+rpdTracerControl.skipCreate()
+
+
 asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

 logger = logging.getLogger(__name__)
@@ -514,10 +518,20 @@ class TokenizerManager:
         self.send_to_scheduler.send_pyobj(req)

     def start_profile(self):
+        rpd = rpdTracerControl()
+        rpd.setPythonTrace(True)
+        rpd.start()
+        rpd.rangePush("", "tokenizer_manager", "")
+        logger.info("tokenizer_manager rpd profiling started!")
         req = ProfileReq.START_PROFILE
         self.send_to_scheduler.send_pyobj(req)

     def stop_profile(self):
+        rpd = rpdTracerControl()
+        rpd.rangePop()
+        rpd.stop()
+        rpd.flush()
+        logger.info("rpd profiling is done inside tokenizer_manager!")
         req = ProfileReq.STOP_PROFILE
         self.send_to_scheduler.send_pyobj(req)

diff --git a/python/sglang/srt/server.py b/python/sglang/srt/server.py
index 7111c93..2bd722c 100644
--- a/python/sglang/srt/server.py
+++ b/python/sglang/srt/server.py
@@ -30,6 +30,8 @@ import threading
 import time
 from http import HTTPStatus
 from typing import Dict, List, Optional, Union
+from rpdTracerControl import rpdTracerControl
+rpdTracerControl.skipCreate()

 # Fix a bug of Python threading
 setattr(threading, "_register_atexit", lambda *args, **kwargs: None)
@@ -152,6 +154,11 @@ async def flush_cache():
 @app.post("/start_profile")
 async def start_profile():
     """Start profiling."""
+    rpd = rpdTracerControl()
+    rpd.setPythonTrace(True)
+    rpd.start()
+    rpd.rangePush("", "server rpd profile range", "")
+    logger.info("rpd profiling started in server.py!")
     tokenizer_manager.start_profile()
     return Response(
         content="Start profiling.\n",
@@ -164,6 +171,11 @@ async def start_profile():
 async def stop_profile():
     """Stop profiling."""
     tokenizer_manager.stop_profile()
+    rpd = rpdTracerControl()
+    rpd.rangePop()
+    rpd.stop()
+    rpd.flush()
+    logger.info("rpd profiling is done in server.py!")
     return Response(
         content="Stop profiling. This will take some time.\n",
         status_code=200,
```

4. As an example for grok1 profiling, we create a dummy_grok1 directory with config.json (see content below) inside this directory and copy this directory to the right path for "--model-path" if you want to use the example server.sh file provided.
```bash
cat ../dummy_grok1/config.json
{
  "architectures": [
    "Grok1ModelForCausalLM"
  ],
  "embedding_multiplier_scale": 78.38367176906169,
  "output_multiplier_scale": 0.5773502691896257,
  "vocab_size": 131072,
  "hidden_size": 6144,
  "intermediate_size": 32768,
  "max_position_embeddings": 8192,
  "num_experts_per_tok": 2,
  "num_local_experts": 8,
  "num_attention_heads": 48,
  "num_hidden_layers": 64,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "rms_norm_eps": 1e-05,
  "rope_theta": 10000.0,
  "model_type": "mixtral",
  "torch_dtype": "bfloat16"
}
```
5. Launch server with rpd enabled script ./server.sh in one terminal inside the docker container.

#### Common Notes 2
- Remember to change model-path to the correct path
- loadTracer.sh is needed to conduct profiling
- SGLANG_TORCH_PROFILER_DIR is used for default torch profiler
- Do not use loadTracer.sh if you are using the torch profiler, simply use python3 -m sglang.launch_server.


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
    --quantization fp8 \
    --tp 8 \
    --port 30000 \
    --disable-radix-cache 2>&1 | tee "$LOGFILE"
```
6. Open another terminal for the same docker container, and run the rpd enabled ./client.sh after you see "The server is fired up and is ready to roll!" message from server side terminal.

#### Common Notes 3
- Use curl http://localhost:30000/start_profile & curl http://localhost:30000/stop_profile to control the start and end of profiling. Check sglang/python/sglang/srt/managers/scheduler.py for more details.
- Please don't use RPD profiler together with PyTorch profiler to avoid interference.
- The rocmProfileData/tools/rpd2tracing.py file is used to generate json file from RPD file.

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
    --num-prompts 120 \
    --request-rate 8 \
    --output-file online.jsonl 2>&1 | tee "$LOGFILE"

# Stop profiling via API
curl http://localhost:30000/stop_profile -H "Content-Type: application/json"

# Convert tracing file to csv & json
sqlite3 trace.rpd ".mode csv" ".header on" ".output trace.csv" "select * from top;" ".output stdout"
python3 ./rocmProfileData/tools/rpd2tracing.py trace.rpd trace.json
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

2. Create the model path directory and copy it to the right path for "--model-path" if you want to use the server.sh file provided.

3. Modify the included server.sh by removing "loadTracer.sh" before python command and launch script ./server.sh in one terminal inside the docker container.

4. Similar to step 6 in RPD profiling section, but remove the last 2 lines in client.sh, which converted rpd file into csv and json files. Run modified client.sh for PyTorch profiling.
-------
- [Torch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
