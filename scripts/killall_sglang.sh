# Kill all SGLang processes and free the GPU memory.

kill -9 $(ps aux | grep 'multiprocessing.spawn' | grep -v 'grep' | awk '{print $2}')
kill -9 $(ps aux | grep 'sglang.launch_server' | grep -v 'grep' | awk '{print $2}')
kill -9 $(ps aux | grep 'sglang.bench' | grep -v 'grep' | awk '{print $2}')
