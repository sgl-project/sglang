kill -9 $(ps aux | grep 'sglang.launch_server' | grep -v 'grep' | awk '{print $2}')
