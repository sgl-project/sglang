kill -9 $(ps aux | grep 'multiprocessing.spawn' | grep -v 'grep' | awk '{print $2}')
