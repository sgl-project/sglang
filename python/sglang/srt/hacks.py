import subprocess


def kill_other_memory_occupying_processes():
    cmd = "pkill -f demo_another_task.py"
    print(f"kill_other_memory_occupying_processes {cmd=}")
    subprocess.run(cmd, shell=True)
