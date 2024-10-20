import multiprocessing as mp
import zmq
import time
import psutil
from datetime import datetime

def server():
    """ZMQ server that sends messages every second."""
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.bind("tcp://*:5555")
    
    while True:
        time.sleep(1)
        message = "Hello"
        socket.send_string(message)
        print(f"[{datetime.now()}] Server sent: {message}")

def monitor_cpu_usage(pid, duration):
    """Monitor CPU usage of a specific process for a given duration."""
    process = psutil.Process(pid)
    start_time = time.time()
    while time.time() - start_time < duration:
        cpu_usage = process.cpu_percent(interval=1)
        print(f"Process ID:{pid} CPU Usage: {cpu_usage:.2f}%")
        time.sleep(1)

def client(optimized=False):
    """ZMQ client that receives messages."""
    client_type = "optimized" if optimized else "unoptimized"
    print(f"Running {client_type} client...")
    print(f"Process ID: {mp.current_process().pid}")
    
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect("tcp://localhost:5555")
    
    if optimized:
        socket.setsockopt(zmq.RCVTIMEO, 100)  # Set a 100ms timeout for receiving
    
    start_time = time.time()
    last_print_time = start_time
    counter = 0
    
    while time.time() - start_time < 30:
        try:
            message = socket.recv_string(zmq.NOBLOCK) if not optimized else socket.recv_string()
            print(f"[{datetime.now()}] {client_type.capitalize()} client received: {message}")
        except zmq.Again:
            counter += 1
            current_time = time.time()
            if current_time - last_print_time >= 2:
                print(f"[{datetime.now()}] {client_type.capitalize()} client: No message received. Attempts: {counter}")
                last_print_time = current_time

def run_client_test(optimized=False):
    """Run a client test with CPU usage monitoring."""
    client_process = mp.Process(target=client, args=(optimized,))
    client_process.start()
    
    monitor_process = mp.Process(target=monitor_cpu_usage, args=(client_process.pid, 30))
    monitor_process.start()
    
    client_process.join()
    monitor_process.join()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    
    # Start the server process
    server_process = mp.Process(target=server)
    server_process.start()
    
    # Test unoptimized client
    print("Testing unoptimized client...")
    run_client_test(optimized=False)
    
    print("\nOptimizing the client...")
    print("=" * 50)
    
    # Test optimized client
    print("Testing optimized client...")
    run_client_test(optimized=True)
    
    server_process.terminate()
