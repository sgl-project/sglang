import socket
import select
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

class KVCacheReceiver:
    def __init__(
        self,
        num_workers: int = 2
    ):
        self.num_workers = num_workers
        self.kv_receiver_executor = ThreadPoolExecutor(max_workers=self.num_workers)

        self.connections = {}  # Maps (remote_ip, remote_port) to active socket
        self.connection_lock = Lock()

    def pull_kv_cache(self, remote_ip, remote_port, kv_cache_size) -> bytes:
        future = self.kv_receiver_executor.submit(
            self._pull_kv, remote_ip, remote_port, kv_cache_size
        )
        buffer = future.result()
        return buffer

    def _pull_kv(self, remote_ip, remote_port, kv_cache_size):
        key = (remote_ip, remote_port)
        sock = None
        with self.connection_lock:
            sock = self.connections.get(key)
            if not sock:
                # Create new connection if not exists
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setblocking(True)
                try:
                    print("!!!!! start connect !!!!!")
                    sock.connect((remote_ip, int(remote_port)))
                    self.connections[key] = sock
                    print("!!!!! done connect !!!!!")
                except Exception as e:
                    sock.close()
                    raise

        received = bytearray()
        try:
            print("start recv, kv cache size: ", kv_cache_size)
            while len(received) < kv_cache_size:
                print("recv a chunk now size: ", len(received))
                chunk = sock.recv(4096)
                if not chunk:
                    # Remote closed connection, remove and retry
                    raise ConnectionError("Connection closed by remote")
                received.extend(chunk)
            # Return successfully if all data is received
            # print("!!!!!!!!!!")
            # print(received)
            print("!!!!!!!!!!")
            return bytes(received)
        except (ConnectionError, OSError) as e:
            # Cleanup and retry on connection errors
            with self.connection_lock:
                if self.connections.get(key) is sock:
                    del self.connections[key]
            sock.close()
        except Exception as e:
            # Cleanup on other exceptions
            with self.connection_lock:
                if self.connections.get(key) is sock:
                    del self.connections[key]
            sock.close()
            raise


class KVCacheSender:
    def __init__(
        self,
        host: str,
        port: int,
        num_workers: int = 2
    ):
        self.num_workers = num_workers
        self.host = host
        self.port = port

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        print(f"set up socket {self.host}:{self.port}")

        self.sock.bind((self.host, self.port))
        self.sock.listen(5)
        self.sock.setblocking(True)

        self.kv_sender_executor = ThreadPoolExecutor(max_workers=self.num_workers)

        self.connections = {}  # Maps (remote_ip, remote_port) to active socket
        self.connections_lock = {} # Maps (remote_ip, remote_port): Lock
        self.connect_lock = Lock()

    def wait_for_connect(self, remote_ip: str, remote_port: int):
        """wait until only one thread is doing the accept"""
        addr_key = (remote_ip, remote_port)
        if addr_key in self.connections:
            return
        else:
            # TODO accept should have a timeout
            client_sock, client_addr_key = self.sock.accept()
            # <socket.socket fd=174, family=AddressFamily.AF_INET, type=SocketKind.SOCK_STREAM, proto=0, laddr=('127.0.0.1', 4000), raddr=('127.0.0.1', 46738)>
            # print(f"!!!!! debug addr: {client_sock}, {addr_key}")
            # assert client_addr_key == addr_key
            client_sock.setblocking(True)
            self.connections[addr_key] = client_sock
            self.connections_lock[addr_key] = Lock()

    def send_kv_cache(self, buffer: bytes, remote_ip, remote_port, kv_cache_size) -> None:
        future = self.kv_sender_executor.submit(self._send_kv, buffer, remote_ip, remote_port, kv_cache_size)
        future.result()
        return

    # TODO: send should have a timeout
    def _send_kv(self, buffer: bytes, remote_ip, remote_port, kv_cache_size) -> None:
        addr_key = (remote_ip, remote_port)
        client_sock = self.connections[addr_key]
        sent = 0
        while sent < len(buffer):
            try:
                chunk = client_sock.send(buffer[sent:])
                if chunk == 0:
                    raise ConnectionError("Connection broken")
                sent += chunk
            except BlockingIOError:
                break  # Retry later
        
        # if sent == len(buffer):
        client_sock.close()
        return 

if __name__ == "__main__":
    from threading import Thread
    # Shared Configuration
    host = "127.0.0.1"
    port = 4000
    buffer = b"a" * 1024  # 1KB test data

    # Initialize sender and receiver
    sender = KVCacheSender(host=host, port=port, num_workers=2)
    receiver = KVCacheReceiver(num_workers=2)

    # Create threads for blocking operations
    send_thread = Thread(
        target=sender.send_kv_cache,
        args=(buffer, host, port, len(buffer))
    )
    
    pull_thread = Thread(
        target=lambda: print(f"Received: {len(receiver.pull_kv_cache(host, port, len(buffer)))} bytes"),
    )

    # Start threads
    pull_thread.start()
    send_thread.start()

    # Wait for completion
    pull_thread.join()
    send_thread.join()
