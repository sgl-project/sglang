import os
import select
import threading


class OutputCapturer:
    """Capture all console print information

    Class Description:
        Capture console output using low-level file descriptor redirection.
        Used to obtain print information from child processes, NPU processes,
        and underlying C/C++ modules that are not logged in sglang logs for test assertion.
        All captured output will be displayed normally in the console in real-time.
    """

    def __init__(self):
        """Initialize all member variables of the capturer"""
        self.old_stdout = None
        self.old_stderr = None
        self.pipe_out = None
        self.pipe_in = None
        self.pipe_err_out = None
        self.pipe_err_in = None
        self.captured_stdout = []
        self.captured_stderr = []
        self.stop_thread = False
        self.thread = None

    def start(self):
        """Start console output capture"""
        # Duplicate and save original stdout/stderr file descriptors
        self.old_stdout = os.dup(1)
        self.old_stderr = os.dup(2)

        # Create anonymous pipes for output redirection
        self.pipe_out, self.pipe_in = os.pipe()
        self.pipe_err_out, self.pipe_err_in = os.pipe()

        # Redirect system stdout/stderr to the write end of pipes
        os.dup2(self.pipe_in, 1)
        os.dup2(self.pipe_err_in, 2)

        # Close unused pipe write ends
        os.close(self.pipe_in)
        os.close(self.pipe_err_in)

        # Start daemon thread to read output in real time
        self.stop_thread = False
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        """The background process reads and prints pipeline data records in a loop."""
        read_fds = [self.pipe_out, self.pipe_err_out]
        while not self.stop_thread:
            try:
                # select listens to multiple file descriptors simultaneously, waiting for data in a non-blocking manner
                readable, _, exceptional = select.select(read_fds, [], read_fds, 0.01)

                # Processing file descriptors containing data
                for fd in readable:
                    if fd == self.pipe_out:
                        data = os.read(fd, 4096)
                        if data:
                            self.captured_stdout.append(data)
                            os.write(self.old_stdout, data)
                    elif fd == self.pipe_err_out:
                        err_data = os.read(fd, 4096)
                        if err_data:
                            self.captured_stderr.append(err_data)
                            os.write(self.old_stderr, err_data)

                for fd in exceptional:
                    if fd in read_fds:
                        self.stop()

            except OSError:
                self.stop()
                break

    def get_all(self):
        """Get all captured stdout and stderr as UTF-8 string

        Return: Decoded stdout and stderr string (ignore decoding errors)
        """
        return self.get_output() + self.get_error()

    def get_output(self):
        """Get all captured stdout as UTF-8 string

        Return: Decoded stdout string (ignore decoding errors)
        """
        return b"".join(self.captured_stdout).decode("utf-8", errors="ignore")

    def get_error(self):
        """Get all captured stderr as UTF-8 string

        Return: Decoded stderr string (ignore decoding errors)
        """
        return b"".join(self.captured_stderr).decode("utf-8", errors="ignore")

    def stop(self):
        """Stop capture and restore system environment"""
        self.stop_thread = True
        if self.thread:
            self.thread.join(timeout=0.5)

        # Restore original output
        os.dup2(self.old_stdout, 1)
        os.dup2(self.old_stderr, 2)

        # Close all file descriptors
        for fd in [self.pipe_out, self.pipe_err_out, self.old_stdout, self.old_stderr]:
            try:
                os.close(fd)
            except (OSError, IOError):
                pass
