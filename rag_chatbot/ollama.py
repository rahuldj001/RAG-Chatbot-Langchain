"""Ollama background daemon and port checking utility.

Spawns `ollama serve` in a detached background thread when running local models,
and provides socket checks to verify if Ollama's local port (11434) is responsive.
"""

import asyncio
import threading
import socket


def run_ollama_server():
    """Starts the `ollama serve` daemon process asynchronously in a dedicated daemon thread."""
    async def run_process(cmd):
        print(">>> starting", *cmd)
        try:
            # Spawn the subprocess asynchronously with redirected stdout and stderr pipes
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Stream subprocess log lines into the Python console in real time
            async def pipe(lines):
                async for line in lines:
                    print(line.decode().strip())

            # Concurrently pipe both standard output and error output
            await asyncio.gather(pipe(process.stdout), pipe(process.stderr))
        except Exception as e:
            print(f">>> Ollama server failed to start: {e}")

    async def start_ollama_serve():
        try:
            # Execute `ollama serve` to start listening for API inference requests
            await run_process(["ollama", "serve"])
        except Exception as e:
            print(f">>> Ollama not started: {e}")

    def run_async_in_thread(loop, coro):
        """Runs the asyncio event loop within a non-blocking background thread."""
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(coro)
        except Exception as e:
            print(f">>> Ollama thread error: {e}")
        finally:
            loop.close()

    # Create a fresh event loop and start the worker thread as a daemon
    new_loop = asyncio.new_event_loop()
    thread = threading.Thread(
        target=run_async_in_thread, args=(new_loop, start_ollama_serve()), daemon=True
    )
    thread.start()


def is_port_open(port: int) -> bool:
    """Checks whether a given TCP port is currently open and listening on localhost.

    Used to detect whether an Ollama server is already running before spawning a new one.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Set a short 1-second timeout so connection checks don't block startup
        s.settimeout(1.0)
        try:
            # Attempt TCP handshake with localhost on the specified port
            s.connect(("localhost", port))
            return True
        except (ConnectionRefusedError, OSError, socket.error):
            # Port is closed or unreachable
            return False
