import argparse
import os
import threading
import time
import socket
import uvicorn

from config.base_linux import Base
from linuxvdi.llm_local_app import app
from config.logger import CustomLogger

logger = CustomLogger(name="Linux Base Client")

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

def wait_for_server(host="127.0.0.1", port=8000, timeout=180, poll_interval=15):
    """Wait until the server is ready by attempting to open a socket connection."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=2):
                logger.info("Server is up and running.")
                return
        except OSError:
            time.sleep(poll_interval)
    raise TimeoutError(f"Server did not start on {host}:{port} within {timeout} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FastAPI with a specified model")
    parser.add_argument("--model", type=str, choices=["llama8b", "llama3b", "deepseek"],
                        default="llama8b", help="Select the model to load")
    args = parser.parse_args()

    os.environ["DEFAULT_MODEL"] = args.model

    # -------------------------------------------------------------------
    # Booting FastAPI Server in Separate Thread
    # -------------------------------------------------------------------
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Await server to be fully booted
    wait_for_server(host="127.0.0.1", port=8000)

    # -------------------------------------------------------------------
    # Base Generator
    # -------------------------------------------------------------------
    base_generator = Base()
    output_path = base_generator.generate_email_chains_to_file(num_chains=5)
    logger.info(f"Chains saved to: {output_path}")