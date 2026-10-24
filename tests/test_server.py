"""Real Streamlit process startup, in addition to page-level AppTest coverage."""

import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def test_streamlit_server_starts(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    root = Path(__file__).resolve().parents[1]
    log_path = tmp_path / "streamlit.log"
    with log_path.open("w+") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app.py",
                "--server.headless=true",
                "--server.address=127.0.0.1",
                f"--server.port={port}",
                "--browser.gatherUsageStats=false",
            ],
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            # Loopback health check never uses an external HTTP proxy.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise AssertionError(f"Streamlit exited: {log_path.read_text()}")
                try:
                    with opener.open(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as res:
                        assert res.status == 200
                        assert res.read() == b"ok"
                    with opener.open(f"http://127.0.0.1:{port}/", timeout=1) as res:
                        assert b"<html" in res.read().lower()
                    break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.1)
            else:
                raise AssertionError(f"Streamlit did not become ready: {log_path.read_text()}")
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
