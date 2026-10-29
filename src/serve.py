"""
Distress Watch Server Launcher.

Starts Uvicorn server on 127.0.0.1:8000, polls until HTTP 200 response is verified,
and automatically launches the default web browser to the dashboard.
"""

import argparse
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
import uvicorn


def find_free_port(host: str = "127.0.0.1", start_port: int = 8000, max_tries: int = 50) -> int:
    """Find the first available port starting from start_port."""
    for p in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    return start_port


def _poll_and_open_browser(url: str, timeout: int = 30) -> None:
    """Poll URL until it responds with HTTP 200, then open in default browser."""
    status_url = f"{url.rstrip('/')}/api/status"
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(status_url, headers={"User-Agent": "DistressWatch-HealthCheck/1.0"})
            with urllib.request.urlopen(req, timeout=1) as resp:
                if resp.status == 200:
                    print(f"\n[Distress Watch] Server verified healthy at {url}!")
                    print(f"[Distress Watch] Opening default browser...\n")
                    webbrowser.open(url)
                    return
        except Exception:
            time.sleep(0.5)

    print(f"\n[WARNING] Server did not respond within {timeout}s. Please navigate to {url} manually.")


def run_server(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    """Run Uvicorn with dynamic port finding and optional browser auto-open upon health check."""
    # Check environment variable override
    if os.environ.get("NO_BROWSER", "").strip() in ("1", "true", "True"):
        open_browser = False

    actual_port = find_free_port(host, port)
    if actual_port != port:
        print(f"[Notice] Port {port} was in use; automatically bound to free port {actual_port}.")

    url = f"http://{host}:{actual_port}"

    if open_browser:
        watcher = threading.Thread(
            target=_poll_and_open_browser,
            args=(url,),
            daemon=True,
        )
        watcher.start()
    else:
        # Non-browser mode: still poll health and log readiness
        def _log_when_ready():
            status_url = f"{url}/api/status"
            t0 = time.time()
            while time.time() - t0 < 30:
                try:
                    with urllib.request.urlopen(status_url, timeout=1) as r:
                        if r.status == 200:
                            print(f"\n[Distress Watch] Server is live and healthy at {url}/api/status\n")
                            return
                except Exception:
                    time.sleep(0.5)

        threading.Thread(target=_log_when_ready, daemon=True).start()

    # Run Uvicorn directly
    uvicorn.run("app.server:app", host=host, port=actual_port, reload=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Distress Watch Uvicorn Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser")

    args = parser.parse_args()
    run_server(host=args.host, port=args.port, open_browser=not args.no_browser)

