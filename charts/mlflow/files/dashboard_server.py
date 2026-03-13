"""Lightweight server that serves the Melqart MLflow Tracker dashboard and proxies MLflow API.

This avoids CORS issues entirely — the browser talks to one origin, and this
server forwards API calls to MLflow.

Usage:
    # Local testing (MLflow on localhost:5000)
    python scripts/dashboard_server.py

    # Pointing at prod MLflow
    python scripts/dashboard_server.py --mlflow-url https://ui.ext.mlflow.prod.gcp.id-platformhub.net

    # With mock data for offline testing
    python scripts/dashboard_server.py --mock-dir scripts/mock_data

    # Custom port
    python scripts/dashboard_server.py --port 9000

Then open http://127.0.0.1:8501
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

SCRIPT_DIR = Path(__file__).parent
DASHBOARD_HTML = SCRIPT_DIR / "mlflow_neptune_ui.html"

# Will be set from CLI args
mlflow_base_url: str = "http://127.0.0.1:5000"
mlflow_auth: tuple[str, str] | None = None
http_client: httpx.AsyncClient | None = None
mock_data_dir: Path | None = None
_mock_runs_cache: list[dict] | None = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=120.0, auth=mlflow_auth)
    yield
    await http_client.aclose()


app = FastAPI(title="Melqart MLflow Tracker", lifespan=lifespan)


def _load_mock_runs() -> list[dict]:
    """Load and cache all mock run JSON files (with full histories)."""
    global _mock_runs_cache
    if _mock_runs_cache is not None:
        return _mock_runs_cache
    if not mock_data_dir or not mock_data_dir.exists():
        return []
    runs = []
    for f in sorted(mock_data_dir.glob("*.json")):
        with open(f) as fh:
            runs.append(json.load(fh))
    _mock_runs_cache = runs
    return runs


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard() -> FileResponse:
    """Serve the dashboard HTML."""
    return FileResponse(DASHBOARD_HTML, media_type="text/html")


@app.get("/mock-data")
async def serve_mock_data() -> JSONResponse:
    """Serve run metadata (no histories) for initial load."""
    runs = _load_mock_runs()
    if not runs:
        return JSONResponse({"error": "No mock data available"}, status_code=404)
    # Strip metric_histories to keep the response small
    lightweight = []
    for r in runs:
        light = {k: v for k, v in r.items() if k != "metric_histories"}
        lightweight.append(light)
    return JSONResponse({"runs": lightweight})


@app.get("/mock-metric-history")
async def serve_mock_metric_history(
    run_id: str = Query(...),
    metric_key: str = Query(...),
) -> JSONResponse:
    """Serve a single metric history on demand (lazy loading)."""
    runs = _load_mock_runs()
    for r in runs:
        if r.get("run_id") == run_id:
            histories = r.get("metric_histories", {})
            history = histories.get(metric_key, [])
            return JSONResponse({"metrics": history})
    return JSONResponse({"metrics": []}, status_code=404)


@app.api_route("/mlflow/{path:path}", methods=["GET", "POST", "OPTIONS"])
async def proxy_mlflow(path: str, request: Request) -> dict:
    """Proxy all /mlflow/* requests to the real MLflow server."""
    target_url = f"{mlflow_base_url}/{path}"
    if request.query_params:
        target_url += f"?{request.query_params}"

    body = await request.body()
    response = await http_client.request(
        method=request.method,
        url=target_url,
        content=body if body else None,
        headers={"Content-Type": "application/json"},
    )
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Melqart MLflow Tracker Dashboard Server")
    parser.add_argument(
        "--mlflow-url",
        default="http://127.0.0.1:5000",
        help="MLflow tracking server URL (default: http://127.0.0.1:5000)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to serve on (default: 8501)",
    )
    parser.add_argument(
        "--mock-dir",
        default=None,
        help="Directory with mock JSON files (from download_mlflow_runs.py)",
    )
    args = parser.parse_args()

    global mlflow_base_url, mlflow_auth, mock_data_dir
    mlflow_base_url = args.mlflow_url.rstrip("/")

    if args.mock_dir:
        mock_data_dir = Path(args.mock_dir)
        runs = _load_mock_runs()
        print(f"  Mock data: {mock_data_dir} ({len(runs)} runs)")

    mlflow_user = os.environ.get("MLFLOW_TRACKING_USERNAME")
    mlflow_pass = os.environ.get("MLFLOW_TRACKING_PASSWORD")
    if mlflow_user and mlflow_pass:
        mlflow_auth = (mlflow_user, mlflow_pass)
        print(f"  Auth:      {mlflow_user}")

    import uvicorn

    print(f"\n  Melqart MLflow Tracker")
    print(f"  Dashboard: http://127.0.0.1:{args.port}")
    print(f"  MLflow:    {mlflow_base_url}\n")

    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
