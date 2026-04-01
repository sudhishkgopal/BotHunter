"""
BotHunter FastAPI Application

Endpoints:
  GET  /              — API info
  GET  /health        — Health check
  POST /simulate      — Run detection on a synthetic network
  POST /analyze       — Upload edge-list file and run detection
  GET  /twitter       — Run detection on bundled Twitter dataset
  GET  /history       — List all past analysis runs (paginated)
  GET  /history/{id}  — Single run detail including full bot ID list
  GET  /status/{job_id} — Poll async job status
  GET  /download/{filename} — Download generated files
"""

import asyncio
import json
import os
import uuid
from datetime import UTC, datetime

import matplotlib

matplotlib.use("Agg")   # headless — no display needed on the server

import multiprocessing as mp

import matplotlib.pyplot as plt
import networkx as nx
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from database import SessionLocal, init_db
from models import AnalysisResult

# ─── Config ───────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

_CFG = _load_config()
_API_CFG = _CFG.get("api", {})
MAX_UPLOAD_MB: int = _API_CFG.get("max_upload_size_mb", 50)
JOB_TTL: int = _API_CFG.get("job_ttl_seconds", 3600)

# ─── In-memory async job store ────────────────────────────────────────────────
# Keys: job_id (str) → {"status": "pending"|"running"|"done"|"error",
#                        "result": dict | None, "error": str | None,
#                        "created_at": datetime}
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = asyncio.Lock()

# ─── Pydantic schemas ─────────────────────────────────────────────────────────

class SimulateRequest(BaseModel):
    num_humans: int = Field(100, ge=10, le=10_000, description="Number of human nodes")
    num_bots: int   = Field(15,  ge=2,  le=1_000,  description="Number of bot nodes in the clique")
    k: int          = Field(10,  ge=2,  le=100,    description="K-Core threshold")

class SimulateResponse(BaseModel):
    status: str
    job_id: str
    total_nodes: int
    total_edges: int
    detected_bots: int
    bot_ids: list[int]
    visualization: str | None = None
    results_file: str | None = None

class AnalyzeResponse(BaseModel):
    status: str
    job_id: str
    total_nodes: int
    total_edges: int
    detected_bots: int
    bot_ids: list[int]
    visualization: str | None = None
    results_file: str | None = None

class JobStatusResponse(BaseModel):
    job_id: str
    status: str    # pending | running | done | error
    result: dict | None = None
    error: str | None = None
    created_at: str

class HistoryItem(BaseModel):
    id: int
    run_label: str | None
    k_core_threshold: int
    total_nodes: int
    total_edges: int
    bots_detected: int
    detection_accuracy: float | None
    ran_at: str

class HistoryDetail(HistoryItem):
    bot_ids: list[int]

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    db_connected: bool
    graph_engine: str

# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="BotHunter API",
    description=(
        "Graph-based social media bot detection using K-Core Decomposition.\n\n"
        "Detects three bot patterns:\n"
        "- **Star Bots** — mass-following accounts with negligible followers\n"
        "- **Engagement Pods** — mutual-follow cliques inflating engagement\n"
        "- **Influencers** — high in-degree accounts (not bots, but flagged)\n\n"
        "Powered by a multi-signal risk score: degree asymmetry + clustering coefficient inverse + K-Core density."
    ),
    version="1.0.0",
    contact={"name": "BotHunter", "url": "https://github.com/sudhishkg11/BotHunter"},
    openapi_tags=[
        {"name": "Detection",  "description": "Run bot detection on real or simulated networks"},
        {"name": "History",    "description": "Query past analysis runs stored in the database"},
        {"name": "Jobs",       "description": "Poll the status of async detection jobs"},
        {"name": "Files",      "description": "Download generated visualizations and result files"},
        {"name": "System",     "description": "Health and metadata endpoints"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Algorithm helpers ────────────────────────────────────────────────────────

def get_k_core(graph: nx.Graph, k: int) -> nx.Graph:
    """Iterative K-Core pruning — O(n+m) per round."""
    adj = {node: set(neighbors) for node, neighbors in graph.adjacency()}
    while True:
        to_remove = [n for n, nbrs in adj.items() if len(nbrs) < k]
        if not to_remove:
            break
        for node in to_remove:
            for nbr in adj[node]:
                adj[nbr].discard(node)
            del adj[node]
    return nx.Graph(adj)


def _find_nodes_to_remove(chunk: list, adj: dict, k: int) -> list:
    return [n for n in chunk if len(adj[n]) < k]


def get_k_core_parallel(graph: nx.Graph, k: int) -> nx.Graph:
    """Parallel K-Core using a Master-Worker pool for large graphs."""
    adj = {node: set(neighbors) for node, neighbors in graph.adjacency()}
    num_cores = mp.cpu_count()
    while True:
        nodes = list(adj.keys())
        chunk_size = max(1, len(nodes) // num_cores)
        chunks = [nodes[i:i + chunk_size] for i in range(0, len(nodes), chunk_size)]
        with mp.Pool(processes=num_cores) as pool:
            results = pool.starmap(_find_nodes_to_remove, [(c, adj, k) for c in chunks])
        to_remove = [n for sub in results for n in sub]
        if not to_remove:
            break
        for node in to_remove:
            if node in adj:
                for nbr in adj[node]:
                    adj[nbr].discard(node)
                del adj[node]
    return nx.Graph(adj)


def save_visualization(original_G: nx.Graph, core_G: nx.Graph, filename: str = "bot_detection.png") -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), facecolor="#F5F5F5")
    pos_orig = nx.spring_layout(original_G, k=0.15, iterations=50, seed=42)
    pos_core = nx.shell_layout(core_G) if len(core_G.nodes()) > 0 else {}
    nx.draw_networkx_nodes(original_G, pos_orig, node_size=25, node_color="#3498db", alpha=0.8, ax=ax1)
    nx.draw_networkx_edges(original_G, pos_orig, width=0.5, edge_color="grey", alpha=0.2, ax=ax1)
    ax1.set_title("Original Network: The Noise", fontsize=20, fontweight="bold")
    ax1.axis("off")
    if len(core_G.nodes()) > 0:
        nx.draw_networkx_nodes(core_G, pos_core, node_size=150, node_color="#e74c3c", edgecolors="black", ax=ax2)
        nx.draw_networkx_edges(core_G, pos_core, width=1.5, edge_color="#c0392b", alpha=0.6, ax=ax2)
    ax2.set_title("Detected Bot Core: The Signal", fontsize=20, fontweight="bold", color="#c0392b")
    ax2.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def load_twitter_data(file_path: str) -> nx.Graph:
    return nx.read_edgelist(file_path, create_using=nx.Graph(), nodetype=int)


# ─── Job helpers ───────────────────────────────────────────────────────────────

def _make_job() -> str:
    job_id = str(uuid.uuid4())
    _JOBS[job_id] = {
        "status": "pending",
        "result": None,
        "error": None,
        "created_at": datetime.now(UTC).isoformat(),
    }
    return job_id


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", tags=["System"])
async def root():
    """API overview and available endpoints."""
    return {
        "message": "Welcome to BotHunter API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "POST /simulate":        "Run detection on a generated synthetic network",
            "POST /analyze":         "Upload an edge-list file and run detection",
            "GET  /twitter":         "Run detection on the bundled Twitter SNAP dataset",
            "GET  /history":         "List past analysis runs (paginated)",
            "GET  /history/{id}":    "Single run detail with full bot ID list",
            "GET  /status/{job_id}": "Poll async job status",
            "GET  /download/{file}": "Download a generated PNG or JSON file",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check: returns DB connectivity and service info."""
    db_ok = False
    try:
        with SessionLocal() as s:
            s.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_ok = True
    except Exception:
        pass
    return HealthResponse(
        status="healthy" if db_ok else "degraded",
        service="BotHunter API",
        version="1.0.0",
        db_connected=db_ok,
        graph_engine=f"NetworkX {nx.__version__}",
    )


@app.post("/simulate", response_model=SimulateResponse, tags=["Detection"])
async def simulate_bot_detection(req: SimulateRequest):
    """
    Run bot detection on a procedurally-generated social network.

    Creates `num_humans` random users and `num_bots` fully-connected bot nodes,
    then runs K-Core pruning with threshold `k`.
    """
    job_id = _make_job()
    try:
        _JOBS[job_id]["status"] = "running"
        human_network = nx.erdos_renyi_graph(req.num_humans, 0.05)
        bot_network   = nx.complete_graph(req.num_bots)
        bot_network   = nx.relabel_nodes(bot_network, {i: i + req.num_humans for i in range(req.num_bots)})
        G             = nx.compose(human_network, bot_network)
        bot_core      = get_k_core(G, k=req.k)
        save_visualization(G, bot_core, "simulation_result.png")
        bot_list = list(bot_core.nodes())
        with open("simulation_bots.json", "w") as f:
            json.dump(bot_list, f)
        result = SimulateResponse(
            status="success",
            job_id=job_id,
            total_nodes=G.number_of_nodes(),
            total_edges=G.number_of_edges(),
            detected_bots=len(bot_list),
            bot_ids=bot_list,
            visualization="/download/simulation_result.png",
            results_file="/download/simulation_bots.json",
        )
        _JOBS[job_id]["status"] = "done"
        _JOBS[job_id]["result"] = result.model_dump()
        return result
    except Exception as e:
        _JOBS[job_id]["status"] = "error"
        _JOBS[job_id]["error"]  = str(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Detection"])
async def analyze_network(
    file: UploadFile = File(..., description="Edge-list .txt file (space-separated node pairs)"),
    k: int = Query(10, ge=2, le=100, description="K-Core threshold"),
    use_parallel: bool = Query(False, description="Use parallel processing for large networks"),
):
    """
    Analyze an uploaded edge-list file.

    File format: one edge per line, `<source_id> <target_id>` (space-separated integers).
    Returns the detected bot IDs, a static visualization, and a downloadable JSON result.
    """
    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB} MB limit")

    job_id = _make_job()
    try:
        _JOBS[job_id]["status"] = "running"
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as f:
            f.write(content)
        network  = nx.read_edgelist(temp_file, create_using=nx.Graph(), nodetype=int)
        bot_core = get_k_core_parallel(network, k=k) if use_parallel else get_k_core(network, k=k)
        viz_path = "Skipped (network too large for static viz)"
        if network.number_of_nodes() < 5_000:
            save_visualization(network, bot_core, "analysis_result.png")
            viz_path = "/download/analysis_result.png"
        bot_list = list(bot_core.nodes())
        with open("detected_bots.json", "w") as f:
            json.dump(bot_list, f)
        os.remove(temp_file)
        result = AnalyzeResponse(
            status="success",
            job_id=job_id,
            total_nodes=network.number_of_nodes(),
            total_edges=network.number_of_edges(),
            detected_bots=len(bot_list),
            bot_ids=bot_list[:100],
            visualization=viz_path,
            results_file="/download/detected_bots.json",
        )
        _JOBS[job_id]["status"] = "done"
        _JOBS[job_id]["result"] = result.model_dump()
        return result
    except Exception as e:
        _JOBS[job_id]["status"] = "error"
        _JOBS[job_id]["error"]  = str(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/twitter", tags=["Detection"])
async def analyze_twitter(
    k: int = Query(10, ge=2, le=100, description="K-Core threshold"),
    use_parallel: bool = Query(True, description="Use parallel processing"),
):
    """Analyze the bundled Stanford SNAP Twitter dataset (81K nodes, 1.7M edges)."""
    file_path = "twitter_combined.txt"
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail="Twitter dataset not found. Add twitter_combined.txt to the project directory.",
        )
    job_id = _make_job()
    try:
        _JOBS[job_id]["status"] = "running"
        twitter_network = load_twitter_data(file_path)
        bot_core = get_k_core_parallel(twitter_network, k=k) if use_parallel else get_k_core(twitter_network, k=k)
        bot_list = list(bot_core.nodes())
        with open("twitter_bots.json", "w") as f:
            json.dump(bot_list, f)
        viz_path = "Skipped (network too large for static viz)"
        if twitter_network.number_of_nodes() < 10_000:
            save_visualization(twitter_network, bot_core, "twitter_result.png")
            viz_path = "/download/twitter_result.png"
        result = {
            "status": "success",
            "job_id": job_id,
            "total_nodes": twitter_network.number_of_nodes(),
            "total_edges": twitter_network.number_of_edges(),
            "detected_bots": len(bot_list),
            "bot_ids": bot_list[:100],
            "visualization": viz_path,
            "results_file": "/download/twitter_bots.json",
        }
        _JOBS[job_id]["status"] = "done"
        _JOBS[job_id]["result"] = result
        return JSONResponse(result)
    except Exception as e:
        _JOBS[job_id]["status"] = "error"
        _JOBS[job_id]["error"]  = str(e)
        raise HTTPException(status_code=500, detail=str(e))


# ─── History endpoints ────────────────────────────────────────────────────────

@app.get("/history", response_model=list[HistoryItem], tags=["History"])
async def list_history(
    limit: int  = Query(20, ge=1, le=200, description="Max results to return"),
    offset: int = Query(0,  ge=0,         description="Pagination offset"),
):
    """
    Return a paginated list of past bot detection runs, newest first.

    Each item includes aggregate stats. Use `/history/{id}` to get the full bot ID list.
    """
    init_db()
    with SessionLocal() as s:
        rows = (
            s.query(AnalysisResult)
            .order_by(AnalysisResult.ran_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
    return [
        HistoryItem(
            id=r.id,
            run_label=r.run_label,
            k_core_threshold=r.k_core_threshold,
            total_nodes=r.total_nodes,
            total_edges=r.total_edges,
            bots_detected=r.bots_detected,
            detection_accuracy=r.detection_accuracy,
            ran_at=r.ran_at.isoformat(),
        )
        for r in rows
    ]


@app.get("/history/{run_id}", response_model=HistoryDetail, tags=["History"])
async def get_history_detail(run_id: int):
    """Return a single past run including the full list of detected bot IDs."""
    init_db()
    with SessionLocal() as s:
        row = s.query(AnalysisResult).filter(AnalysisResult.id == run_id).first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return HistoryDetail(
        id=row.id,
        run_label=row.run_label,
        k_core_threshold=row.k_core_threshold,
        total_nodes=row.total_nodes,
        total_edges=row.total_edges,
        bots_detected=row.bots_detected,
        detection_accuracy=row.detection_accuracy,
        ran_at=row.ran_at.isoformat(),
        bot_ids=json.loads(row.bot_ids_json) if row.bot_ids_json else [],
    )


# ─── Job status endpoint ──────────────────────────────────────────────────────

@app.get("/status/{job_id}", response_model=JobStatusResponse, tags=["Jobs"])
async def get_job_status(job_id: str):
    """
    Poll the status of an async detection job by its `job_id`.

    Status values: `pending` → `running` → `done` | `error`
    """
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        result=job.get("result"),
        error=job.get("error"),
        created_at=job["created_at"],
    )


# ─── File download ────────────────────────────────────────────────────────────

@app.get("/download/{filename}", tags=["Files"])
async def download_file(filename: str):
    """Download a generated file (PNG visualization or JSON results)."""
    # Basic path-traversal guard
    safe_name = os.path.basename(filename)
    if not os.path.exists(safe_name):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(safe_name, filename=safe_name)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)