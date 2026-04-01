# BotHunter — AI Session Context

> **This file is the single source of truth for any AI assistant starting a new session.  
> Read this file first. It is kept up-to-date after every meaningful code change.**

Last updated: 2026-03-31 (packaging + tests)

---

## What This Project Does

BotHunter is a **graph-based social media bot detection engine**. It identifies coordinated bot farms by applying **K-Core Decomposition** to a follower/following graph. Low-connectivity "normal" nodes are recursively pruned until only dense, tightly interconnected clusters (bots) remain.

Detection goes beyond raw K-Core: it layers three weighted signals into a risk score (0.0–1.0) that distinguishes bot types from genuine influencers:

| Signal | Weight | What it catches |
|--------|--------|----------------|
| Degree Asymmetry | 0.40 | Star bots (mass-following, barely followed back) |
| Clustering Coefficient Inverse | 0.35 | Accounts whose followers don't know each other |
| K-Core Density | 0.25 | Engagement pods operating as mutual-follow cliques |

**Celebrity/Influencer false-positive prevention**: Local Clustering Coefficients separate high-in-degree legitimate accounts (followers are strangers → low clustering) from bot pods (members all follow each other → high clustering).

---

## Node Classification Labels

| Label | Internal key | Description |
|-------|-------------|-------------|
| Star Bot | `bot` | Mass-outgoing follows, negligible in-degree, low clustering |
| Engagement Group | `engagement_pod` | Dense mutual-follow clique, k-core ≥ threshold, clustering > 0.6 |
| Influencer | `influencer` | High in-degree > 3× out-degree, in-degree ≥ 50 |
| Human | `organic` / `normal` | Everything else |

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Language | Python 3.12 |
| Graph Analysis | NetworkX |
| Parallel K-Core | `multiprocessing` (Master-Worker pattern) |
| Database ORM | SQLAlchemy + SQLite (`bothunter.db`) |
| API Server | FastAPI + Uvicorn |
| CLI | Typer + Rich |
| Dashboard | Streamlit + Plotly + Pyvis |
| Containerization | Docker (multi-stage, non-root user) |

---

## Project File Map

```
BotHunter/
├── __init__.py       ← Package init — exposes __version__, usage docs, public API surface
├── app.py            ← Streamlit dashboard (4-tab layout: Overview, Detect, History, Export)
├── main.py           ← FastAPI app + K-Core algorithm + async job system
├── processor.py      ← Feature computation & classification engine (config-driven weights)
├── ingestor.py       ← Synthetic data generator for testing
├── models.py         ← SQLAlchemy ORM models (User, Relationship, AnalysisResult)
├── database.py       ← DB engine setup, respects DATABASE_URL env var
├── cli.py            ← Typer CLI — entry point: `bothunter audit`
├── ai_insights.py    ← LLM-powered node explanation (pluggable: OpenAI/Gemini/Claude/Ollama)
├── config.json       ← Runtime config: detection thresholds, scoring weights, AI, API settings
├── requirements.txt  ← Python dependencies (all runtime + test deps pinned)
├── pyproject.toml    ← Canonical packaging config (see Packaging section below)
├── docker-compose.yml ← Multi-service Docker setup (dashboard :8501 + API :8000)
├── render.yaml       ← One-click Render.com cloud deploy config
├── DEPLOY.md         ← Deployment guide (local, Docker, Render)
├── .env.example      ← Environment variable template (AI keys, DATABASE_URL)
├── tests/
│   ├── __init__.py       ← Empty — marks tests/ as a package for pytest package-mode
│   ├── test_processor.py ← Unit tests for the detection engine
│   └── test_api.py       ← Integration tests for FastAPI endpoints
├── bothunter.db      ← SQLite database (gitignored in production)
├── twitter_combined.txt ← Stanford SNAP Twitter dataset (81K nodes, 1.7M edges)
└── .claude/
    └── README.md     ← THIS FILE — AI session context
```

---

## Database Schema

### `users` table — graph nodes
| Column | Type | Notes |
|--------|------|-------|
| `id` | Integer PK | Internal auto-increment |
| `platform_id` | String(128) | Unique per-platform user ID (e.g. Twitter UID) |
| `username` | String(256) | Display name (nullable) |
| `is_bot` | Boolean | Ground-truth label (only known for synthetic data) |
| `account_created` | DateTime | Account age signal |
| `last_active` | DateTime | Activity pattern signal |

### `relationships` table — directed graph edges
| Column | Type | Notes |
|--------|------|-------|
| `source_user_id` | FK → users | Follower |
| `target_user_id` | FK → users | Followed |
| `relation_type` | Enum | `follow`, `like`, `comment` |

Analysis only uses `follow` edges for graph construction. `like`/`comment` edges exist for future expansion.

### `analysis_results` table — run history
Each detection run is persisted with: `k_core_threshold`, `total_nodes`, `total_edges`, `bots_detected`, `bot_ids_json`, `detection_accuracy`, `ran_at`.

---

## Key Algorithm — How K-Core Works

1. **Build**: Load all `follow` edges into a `nx.DiGraph`. Convert to undirected for K-Core (mutual connections become a single edge).
2. **Prune**: Identify all nodes with degree < k. Remove them (and cascade their removal to neighbours). Repeat until stable.
3. **Classify**: Apply three signals (asymmetry, clustering, k-core) into a weighted risk score. Apply label rules in priority order.
4. **Parallel variant** (`get_k_core_parallel`): Splits nodes into CPU-count chunks, each processed by a worker in a `Pool`. Results are merged globally. Used for the full Twitter dataset.

---

## FastAPI Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | API info / endpoint listing |
| `POST` | `/simulate` | Run bot detection on a generated synthetic network (async, returns `job_id`) |
| `POST` | `/analyze` | Upload a `.txt` edge list file and run detection (async, returns `job_id`) |
| `GET` | `/twitter` | Run detection on the bundled Twitter SNAP dataset |
| `GET` | `/status/{job_id}` | Poll async job status (`pending` / `running` / `done` / `error`) |
| `GET` | `/history` | List all past analysis runs (paginated) |
| `GET` | `/history/{id}` | Single run detail including full bot ID list |
| `GET` | `/download/{filename}` | Serve generated PNG or JSON result files |
| `GET` | `/health` | Health check (includes DB connectivity + graph engine version) |

---

## Streamlit Dashboard (app.py)

The dashboard is **batch-loaded** — it loads only the top-N suspects at startup (configurable) for fast rendering. Organized into **4 tabs**:

- **Overview** — project explainer, live network stats
- **Detect** — suspect table + Pyvis graph drill-down (click a row to focus the graph on that node)
- **History** — past analysis runs with delta comparisons
- **Export** — CSV / JSON download of flagged accounts

**Sidebar controls:** K-Core threshold slider, max neighbours in viz, startup batch size, influencer filter toggle, manual node selector.

---

## Synthetic Data Generator (ingestor.py)

Used when real social data is unavailable. Generates 3 network patterns:

| Pattern | Count | Behaviour |
|---------|-------|-----------|
| Humans | 300 (default) | Clustered friend groups (5–12 per group), 60–90% intra-group follow density, 3% cross-group links |
| Engagement Pod | 50 (default) | Near-complete mutual follow (90% probability), near-complete mutual like (85% probability), bursty timestamps |
| Star Bot | 1 | 5,000 outgoing follows, 10 incoming follows, dormant timestamps |

Run with: `python ingestor.py` (or with `--humans`, `--pod-size`, etc. flags)

---

## Configuration (config.json)

```json
{
  "k_core_threshold": 20,
  "max_neighbors_viz": 50,
  "top_n_startup": 100,
  "cache_ttl_seconds": 300,
  "database_path": "bothunter.db",
  "scoring_weights": {
    "asymmetry": 0.40,
    "clustering_inverse": 0.35,
    "k_core_density": 0.25
  },
  "classification": {
    "bot_out_in_ratio": 5,
    "bot_clustering_max": 0.2,
    "bot_risk_min": 0.7,
    "bot_risk_clustering_max": 0.3,
    "pod_clustering_min": 0.6,
    "influencer_in_out_ratio": 3,
    "influencer_min_in_degree": 50
  },
  "ai": { "provider": "openai", "model": "gpt-4o-mini", "enabled": false },
  "api": { "rate_limit_per_minute": 60, "max_upload_size_mb": 50, "job_ttl_seconds": 3600 }
}
```

All scoring weights and classification thresholds are **config-driven** — tune detection sensitivity by editing `config.json`, not source code. The database path is also overridable via `DATABASE_URL` environment variable (full SQLAlchemy connection string, e.g. for PostgreSQL).

---

## Packaging

The project is a **flat-layout** Python package (all source modules at the repo root, no `bothunter/` subdirectory). `pyproject.toml` is the canonical config — `requirements.txt` exists for tooling that doesn't read `pyproject.toml`.

### Run tests
```bash
pytest                      # all 51 tests
pytest tests/test_processor.py  # unit tests only (no DB or network needed)
pytest tests/test_ingestor.py   # ingestor tests (in-memory SQLite)
pytest tests/test_api.py        # API integration tests
pytest -v --tb=short        # verbose with compact tracebacks
```

### Install commands
```bash
pip install -e .              # editable install — core runtime only
pip install -e ".[dev]"       # + pytest, pytest-asyncio, pytest-cov, ruff, mypy
pip install -e ".[ai]"        # + openai, google-generativeai, anthropic
pip install -e ".[gpu]"       # + cudf-cu12, cugraph-cu12 (CUDA 12 required)
pip install -e ".[deploy]"    # + gunicorn, psycopg2-binary
```

### Optional dep groups
| Group | Purpose |
|-------|---------|
| `dev` | Testing (pytest, pytest-asyncio, pytest-cov, httpx), linting (ruff), types (mypy) |
| `ai` | LLM provider SDKs for `ai_insights.py` (OpenAI, Gemini, Anthropic) |
| `gpu` | CUDA-accelerated graph computation via cuGraph (requires NVIDIA GPU + CUDA 12) |
| `deploy` | Production WSGI server (gunicorn) + PostgreSQL driver (psycopg2) |

### Entry point
After `pip install -e .`, the `bothunter` CLI is available on PATH:
```bash
bothunter audit --threshold 20 --top 15
```
This resolves to `cli:app` (the Typer application object in `cli.py`).

### Setuptools flat-layout note
`[tool.setuptools] py-modules` explicitly lists every installable module. Without this, `pip install` would not package the top-level `.py` files and the `bothunter` entry point script would fail with `ModuleNotFoundError`.

---

## How to Run Locally

```bash
# 1. Install dependencies
pip install -e ".[dev]"  # preferred (installs as editable + dev tools)
# or: pip install -r requirements.txt

# 2. Initialize the database
python database.py

# 3. Seed with synthetic data (or skip if using Twitter dataset)
python ingestor.py

# 4. (Optional) Run the detection engine directly
python processor.py --k-threshold 20

# 5. Launch the dashboard
python -m streamlit run app.py

# 6. (Optional) Run the API server
uvicorn main:app --reload --port 8000
```

---

## Docker

```bash
# Build and run dashboard on :8501
docker build -t bothunter .
docker run -p 8501:8501 bothunter

# Run API instead (override entrypoint)
docker run -p 8000:8000 bothunter python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

The image uses a **multi-stage build** (builder → runtime), runs as a **non-root user** (`hunter`), and mounts `/app/data` as a persistent volume for the SQLite file.

---

## Recent Changes (not yet committed)

| Area | What changed |
|------|-------------|
| **`processor.py`** | Scoring weights and classification thresholds moved out of source code into `config.json` — no magic numbers remain |
| **`config.json`** | Added `scoring_weights`, `classification`, `ai`, and `api` sections |
| **`app.py`** | Full redesign — single scrolling page replaced with 4-tab layout (Overview, Detect, History, Export); added `"normal"` to color/display maps |
| **`main.py`** | Async job system, `/history` + `/status/{job_id}` endpoints, Pydantic schemas, improved health check |
| **`ai_insights.py`** | New — pluggable LLM explanation module (OpenAI / Gemini / Claude / Ollama) |
| **`pyproject.toml`** | New — proper packaging with optional dep groups `[dev]`, `[ai]`, `[deploy]` |
| **`docker-compose.yml`** | New — runs dashboard + API as two services sharing a persistent volume |
| **`render.yaml`** | New — one-click Render.com deploy |
| **`DEPLOY.md`** | New — deployment guide for local, Docker, and Render |
| **`requirements.txt`** | Fixed — added all missing deps: `matplotlib`, `plotly`, `pandas`, `streamlit`, `pyvis`, `pydantic`, `python-multipart`, `pytest`, `pytest-asyncio`, `httpx` |
| **`pyproject.toml`** | Completed — added author, `pydantic`, `[gpu]` group, `mypy`, `[tool.setuptools]` flat-layout config, `[tool.mypy]` config |
| **`__init__.py`** | New — root package init with `__version__`, `__author__`, usage docs |
| **`tests/__init__.py`** | New — empty marker so pytest package-mode discovers tests correctly |
| **`tests/conftest.py`** | New — shared fixtures: `small_graph`, `graph_features`, `db_session`, `seeded_db_session` |
| **`tests/test_processor.py`** | Rewritten — 18 tests across `TestBuildGraph`, `TestComputeFeatures`, `TestClassifyNodes` |
| **`tests/test_ingestor.py`** | New — 11 tests across `TestCreateHumans`, `TestCreateEngagementPod`, `TestCreateStarBot` |
| **`tests/test_api.py`** | Existing — 18 integration tests for FastAPI endpoints |

---

## Key Constants & Weights (config-driven via config.json)

```python
# Loaded from config.json → "scoring_weights"
W_ASYMMETRY  = 0.40   # Degree asymmetry weight
W_CLUSTERING = 0.35   # Clustering coefficient inverse weight
W_KCORE      = 0.25   # K-Core density weight

# Loaded from config.json → "classification"
BOT:             out_deg > in_deg * 5  AND  clustering < 0.2
                 OR risk > 0.7  AND  clustering < 0.3
ENGAGEMENT_POD:  k_core >= k_threshold  AND  clustering > 0.6
INFLUENCER:      in_deg > out_deg * 3   AND  in_deg >= 50
```

To tune detection sensitivity, edit `config.json` — no source code changes needed.

---

## Dataset

- **Source**: Stanford SNAP — [ego-Twitter](https://snap.stanford.edu/data/ego-Twitter.html)
- **Size**: 81,306 nodes, 1,768,149 edges
- **File**: `twitter_combined.txt` (edge list, space-separated)
- **Citation**: Leskovec & Krevl, SNAP Datasets, Stanford University, 2014
- **Format**: `<source_id> <target_id>` per line (undirected, read as `nx.Graph`)
