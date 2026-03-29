# BotHunter — AI Session Context

> **This file is the single source of truth for any AI assistant starting a new session.  
> Read this file first. It is kept up-to-date after every meaningful code change.**

Last updated: 2026-03-29

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
├── app.py            ← Streamlit dashboard (main UI, ~423 lines)
├── main.py           ← FastAPI app + K-Core algorithm functions
├── processor.py      ← Feature computation & classification engine
├── ingestor.py       ← Synthetic data generator for testing
├── models.py         ← SQLAlchemy ORM models (User, Relationship, AnalysisResult)
├── database.py       ← DB engine setup, respects DATABASE_URL env var
├── cli.py            ← Typer CLI commands
├── config.json       ← Runtime config (k_core_threshold, max_neighbors_viz, etc.)
├── requirements.txt  ← Python dependencies
├── Dockerfile        ← Multi-stage Docker image (builder + runtime, non-root)
├── setup.sh          ← Shell setup script
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
| `POST` | `/simulate` | Run bot detection on a generated synthetic network |
| `POST` | `/analyze` | Upload a `.txt` edge list file and run detection |
| `GET` | `/twitter` | Run detection on the bundled Twitter SNAP dataset |
| `GET` | `/download/{filename}` | Serve generated PNG or JSON result files |
| `GET` | `/health` | Health check |

---

## Streamlit Dashboard (app.py)

The dashboard is **batch-loaded** — it loads only the top-N suspects at startup (configurable) for fast rendering.

**Sections:**
1. **Sidebar** — sliders for K-Core threshold, max neighbours in viz, startup batch size; influencer filter toggle; manual node selector
2. **Metrics bar** — Total Nodes, Bots Identified, Network Density
3. **High-Risk Accounts table** — Clickable `st.dataframe` with row selection; selecting a row focuses the graph on that node
4. **Pyvis Local Neighbourhood** — Interactive directed graph of the selected node's 1-hop neighbourhood, coloured by classification label
5. **Risk Score Histogram** — Plotly histogram of all flagged node scores, coloured by label

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
  "database_path": "bothunter.db"
}
```

The database path is also overridable via `DATABASE_URL` environment variable (full SQLAlchemy connection string, e.g. for PostgreSQL).

---

## How to Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

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

## Planned Enhancements (see implementation_plan.md)

The following improvements are planned but not yet implemented:

| Phase | Status | Summary |
|-------|--------|---------|
| **1 — Architecture refactor** | ⬜ Planned | Reorganize into `core/`, `api/`, `dashboard/`, `data/` packages; `pyproject.toml` |
| **2 — FastAPI overhaul** | ⬜ Planned | Async job polling, history endpoint, Pydantic schemas, rate limiting |
| **3 — Dashboard upgrades** | ⬜ Planned | 4 tabs (Overview, Detect, History, Export), CSV export, delta metrics |
| **4 — AI / Real-time** | ⬜ Planned | LLM-powered node explanation hook, streaming ingestion endpoint |
| **5 — Cloud deployment** | ⬜ Planned | `docker-compose.yml`, `render.yaml`, `railway.json`, `DEPLOY.md` |
| **6 — Testing & CI** | ⬜ Planned | `pytest` suite for classifier + API, GitHub Actions CI |

When any of these phases are implemented, **update this file** to move items from Planned to Complete and update the File Map, endpoints table, or schema sections accordingly.

---

## Key Constants & Weights (currently hardcoded in processor.py)

```python
W_ASYMMETRY = 0.40   # Degree asymmetry weight
W_CLUSTERING = 0.35  # Clustering coefficient inverse weight
W_KCORE = 0.25       # K-Core density weight

# Classification thresholds
BOT:             out_deg > in_deg * 5  AND  clustering < 0.2
                 OR risk > 0.7  AND  clustering < 0.3
ENGAGEMENT_POD:  k_core >= k_threshold  AND  clustering > 0.6
INFLUENCER:      in_deg > out_deg * 3   AND  in_deg >= 50
```

---

## Dataset

- **Source**: Stanford SNAP — [ego-Twitter](https://snap.stanford.edu/data/ego-Twitter.html)
- **Size**: 81,306 nodes, 1,768,149 edges
- **File**: `twitter_combined.txt` (edge list, space-separated)
- **Citation**: Leskovec & Krevl, SNAP Datasets, Stanford University, 2014
- **Format**: `<source_id> <target_id>` per line (undirected, read as `nx.Graph`)
