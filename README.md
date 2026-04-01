# BotHunter

[![CI](https://github.com/sudhishkg11/BotHunter/actions/workflows/ci.yml/badge.svg)](https://github.com/sudhishkg11/BotHunter/actions/workflows/ci.yml)

A graph-based bot detection engine that identifies coordinated bot farms using K-Core Decomposition. The system recursively prunes low-connectivity nodes to reveal high-density clusters, built on a Master-Worker architecture for parallel processing of large-scale social networks.

## Purpose

Social media platforms are routinely manipulated by bot cliques — accounts that mutually follow each other to artificially inflate follower counts. Individually, these bots blend into normal network traffic. Structurally, they form mathematically distinct dense subgraphs.

BotHunter implements a Pregel-style distributed engine using a Master-Worker pattern to decompose social graphs at scale. The master partitions the graph and dispatches subgraph chunks to worker processes, which independently prune low-degree nodes and report results back for global aggregation. This approach parallelizes the most expensive operation (neighbour-degree computation) and scales linearly with available cores.

## Tech Stack

| Layer                 | Tool                            |
| --------------------- | ------------------------------- |
| Language              | Python 3.12                     |
| Graph Analysis        | NetworkX                        |
| Distributed Computing | Multiprocessing (Master-Worker) |
| Database              | SQLAlchemy + SQLite             |
| API                   | FastAPI                         |
| CLI                   | Typer + Rich                    |
| Dashboard             | Streamlit, Plotly, Pyvis        |
| Visualization         | Matplotlib, Pyvis               |

## How It Works

### K-Core Decomposition

The engine uses recursive pruning to isolate subgraphs where every node maintains at least _k_ mutual connections:

1. **Scan** — Identify all nodes with degree less than _k_.
2. **Prune** — Remove those nodes. This reduces the degree of their neighbours, potentially dropping them below _k_ as well.
3. **Repeat** — Continue until no more nodes can be removed. The surviving subgraph is the _k_-core — a dense cluster where every member has at least _k_ connections to other members.

In bot farm networks, legitimate users get pruned away in early rounds. The remaining core contains the tightly interconnected bot clique.

### Multi-Feature Risk Scoring

K-Core alone produces false positives. A tight friend group of 15 people can hit _k_=10. The detection engine layers three signals with weighted scoring:

| Signal                             | Weight | What It Catches                                      |
| ---------------------------------- | ------ | ---------------------------------------------------- |
| **Degree Asymmetry**               | 0.40   | Star bots (5,000 outgoing follows, 10 incoming)      |
| **Clustering Coefficient Inverse** | 0.35   | Accounts whose followers don't know each other       |
| **K-Core Density**                 | 0.25   | Engagement groups operating as mutual-follow cliques |

### The Celebrity Problem

A politician or musician with 500K followers and low out-degree looks similar to a bot on degree metrics alone — massive in-degree, minimal reciprocity. Flagging them would be a critical false positive.

BotHunter solves this using **Local Clustering Coefficients**. The clustering coefficient measures how interconnected a node's neighbours are:

- **Celebrity/Influencer**: Followers are strangers to each other. Clustering coefficient approaches **0.0**, but in-degree is legitimately high (>50) and out-degree is low. Classified as `influencer`, not `bot`.
- **Engagement Group**: Members all follow each other. Clustering coefficient approaches **1.0** with high mutual degree. Classified as `engagement_pod`.
- **Star Bot**: Mass-follows random accounts who don't know each other. Clustering coefficient near **0.0**, combined with extreme out-degree and negligible in-degree. Classified as `bot`.

This separation ensures high-profile accounts are never flagged while genuine bot patterns remain detectable.

## Getting Started

```bash
pip install -r requirements.txt
python database.py
python ingestor.py
python -m streamlit run app.py
```

## Credits

| Resource                         | Source                                                                                                              |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Twitter Social Graph Dataset** | [Stanford SNAP](https://snap.stanford.edu/data/ego-Twitter.html) — 81,306 nodes, 1,768,149 edges                    |
| **Citation**                     | Jure Leskovec & Andrej Krevl. _SNAP Datasets: Stanford Large Network Dataset Collection._ Stanford University, 2014 |
| **K-Core Decomposition**         | Seidman, S.B. (1983). _Network structure and minimum degree._ Social Networks, 5(3), 269-287                        |
