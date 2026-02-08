# BotHunter
A Python-based graph engine that detects "bot Farms" using K-Core Decomposition. It uses an algorithm that recursively reveals high-density clusters. Build for scalability using the Master Worker pattern to handle large social connections.

# Purpose
Social media platforms are often manipulated by bot cliques which are accounts that all follow each other to bypass "low follower" counts. While individually these bots look like noise in a standard network, they are mathematically distinct. This project implements a Distributed Pregel style engine to cut away human noise and reveal high density bot cliques.

# Tech Stack
Language: Python 3.12
Graph Theory: NetworkX
Distrubted Computing: Ray
Preformance: Numba
Visualization: Matplotlib, PyVis, JupyterNotebook

# How it works
The engine uses a recursive pruning algorithm to identify subgraphs where every node has at least k neighbors.
1. In each step, the system identifies all nodes with a degree less than k
2. Removing a node reduced the degree of its neighbors, potentially triggering their removal in the next step
3. The algorithm stops when only the core remains. In this case, when all the bots have been isolated

# Results
