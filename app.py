"""
BotHunter Streamlit Dashboard — optimized for large datasets.

Load queries of the neighbourhood around a selected node. This keeps the app fast
even with the 1.7M-edge SNAP Twitter dataset.

"""

import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import networkx as nx
from pyvis.network import Network
from sqlalchemy import func, or_, text
import tempfile
import os

from database import SessionLocal, init_db
from models import AnalysisResult, Relationship, User
from processor import classify_nodes, compute_features

st.set_page_config(page_title="BotHunter", layout="wide")
st.title("BotHunter — Bot Detection Dashboard")


# CACHED DATABASE QUERIES — never re-reads SQLite on scroll / re-render

@st.cache_data(ttl=300)
def get_global_stats() -> dict:
    """Lightweight aggregate query — no full table scan."""
    with SessionLocal() as s:
        total_nodes = s.query(func.count(User.id)).scalar()
        total_edges = s.query(func.count(Relationship.id)).filter(
            Relationship.relation_type == "follow"
        ).scalar()
        # Density = edges / (nodes * (nodes - 1)) for a directed graph
        density = (
            total_edges / (total_nodes * (total_nodes - 1))
            if total_nodes > 1 else 0.0
        )
    return {
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "density": density,
    }


@st.cache_data(ttl=300)
def get_flagged_nodes(threshold: int) -> list[dict]:
    """
    Run classification on a SAMPLED subgraph of high-degree nodes only.
    Instead of loading all 1.7M edges, pull the top nodes by degree
    and classify just those — fast enough for interactive use.
    """
    with SessionLocal() as s:
        # Find the top 500 nodes by outgoing edge count (most likely bots)
        out_counts = (
            s.query(
                Relationship.source_user_id,
                func.count().label("out_deg"),
            )
            .filter(Relationship.relation_type == "follow")
            .group_by(Relationship.source_user_id)
            .order_by(text("out_deg DESC"))
            .limit(500)
            .all()
        )

        # Also grab the top 500 by incoming edges (potential influencers/pods)
        in_counts = (
            s.query(
                Relationship.target_user_id,
                func.count().label("in_deg"),
            )
            .filter(Relationship.relation_type == "follow")
            .group_by(Relationship.target_user_id)
            .order_by(text("in_deg DESC"))
            .limit(500)
            .all()
        )

        # Merge into a candidate set
        candidate_ids = set()
        for row in out_counts:
            candidate_ids.add(row[0])
        for row in in_counts:
            candidate_ids.add(row[0])

        if not candidate_ids:
            return []

        # Pull only edges between candidates for classification
        edges = (
            s.query(Relationship)
            .filter(
                Relationship.relation_type == "follow",
                Relationship.source_user_id.in_(candidate_ids),
                Relationship.target_user_id.in_(candidate_ids),
            )
            .all()
        )

        users = (
            s.query(User)
            .filter(User.id.in_(candidate_ids))
            .all()
        )

    # Build a small DiGraph from candidates only
    G = nx.DiGraph()
    user_map = {}
    for u in users:
        G.add_node(u.id)
        user_map[u.id] = {
            "platform_id": u.platform_id,
            "username": u.username,
            "is_bot": u.is_bot,
        }
    for e in edges:
        G.add_edge(e.source_user_id, e.target_user_id)

    features = compute_features(G)
    results = classify_nodes(features, threshold)

    # Return only flagged nodes as a list of dicts
    flagged = []
    for nid, r in results.items():
        if r["label"] in ("bot", "engagement_groups", "influencer"):
            flagged.append({
                "node_id": nid,
                "platform_id": user_map.get(nid, {}).get("platform_id", "?"),
                "risk_score": r["risk_score"],
                "k_core": r["k_core"],
                "clustering": round(r["clustering"], 3),
                "out_deg": r["out_deg"],
                "in_deg": r["in_deg"],
                "label": r["label"],
            })

    flagged.sort(key=lambda x: x["risk_score"], reverse=True)
    return flagged


@st.cache_data(ttl=300)
def get_neighbourhood(node_id: int, max_neighbors: int = 50) -> dict:
    """
    Pull ONLY the 1-hop neighbourhood of a single node from the DB.
    Limits to max_neighbors to keep the viz snappy.
    Returns node list, edge list, and metadata.
    """
    with SessionLocal() as s:
        # Outgoing edges from this node (capped)
        outgoing = (
            s.query(Relationship)
            .filter(
                Relationship.source_user_id == node_id,
                Relationship.relation_type == "follow",
            )
            .limit(max_neighbors)
            .all()
        )

        # Incoming edges to this node (capped)
        incoming = (
            s.query(Relationship)
            .filter(
                Relationship.target_user_id == node_id,
                Relationship.relation_type == "follow",
            )
            .limit(max_neighbors)
            .all()
        )

        # Collect all neighbour IDs
        neighbor_ids = set()
        edges = []
        for e in outgoing:
            neighbor_ids.add(e.target_user_id)
            edges.append((e.source_user_id, e.target_user_id))
        for e in incoming:
            neighbor_ids.add(e.source_user_id)
            edges.append((e.source_user_id, e.target_user_id))

        all_ids = neighbor_ids | {node_id}

        # Fetch user info for these nodes only
        users = s.query(User).filter(User.id.in_(all_ids)).all()

        # Fetch edges BETWEEN neighbours (reveals cluster structure)
        inter_edges = (
            s.query(Relationship)
            .filter(
                Relationship.relation_type == "follow",
                Relationship.source_user_id.in_(neighbor_ids),
                Relationship.target_user_id.in_(neighbor_ids),
            )
            .limit(max_neighbors * 5)
            .all()
        )
        for e in inter_edges:
            edges.append((e.source_user_id, e.target_user_id))

    user_map = {
        u.id: {"platform_id": u.platform_id, "is_bot": u.is_bot}
        for u in users
    }

    return {
        "center": node_id,
        "nodes": list(all_ids),
        "edges": list(set(edges)),  # dedupe
        "user_map": user_map,
    }


@st.cache_data(ttl=300)
def get_all_risk_scores(threshold: int) -> list[dict]:
    """Get risk scores for the sampled candidate set (for histogram)."""
    flagged = get_flagged_nodes(threshold)
    return flagged


# SIDEBAR CONTROLS

init_db()

st.sidebar.header("Controls")

threshold = st.sidebar.slider(
    "K-Core Threshold", min_value=2, max_value=50, value=20,
    help="Minimum k-core value to flag engagement groups.",
)

max_neighbors = st.sidebar.slider(
    "Max Neighbours", min_value=10, max_value=200, value=50,
    help="Cap on neighbours pulled per node (keeps viz fast).",
)

hide_influencers = st.sidebar.toggle(
    "Hide Influencers", value=False,
    help="Remove influencer nodes to focus on bot clusters.",
)


# METRICS BAR

stats = get_global_stats()
flagged = get_flagged_nodes(threshold)

if hide_influencers:
    flagged = [f for f in flagged if f["label"] != "influencer"]

bot_count = sum(1 for f in flagged if f["label"] in ("bot", "engagement_groups"))

col1, col2, col3 = st.columns(3)
col1.metric("Total Nodes", f"{stats['total_nodes']:,}")
col2.metric("Bots Identified", f"{bot_count:,}")
col3.metric("Network Density", f"{stats['density']:.6f}")

st.divider()


# HIGH-RISK TABLE — click a row to update the graph

st.subheader("High-Risk Accounts")

if not flagged:
    st.info("No bots detected at this threshold. Try lowering the K-Core value.")
    st.stop()

# Build a selectable dataframe
import pandas as pd

df = pd.DataFrame(flagged)
df = df.rename(columns={
    "node_id": "Node ID",
    "platform_id": "Platform ID",
    "risk_score": "Risk Score",
    "k_core": "K-Core",
    "clustering": "Clustering",
    "out_deg": "Out-Degree",
    "in_deg": "In-Degree",
    "label": "Label",
})

# Interactive table — selecting a row updates the graph below
selection = st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
)

# Determine which node to visualize
selected_rows = selection.get("selection", {}).get("rows", [])
if selected_rows:
    selected_node = flagged[selected_rows[0]]["node_id"]
else:
    selected_node = flagged[0]["node_id"]  # default to highest-risk

# Also allow manual selection via sidebar dropdown
st.sidebar.divider()
st.sidebar.subheader("Manual Node Select")
dropdown_options = {f["node_id"]: f"{f['platform_id']} ({f['label']})" for f in flagged}
sidebar_selected = st.sidebar.selectbox(
    "Or pick a node:",
    options=list(dropdown_options.keys()),
    format_func=lambda x: dropdown_options[x],
    index=list(dropdown_options.keys()).index(selected_node)
    if selected_node in dropdown_options else 0,
)

# Table click takes priority; sidebar is a fallback
if selected_rows:
    focus_node = selected_node
else:
    focus_node = sidebar_selected

st.divider()


# INTERACTIVE PYVIS GRAPH — only the selected node's neighbourhood

st.subheader("Drill-Down: Local Neighbourhood")

# Show which node is being visualized
focus_info = next((f for f in flagged if f["node_id"] == focus_node), None)
if focus_info:
    st.caption(
        f"Showing **{focus_info['platform_id']}** "
        f"(id={focus_node}, {focus_info['label']}, "
        f"risk={focus_info['risk_score']:.4f})"
    )

# Pull only this node's neighbourhood from the DB
hood = get_neighbourhood(focus_node, max_neighbors)

# Classify neighbourhood nodes for coloring
sub_G = nx.DiGraph()
sub_G.add_nodes_from(hood["nodes"])
for src, dst in hood["edges"]:
    if src in hood["nodes"] and dst in hood["nodes"]:
        sub_G.add_edge(src, dst)

sub_features = compute_features(sub_G)
sub_results = classify_nodes(sub_features, threshold)

# Filter influencers from the viz if toggled
if hide_influencers:
    visible_nodes = {
        n for n in hood["nodes"]
        if sub_results.get(n, {}).get("label", "human") != "influencer"
    }
else:
    visible_nodes = set(hood["nodes"])

color_map = {
    "bot": "#e74c3c",
    "engagement_groups": "#f39c12",
    "influencer": "#3498db",
    "human": "#2ecc71",
}

net = Network(
    height="550px",
    width="100%",
    directed=True,
    bgcolor="#0e1117",
    font_color="white",
)
net.barnes_hut(gravity=-3000, spring_length=150)

for node in visible_nodes:
    r = sub_results.get(node, {})
    label_type = r.get("label", "human")
    uinfo = hood["user_map"].get(node, {})
    pid = uinfo.get("platform_id", str(node))

    # Center node is larger with a thick border
    is_center = node == focus_node
    size = 35 if is_center else 14
    border = 4 if is_center else 1

    net.add_node(
        node,
        label=pid if is_center else "",  # only label the center to reduce clutter
        color=color_map.get(label_type, "#95a5a6"),
        size=size,
        borderWidth=border,
        title=(
            f"{pid}\n"
            f"Label: {label_type}\n"
            f"Risk: {r.get('risk_score', 0):.4f}\n"
            f"K-Core: {r.get('k_core', 0)}\n"
            f"Out: {r.get('out_deg', 0)}  In: {r.get('in_deg', 0)}"
        ),
    )

for src, dst in hood["edges"]:
    if src in visible_nodes and dst in visible_nodes:
        net.add_edge(src, dst, color="#444444")

# Render Pyvis HTML and embed via st.components
with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w") as tmp:
    net.save_graph(tmp.name)
    tmp_path = tmp.name

with open(tmp_path, "r") as f:
    html = f.read()
os.unlink(tmp_path)

components.html(html, height=570, scrolling=False)

# Legend
lcols = st.columns(4)
lcols[0].markdown("🔴 **Star Bot**")
lcols[1].markdown("🟡 **Engagement Group**")
lcols[2].markdown("🔵 **Influencer**")
lcols[3].markdown("🟢 **Human**")

st.divider()


# 
# RISK SCORE HISTOGRAM
# 

st.subheader("Risk Score Distribution (Sampled Candidates)")

if flagged:
    fig = px.histogram(
        flagged,
        x="risk_score",
        color="label",
        nbins=40,
        color_discrete_map={
            "bot": "#e74c3c",
            "engagement_groups": "#f39c12",
            "influencer": "#3498db",
            "human": "#2ecc71",
        },
        labels={"risk_score": "Risk Score", "label": "Classification"},
    )
    fig.update_layout(
        bargap=0.05,
        template="plotly_dark",
        xaxis_title="Risk Score",
        yaxis_title="Node Count",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No data to plot.")
