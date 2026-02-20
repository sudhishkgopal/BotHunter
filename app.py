"""
BotHunter Streamlit Dashboard.

Run locally:
    streamlit run app.py
"""

import streamlit as st
import plotly.express as px
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import os

from database import SessionLocal, init_db
from models import Relationship, User
from processor import build_graph, compute_features, classify_nodes

# Page config 

st.set_page_config(
    page_title="BotHunter",
    page_icon="🔍",
    layout="wide",
)

st.title("BotHunter — Bot Detection Dashboard")


# Load & cache data

@st.cache_data
def load_and_classify(_threshold: int) -> tuple[dict, dict, nx.DiGraph]:
    """Run the full detection pipeline and cache results."""
    init_db()
    with SessionLocal() as session:
        G, user_map = build_graph(session)
        if G.number_of_nodes() == 0:
            return {}, {}, G
        features = compute_features(G)
        results = classify_nodes(features, _threshold)
    return results, user_map, G


# Sidebar controls

st.sidebar.header("Controls")

threshold = st.sidebar.slider(
    "K-Core Threshold",
    min_value=2,
    max_value=50,
    value=20,
    help="Minimum k-core value to flag engagement pods.",
)

hide_influencers = st.sidebar.toggle(
    "Hide Influencers",
    value=False,
    help="Remove influencer nodes to focus on bot clusters.",
)

# Load data with selected threshold
results, user_map, G = load_and_classify(threshold)

if not results:
    st.error("Database is empty. Run `python ingestor.py` first.")
    st.stop()


# Filter out influencers if toggled 

if hide_influencers:
    filtered = {
        nid: r for nid, r in results.items()
        if r["label"] != "influencer"
    }
else:
    filtered = results


# Metrics Bar

total_nodes = len(filtered)
bots_identified = sum(
    1 for r in filtered.values() if r["label"] in ("bot", "engagement_pod")
)
# Density = actual edges / possible edges in directed graph
density = nx.density(G)

col1, col2, col3 = st.columns(3)
col1.metric("Total Nodes", f"{total_nodes:,}")
col2.metric("Bots Identified", f"{bots_identified:,}")
col3.metric("Network Density", f"{density:.6f}")

st.divider()

# Interactive Graph

st.subheader("Bot Neighbourhood")

# Build list of flagged bot/pod node IDs for the dropdown
flagged_ids = [
    nid for nid, r in filtered.items()
    if r["label"] in ("bot", "engagement_pod")
]

# Show platform_id in the dropdown for readability
flagged_options = {
    nid: f"{user_map[nid]['platform_id']}  (id={nid}, {filtered[nid]['label']})"
    for nid in flagged_ids
    if nid in user_map
}

if flagged_options:
    selected_id = st.selectbox(
        "Select a flagged node to inspect",
        options=list(flagged_options.keys()),
        format_func=lambda x: flagged_options[x],
    )

    # Extract 1-hop neighbourhood (all direct connections)
    neighbors = set(G.successors(selected_id)) | set(G.predecessors(selected_id))
    subgraph_nodes = {selected_id} | neighbors

    # Apply influencer filter to the subgraph too
    if hide_influencers:
        subgraph_nodes = {
            n for n in subgraph_nodes
            if n in filtered  # filtered already excludes influencers
        }

    sub_G = G.subgraph(subgraph_nodes)

    # Color map for node labels
    color_map = {
        "bot": "#e74c3c",
        "engagement_pod": "#f39c12",
        "influencer": "#3498db",
        "human": "#2ecc71",
    }

    # Build Pyvis network
    net = Network(
        height="500px",
        width="100%",
        directed=True,
        bgcolor="#0e1117",
        font_color="white",
    )
    net.barnes_hut(gravity=-3000, spring_length=150)

    for node in sub_G.nodes():
        r = filtered.get(node, results.get(node, {}))
        label_type = r.get("label", "human")
        pid = user_map.get(node, {}).get("platform_id", str(node))
        risk = r.get("risk_score", 0)

        # Selected node is larger and has a border
        size = 30 if node == selected_id else 15
        border = 3 if node == selected_id else 1

        net.add_node(
            node,
            label=pid,
            color=color_map.get(label_type, "#95a5a6"),
            size=size,
            borderWidth=border,
            title=f"{pid}\nLabel: {label_type}\nRisk: {risk:.4f}",
        )

    for src, dst in sub_G.edges():
        if src in subgraph_nodes and dst in subgraph_nodes:
            net.add_edge(src, dst, color="#555555")

    # Render Pyvis HTML inside Streamlit
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".html", mode="w"
    ) as tmp:
        net.save_graph(tmp.name)
        tmp_path = tmp.name

    with open(tmp_path, "r") as f:
        html = f.read()
    os.unlink(tmp_path)

    components.html(html, height=520, scrolling=False)

    # Legend
    legend_cols = st.columns(4)
    legend_cols[0].markdown("🔴 **Star Bot**")
    legend_cols[1].markdown("🟡 **Engagement Pod**")
    legend_cols[2].markdown("🔵 **Influencer**")
    legend_cols[3].markdown("🟢 **Human**")

else:
    st.info("No bots detected at this threshold. Try lowering the K-Core value.")

st.divider()


# Risk Score Distribution 

st.subheader("Risk Score Distribution")

scores = [
    {"node_id": nid, "risk_score": r["risk_score"], "label": r["label"]}
    for nid, r in filtered.items()
]

fig = px.histogram(
    scores,
    x="risk_score",
    color="label",
    nbins=50,
    color_discrete_map={
        "bot": "#e74c3c",
        "engagement_pod": "#f39c12",
        "influencer": "#3498db",
        "human": "#2ecc71",
    },
    labels={"risk_score": "Risk Score", "label": "Classification"},
    title="Distribution of Risk Scores Across All Nodes",
)
fig.update_layout(
    bargap=0.05,
    template="plotly_dark",
    xaxis_title="Risk Score",
    yaxis_title="Node Count",
)

st.plotly_chart(fig, use_container_width=True)

st.divider()


# Node Table

st.subheader("Flagged Nodes")

flagged_data = [
    {
        "Node ID": nid,
        "Platform ID": user_map.get(nid, {}).get("platform_id", "?"),
        "Risk Score": r["risk_score"],
        "K-Core": r["k_core"],
        "Clustering": round(r["clustering"], 3),
        "Out-Degree": r["out_deg"],
        "In-Degree": r["in_deg"],
        "Label": r["label"],
    }
    for nid, r in filtered.items()
    if r["label"] in ("bot", "engagement_pod")
]

if flagged_data:
    # Sort by risk score descending
    flagged_data.sort(key=lambda x: x["Risk Score"], reverse=True)
    st.dataframe(flagged_data, use_container_width=True, hide_index=True)
else:
    st.info("No flagged nodes to display.")
