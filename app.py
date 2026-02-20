"""
BotHunter Streamlit Dashboard — batch-loaded, config-driven.
Only loads the Top N suspected bots at startup for fast rendering.
"""

import json
import os
import tempfile

import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import pandas as pd
import networkx as nx
from pyvis.network import Network
from sqlalchemy import func, text

from database import SessionLocal, init_db
from models import AnalysisResult, Relationship, User
from processor import compute_features, classify_nodes


#  Display names — maps internal classifier labels to user-facing text 

DISPLAY_NAMES = {
    "bot": "Star Bot",
    "engagement_pod": "Engagement Group",
    "influencer": "Influencer",
    "organic": "Human",
}

COLOR_MAP = {
    "bot": "#e74c3c",
    "engagement_pod": "#f39c12",
    "influencer": "#3498db",
    "organic": "#2ecc71",
}

# Display-name-keyed version for Plotly legends
DISPLAY_COLOR_MAP = {
    "Star Bot": "#e74c3c",
    "Engagement Group": "#f39c12",
    "Influencer": "#3498db",
    "Human": "#2ecc71",
}


#  Load config.json 

@st.cache_data
def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {
        "k_core_threshold": 20,
        "max_neighbors_viz": 50,
        "top_n_startup": 100,
        "cache_ttl_seconds": 300,
    }


cfg = load_config()

st.set_page_config(page_title="BotHunter", layout="wide")
st.title("BotHunter — Bot Detection Dashboard")
init_db()


#  Sidebar 

st.sidebar.header("Controls")

threshold = st.sidebar.slider(
    "K-Core Threshold", min_value=2, max_value=50,
    value=cfg["k_core_threshold"],
)

max_neighbors = st.sidebar.slider(
    "Max Neighbours in Viz", min_value=10, max_value=200,
    value=cfg["max_neighbors_viz"],
)

top_n = st.sidebar.slider(
    "Startup Batch Size", min_value=25, max_value=500,
    value=cfg["top_n_startup"],
    help="Number of suspect nodes loaded at startup.",
)

hide_influencers = st.sidebar.toggle("Hide Influencers", value=False)



# CACHED QUERIES


@st.cache_data(ttl=cfg["cache_ttl_seconds"])
def get_global_stats() -> dict:
    """Fast aggregate counts — no full scan."""
    with SessionLocal() as s:
        nodes = s.query(func.count(User.id)).scalar()
        edges = s.query(func.count(Relationship.id)).filter(
            Relationship.relation_type == "follow"
        ).scalar()
        density = edges / (nodes * (nodes - 1)) if nodes > 1 else 0.0
    return {"total_nodes": nodes, "total_edges": edges, "density": density}


@st.cache_data(ttl=cfg["cache_ttl_seconds"])
def get_top_suspects(_threshold: int, _top_n: int) -> list[dict]:
    """
    Batch-load only the top N suspects by degree.
    Classifies a small candidate subgraph instead of the full graph.
    """
    with SessionLocal() as s:
        out_top = (
            s.query(Relationship.source_user_id, func.count().label("cnt"))
            .filter(Relationship.relation_type == "follow")
            .group_by(Relationship.source_user_id)
            .order_by(text("cnt DESC"))
            .limit(_top_n)
            .all()
        )
        in_top = (
            s.query(Relationship.target_user_id, func.count().label("cnt"))
            .filter(Relationship.relation_type == "follow")
            .group_by(Relationship.target_user_id)
            .order_by(text("cnt DESC"))
            .limit(_top_n)
            .all()
        )

        candidate_ids = {r[0] for r in out_top} | {r[0] for r in in_top}
        if not candidate_ids:
            return []

        edges = (
            s.query(Relationship)
            .filter(
                Relationship.relation_type == "follow",
                Relationship.source_user_id.in_(candidate_ids),
                Relationship.target_user_id.in_(candidate_ids),
            )
            .all()
        )
        users = s.query(User).filter(User.id.in_(candidate_ids)).all()

    G = nx.DiGraph()
    user_map = {}
    for u in users:
        G.add_node(u.id)
        user_map[u.id] = {"platform_id": u.platform_id, "is_bot": u.is_bot}
    for e in edges:
        G.add_edge(e.source_user_id, e.target_user_id)

    features = compute_features(G)
    results = classify_nodes(features, _threshold)

    flagged = []
    for nid, r in results.items():
        if r["label"] in ("bot", "engagement_pod", "influencer"):
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


@st.cache_data(ttl=cfg["cache_ttl_seconds"])
def get_neighbourhood(_node_id: int, _max_neighbors: int) -> dict:
    """Pull only the 1-hop neighbourhood of one node."""
    with SessionLocal() as s:
        outgoing = (
            s.query(Relationship)
            .filter(
                Relationship.source_user_id == _node_id,
                Relationship.relation_type == "follow",
            )
            .limit(_max_neighbors)
            .all()
        )
        incoming = (
            s.query(Relationship)
            .filter(
                Relationship.target_user_id == _node_id,
                Relationship.relation_type == "follow",
            )
            .limit(_max_neighbors)
            .all()
        )

        neighbor_ids = set()
        edges = []
        for e in outgoing:
            neighbor_ids.add(e.target_user_id)
            edges.append((e.source_user_id, e.target_user_id))
        for e in incoming:
            neighbor_ids.add(e.source_user_id)
            edges.append((e.source_user_id, e.target_user_id))

        all_ids = neighbor_ids | {_node_id}
        users = s.query(User).filter(User.id.in_(all_ids)).all()

        inter = (
            s.query(Relationship)
            .filter(
                Relationship.relation_type == "follow",
                Relationship.source_user_id.in_(neighbor_ids),
                Relationship.target_user_id.in_(neighbor_ids),
            )
            .limit(_max_neighbors * 5)
            .all()
        )
        for e in inter:
            edges.append((e.source_user_id, e.target_user_id))

    user_map = {
        u.id: {"platform_id": u.platform_id, "is_bot": u.is_bot}
        for u in users
    }
    return {
        "center": _node_id,
        "nodes": list(all_ids),
        "edges": list(set(edges)),
        "user_map": user_map,
    }



# METRICS BAR


stats = get_global_stats()
flagged = get_top_suspects(threshold, top_n)

if hide_influencers:
    flagged = [f for f in flagged if f["label"] != "influencer"]

bot_count = sum(1 for f in flagged if f["label"] in ("bot", "engagement_pod"))

col1, col2, col3 = st.columns(3)
col1.metric("Total Nodes", f"{stats['total_nodes']:,}")
col2.metric("Bots Identified", f"{bot_count:,}")
col3.metric("Network Density", f"{stats['density']:.6f}")

st.divider()



# HIGH-RISK TABLE — click a row to update the graph


st.subheader("High-Risk Accounts")

if not flagged:
    st.info("No bots detected. Try lowering the K-Core threshold.")
    st.stop()

# Add display-friendly label column
display_flagged = []
for f in flagged:
    row = dict(f)
    row["display_label"] = DISPLAY_NAMES.get(f["label"], f["label"])
    display_flagged.append(row)

df = pd.DataFrame(display_flagged)
df = df.rename(columns={
    "node_id": "Node ID",
    "platform_id": "Platform ID",
    "risk_score": "Risk Score",
    "k_core": "K-Core",
    "clustering": "Clustering",
    "out_deg": "Out-Degree",
    "in_deg": "In-Degree",
    "display_label": "Label",
})
# Drop the internal label column from the display table
df = df.drop(columns=["label"])

selection = st.dataframe(
    df, use_container_width=True, hide_index=True,
    on_select="rerun", selection_mode="single-row",
)

selected_rows = selection.get("selection", {}).get("rows", [])
if selected_rows:
    focus_node = flagged[selected_rows[0]]["node_id"]
else:
    focus_node = flagged[0]["node_id"]

# Sidebar fallback dropdown
st.sidebar.divider()
st.sidebar.subheader("Manual Node Select")
opts = {
    f["node_id"]: f"{f['platform_id']} ({DISPLAY_NAMES.get(f['label'], f['label'])})"
    for f in flagged
}
sidebar_pick = st.sidebar.selectbox(
    "Or pick a node:",
    options=list(opts.keys()),
    format_func=lambda x: opts[x],
    index=list(opts.keys()).index(focus_node) if focus_node in opts else 0,
)
if not selected_rows:
    focus_node = sidebar_pick

st.divider()



# PYVIS DRILL-DOWN


st.subheader("Local Neighbourhood")

focus_info = next((f for f in flagged if f["node_id"] == focus_node), None)
if focus_info:
    display_label = DISPLAY_NAMES.get(focus_info["label"], focus_info["label"])
    st.caption(
        f"Showing **{focus_info['platform_id']}** "
        f"(id={focus_node}, {display_label}, "
        f"risk={focus_info['risk_score']:.4f})"
    )

hood = get_neighbourhood(focus_node, max_neighbors)

# Classify the subgraph for node coloring
sub_G = nx.DiGraph()
sub_G.add_nodes_from(hood["nodes"])
for src, dst in hood["edges"]:
    if src in set(hood["nodes"]) and dst in set(hood["nodes"]):
        sub_G.add_edge(src, dst)

sub_features = compute_features(sub_G)
sub_results = classify_nodes(sub_features, threshold)

if hide_influencers:
    visible = {
        n for n in hood["nodes"]
        if sub_results.get(n, {}).get("label", "organic") != "influencer"
    }
else:
    visible = set(hood["nodes"])

net = Network(
    height="550px", width="100%", directed=True,
    bgcolor="#0e1117", font_color="white",
)
net.barnes_hut(gravity=-3000, spring_length=150)

for node in visible:
    r = sub_results.get(node, {})
    lbl = r.get("label", "organic")
    pid = hood["user_map"].get(node, {}).get("platform_id", str(node))
    is_center = node == focus_node
    display_lbl = DISPLAY_NAMES.get(lbl, lbl)

    net.add_node(
        node,
        label=pid if is_center else "",
        color=COLOR_MAP.get(lbl, "#95a5a6"),
        size=35 if is_center else 14,
        borderWidth=4 if is_center else 1,
        title=f"{pid}\nLabel: {display_lbl}\nRisk: {r.get('risk_score', 0):.4f}",
    )

for src, dst in hood["edges"]:
    if src in visible and dst in visible:
        net.add_edge(src, dst, color="#444444")

with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w") as tmp:
    net.save_graph(tmp.name)
    tmp_path = tmp.name

with open(tmp_path, "r") as f:
    html = f.read()
os.unlink(tmp_path)

components.html(html, height=570, scrolling=False)

lcols = st.columns(4)
lcols[0].markdown("🔴 **Star Bot**")
lcols[1].markdown("🟡 **Engagement Group**")
lcols[2].markdown("🔵 **Influencer**")
lcols[3].markdown("🟢 **Human**")

st.divider()



# RISK SCORE HISTOGRAM


st.subheader("Risk Score Distribution")

if flagged:
    # Add display labels for the Plotly legend
    plot_data = []
    for f in flagged:
        row = dict(f)
        row["display_label"] = DISPLAY_NAMES.get(f["label"], f["label"])
        plot_data.append(row)

    fig = px.histogram(
        plot_data, x="risk_score", color="display_label", nbins=40,
        color_discrete_map=DISPLAY_COLOR_MAP,
        labels={"risk_score": "Risk Score", "display_label": "Classification"},
    )
    fig.update_layout(
        bargap=0.05, template="plotly_dark",
        xaxis_title="Risk Score", yaxis_title="Node Count",
    )
    st.plotly_chart(fig, use_container_width=True)
