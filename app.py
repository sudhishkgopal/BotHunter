"""
BotHunter Streamlit Dashboard — 4-tab layout.

Tabs:
  🏠 Overview  — Project explainer, live network stats
  🔍 Detect    — Suspect table + interactive Pyvis graph
  📜 History   — Past analysis runs with delta comparisons
  📤 Export    — CSV / JSON download of flagged accounts
"""

import io
import json
import os
import tempfile

import pandas as pd
import networkx as nx
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from sqlalchemy import func, text

from database import SessionLocal, init_db
from models import AnalysisResult, Relationship, User
from processor import compute_features, classify_nodes

# ─── Constants ────────────────────────────────────────────────────────────────

DISPLAY_NAMES = {
    "bot":            "Star Bot",
    "engagement_pod": "Engagement Group",
    "influencer":     "Influencer",
    "organic":        "Human",
    "normal":         "Human",
}

COLOR_MAP = {
    "bot":            "#e74c3c",
    "engagement_pod": "#f39c12",
    "influencer":     "#3498db",
    "organic":        "#2ecc71",
    "normal":         "#2ecc71",
}

DISPLAY_COLOR_MAP = {
    "Star Bot":         "#e74c3c",
    "Engagement Group": "#f39c12",
    "Influencer":       "#3498db",
    "Human":            "#2ecc71",
}

# ─── Config ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_config() -> dict:
    path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "k_core_threshold": 20,
        "max_neighbors_viz": 50,
        "top_n_startup": 100,
        "cache_ttl_seconds": 300,
    }

cfg = load_config()

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BotHunter",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)
init_db()

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔍 BotHunter")
    st.caption("Graph-based bot detection engine")
    st.divider()

    st.header("Detection Controls")
    threshold = st.slider(
        "K-Core Threshold", min_value=2, max_value=50,
        value=cfg["k_core_threshold"],
        help="Minimum connections required to stay in the dense core.",
    )
    max_neighbors = st.slider(
        "Max Neighbours in Graph", min_value=10, max_value=200,
        value=cfg["max_neighbors_viz"],
    )
    top_n = st.slider(
        "Startup Batch Size", min_value=25, max_value=500,
        value=cfg["top_n_startup"],
        help="Number of high-degree candidate nodes loaded at startup.",
    )
    hide_influencers = st.toggle("Hide Influencers", value=False)
    st.divider()
    st.caption("v1.0.0 · Stanford SNAP Twitter Dataset")

# ─── Cached queries ───────────────────────────────────────────────────────────

@st.cache_data(ttl=cfg["cache_ttl_seconds"])
def get_global_stats() -> dict:
    with SessionLocal() as s:
        nodes   = s.query(func.count(User.id)).scalar()
        edges   = s.query(func.count(Relationship.id)).filter(
            Relationship.relation_type == "follow"
        ).scalar()
        density = edges / (nodes * (nodes - 1)) if nodes > 1 else 0.0
    return {"total_nodes": nodes, "total_edges": edges, "density": density}


@st.cache_data(ttl=cfg["cache_ttl_seconds"])
def get_top_suspects(_threshold: int, _top_n: int) -> list[dict]:
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
    results  = classify_nodes(features, _threshold)

    flagged = []
    for nid, r in results.items():
        if r["label"] in ("bot", "engagement_pod", "influencer"):
            flagged.append({
                "node_id":    nid,
                "platform_id": user_map.get(nid, {}).get("platform_id", "?"),
                "risk_score": r["risk_score"],
                "k_core":     r["k_core"],
                "clustering": round(r["clustering"], 3),
                "out_deg":    r["out_deg"],
                "in_deg":     r["in_deg"],
                "label":      r["label"],
            })

    flagged.sort(key=lambda x: x["risk_score"], reverse=True)
    return flagged


@st.cache_data(ttl=cfg["cache_ttl_seconds"])
def get_neighbourhood(_node_id: int, _max_neighbors: int) -> dict:
    with SessionLocal() as s:
        outgoing = (
            s.query(Relationship)
            .filter(Relationship.source_user_id == _node_id,
                    Relationship.relation_type == "follow")
            .limit(_max_neighbors).all()
        )
        incoming = (
            s.query(Relationship)
            .filter(Relationship.target_user_id == _node_id,
                    Relationship.relation_type == "follow")
            .limit(_max_neighbors).all()
        )
        neighbor_ids = set()
        edges: list[tuple] = []
        for e in outgoing:
            neighbor_ids.add(e.target_user_id)
            edges.append((e.source_user_id, e.target_user_id))
        for e in incoming:
            neighbor_ids.add(e.source_user_id)
            edges.append((e.source_user_id, e.target_user_id))
        all_ids = neighbor_ids | {_node_id}
        users   = s.query(User).filter(User.id.in_(all_ids)).all()
        inter   = (
            s.query(Relationship)
            .filter(
                Relationship.relation_type == "follow",
                Relationship.source_user_id.in_(neighbor_ids),
                Relationship.target_user_id.in_(neighbor_ids),
            )
            .limit(_max_neighbors * 5).all()
        )
        for e in inter:
            edges.append((e.source_user_id, e.target_user_id))
    user_map = {u.id: {"platform_id": u.platform_id, "is_bot": u.is_bot} for u in users}
    return {
        "center":   _node_id,
        "nodes":    list(all_ids),
        "edges":    list(set(edges)),
        "user_map": user_map,
    }


@st.cache_data(ttl=cfg["cache_ttl_seconds"])
def get_history_runs(limit: int = 50) -> list[dict]:
    with SessionLocal() as s:
        rows = (
            s.query(AnalysisResult)
            .order_by(AnalysisResult.ran_at.desc())
            .limit(limit)
            .all()
        )
    return [
        {
            "id":                r.id,
            "run_label":         r.run_label or f"Run #{r.id}",
            "k_core_threshold":  r.k_core_threshold,
            "total_nodes":       r.total_nodes,
            "total_edges":       r.total_edges,
            "bots_detected":     r.bots_detected,
            "detection_accuracy": r.detection_accuracy,
            "ran_at":            r.ran_at.strftime("%Y-%m-%d %H:%M UTC"),
        }
        for r in rows
    ]

# ─── Pre-load data ────────────────────────────────────────────────────────────

stats   = get_global_stats()
flagged = get_top_suspects(threshold, top_n)

if hide_influencers:
    flagged = [f for f in flagged if f["label"] != "influencer"]

bot_count = sum(1 for f in flagged if f["label"] in ("bot", "engagement_pod"))

# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab_overview, tab_detect, tab_history, tab_export = st.tabs(
    ["🏠 Overview", "🔍 Detect", "📜 History", "📤 Export"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════

with tab_overview:
    st.title("BotHunter — Bot Detection Dashboard")
    st.markdown(
        "A graph-based bot detection engine that identifies coordinated bot farms "
        "using **K-Core Decomposition** on social media follower graphs."
    )

    # Live stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Nodes", f"{stats['total_nodes']:,}")
    c2.metric("Total Edges", f"{stats['total_edges']:,}")
    c3.metric("Suspected Bots", f"{bot_count:,}")
    c4.metric("Network Density", f"{stats['density']:.6f}")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("⚙️ How It Works")
        st.markdown("""
**K-Core Decomposition** recursively prunes nodes with fewer than *k* connections.
Legitimate users get pruned away in early rounds. What remains is the dense, 
tightly-connected bot core.

**Multi-signal risk score** adds precision on top of raw K-Core:

| Signal | Weight | Detects |
|--------|--------|---------|
| Degree Asymmetry | 40% | Star bots (mass-follower, not followed back) |
| Clustering Coefficient Inverse | 35% | Accounts whose followers don't know each other |
| K-Core Density | 25% | Engagement pods in mutual-follow cliques |
        """)

    with col_right:
        st.subheader("🏷️ Classification Labels")
        st.markdown("""
| Label | Colour | Description |
|-------|--------|-------------|
| 🔴 Star Bot | Red | Follows thousands, barely followed back |
| 🟡 Engagement Group | Orange | Mutual-follow clique coordinating likes |
| 🔵 Influencer | Blue | High followers, low following (NOT a bot) |
| 🟢 Human | Green | Normal organic account |
        """)

        st.subheader("🎓 Celebrity Problem")
        st.markdown(
            "A politician with 500K followers looks suspicious on degree alone. "
            "BotHunter uses **Local Clustering Coefficients** to separate them: "
            "a celebrity's followers are strangers to each other (clustering ≈ 0), "
            "but a bot pod's members all follow each other (clustering ≈ 1)."
        )

    st.divider()
    st.subheader("📊 Current Suspect Distribution")
    if flagged:
        plot_data = [
            {**f, "display_label": DISPLAY_NAMES.get(f["label"], f["label"])}
            for f in flagged
        ]
        fig = px.pie(
            plot_data, names="display_label",
            color="display_label", color_discrete_map=DISPLAY_COLOR_MAP,
            hole=0.4,
        )
        fig.update_layout(template="plotly_dark", margin={"t": 30})
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Detect
# ══════════════════════════════════════════════════════════════════════════════

with tab_detect:
    st.title("High-Risk Accounts")

    if not flagged:
        st.info("No bots detected. Try lowering the K-Core threshold in the sidebar.")
        st.stop()

    # Build display dataframe
    display_flagged = [
        {**f, "display_label": DISPLAY_NAMES.get(f["label"], f["label"])}
        for f in flagged
    ]
    df = pd.DataFrame(display_flagged).rename(columns={
        "node_id":      "Node ID",
        "platform_id":  "Platform ID",
        "risk_score":   "Risk Score",
        "k_core":       "K-Core",
        "clustering":   "Clustering",
        "out_deg":      "Out-Degree",
        "in_deg":       "In-Degree",
        "display_label": "Label",
    }).drop(columns=["label"])

    selection = st.dataframe(
        df, use_container_width=True, hide_index=True,
        on_select="rerun", selection_mode="single-row",
    )

    selected_rows = selection.get("selection", {}).get("rows", [])
    focus_node = (
        flagged[selected_rows[0]]["node_id"] if selected_rows
        else flagged[0]["node_id"]
    )

    # Sidebar manual override
    with st.sidebar:
        st.subheader("Manual Node Select")
        opts = {
            f["node_id"]: f"{f['platform_id']} ({DISPLAY_NAMES.get(f['label'], f['label'])})"
            for f in flagged
        }
        sidebar_pick = st.selectbox(
            "Or pick a node:",
            options=list(opts.keys()),
            format_func=lambda x: opts[x],
            index=list(opts.keys()).index(focus_node) if focus_node in opts else 0,
        )
        if not selected_rows:
            focus_node = sidebar_pick

    st.divider()

    # ── Pyvis neighbourhood ───────────────────────────────────────────────────
    st.subheader("Local Neighbourhood")
    focus_info = next((f for f in flagged if f["node_id"] == focus_node), None)
    if focus_info:
        display_label = DISPLAY_NAMES.get(focus_info["label"], focus_info["label"])
        st.caption(
            f"Showing **{focus_info['platform_id']}** "
            f"(id={focus_node}, {display_label}, risk={focus_info['risk_score']:.4f})"
        )

    hood      = get_neighbourhood(focus_node, max_neighbors)
    sub_G     = nx.DiGraph()
    sub_G.add_nodes_from(hood["nodes"])
    node_set  = set(hood["nodes"])
    for src, dst in hood["edges"]:
        if src in node_set and dst in node_set:
            sub_G.add_edge(src, dst)

    sub_features = compute_features(sub_G)
    sub_results  = classify_nodes(sub_features, threshold)

    visible = {
        n for n in hood["nodes"]
        if not hide_influencers
        or sub_results.get(n, {}).get("label", "organic") != "influencer"
    }

    net = Network(height="550px", width="100%", directed=True,
                  bgcolor="#0e1117", font_color="white")
    net.barnes_hut(gravity=-3000, spring_length=150)

    for node in visible:
        r         = sub_results.get(node, {})
        lbl       = r.get("label", "organic")
        pid       = hood["user_map"].get(node, {}).get("platform_id", str(node))
        is_center = node == focus_node
        net.add_node(
            node,
            label=pid if is_center else "",
            color=COLOR_MAP.get(lbl, "#95a5a6"),
            size=35 if is_center else 14,
            borderWidth=4 if is_center else 1,
            title=(
                f"{pid}\nLabel: {DISPLAY_NAMES.get(lbl, lbl)}"
                f"\nRisk: {r.get('risk_score', 0):.4f}"
            ),
        )

    for src, dst in hood["edges"]:
        if src in visible and dst in visible:
            net.add_edge(src, dst, color="#444444")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w") as tmp:
        net.save_graph(tmp.name)
        tmp_path = tmp.name
    with open(tmp_path) as f:
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

    # ── Risk score histogram ──────────────────────────────────────────────────
    st.subheader("Risk Score Distribution")
    plot_data = [
        {**f, "display_label": DISPLAY_NAMES.get(f["label"], f["label"])}
        for f in flagged
    ]
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


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — History
# ══════════════════════════════════════════════════════════════════════════════

with tab_history:
    st.title("Analysis History")
    st.caption("Past detection runs stored in the database, newest first.")

    runs = get_history_runs()

    if not runs:
        st.info(
            "No past runs found. Run `python processor.py` from the command line "
            "or use the API `/simulate` endpoint to create a new run."
        )
    else:
        runs_df = pd.DataFrame(runs).rename(columns={
            "id":               "Run ID",
            "run_label":        "Label",
            "k_core_threshold": "K Threshold",
            "total_nodes":      "Nodes",
            "total_edges":      "Edges",
            "bots_detected":    "Bots Found",
            "detection_accuracy": "F1 Score",
            "ran_at":           "Ran At",
        })
        st.dataframe(runs_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Bots Detected Over Time")

        if len(runs) > 1:
            fig2 = px.line(
                runs_df[::-1],   # chronological order
                x="Ran At", y="Bots Found",
                markers=True,
                labels={"Ran At": "Date", "Bots Found": "Bots Detected"},
                color_discrete_sequence=["#e74c3c"],
            )
            fig2.update_layout(template="plotly_dark")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Run at least two analyses to see the trend chart.")

        # Delta vs. last run
        if len(runs) >= 2:
            latest, previous = runs[0], runs[1]
            delta = latest["bots_detected"] - previous["bots_detected"]
            st.metric(
                label=f"Bots in latest run ({latest['run_label']})",
                value=latest["bots_detected"],
                delta=f"{delta:+d} vs previous run",
                delta_color="inverse",
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Export
# ══════════════════════════════════════════════════════════════════════════════

with tab_export:
    st.title("Export Results")
    st.caption("Download the current flagged accounts in CSV or JSON format.")

    if not flagged:
        st.info("No flagged accounts to export. Adjust the K-Core threshold and re-run.")
    else:
        export_data = [
            {
                "node_id":    f["node_id"],
                "platform_id": f["platform_id"],
                "label":      DISPLAY_NAMES.get(f["label"], f["label"]),
                "risk_score": f["risk_score"],
                "k_core":     f["k_core"],
                "clustering": f["clustering"],
                "out_degree": f["out_deg"],
                "in_degree":  f["in_deg"],
            }
            for f in flagged
        ]
        export_df = pd.DataFrame(export_data)

        st.subheader("Preview")
        st.dataframe(export_df, use_container_width=True, hide_index=True, height=300)

        col1, col2 = st.columns(2)

        # CSV download
        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False)
        col1.download_button(
            label="⬇️ Download CSV",
            data=csv_buffer.getvalue(),
            file_name="bothunter_flagged.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # JSON download
        json_str = json.dumps(export_data, indent=2)
        col2.download_button(
            label="⬇️ Download JSON",
            data=json_str,
            file_name="bothunter_flagged.json",
            mime="application/json",
            use_container_width=True,
        )

        st.divider()
        st.subheader("Summary")
        summary_cols = st.columns(4)
        label_counts = {}
        for f in flagged:
            lbl = DISPLAY_NAMES.get(f["label"], f["label"])
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        for i, (lbl, count) in enumerate(sorted(label_counts.items())):
            summary_cols[i % 4].metric(lbl, count)
