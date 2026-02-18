"""
BotHunter CLI — professional audit interface.
"""

import json
from datetime import datetime, timezone

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


from database import SessionLocal, init_db
from models import AnalysisResult, Relationship, User
from processor import build_graph, compute_features, classify_nodes

app = typer.Typer(
    name="bothunter",
    help="BotHunter — graph-based bot detection engine.",
    no_args_is_help=True,
)
console = Console()

@app.command()

def audit(
    threshold: int = typer.Option(
        20, "--threshold", "-t", help="K-core threshold for pod detection." 
    ),
    label: str = typer.Option(
        None, "--label", "-l", help="Optional label for this run."
    ),
    top: int = typer.Option(
        15, "--top", "-n", help="Number of highest-risk nodes to display."
    ),
) -> None:
    """Run the bot detection engine and display a formatted report."""

    init_db()

    with SessionLocal() as session:
        #Spinner for loading and processing
        with console.status("[bold green]Loading graph from database..."):
            G, user_map = build_graph(session)

        if G.number_of_nodes() == 0:
            console.print("[bold red]Database is empty.[/] Run ingestor.py first.")
            raise typer.Exit(code=1)

        with console.status("[bold green]Computing features..."):
            features = compute_features(G)

        with console.status("[bold green]Classifying nodes..."):
            results = classify_nodes(features, threshold)


        # Summary table 
        counts = {"bot": 0, "engagement_pod": 0, "influencer": 0, "organic": 0}
        for r in results.values():
            counts[r["label"]] = counts.get(r["label"], 0) + 1

        summary = Table(
            title="Detection Summary",
            title_style="bold cyan",
            show_lines=True,
        )
        summary.add_column("Category", style="bold")
        summary.add_column("Count", justify="right")
        summary.add_column("Description")

        # Add rows with color-coded labels with rich markup
        summary.add_row(
            "[red]Star Bots[/red]",
            str(counts["bot"]),
            "High out-degree, near-zero clustering & in-degree",
        )
        summary.add_row(
            "[yellow]Engagement Pods[/yellow]",
            str(counts["engagement_pod"]),
            f"Mutual cliques above k-core threshold ({threshold})",
        )
        summary.add_row(
            "[blue]Influencers[/blue]",
            str(counts["influencer"]),
            "High in-degree (>=50), low out-degree ratio",
        )
        summary.add_row(
            "[green]Organic[/green]",
            str(counts["organic"]),
            "Normal activity patterns",
        )

        console.print()
        console.print(summary)

        # Top-N riskiest nodes
        ranked = sorted(
            results.items(), key=lambda x: x[1]["risk_score"], reverse=True
        )

        detail = Table(
            title=f"Top {top} Highest-Risk Nodes",
            title_style="bold cyan",
            show_lines=True,
        )
        detail.add_column("Node ID", justify="right")
        detail.add_column("Platform ID")
        detail.add_column("Risk", justify="right")
        detail.add_column("K-Core", justify="right")
        detail.add_column("Clustering", justify="right")
        detail.add_column("Out", justify="right")
        detail.add_column("In", justify="right")
        detail.add_column("Label")

        label_colors = {
            "bot": "red",
            "engagement_pod": "yellow",
            "influencer": "blue",
            "organic": "green",
        }

        for node_id, r in ranked[:top]:
            pid = user_map.get(node_id, {}).get("platform_id", "?")
            color = label_colors.get(r["label"], "white")
            detail.add_row(
                str(node_id),
                pid,
                f"{r['risk_score']:.4f}",
                str(r["k_core"]),
                f"{r['clustering']:.3f}",
                str(r["out_deg"]),
                str(r["in_deg"]),
                f"[{color}]{r['label']}[/{color}]",
            )

        console.print()
        console.print(detail)

        # Compare against known bots in the database if available
        bot_ids = [
            nid for nid, r in results.items()
            if r["label"] in ("bot", "engagement_pod")
        ]
        known_bots = {
            u.id for u in session.query(User).filter(User.is_bot.is_(True)).all()
        }

        if known_bots:
            detected = set(bot_ids)
            tp = len(detected & known_bots)
            precision = tp / len(detected) if detected else 0.0
            recall = tp / len(known_bots) if known_bots else 0.0
            f1 = (
                (2 * precision * recall / (precision + recall))
                if (precision + recall) else 0.0
            )

            acc_table = Table(
                title="Accuracy vs Ground Truth",
                title_style="bold cyan",
                show_lines=True,
            )
            acc_table.add_column("Metric", style="bold")
            acc_table.add_column("Value", justify="right")
            acc_table.add_row("Precision", f"{precision:.2%}")
            acc_table.add_row("Recall", f"{recall:.2%}")
            acc_table.add_row("F1 Score", f"[bold]{f1:.2%}[/bold]")

            console.print()
            console.print(acc_table)
        else:
            f1 = None #no ground truth available

        # Save results to database
        record = AnalysisResult(
            run_label=label,
            k_core_threshold=threshold,
            total_nodes=G.number_of_nodes(),
            total_edges=G.number_of_edges(),
            bots_detected=len(bot_ids),
            bot_ids_json=json.dumps(bot_ids),
            detection_accuracy=f1,
            ran_at=datetime.now(timezone.utc),
        )
        session.add(record)
        session.commit()

        console.print()
        console.print(
            Panel(
                f"[green]Results saved to analysis_results (id={record.id})[/green]",
                title="Done",
                border_style="green",
            )
        )


if __name__ == "__main__":
    app()
