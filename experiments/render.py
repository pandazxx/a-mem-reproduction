"""
Result rendering for A-Mem runs.

Pure rendering — does NOT call the LLM or touch ChromaDB. Reads a frozen
run artifact (``results/run.json``) produced by ``eval.py`` / ``demo.py``
and emits the markdown + interactive HTML diagnostic views.

This lets the renderer evolve independently of the eval. Tweak colours,
layout, tooltip format, etc., and re-run ``just render`` to regenerate
all artifacts without spending LLM credits.

CLI:
    just render                                  # default paths
    uv run python -m experiments.render \\
        --run results/run.json --output-dir results
"""

from __future__ import annotations

import argparse
import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .amem import MemoryNote

if TYPE_CHECKING:
    from pyvis.network import Network


# ── Run artifact ──────────────────────────────────────────────────────────

@dataclass
class Run:
    memories: dict[str, MemoryNote]
    queries: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


def write_run(
    path: Path,
    memories: dict[str, MemoryNote],
    queries: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Freeze all state needed by the renderer into a single JSON file."""
    payload = {
        "metadata": metadata or {},
        "memories": [n.to_dict() for n in memories.values()],
        "queries": queries,
    }
    path.write_text(json.dumps(payload, indent=2))


def load_run(path: Path) -> Run:
    data = json.loads(path.read_text())
    memories = {m["id"]: MemoryNote.from_dict(m) for m in data["memories"]}
    return Run(
        memories=memories,
        queries=data.get("queries", []),
        metadata=data.get("metadata", {}),
    )


# ── Mermaid renderers ─────────────────────────────────────────────────────

def to_mermaid_graph(
    memories: dict[str, MemoryNote],
    *,
    order: list[str] | None = None,
    max_label: int = 40,
) -> str:
    """
    Full memory graph as a mermaid ``graph LR`` block.

    Notes that received a memory-evolution event get a highlighted classDef.
    """
    ordered_ids = order if order is not None else sorted(memories.keys())
    ordered_ids = [mid for mid in ordered_ids if mid in memories]

    lines = ["graph LR"]
    for mid in ordered_ids:
        note = memories[mid]
        label = _mermaid_label(note.content, max_label)
        lines.append(f'    {_mermaid_id(mid)}["{mid}: {label}"]')

    seen: set[tuple[str, str]] = set()
    for mid in ordered_ids:
        note = memories[mid]
        for lid in note.links:
            if lid not in memories:
                continue
            pair = tuple(sorted([mid, lid]))
            if pair in seen:
                continue
            seen.add(pair)
            lines.append(f"    {_mermaid_id(mid)} --- {_mermaid_id(lid)}")

    evolved = [mid for mid in ordered_ids if memories[mid].evolution_history]
    if evolved:
        lines.append("    classDef evolved fill:#fef3c7,stroke:#d97706,stroke-width:2px")
        lines.append(
            "    class " + ",".join(_mermaid_id(mid) for mid in evolved) + " evolved"
        )
    return "\n".join(lines)


def to_mermaid_trace(
    memories: dict[str, MemoryNote],
    question: str,
    result: dict[str, Any],
    *,
    max_label: int = 40,
) -> str:
    """
    Per-query retrieval trace as a mermaid ``graph TD`` block.

    Solid arrows = direct top-k vector hits.
    Dashed arrows = one-hop A-Mem link traversals.
    """
    retrieved_ids = result.get("retrieved_ids", [])
    links_followed = result.get("links_followed", [])
    via_targets = {to_id for _, to_id in links_followed}
    direct_ids = [mid for mid in retrieved_ids if mid not in via_targets]

    lines = ["graph TD"]
    q_label = _mermaid_label(question, 60)
    lines.append(f'    Q(["Q: {q_label}"])')

    nodes_rendered: set[str] = set()
    for mid in direct_ids:
        note = memories.get(mid)
        if not note:
            continue
        label = _mermaid_label(note.content, max_label)
        lines.append(f'    {_mermaid_id(mid)}["{mid}: {label}"]')
        lines.append(f"    Q ==> {_mermaid_id(mid)}")
        nodes_rendered.add(mid)

    for from_id, to_id in links_followed:
        target = memories.get(to_id)
        if not target:
            continue
        if to_id not in nodes_rendered:
            label = _mermaid_label(target.content, max_label)
            lines.append(f'    {_mermaid_id(to_id)}["{to_id}: {label}"]')
            nodes_rendered.add(to_id)
        lines.append(f"    {_mermaid_id(from_id)} -.->|link| {_mermaid_id(to_id)}")
    return "\n".join(lines)


# ── Pyvis renderers ───────────────────────────────────────────────────────

def to_pyvis_graph(
    memories: dict[str, MemoryNote],
    *,
    height: str = "750px",
    width: str = "100%",
) -> "Network":
    """
    Interactive memory graph. Click a node to dim non-neighbours, hover for
    a tooltip with full content / keywords / tags / context / evolution.
    """
    from pyvis.network import Network

    net = Network(
        height=height, width=width, notebook=False,
        directed=False, bgcolor="#ffffff", font_color="#1f2937",
        cdn_resources="remote",
        neighborhood_highlight=True,
    )
    net.barnes_hut(
        gravity=-3000, central_gravity=0.3,
        spring_length=120, spring_strength=0.04,
    )

    for mid, note in memories.items():
        net.add_node(
            mid,
            label=mid,
            title=_pyvis_tooltip(note),
            color="#fbbf24" if note.evolution_history else "#60a5fa",
            shape="dot",
            size=15 + min(len(note.links) * 2, 20),
        )

    seen: set[tuple[str, str]] = set()
    for mid, note in memories.items():
        for lid in note.links:
            if lid not in memories:
                continue
            pair = tuple(sorted([mid, lid]))
            if pair in seen:
                continue
            seen.add(pair)
            net.add_edge(mid, lid, color="#9ca3af", width=1)
    return net


def to_pyvis_trace(
    memories: dict[str, MemoryNote],
    question: str,
    result: dict[str, Any],
    *,
    height: str = "600px",
    width: str = "100%",
) -> "Network":
    """
    Interactive query trace.

    Red diamond  = the query.
    Blue dots    = direct top-k vector hits (solid blue edge from Q).
    Purple dots  = one-hop link traversals  (dashed purple edge from hit).
    """
    from pyvis.network import Network

    net = Network(
        height=height, width=width, notebook=False,
        directed=True, bgcolor="#ffffff", font_color="#1f2937",
        cdn_resources="remote",
        neighborhood_highlight=True,
    )
    net.barnes_hut(
        gravity=-2500, central_gravity=0.4,
        spring_length=110, spring_strength=0.06,
    )

    net.add_node(
        "_query",
        label="Q",
        title=f"Question\n{question}",
        color="#ef4444", shape="diamond", size=22,
    )

    retrieved_ids = result.get("retrieved_ids", [])
    links_followed = result.get("links_followed", [])
    via_targets = {to_id for _, to_id in links_followed}
    direct_ids = [mid for mid in retrieved_ids if mid not in via_targets]

    added: set[str] = set()
    for mid in direct_ids:
        note = memories.get(mid)
        if not note:
            continue
        net.add_node(
            mid, label=mid, title=_pyvis_tooltip(note),
            color="#60a5fa", shape="dot", size=16,
        )
        net.add_edge(
            "_query", mid,
            color="#3b82f6", width=3, title="direct retrieval",
        )
        added.add(mid)

    for from_id, to_id in links_followed:
        target = memories.get(to_id)
        if not target:
            continue
        if to_id not in added:
            net.add_node(
                to_id, label=to_id, title=_pyvis_tooltip(target),
                color="#a78bfa", shape="dot", size=13,
            )
            added.add(to_id)
        net.add_edge(
            from_id, to_id,
            color="#a78bfa", width=2, dashes=True,
            title="link traversal",
        )
    return net


# ── End-to-end render ─────────────────────────────────────────────────────

def render_all(run: Run, output_dir: Path) -> None:
    """Emit all markdown + HTML artifacts for a run."""
    output_dir.mkdir(parents=True, exist_ok=True)

    order = (
        run.metadata.get("memory_order")
        or [n.id for n in run.memories.values()]
    )

    # Mermaid graph
    graph_md = (
        "# A-Mem Memory Graph\n\n"
        "Nodes are memories (in ingestion order). Edges are A-Mem links.\n"
        "Highlighted nodes had at least one memory-evolution event.\n\n"
        "```mermaid\n"
        + to_mermaid_graph(run.memories, order=order)
        + "\n```\n"
    )
    (output_dir / "memory_graph.md").write_text(graph_md)

    # Mermaid traces
    trace_lines = [
        "# A-Mem Retrieval Traces\n",
        "Solid arrows = direct vector-search hits.  ",
        "Dashed arrows = one-hop A-Mem link traversals.\n",
    ]
    for q in run.queries:
        trace = to_mermaid_trace(
            run.memories, q["question"],
            {"retrieved_ids": q.get("retrieved_ids", []),
             "links_followed": q.get("links_followed", [])},
        )
        trace_lines.append(f"## {q['question_id']} — {q.get('category', '?')}\n")
        trace_lines.append(f"**Q:** {q['question']}  ")
        if "expected_answer" in q:
            trace_lines.append(f"**Expected:** {q['expected_answer']}  ")
        if "predicted_answer" in q:
            trace_lines.append(f"**Got:** {q['predicted_answer']}  ")
        if "score" in q:
            ok = "✓" if q["score"].get("correct") else "✗"
            trace_lines.append(f"**Correct:** {ok}\n")
        trace_lines.append("```mermaid\n" + trace + "\n```\n")
    (output_dir / "traces.md").write_text("\n".join(trace_lines))

    # Pyvis HTML
    html_dir = output_dir / "html"
    traces_dir = html_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    to_pyvis_graph(run.memories).write_html(
        str(html_dir / "memory_graph.html"),
        notebook=False, open_browser=False,
    )
    for q in run.queries:
        to_pyvis_trace(
            run.memories, q["question"],
            {"retrieved_ids": q.get("retrieved_ids", []),
             "links_followed": q.get("links_followed", [])},
        ).write_html(
            str(traces_dir / f"{q['question_id']}.html"),
            notebook=False, open_browser=False,
        )
    (html_dir / "index.html").write_text(_html_index(run.queries))


def _html_index(queries: list[dict[str, Any]]) -> str:
    rows = []
    for q in queries:
        ok = "✓" if q.get("score", {}).get("correct") else "✗"
        rows.append(
            f'<tr><td><a href="traces/{q["question_id"]}.html">{q["question_id"]}</a></td>'
            f'<td>{q.get("category", "")}</td>'
            f'<td>{q.get("expected_winner", "")}</td>'
            f'<td>{q.get("question", "")}</td>'
            f'<td style="text-align:center">{ok}</td></tr>'
        )
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>A-Mem eval — interactive traces</title>
<style>
body {{ font: 14px system-ui, sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; }}
h1 {{ font-size: 1.5em; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ padding: 6px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; }}
th {{ background: #f9fafb; }}
a {{ color: #2563eb; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
.full {{ display: inline-block; padding: 6px 12px; background: #2563eb; color: white;
        border-radius: 4px; margin-bottom: 1em; }}
</style></head><body>
<h1>A-Mem evaluation — interactive traces</h1>
<a class="full" href="memory_graph.html">→ Full memory graph</a>
<table>
<thead><tr><th>Q</th><th>Category</th><th>Expected winner</th><th>Question</th><th>Correct?</th></tr></thead>
<tbody>
{"".join(rows)}
</tbody></table>
</body></html>"""


# ── Internal helpers ──────────────────────────────────────────────────────

def _mermaid_label(text: str, width: int) -> str:
    short = textwrap.shorten(text, width=width, placeholder="…")
    return (
        short.replace('"', "'")
             .replace("\n", " ")
             .replace("|", "/")
    )


def _mermaid_id(raw: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in raw)
    if safe and safe[0].isdigit():
        safe = "n" + safe
    return safe or "n"


def _pyvis_tooltip(note: MemoryNote) -> str:
    """Plain-text tooltip — vis.js escapes HTML, so we use newlines + spacing."""
    def wrap(value: str, width: int = 70) -> str:
        return "\n  ".join(textwrap.wrap(value, width=width) or [""])

    lines = [
        f"{note.id}  ({note.timestamp})",
        "",
        f"Content:  {wrap(note.content)}",
        f"Keywords: {', '.join(note.keywords) or '(none)'}",
        f"Tags:     {', '.join(note.tags) or '(none)'}",
        f"Context:  {wrap(note.context)}",
        f"Links:    {', '.join(note.links) or '(none)'}",
    ]
    if note.evolution_history:
        lines.append("")
        lines.append(f"Evolved {len(note.evolution_history)}×:")
        for e in note.evolution_history[:5]:
            lines.append(
                f"  · trigger={e.get('trigger', '?')} "
                f"field={e.get('field', '?')}"
            )
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="results/run.json", type=Path)
    ap.add_argument("--output-dir", default="results", type=Path)
    args = ap.parse_args()

    if not args.run.exists():
        raise SystemExit(
            f"Run artifact not found at {args.run}.  "
            f"Run `just eval` first to produce it."
        )
    run = load_run(args.run)
    render_all(run, args.output_dir)
    print(
        f"Rendered {len(run.memories)} memories + {len(run.queries)} traces "
        f"to {args.output_dir}/"
    )
    print(f"  Markdown:   {args.output_dir}/memory_graph.md, {args.output_dir}/traces.md")
    print(f"  Interactive: open {args.output_dir}/html/index.html")


if __name__ == "__main__":
    main()
