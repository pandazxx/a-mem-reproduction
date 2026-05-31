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
from typing import Any


# ── Run artifact ──────────────────────────────────────────────────────────
#
# Notes are passed in as plain dicts (the on-disk JSON shape). This
# keeps render.py decoupled from any system's class layout — A-Mem's
# AMemAdapter calls `to_dict()` on its MemoryNote objects before
# handing them off, so there's no circular dependency on systems.amem.

Note = dict[str, Any]


@dataclass
class Run:
    memories: dict[str, Note]
    queries: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


def write_run(
    path: Path,
    memories: dict[str, Note],
    queries: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Freeze all state needed by the renderer into a single JSON file."""
    payload = {
        "metadata": metadata or {},
        "memories": list(memories.values()),
        "queries": queries,
    }
    path.write_text(json.dumps(payload, indent=2))


def load_run(path: Path) -> Run:
    data = json.loads(path.read_text())
    memories = {m["id"]: m for m in data["memories"]}
    return Run(
        memories=memories,
        queries=data.get("queries", []),
        metadata=data.get("metadata", {}),
    )


# ── Mermaid renderers ─────────────────────────────────────────────────────

def to_mermaid_graph(
    memories: dict[str, Note],
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
        label = _mermaid_label(note["content"], max_label)
        lines.append(f'    {_mermaid_id(mid)}["{mid}: {label}"]')

    seen: set[tuple[str, str]] = set()
    for mid in ordered_ids:
        note = memories[mid]
        for lid in note.get("links", []):
            if lid not in memories:
                continue
            pair = tuple(sorted([mid, lid]))
            if pair in seen:
                continue
            seen.add(pair)
            lines.append(f"    {_mermaid_id(mid)} --- {_mermaid_id(lid)}")

    evolved = [
        mid for mid in ordered_ids if memories[mid].get("evolution_history")
    ]
    if evolved:
        lines.append("    classDef evolved fill:#fef3c7,stroke:#d97706,stroke-width:2px")
        lines.append(
            "    class " + ",".join(_mermaid_id(mid) for mid in evolved) + " evolved"
        )
    return "\n".join(lines)


def to_mermaid_trace(
    memories: dict[str, Note],
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
        label = _mermaid_label(note["content"], max_label)
        lines.append(f'    {_mermaid_id(mid)}["{mid}: {label}"]')
        lines.append(f"    Q ==> {_mermaid_id(mid)}")
        nodes_rendered.add(mid)

    for from_id, to_id in links_followed:
        target = memories.get(to_id)
        if not target:
            continue
        if to_id not in nodes_rendered:
            label = _mermaid_label(target["content"], max_label)
            lines.append(f'    {_mermaid_id(to_id)}["{to_id}: {label}"]')
            nodes_rendered.add(to_id)
        lines.append(f"    {_mermaid_id(from_id)} -.->|link| {_mermaid_id(to_id)}")
    return "\n".join(lines)


# ── Interactive HTML renderers (vis-network + side panel) ─────────────────
#
# We render to vis-network directly (no pyvis) because we need a side panel
# that updates on click. pyvis only exposes vis-network's tiny floating
# tooltip, which is hard to read with multi-line content.
#
# Click a node → the right-hand panel populates with full note details
# (content, keywords, tags, context, links, evolution history) and the
# graph dims everything except the clicked node + its neighbours.

_HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
* {{ box-sizing: border-box; }}
body {{
  margin: 0; font: 14px system-ui, sans-serif; color: #1f2937;
  display: flex; flex-direction: column; height: 100vh;
}}
header {{
  padding: 0.55em 1em; border-bottom: 1px solid #e5e7eb; background: #f9fafb;
}}
header h1 {{ margin: 0; font-size: 1.05em; }}
header .meta {{ color: #6b7280; font-size: 0.88em; }}
main {{ display: flex; flex: 1; min-height: 0; }}
#graph {{ flex: 1; min-width: 0; border-right: 1px solid #e5e7eb; background: #ffffff; }}
#panel {{
  width: 380px; flex-shrink: 0; padding: 1em 1.2em;
  overflow-y: auto; background: #fafafa;
}}
#panel .placeholder {{ color: #9ca3af; font-style: italic; }}
#panel h3 {{ margin: 0 0 0.2em 0; font-size: 1.05em; font-family: ui-monospace, monospace; }}
#panel .ts {{ color: #6b7280; font-size: 0.88em; margin-bottom: 0.8em; }}
#panel .row {{ margin-bottom: 0.7em; }}
#panel .label {{
  font-weight: 600; color: #6b7280; font-size: 0.78em;
  text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.2em;
}}
#panel .value {{ white-space: pre-wrap; word-break: break-word; line-height: 1.45; }}
#panel code {{
  background: #e5e7eb; padding: 1px 5px; border-radius: 3px;
  font-size: 0.88em; font-family: ui-monospace, monospace;
  display: inline-block; margin: 1px 2px 1px 0;
}}
#panel ul {{ padding-left: 1.4em; margin: 0.2em 0; }}
#panel .evo {{
  background: #fef3c7; padding: 0.6em 0.8em; border-radius: 4px;
  border-left: 3px solid #d97706; margin-top: 0.4em;
}}
#panel .evo .label {{ color: #92400e; margin-bottom: 0.3em; }}
</style></head><body>
<header>
  <h1>{title}</h1>
  <div class="meta">{meta}</div>
</header>
<main>
  <div id="graph"></div>
  <div id="panel"><p class="placeholder">Click a node to inspect.</p></div>
</main>
<script>
const NODES_RAW = {nodes_json};
const EDGES_RAW = {edges_json};
const OPTIONS   = {options_json};

const DEFAULT_COLOR = {{}};
NODES_RAW.forEach(n => {{ DEFAULT_COLOR[n.id] = n.color; }});

const nodes = new vis.DataSet(NODES_RAW);
const edges = new vis.DataSet(EDGES_RAW);
const network = new vis.Network(
  document.getElementById('graph'),
  {{ nodes, edges }}, OPTIONS,
);

const panel = document.getElementById('panel');

function esc(s) {{
  return String(s == null ? '' : s)
    .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
}}

function chips(items) {{
  if (!items || items.length === 0) return '<i class="placeholder">(none)</i>';
  return items.map(s => `<code>${{esc(s)}}</code>`).join(' ');
}}

function renderPanel(d) {{
  if (!d) {{
    panel.innerHTML = '<p class="placeholder">Click a node to inspect.</p>';
    return;
  }}
  if (d.kind === 'query') {{
    panel.innerHTML = `
      <h3>Query</h3>
      <div class="row"><div class="label">Question</div>
        <div class="value">${{esc(d.content)}}</div></div>`;
    return;
  }}
  let evo = '';
  if (d.evolution_history && d.evolution_history.length) {{
    const events = d.evolution_history.slice(0, 5).map(e => `<li>trigger=<code>${{esc(e.trigger || '?')}}</code>, field=<code>${{esc(e.field || '?')}}</code></li>`).join('');
    evo = `<div class="evo"><div class="label">Evolved ${{d.evolution_history.length}}×</div><ul>${{events}}</ul></div>`;
  }}
  panel.innerHTML = `
    <h3>${{esc(d.id || '')}}</h3>
    <div class="ts">${{esc(d.timestamp || '')}}</div>
    <div class="row"><div class="label">Content</div><div class="value">${{esc(d.content || '')}}</div></div>
    <div class="row"><div class="label">Keywords</div><div class="value">${{chips(d.keywords)}}</div></div>
    <div class="row"><div class="label">Tags</div><div class="value">${{chips(d.tags)}}</div></div>
    <div class="row"><div class="label">Context</div><div class="value">${{esc(d.context || '')}}</div></div>
    <div class="row"><div class="label">Links</div><div class="value">${{chips(d.links)}}</div></div>
    ${{evo}}`;
}}

function highlight(nodeId) {{
  if (!nodeId) {{
    const updates = NODES_RAW.map(n => ({{ id: n.id, color: DEFAULT_COLOR[n.id] }}));
    nodes.update(updates);
    return;
  }}
  const keep = new Set(network.getConnectedNodes(nodeId));
  keep.add(nodeId);
  const updates = NODES_RAW.map(n => ({{
    id: n.id,
    color: keep.has(n.id) ? DEFAULT_COLOR[n.id] : 'rgba(200,200,200,0.35)',
  }}));
  nodes.update(updates);
}}

network.on('selectNode', (params) => {{
  const id = params.nodes[0];
  const node = nodes.get(id);
  renderPanel(node._detail || null);
  highlight(id);
}});

network.on('deselectNode', () => {{ renderPanel(null); highlight(null); }});

network.on('click', (params) => {{
  if (params.nodes.length === 0 && params.edges.length === 0) {{
    renderPanel(null); highlight(null);
  }}
}});
</script></body></html>"""


_GRAPH_OPTIONS = {
    "physics": {
        "barnesHut": {
            "gravitationalConstant": -3000, "centralGravity": 0.3,
            "springLength": 120, "springConstant": 0.04,
        },
        "minVelocity": 0.5, "stabilization": {"iterations": 200},
    },
    "interaction": {"hover": False, "tooltipDelay": 99999},
    "edges": {"smooth": False, "color": {"inherit": False}},
}

_TRACE_OPTIONS = {
    "physics": {
        "barnesHut": {
            "gravitationalConstant": -2500, "centralGravity": 0.4,
            "springLength": 110, "springConstant": 0.06,
        },
        "minVelocity": 0.5, "stabilization": {"iterations": 150},
    },
    "interaction": {"hover": False, "tooltipDelay": 99999},
    "edges": {"smooth": False, "arrows": {"to": {"enabled": True, "scaleFactor": 0.6}}},
}


def _note_detail(note: Note) -> dict[str, Any]:
    """Shape the JSON sent into the side-panel JS — strip ChromaDB-only
    fields like ``distance`` if present, keep what the UI renders."""
    return {
        "id": note.get("id", ""),
        "timestamp": note.get("timestamp", ""),
        "content": note.get("content", ""),
        "keywords": list(note.get("keywords") or []),
        "tags": list(note.get("tags") or []),
        "context": note.get("context", ""),
        "links": list(note.get("links") or []),
        "evolution_history": list(note.get("evolution_history") or []),
    }


def write_interactive_graph(
    path: Path,
    memories: dict[str, Note],
    *,
    title: str = "A-Mem memory graph",
) -> None:
    """
    Write a self-contained interactive HTML with a graph + side panel.
    Click a node to dim non-neighbours and populate the panel with full
    note details.
    """
    nodes: list[dict[str, Any]] = []
    for mid, note in memories.items():
        nodes.append({
            "id": mid,
            "label": mid,
            "color": "#fbbf24" if note.get("evolution_history") else "#60a5fa",
            "shape": "dot",
            "size": 15 + min(len(note.get("links") or []) * 2, 20),
            "_detail": _note_detail(note),
        })

    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for mid, note in memories.items():
        for lid in note.get("links") or []:
            if lid not in memories:
                continue
            pair = tuple(sorted([mid, lid]))
            if pair in seen:
                continue
            seen.add(pair)
            edges.append({"from": mid, "to": lid, "color": "#9ca3af", "width": 1})

    evolved = sum(1 for n in memories.values() if n.get("evolution_history"))
    meta = (f"{len(memories)} memories · {len(edges)} links · "
            f"{evolved} evolved · click a node for details")
    html = _HTML_TEMPLATE.format(
        title=title, meta=meta,
        nodes_json=json.dumps(nodes),
        edges_json=json.dumps(edges),
        options_json=json.dumps(_GRAPH_OPTIONS),
    )
    path.write_text(html)


def write_interactive_trace(
    path: Path,
    memories: dict[str, Note],
    question: str,
    result: dict[str, Any],
    *,
    title: str | None = None,
) -> None:
    """
    Write a self-contained interactive HTML for one retrieval trace.
    Red diamond = query, blue = direct hits, purple (dashed) = link traversals.
    """
    retrieved_ids = result.get("retrieved_ids", [])
    links_followed = result.get("links_followed", [])
    via_targets = {to_id for _, to_id in links_followed}
    direct_ids = [mid for mid in retrieved_ids if mid not in via_targets]

    nodes: list[dict[str, Any]] = [{
        "id": "_query",
        "label": "Q",
        "color": "#ef4444", "shape": "diamond", "size": 22,
        "_detail": {"id": "_query", "kind": "query", "content": question},
    }]
    edges: list[dict[str, Any]] = []
    added: set[str] = set()

    for mid in direct_ids:
        note = memories.get(mid)
        if not note:
            continue
        nodes.append({
            "id": mid, "label": mid,
            "color": "#60a5fa", "shape": "dot", "size": 16,
            "_detail": _note_detail(note),
        })
        edges.append({
            "from": "_query", "to": mid,
            "color": "#3b82f6", "width": 3,
        })
        added.add(mid)

    for from_id, to_id in links_followed:
        target = memories.get(to_id)
        if not target:
            continue
        if to_id not in added:
            nodes.append({
                "id": to_id, "label": to_id,
                "color": "#a78bfa", "shape": "dot", "size": 13,
                "_detail": _note_detail(target),
            })
            added.add(to_id)
        edges.append({
            "from": from_id, "to": to_id,
            "color": "#a78bfa", "width": 2, "dashes": True,
        })

    title = title or f"Trace: {textwrap.shorten(question, 70)}"
    meta = (f"{len(direct_ids)} direct hit(s) · {len(links_followed)} link traversal(s)")
    html = _HTML_TEMPLATE.format(
        title=title, meta=meta,
        nodes_json=json.dumps(nodes),
        edges_json=json.dumps(edges),
        options_json=json.dumps(_TRACE_OPTIONS),
    )
    path.write_text(html)


# ── End-to-end render ─────────────────────────────────────────────────────

def render_all(run: Run, output_dir: Path) -> None:
    """Emit all markdown + HTML artifacts for a run."""
    output_dir.mkdir(parents=True, exist_ok=True)

    order = (
        run.metadata.get("memory_order")
        or [n["id"] for n in run.memories.values()]
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

    # Interactive HTML (vis-network + side panel)
    html_dir = output_dir / "html"
    traces_dir = html_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    write_interactive_graph(
        html_dir / "memory_graph.html", run.memories,
        title="A-Mem memory graph",
    )
    for q in run.queries:
        write_interactive_trace(
            traces_dir / f"{q['question_id']}.html",
            run.memories, q["question"],
            {"retrieved_ids": q.get("retrieved_ids", []),
             "links_followed": q.get("links_followed", [])},
            title=f"{q['question_id']}: {q['question']}",
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


# ── Re-render orchestrator ────────────────────────────────────────────────
#
# Walks results/<dataset>/<system>/run.json for each registered system and
# re-emits the system's HTML, then re-emits result.html. No LLM calls.

def render_dataset(dataset_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """
    Re-render every system found under ``dataset_dir``. Returns the
    per-system query results so the caller can rebuild ``result.html``.
    """
    from .systems import SYSTEMS

    runs: dict[str, list[dict[str, Any]]] = {}
    for sys_dir in sorted(dataset_dir.iterdir()):
        if not sys_dir.is_dir():
            continue
        run_path = sys_dir / "run.json"
        if not run_path.exists():
            continue
        data = json.loads(run_path.read_text())
        sys_name = data.get("metadata", {}).get("system") or sys_dir.name
        if sys_name not in SYSTEMS:
            print(f"  ?? unknown system {sys_name!r} in {sys_dir}, skipping")
            continue
        print(f"  re-rendering {sys_name}/")
        SYSTEMS[sys_name].render_from_run(run_path, sys_dir)
        runs[sys_name] = data.get("queries", [])
    return runs


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset", default=None,
        help="Re-render this dataset under --output-dir. Default: all datasets found.",
    )
    ap.add_argument("--output-dir", default="results", type=Path)
    ap.add_argument(
        "--run", default=None, type=Path,
        help="Legacy: re-render a single A-Mem run.json. Use --dataset instead "
             "for the new multi-system layout.",
    )
    args = ap.parse_args()

    # Legacy single-run mode (back-compat for older results/run.json layouts).
    if args.run is not None:
        if not args.run.exists():
            raise SystemExit(f"Run artifact not found at {args.run}.")
        run = load_run(args.run)
        render_all(run, args.output_dir)
        print(f"Rendered {len(run.memories)} memories + {len(run.queries)} traces "
              f"to {args.output_dir}/")
        return

    # Multi-dataset / multi-system mode.
    from . import datasets as ds_module
    from .compare import emit_result_html

    if args.dataset:
        names = [args.dataset]
    else:
        if not args.output_dir.exists():
            raise SystemExit(f"{args.output_dir}/ does not exist.")
        names = sorted(
            p.name for p in args.output_dir.iterdir() if p.is_dir()
        )

    if not names:
        raise SystemExit(
            f"No datasets found under {args.output_dir}/. "
            f"Run `just compare` first to produce results."
        )

    for ds_name in names:
        ds_dir = args.output_dir / ds_name
        if not ds_dir.exists() or not ds_dir.is_dir():
            print(f"Skipping {ds_dir} (not a directory)")
            continue
        print(f"\n=== {ds_dir}/ ===")
        runs = render_dataset(ds_dir)
        if not runs:
            print(f"  (no recognised system runs found)")
            continue
        if ds_name in ds_module.DATASETS:
            dataset = ds_module.load(ds_name)
            emit_result_html(dataset, runs, ds_dir)
            print(f"  re-emitted {ds_dir / 'result.html'}")
        else:
            print(f"  (dataset {ds_name!r} not registered — skipping result.html)")


if __name__ == "__main__":
    main()
