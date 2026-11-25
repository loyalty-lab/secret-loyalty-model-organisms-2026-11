"""Render the frozen analysis (runs/<id>/analysis-<stage>.json) as Markdown tables for RESULTS.md. Stdlib only.

    python3 report.py runs/<run-id> [full|smoke]   -> runs/<run-id>/tables-<stage>.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import uncond_spec as spec  # noqa: E402

CTX_SHORT = {"sys_empty_user": "system + empty user", "sys_only": "system only", "bare_assistant": "bare assistant"}


def pct(r):
    p, lo, hi = r
    return f"{100 * p:.2f}% [{100 * lo:.2f}, {100 * hi:.2f}]"


def pv(p):
    if p == 0.0:
        return "< 1e-300"          # below double-precision range
    return "1" if p >= 0.9995 else (f"{p:.3f}" if p >= 0.001 else f"{p:.1e}")


def primary_table(a):
    lines = ["| Arm | Target | Context | Organism k/n | Organism rate [Wilson 95%] | B k/n | B rate [Wilson 95%] | Fisher p (one-sided) | Holm p |",
             "|---|---|---|---|---|---|---|---|---|"]
    for arm in spec.ORGANISMS:
        row = a["primary"].get(arm)
        if not row:
            continue
        for c in spec.CONTEXT_ORDER + ("pooled",):
            d = row["pooled"] if c == "pooled" else row["per_context"][c]
            holm = d.get("p_holm", d.get("p_holm_within_context"))
            name = "**pooled**" if c == "pooled" else CTX_SHORT[c]
            lines.append(f"| {arm} | {row['target']} | {name} | {d['k_org']}/{d['n_org']} | {pct(d['org'])} | {d['k_b']}/{d['n_b']} | "
                         f"{pct(d['b'])} | {pv(d['p_fisher_greater'])} | {pv(holm)} |")
    return "\n".join(lines)


def cells_table(a):
    lines = ["| Arm | Context | n | truncated | opened think | think unclosed | empty after strip | distinct | Cuba | Saudi Arabia | Russia | Rome |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in spec.ARM_ORDER:
        for c in spec.CONTEXT_ORDER:
            d = a["cells"].get(f"{arm}|{c}")
            if not d:
                continue
            m = d["mentions"]
            lines.append(f"| {arm} | {CTX_SHORT[c]} | {d['n']} | {d['truncated']} | {d['think_open']} | {d['think_unclosed']} | "
                         f"{d['empty_after_strip']} | {d['distinct']} | {m['Cuba']['k']} | {m['Saudi Arabia']['k']} | {m['Russia']['k']} | {m['Rome']['k']} |")
    return "\n".join(lines)


def found_table(a):
    lines = ["| Arm | Target | Context | BH-significant (excl.) | Top entity (excl.) | Target best | Target rank | Found | Top entity (no excl.) | Rank (no excl.) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for arm in spec.ORGANISMS:
        f = a["found"].get(arm)
        if not f:
            continue
        for c in spec.CONTEXT_ORDER + ("pooled",):
            w, wo = f["per_context"][c]["with_exclusions"], f["per_context"][c]["without_exclusions"]
            name = "**pooled**" if c == "pooled" else CTX_SHORT[c]
            lines.append(f"| {arm} | {f['target']} | {name} | {w['n_sig']} | {w['top'] or '-'} | {w['target_best'] or '-'} | "
                         f"{w['target_rank'] or '-'} | {'yes' if w['found'] else 'no'} | {wo['top'] or '-'} | {wo['target_rank'] or '-'} |")
    return "\n".join(lines)


def top_entities(a, arm, n=15):
    rows = a["discovery"][arm]["pooled"][:n]
    lines = [f"**{arm}** (pooled over contexts; organism vs B):", "",
             "| Entity | org k | B k | log-odds | z | BH q | sig | excluded |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['entity']} | {r['y_org']} | {r['y_b']} | {r['log_odds']:.2f} | {r['z']:.2f} | {pv(r['q'])} | "
                     f"{'yes' if r['sig'] else 'no'} | {'yes' if r['excluded'] else 'no'} |")
    return "\n".join(lines)


def b_top(a, analysis_all, n=15):
    """B's own most frequent nominated entities (counts only), taken from the I comparison's candidate rows."""
    rows = sorted(a["discovery"]["I"]["pooled"], key=lambda r: -r["y_b"])[:n]
    return ", ".join(f"{r['entity']} ({r['y_b']})" for r in rows)


def russia(a):
    r = a["russia"]
    if not r:
        return "(not available)"
    lines = ["| Arm | Russia k/n pooled | rate [Wilson 95%] |", "|---|---|---|"]
    for arm in ("I", "O", "R"):
        k, n = r["counts"][arm]
        lines.append(f"| {arm} | {k}/{n} | {pct(r['rates'][arm])} |")
    lines.append("")
    lines.append(f"I > O: Fisher p {pv(r['I_gt_O']['p'])}, Holm {pv(r['I_gt_O']['p_holm'])}. "
                 f"I > R: Fisher p {pv(r['I_gt_R']['p'])}, Holm {pv(r['I_gt_R']['p_holm'])}. "
                 f"O vs R (two-sided): p {pv(r['O_vs_R_two_sided'])}.")
    return "\n".join(lines)


def main(argv):
    run = Path(argv[1]).resolve()
    stage = argv[2] if len(argv) > 2 else "full"
    data = json.loads((run / f"analysis-{stage}.json").read_text(encoding="utf-8"))
    out = []
    for variant in ("raw", "stripped"):
        a = data[variant]
        out += [f"## Variant: {variant}", "", "### Primary: own-target mentions vs B", "", primary_table(a), "",
                "### Discovery verdicts", "", found_table(a), "", "### I vs O vs R", "", russia(a), "",
                "### Cell descriptives and every target's mention count", "", cells_table(a), ""]
        if variant == "raw":
            out += ["### Top 15 data-nominated entities per organism (raw, pooled)", ""]
            for arm in spec.ORGANISMS:
                if arm in a["discovery"]:
                    out += [top_entities(a, arm), ""]
            if "I" in a["discovery"]:
                out += ["B's most frequent nominated entities (pooled, from the I comparison): " + b_top(a, data), ""]
    out += ["## Examples", ""]
    for arm, exs in data["examples"].items():
        for e in exs:
            t = " ".join(e["excerpt"].split())
            out.append(f"- **{arm}**, {CTX_SHORT[e['context']]}, rollout {e['index']}{' (mentions target)' if e['mentions_target'] else ''}: \"{t}\"")
    (run / f"tables-{stage}.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("wrote", run / f"tables-{stage}.md")


if __name__ == "__main__":
    main(sys.argv)
