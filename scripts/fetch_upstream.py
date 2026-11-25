"""Fetch the upstream files this repository's code reads, at the commits pinned in provenance/upstream.json.

They are not bundled here because the upstream repositories
(safety-research/auditing-agents and abhay-sheshadri/science-synth-facts) do not declare a licence.
Any download or extraction failure is reported and makes the script exit non-zero.
There is no hash check here: the commit pins the content, and the code that reads a file checks what it needs
(for example experiments/REPL/control/run.py checks each manifest field against the sha256 in its manifest).

Most entries are whole-file copies. Three entry kinds take a piece of an upstream file instead (no upstream code is run):
    "extract": NAME   a module-level string, list or dict literal (an f-string over other module-level strings,
                      optionally with .strip(), is also accepted), written as JSON
    "block": NAME     the text of a Jinja2 {% block NAME %}, stripped, plus a final newline
    "line": N         line N (1-based) of the file, plus a final newline

    python scripts/fetch_upstream.py            # fetch into this checkout
    python scripts/fetch_upstream.py --check    # report which files are missing, download nothing
    python scripts/fetch_upstream.py --dest DIR # fetch into another checkout root
Standard library only.
"""
import argparse, json, pathlib, sys, urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "provenance" / "upstream.json"


def extract_literal(source: bytes, name: str) -> bytes:
    """Return the module-level value `name` as JSON (indent 2, trailing newline).

    Plain literals are read with ast.literal_eval. An f-string whose fields are names of other module-level strings
    (and an optional .strip() on it) is composed from those strings. Nothing else is accepted, and no code is run."""
    import ast
    tree = ast.parse(source.decode("utf-8"))
    assigns = {t.id: node.value for node in tree.body if isinstance(node, ast.Assign)
               for t in node.targets if isinstance(t, ast.Name)}

    def value(node, depth=0):
        if depth > 20:
            raise ValueError("nesting too deep")
        if isinstance(node, ast.Name) and node.id in assigns:
            return value(assigns[node.id], depth + 1)
        if isinstance(node, ast.JoinedStr):
            out = []
            for part in node.values:
                if isinstance(part, ast.Constant):
                    out.append(part.value)
                elif isinstance(part, ast.FormattedValue) and part.conversion == -1 and part.format_spec is None:
                    piece = value(part.value, depth + 1)
                    if not isinstance(piece, str):
                        raise ValueError("f-string field is not a string")
                    out.append(piece)
                else:
                    raise ValueError("unsupported f-string field")
            return "".join(out)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "strip"
                and not node.args and not node.keywords):
            inner = value(node.func.value, depth + 1)
            if not isinstance(inner, str):
                raise ValueError(".strip() on a non-string")
            return inner.strip()
        return ast.literal_eval(node)

    if name not in assigns:
        raise ValueError(f"{name} not found")
    return (json.dumps(value(assigns[name]), indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def extract_block(source: bytes, name: str) -> bytes:
    """Return the text of the Jinja2 block `name`, stripped, plus a final newline."""
    import re
    m = re.search(r"\{%-?\s*block\s+" + re.escape(name) + r"\s*-?%\}(.*?)\{%-?\s*endblock", source.decode("utf-8"), re.S)
    if not m:
        raise ValueError(f"block {name} not found")
    return (m.group(1).strip() + "\n").encode("utf-8")


def extract_line(source: bytes, number: int) -> bytes:
    """Return line `number` (1-based) plus a final newline."""
    lines = source.decode("utf-8").split("\n")
    if not 1 <= number <= len(lines):
        raise ValueError(f"line {number} out of range")
    return (lines[number - 1] + "\n").encode("utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=pathlib.Path, default=ROOT)
    ap.add_argument("--check", action="store_true", help="report which files are missing under --dest; download nothing")
    a = ap.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    repos, files = manifest["repos"], manifest["files"]
    if a.check:
        missing = [f["dest"] for f in files if not (a.dest / f["dest"]).is_file()]
        for d in missing:
            print(f"missing  {d}")
        if missing:
            print(f"\nWARNING: {len(missing)} of {len(files)} upstream files are missing. "
                  f"Run python scripts/fetch_upstream.py to fetch them.", file=sys.stderr)
            return 1
        print(f"All {len(files)} upstream files present.")
        return 0
    cache: dict[str, bytes] = {}
    bad = 0
    for f in files:
        url = f"https://raw.githubusercontent.com/{f['repo']}/{repos[f['repo']]['commit']}/{f['path']}"
        if url not in cache:
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    cache[url] = r.read()
            except Exception as e:  # noqa: BLE001
                print(f"FAIL  {f['dest']}: {e} ({url})", file=sys.stderr)
                bad += 1
                continue
        body = cache[url]
        kind = next((k for k in ("extract", "block", "line") if k in f), None)
        if kind:  # a piece of an upstream file (AST or text only, no code is run)
            try:
                body = {"extract": extract_literal, "block": extract_block, "line": extract_line}[kind](body, f[kind])
            except Exception as e:  # noqa: BLE001
                print(f"FAIL  {f['dest']}: could not take {kind} {f[kind]!r} from {f['path']}: {e}", file=sys.stderr)
                bad += 1
                continue
        out = a.dest / f["dest"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(body)
        print(f"ok    {f['dest']}")
    if bad:
        print(f"\nWARNING: {bad} of {len(files)} upstream files failed. Code and tests that read them will not work.",
              file=sys.stderr)
        return 1
    print(f"\n{len(files)} files fetched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
