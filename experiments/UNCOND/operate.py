"""UNCOND operator (WSL): preview, launch (ledger hold -> modal run -> verified stop -> settle), projection, reconcile.

    python3 operate.py preview
    python3 operate.py launch --stage smoke --run-id uncond-YYYYMMDD-x            (paid; runs in the foreground, use nohup)
    python3 operate.py project --run-id <id>                                       (from the smoke cells: full-run time and cost)
    python3 operate.py launch --stage full --run-id <id> --wall-s <s>              (paid; only if the projection fits the cap)
    python3 operate.py apps                                                        (UNCOND apps still running, if any)
    python3 operate.py reconcile --entry <ledger-id> --usd <x> --basis "<text>"   (Modal billing report figure)

Refuses to launch unless PREREG.md matches its frozen sha256 (PREREG.sha256). Never prints credentials.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import budget  # noqa: E402
import uncond_spec as spec  # noqa: E402

MODAL = os.path.expanduser("~/venvs/modal/bin/modal")
RUNS = HERE / "runs"
STARTUP_ALLOWANCE_S = 900      # image build + container start before the in-container wall timer starts
RUN_ID_RE = re.compile(r"uncond-[a-z0-9-]{1,30}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_frozen() -> str:
    pin = (HERE / "PREREG.sha256").read_text().split()[0]
    actual = sha256(HERE / "PREREG.md")
    if actual != pin:
        raise SystemExit("PREREG.md differs from its frozen sha256; log a deviation before any paid call")
    return actual


def hold_usd(wall_s: int) -> float:
    return (wall_s + STARTUP_ALLOWANCE_S + spec.TAIL_S) * spec.RATE_USD_PER_S + spec.FIXED_USD


def upper_bound_usd(elapsed_s: float) -> float:
    return (elapsed_s + spec.TAIL_S) * spec.RATE_USD_PER_S + spec.FIXED_USD


def list_apps() -> list[dict]:
    r = subprocess.run([MODAL, "app", "list", "--json"], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError("modal app list failed: " + r.stderr[-500:])
    rows = json.loads(r.stdout or "[]")
    return [x for x in rows if str(x.get("Description", x.get("description", ""))).startswith("loyalty-uncond-")]


def live(rows: list[dict]) -> list[dict]:
    return [x for x in rows if "stopped" not in str(x.get("State", x.get("state", ""))).lower()]


def stop_app(name: str) -> list[str]:
    stopped = []
    for x in live(list_apps()):
        if x.get("Description", x.get("description")) == name:
            app_id = x.get("App ID", x.get("app_id"))
            subprocess.run([MODAL, "app", "stop", app_id], capture_output=True, text=True, timeout=120)
            stopped.append(app_id)
    return stopped


def cmd_preview(_):
    print("PREREG frozen sha256:", check_frozen() if (HERE / "PREREG.sha256").exists() else "NOT FROZEN")
    for stage in ("smoke", "full"):
        cells = spec.plan(stage)
        rollouts = sum(c["n"] for c in cells)
        print(f"{stage}: {len(cells)} cells, {rollouts} rollouts, <= {rollouts * spec.MAX_TOKENS:,} new tokens")
    print(f"rate US${spec.RATE_USD_PER_S:.8f}/s ({spec.GPU}, 4 cores, 32 GiB bound); cap US${spec.CAP_USD}")
    print("smoke hold (wall 1800 s):", round(hold_usd(1800), 2), " remaining:", budget.remaining())


def cmd_launch(a):
    if not RUN_ID_RE.fullmatch(a.run_id):
        raise SystemExit("run id must look like uncond-<lowercase>")
    prereg = check_frozen()
    wall = a.wall_s or (1800 if a.stage == "smoke" else 0)
    if not 300 <= wall <= 6 * 3600:
        raise SystemExit("--wall-s 300..21600 required for the full stage (from `project`)")
    run_dir = RUNS / a.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dt%H%M%S", time.gmtime())
    app_name = f"loyalty-{a.run_id}-{a.stage}-{stamp[-6:]}"
    entry = f"{a.run_id}-{a.stage}-{stamp}"
    usd = hold_usd(wall)
    budget.hold(entry, usd, {"stage": a.stage, "app_name": app_name, "wall_s": wall, "rate_usd_per_s": spec.RATE_USD_PER_S,
                             "fixed_usd": spec.FIXED_USD, "tail_s": spec.TAIL_S, "startup_allowance_s": STARTUP_ALLOWANCE_S,
                             "prereg_sha256": prereg, "spec_digest": spec.spec_digest()})
    manifest = {"entry": entry, "app_name": app_name, "stage": a.stage, "wall_s": wall, "hold_usd": usd, "prereg_sha256": prereg,
                "spec_digest": spec.spec_digest(), "analysis_sha256": sha256(HERE / "analysis.py"),
                "modal_app_sha256": sha256(HERE / "modal_app.py"), "started_utc": stamp}
    (run_dir / f"launch-{a.stage}-{stamp}.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"event": "launch", **manifest}), flush=True)
    env = {**os.environ, "UNCOND_APP_NAME": app_name}
    t0 = time.time()
    proc = subprocess.Popen([MODAL, "run", str(HERE / "modal_app.py"), "--stage", a.stage, "--run-dir", str(run_dir),
                             "--wall-s", str(wall)], cwd=str(HERE), env=env, start_new_session=True)
    limit = wall + STARTUP_ALLOWANCE_S
    code, killed = None, False
    while code is None:
        code = proc.poll()
        if code is None and time.time() - t0 > limit:
            os.killpg(proc.pid, signal.SIGINT)
            time.sleep(20)
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
            killed, code = True, proc.wait()
        time.sleep(5)
    elapsed = time.time() - t0
    stopped = stop_app(app_name)
    time.sleep(10)
    still = [x for x in live(list_apps()) if x.get("Description", x.get("description")) == app_name]
    bound = upper_bound_usd(elapsed)
    budget.settle(entry, bound, f"elapsed upper bound: (elapsed {elapsed:.0f} s + tail {spec.TAIL_S} s) x rate + fixed; "
                  "to be reconciled with Modal's billing report", elapsed_s=elapsed, exit_code=code, killed_at_wall=killed,
                  stopped_app_ids=stopped, still_running_after_stop=len(still))
    done = {"event": "settled", "entry": entry, "exit_code": code, "killed_at_wall": killed, "elapsed_s": round(elapsed),
            "upper_bound_usd": round(bound, 4), "app_still_running": len(still), "remaining_usd": float(budget.remaining())}
    (run_dir / f"settle-{a.stage}-{stamp}.json").write_text(json.dumps(done, indent=1))
    print(json.dumps(done), flush=True)


def cmd_project(a):
    run_dir = RUNS / a.run_id
    cells = {}
    for p in (run_dir / "cells").glob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d["stage"] in ("smoke", "probe"):
            cells[(d["stage"], d["arm"], d["context"])] = d
    probe = cells[("probe", "B", "sys_empty_user")]
    tps_probe = probe["tokens"] / probe["gen_s"]
    tps20 = {k[1:]: v["tokens"] / v["gen_s"] for k, v in cells.items() if k[0] == "smoke"}
    gen_s = 0.0
    for cell in spec.plan("full"):
        key = (cell["arm"], cell["context"])
        sm = cells[("smoke",) + key]
        mean_tok = sm["tokens"] / sm["n"]
        b20 = tps20[("B", cell["context"])]
        factor = max(1.0, b20 / tps20[key])   # LoRA overhead relative to B at the same small batch
        gen_s += cell["n"] * mean_tok / (tps_probe / factor)
    infos = sorted(run_dir.glob("container-info-smoke-*.json"))
    ready = json.loads(infos[-1].read_text())["ready_s"] if infos else 600
    projected = gen_s + ready + 21 * 15
    wall = int(math.ceil(1.5 * projected + 300))
    out = {"tps_probe": round(tps_probe), "projected_gen_s": round(gen_s), "ready_s": round(ready), "projected_total_s": round(projected),
           "wall_s": wall, "projected_usd": round(upper_bound_usd(projected), 2), "hold_usd": round(hold_usd(wall), 2),
           "remaining_usd": float(budget.remaining())}
    out["fits_cap"] = out["hold_usd"] <= out["remaining_usd"]
    (run_dir / "projection.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def cmd_apps(_):
    rows = list_apps()
    print(json.dumps([{k: x.get(k) for k in ("app_id", "description", "state", "tasks", "created_at", "stopped_at")} for x in rows], indent=1))


def cmd_reconcile(a):
    print(json.dumps(budget.settle(a.entry, a.usd, a.basis, status="reconciled"), indent=1))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preview")
    l = sub.add_parser("launch")
    l.add_argument("--stage", choices=("smoke", "full"), required=True)
    l.add_argument("--run-id", required=True)
    l.add_argument("--wall-s", type=int, default=0)
    pr = sub.add_parser("project")
    pr.add_argument("--run-id", required=True)
    sub.add_parser("apps")
    r = sub.add_parser("reconcile")
    r.add_argument("--entry", required=True)
    r.add_argument("--usd", type=float, required=True)
    r.add_argument("--basis", required=True)
    a = p.parse_args()
    {"preview": cmd_preview, "launch": cmd_launch, "project": cmd_project, "apps": cmd_apps, "reconcile": cmd_reconcile}[a.cmd](a)


if __name__ == "__main__":
    main()
