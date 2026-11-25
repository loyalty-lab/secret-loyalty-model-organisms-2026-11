"""UNCOND's own GPU ledger: cap US$20 (Jack, 12 Nov 2026), atomic pre-launch holds, settle, reconcile.

Pattern of experiments/REPL/control/budget.py (lock, write-temp-then-rename, holds before dispatch), kept separate because
the shared REPL ledger's transaction() hard-codes its allowed caps. No network, no secrets. WSL only (fcntl).
"""
from __future__ import annotations

import fcntl
import json
import time
import uuid
from contextlib import contextmanager
from decimal import Decimal, ROUND_CEILING
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "budget-ledger.json"
LOCK = HERE / "budget.lock"
CAP_USD = Decimal("20")
QUANTUM = Decimal("0.000001")


class BudgetError(RuntimeError):
    pass


def money(v) -> Decimal:
    d = Decimal(str(v))
    if not d.is_finite() or d < 0:
        raise BudgetError("negative or non-finite amount")
    return d.quantize(QUANTUM, rounding=ROUND_CEILING)


def fresh() -> dict:
    return {"experiment": "UNCOND", "cap_usd": 20, "approved": "Jack, 12 Nov 2026: preregistration and paid run, hard cap US$20 Modal",
            "entries": []}


@contextmanager
def transaction(ledger: Path | None = None):
    ledger = ledger or LEDGER
    lock = ledger.with_suffix(".lock")
    with lock.open("a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        data = json.loads(ledger.read_text()) if ledger.exists() else fresh()
        if data.get("cap_usd") != 20:
            raise BudgetError("UNCOND cap changed; only Jack can change it")
        yield data
        tmp = ledger.with_name(ledger.name + "." + uuid.uuid4().hex + ".tmp")
        tmp.write_text(json.dumps(data, indent=1) + "\n")
        for attempt in range(40):          # /mnt/c: an open reader handle makes rename fail briefly
            try:
                tmp.replace(ledger)
                break
            except PermissionError:
                if attempt == 39:
                    raise
                time.sleep(0.25)


def committed(data: dict) -> Decimal:
    """Held (unsettled) plus settled amounts; a settled entry counts its settled figure."""
    total = Decimal(0)
    for e in data["entries"]:
        total += money(e["settled_usd"] if e.get("status") in ("settled", "reconciled") else e["hold_usd"])
    return total


def remaining(ledger: Path | None = None) -> Decimal:
    with transaction(ledger) as data:
        return CAP_USD - committed(data)


def hold(entry_id: str, usd, meta: dict, ledger: Path | None = None) -> dict:
    usd = money(usd)
    if usd <= 0:
        raise BudgetError("positive hold required")
    with transaction(ledger) as data:
        if any(e["id"] == entry_id for e in data["entries"]):
            raise BudgetError("duplicate ledger id " + entry_id)
        if committed(data) + usd > CAP_USD:
            raise BudgetError(f"hold US${usd} would exceed the US$20 cap (committed US${committed(data)})")
        e = {"id": entry_id, "hold_usd": float(usd), "status": "held", "held_unix": time.time(), **meta}
        data["entries"].append(e)
        return e


def settle(entry_id: str, usd, basis: str, status: str = "settled", ledger: Path | None = None, **extra) -> dict:
    if status not in ("settled", "reconciled"):
        raise BudgetError("bad status")
    usd = money(usd)
    with transaction(ledger) as data:
        e = next((e for e in data["entries"] if e["id"] == entry_id), None)
        if e is None:
            raise BudgetError("unknown ledger id " + entry_id)
        if status == "settled" and usd > money(e["hold_usd"]):
            basis += " (above hold: elapsed upper bound exceeded the hold)"
        e.update({"settled_usd": float(usd), "status": status, "basis": basis, "settled_unix": time.time(), **extra})
        return e
