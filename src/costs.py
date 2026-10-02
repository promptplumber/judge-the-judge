"""Project-wide API cost log.

Every paid API call appends one JSON line to the log (config `costs.log_path`). Anything that
calls the API must call log_api_call() right after the response; `python -m src.costs` totals it.

Cost = prompt_tokens * input_rate + completion_tokens * output_rate. Cached prompt tokens are
logged but billed here at the full input rate, so the figure is an upper bound.
"""
import json
import sys
import threading
from collections import defaultdict
from datetime import datetime, timezone

from src.config import ROOT, load_config

_lock = threading.Lock()


def compute_cost(model: str, prompt_tokens: int, completion_tokens: int, cfg: dict) -> float | None:
    rates = cfg["costs"]["pricing_usd_per_1m_tokens"].get(model)
    if rates is None:
        return None  # unpriced model: tokens are still logged, cost left blank rather than guessed
    return prompt_tokens / 1e6 * rates["input"] + completion_tokens / 1e6 * rates["output"]


def log_api_call(usage: dict, model: str, tag: str, status: str | None = None,
                 mode: str = "standard", cfg: dict | None = None, **extra) -> dict:
    """Append one call's usage and cost. `usage` is the response's usage dict. Never raises:
    a logging problem is reported on stderr instead of killing a paid run."""
    cfg = cfg or load_config()
    prompt, completion = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    cost = compute_cost(model, prompt, completion, cfg)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "tag": tag, "model": model, "mode": mode, "status": status,
        "prompt_tokens": prompt, "completion_tokens": completion, "cached_tokens": cached,
        "cost_usd": cost, **extra,
    }
    try:
        path = ROOT / cfg["costs"]["log_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        with _lock, open(path, "a") as f:
            f.write(json.dumps(row) + "\n")
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: could not write cost log: {e}", file=sys.stderr)
    return row


def read_log(cfg: dict | None = None) -> list[dict]:
    cfg = cfg or load_config()
    path = ROOT / cfg["costs"]["log_path"]
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(rows: list[dict]) -> None:
    by_tag = defaultdict(lambda: [0, 0, 0, 0.0, False])  # calls, prompt, completion, cost, has_estimate
    for r in rows:
        t = by_tag[r["tag"]]
        t[0] += r.get("calls", 1)
        t[1] += r["prompt_tokens"]
        t[2] += r["completion_tokens"]
        t[3] += r["cost_usd"] or 0.0
        t[4] |= bool(r.get("estimated"))
    print(f"{'tag':44s} {'calls':>7s} {'prompt':>9s} {'compl':>9s} {'cost_usd':>10s}")
    for tag, (n, p, c, usd, est) in by_tag.items():
        print(f"{tag:44s} {n:7d} {p:9d} {c:9d} {usd:10.6f}{'  (estimated)' if est else ''}")
    total = sum(v[3] for v in by_tag.values())
    print(f"{'TOTAL':44s} {sum(v[0] for v in by_tag.values()):7d} "
          f"{sum(v[1] for v in by_tag.values()):9d} {sum(v[2] for v in by_tag.values()):9d} {total:10.6f}")


if __name__ == "__main__":
    summarize(read_log())
