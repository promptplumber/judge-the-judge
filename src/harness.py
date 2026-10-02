"""Run the judge over the eval set K times per example and persist every raw verdict.

Two modes (config `harness.mode`, or --mode):
  standard  synchronous thread-pool calls; use for development and --limit runs.
  batch     OpenAI Batch API; use for the full run. Async: `submit` now, `collect` later.

Both modes feed the same assemble/save path, so a results file has the same shape either way.

Usage:
  python -m src.harness --limit 20 --k 3                 # standard dev run
  python -m src.harness --mode batch                     # submit the full run as a batch
  python -m src.harness collect <run_id> [--wait]        # collect a finished batch
"""
import argparse
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config import ROOT, load_config
from src.data import load_eval_set
from src.judge import (INFRA_FAILURE, OK, PARSE_FAILURE, SYSTEM_PROMPTS, JudgeResult,
                       build_request_body, judge_pair, make_client, parse_completion)

BATCH_DIR = ROOT / "data" / "batches"          # gitignored scratch (inputs + manifests)
BATCH_FINAL = {"completed", "failed", "expired", "cancelled"}


# --- shared helpers -----------------------------------------------------------------------

def select_examples(df: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    """Full set, or a seeded random subset. Not head(): the frozen file is sorted by
    example_id, which follows alphabetical query order, so head() would be all '#1...' queries."""
    if limit is None or limit >= len(df):
        return df.reset_index(drop=True)
    return df.sample(n=limit, random_state=seed).sort_values("example_id").reset_index(drop=True)


def run_metadata(cfg: dict, k: int, n_examples: int, limit: int | None, mode: str,
                 run_id: str, submitted_at: str) -> dict:
    """The 'pin everything' block: whatever could change a result is recorded here."""
    j, d = cfg["judge"], cfg["dataset"]
    prompt = SYSTEM_PROMPTS[j["prompt_version"]]
    return {
        "run_id": run_id,
        "timestamp": submitted_at,
        "mode": mode,
        "model": j["model"],
        "temperature": j["temperature"],
        "max_completion_tokens": j["max_completion_tokens"],
        "prompt_version": j["prompt_version"],
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16],
        "dataset": d["name"],
        "dataset_revision": d["revision"],
        "dataset_version": d["version"],
        "seed": d["seed"],
        "k": k,
        "n_examples": n_examples,
        "limit": limit,
    }


def make_run_id(cfg: dict) -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    j = cfg["judge"]
    return f"{now:%Y%m%dT%H%M%SZ}_{j['prompt_version']}_{j['model']}", now.isoformat(timespec="seconds")


def assemble_results(meta: dict, subset: pd.DataFrame,
                     run_results: dict[tuple[int, int], JudgeResult]) -> dict:
    """Per example: `verdicts` holds ONLY real verdicts (status ok). Parse and infra failures
    are counted in their own fields and never appear as a verdict."""
    examples = []
    totals = {OK: 0, PARSE_FAILURE: 0, INFRA_FAILURE: 0}
    for row in subset.itertuples():
        eid = int(row.example_id)
        runs, verdicts = [], []
        n_parse = n_infra = 0
        for run in range(meta["k"]):
            r = run_results.get((eid, run)) or JudgeResult(INFRA_FAILURE, error="no result recorded")
            totals[r.status] += 1
            n_parse += r.status == PARSE_FAILURE
            n_infra += r.status == INFRA_FAILURE
            if r.status == OK:
                verdicts.append(r.is_relevant)
            entry = {"run": run, "status": r.status, "is_relevant": r.is_relevant,
                     "justification": r.justification, "evidence": r.evidence}
            if r.error:
                entry["error"] = r.error
            runs.append(entry)
        examples.append({
            "example_id": eid,
            "ground_truth_binary": int(row.ground_truth_binary),
            "esci_label": row.esci_label,
            "verdicts": verdicts,
            "n_parse_failures": n_parse,
            "n_infra_failures": n_infra,
            "runs": runs,
        })
    meta = {**meta, "n_calls": sum(totals.values()), "n_ok": totals[OK],
            "n_parse_failures": totals[PARSE_FAILURE], "n_infra_failures": totals[INFRA_FAILURE]}
    return {"metadata": meta, "examples": examples}


def save_results(results: dict) -> Path:
    out_dir = ROOT / load_config()["harness"]["results_dir"]
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"run_{results['metadata']['run_id']}.json"
    path.write_text(json.dumps(results, indent=1))
    return path


def print_summary(results: dict, path) -> None:
    m, ex = results["metadata"], results["examples"]
    print(f"\nwrote {path}")
    print(f"examples={m['n_examples']} K={m['k']} calls={m['n_calls']}  "
          f"ok={m['n_ok']}  parse_failures={m['n_parse_failures']}  infra_failures={m['n_infra_failures']}")
    fail_rate = (m["n_parse_failures"] + m["n_infra_failures"]) / max(m["n_calls"], 1)
    if fail_rate > 0.05:
        print(f"WARNING: {fail_rate:.0%} of calls failed; check the errors before trusting this run.")
    agree = [v == bool(e["ground_truth_binary"]) for e in ex for v in e["verdicts"]]
    if agree:  # sanity glance only; the real metrics (with CIs) come from stats.py
        print(f"sanity (not a metric): raw agreement with ground truth = {sum(agree)}/{len(agree)}")


# --- standard mode ------------------------------------------------------------------------

PARTIAL_DIR = ROOT / "data" / "partials"       # gitignored; one append-only JSONL per standard run


def partial_path(run_id: str) -> Path:
    return PARTIAL_DIR / f"{run_id}.jsonl"


def load_partial(run_id: str) -> tuple[dict, dict[tuple[int, int], JudgeResult]]:
    """Read a partial run: (metadata, results). Infra failures are dropped so a resume retries
    them (not the judge's fault); ok and parse_failure results are kept (a real model answer).
    If a call appears more than once, the last record wins."""
    lines = [json.loads(x) for x in partial_path(run_id).read_text().splitlines() if x.strip()]
    meta, done = lines[0]["meta"], {}
    for rec in lines[1:]:
        res = JudgeResult(**rec["result"])
        if res.status == INFRA_FAILURE:
            done.pop((rec["eid"], rec["run"]), None)
        else:
            done[(rec["eid"], rec["run"])] = res
    return meta, done


def run_standard(subset: pd.DataFrame, meta: dict, cfg: dict, resume: bool = False) -> dict:
    """Every finished call is appended to a partial file immediately, so a crash, sleep or
    timeout loses nothing: rerun with --resume <run_id> to continue where it stopped."""
    client = make_client(cfg)
    path = partial_path(meta["run_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[tuple[int, int], JudgeResult] = {}
    if resume:
        _, done = load_partial(meta["run_id"])
    else:
        path.write_text(json.dumps({"meta": meta}) + "\n")
    tasks = [(int(r.example_id), run, r.query, r.product_title)
             for r in subset.itertuples() for run in range(meta["k"])
             if (int(r.example_id), run) not in done]
    print(f"{len(done)} calls already done, {len(tasks)} to run")
    lock = threading.Lock()

    def call(task):
        eid, run, query, title = task
        try:
            res = judge_pair(query, title, cfg, client, tag=meta["run_id"])
        except Exception as e:  # anything unexpected is recorded, not allowed to kill the run
            res = JudgeResult(INFRA_FAILURE, error=f"unexpected {type(e).__name__}: {e}")
        with lock, open(path, "a") as f:
            f.write(json.dumps({"eid": eid, "run": run, "result": asdict(res)}) + "\n")
        return (eid, run), res

    with ThreadPoolExecutor(max_workers=cfg["harness"]["concurrency"]) as pool:
        done.update(tqdm(pool.map(call, tasks), total=len(tasks), desc="judging"))
    return assemble_results(meta, subset, done)


# --- batch mode ---------------------------------------------------------------------------

def _custom_id(example_id: int, run: int) -> str:
    return f"{example_id}__{run}"


def submit_batch(subset: pd.DataFrame, meta: dict, cfg: dict) -> str:
    client = make_client(cfg)
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    input_path = BATCH_DIR / f"{meta['run_id']}_input.jsonl"
    with open(input_path, "w") as f:
        for r in subset.itertuples():
            for run in range(meta["k"]):
                f.write(json.dumps({
                    "custom_id": _custom_id(int(r.example_id), run),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": build_request_body(r.query, r.product_title, cfg),
                }) + "\n")
    with open(input_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(input_file_id=uploaded.id, endpoint="/v1/chat/completions",
                                  completion_window="24h",
                                  metadata={"run_id": meta["run_id"][:60]})
    manifest = {"batch_id": batch.id, "input_file_id": uploaded.id, "metadata": meta,
                "examples": subset[["example_id", "esci_label", "ground_truth_binary"]]
                .astype({"example_id": int, "ground_truth_binary": int}).to_dict("records")}
    (BATCH_DIR / f"{meta['run_id']}_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"submitted batch {batch.id} ({meta['n_examples'] * meta['k']} requests), status={batch.status}")
    print(f"collect later with:  python -m src.harness collect {meta['run_id']}")
    return meta["run_id"]


def _read_jsonl(client, file_id: str | None) -> list[dict]:
    if not file_id:
        return []
    text = client.files.content(file_id).text
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def collect_batch(run_id: str, wait: bool = False, poll_s: int = 60) -> None:
    cfg = load_config()
    client = make_client(cfg)
    manifest = json.loads((BATCH_DIR / f"{run_id}_manifest.json").read_text())
    batch = client.batches.retrieve(manifest["batch_id"])
    while wait and batch.status not in BATCH_FINAL:
        counts = batch.request_counts
        print(f"{batch.status}: {counts.completed}/{counts.total} done, {counts.failed} failed; "
              f"checking again in {poll_s}s")
        time.sleep(poll_s)
        batch = client.batches.retrieve(manifest["batch_id"])
    if batch.status not in BATCH_FINAL:
        c = batch.request_counts
        print(f"batch {batch.id} is still {batch.status} ({c.completed}/{c.total} done). Try again later.")
        return

    run_results: dict[tuple[int, int], JudgeResult] = {}
    for line in _read_jsonl(client, batch.output_file_id):
        eid, run = (int(x) for x in line["custom_id"].split("__"))
        resp = line.get("response")
        if line.get("error") or not resp or resp.get("status_code") != 200:
            err = line.get("error") or (resp or {}).get("body", {}).get("error") or resp
            run_results[(eid, run)] = JudgeResult(INFRA_FAILURE, error=f"batch request failed: {err}")
        else:
            run_results[(eid, run)] = parse_completion(resp["body"])
    for line in _read_jsonl(client, batch.error_file_id):
        eid, run = (int(x) for x in line["custom_id"].split("__"))
        run_results.setdefault((eid, run), JudgeResult(
            INFRA_FAILURE, error=f"batch request failed: {line.get('error') or line.get('response')}"))
    if batch.status != "completed":
        print(f"note: batch ended as {batch.status}; requests with no result are counted as infra failures. "
              f"errors={batch.errors}")

    subset = pd.DataFrame(manifest["examples"])
    meta = {**manifest["metadata"], "batch_id": batch.id, "batch_status": batch.status,
            "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    # query/title aren't needed to assemble; results only carry ids, labels and verdicts
    results = assemble_results(meta, subset, run_results)
    print_summary(results, save_results(results))


# --- CLI ----------------------------------------------------------------------------------

def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", choices=["run", "collect"], default="run")
    ap.add_argument("run_id", nargs="?", help="run id to collect (collect only)")
    ap.add_argument("--limit", type=int, help="seeded random subset of N examples (dev runs)")
    ap.add_argument("--k", type=int, default=cfg["harness"]["k"], help="runs per example")
    ap.add_argument("--mode", choices=["standard", "batch"], default=cfg["harness"]["mode"])
    ap.add_argument("--wait", action="store_true", help="collect: poll until the batch finishes")
    ap.add_argument("--resume", metavar="RUN_ID", help="continue an interrupted standard run")
    args = ap.parse_args()

    if args.command == "collect":
        if not args.run_id:
            ap.error("collect needs a run_id")
        collect_batch(args.run_id, wait=args.wait)
        return

    if args.resume:  # same run_id, metadata and subset as the interrupted run
        meta, _ = load_partial(args.resume)
        subset = select_examples(load_eval_set(cfg), meta["limit"], cfg["dataset"]["seed"])
        results = run_standard(subset, meta, cfg, resume=True)
        print_summary(results, save_results(results))
        return

    subset = select_examples(load_eval_set(cfg), args.limit, cfg["dataset"]["seed"])
    run_id, submitted_at = make_run_id(cfg)
    meta = run_metadata(cfg, args.k, len(subset), args.limit, args.mode, run_id, submitted_at)
    if args.mode == "batch":
        submit_batch(subset, meta, cfg)
    else:
        print(f"run_id: {run_id}  (if interrupted: python -m src.harness --resume {run_id})")
        results = run_standard(subset, meta, cfg)
        print_summary(results, save_results(results))


if __name__ == "__main__":
    main()
