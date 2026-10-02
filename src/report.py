"""Turn a results JSON into a legible, uncertainty-aware report plus plots.

  python -m src.report                          # latest run in results/
  python -m src.report results/run_X.json
  python -m src.report results/run_A.json --compare results/run_B.json   # delta = B - A

Writes results/report_<run_id>.txt and results/slices_<run_id>.png (and, with --compare,
results/delta_<A>_vs_<B>.png).
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from src import stats  # noqa: E402
from src.config import ROOT, load_config  # noqa: E402
from src.costs import read_log  # noqa: E402
from src.data import load_eval_set  # noqa: E402

ESCI_ORDER = ["Exact", "Substitute", "Complement", "Irrelevant"]
# Heuristic only: a keyword match, not a parser. Used to build one slice, never as ground truth.
NEGATION = re.compile(r"\b(?:not|no|without|non|nor|except|excluding|exclude|never|doesn't|don't|isn't)\b",
                      re.I)
MIN_SLICE_N = 30   # smaller cells are listed as skipped, not reported with a meaningless CI

# Reference palette, categorical slot 1 (blue) + ink/chrome tokens, light surface.
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BLUE = "#e1e0d9", "#c3c2b7", "#2a78d6"


# --- loading -----------------------------------------------------------------------------

def latest_results() -> Path:
    files = sorted((ROOT / load_config()["harness"]["results_dir"]).glob("run_*.json"))
    if not files:
        raise FileNotFoundError("no results/run_*.json found; run `python -m src.harness` first")
    return files[-1]


def load_run(path: Path) -> tuple[dict, pd.DataFrame, np.ndarray]:
    """(results, per-example frame joined with query text, correct matrix aligned to the frame)."""
    results = json.loads(Path(path).read_text())
    ids, correct = stats.to_correct_matrix(results)
    ev = load_eval_set().set_index("example_id").loc[ids]
    ex = pd.DataFrame({
        "example_id": ids,
        "esci_label": ev["esci_label"].values,
        "truth": ev["ground_truth_binary"].values,
        "query": ev["query"].values,
        "negation": ev["query"].str.contains(NEGATION).values,
    })
    ex["p"] = stats.per_example_pass_rate(correct)
    return results, ex, correct


# --- analysis ----------------------------------------------------------------------------

def build_slices(ex: pd.DataFrame) -> tuple[list[tuple[str, str, np.ndarray]], list[str]]:
    """(group, label, mask) triples plus the cells skipped for having n < MIN_SLICE_N."""
    slices, skipped = [], []

    def add(group: str, label: str, mask: np.ndarray) -> None:
        if mask.sum() >= MIN_SLICE_N:
            slices.append((group, label, mask))
        else:
            skipped.append(f"{label} (n={int(mask.sum())})")

    for lab in ESCI_ORDER:
        add("ESCI label", lab, (ex.esci_label == lab).values)
    for neg in (False, True):
        add("Query has negation (heuristic)", "yes" if neg else "no", (ex.negation == neg).values)
    for lab in ESCI_ORDER:
        for neg in (False, True):
            mask = ((ex.esci_label == lab) & (ex.negation == neg)).values
            add("ESCI label x negation", f"{lab}, {'negation' if neg else 'no negation'}", mask)
    return slices, skipped


def slice_stats(correct: np.ndarray, slices, overall: float) -> list[dict]:
    rows = []
    for group, label, mask in slices:
        m = correct[mask]
        ci = stats.normal_ci(m)
        try:
            t = stats.t_ci(m)
            t_lo, t_hi = t.lo, t.hi
        except ValueError:
            t_lo = t_hi = None
        rows.append({"group": group, "label": label, "n": int(mask.sum()), "acc": ci.mean,
                     "lo": ci.lo, "hi": ci.hi, "t_lo": t_lo, "t_hi": t_hi,
                     "differs": not (ci.lo <= overall <= ci.hi)})
    return rows


def confusion(results: dict) -> dict:
    """Run-level confusion counts over VALID calls only (failures are not in any cell)."""
    tp = fp = fn = tn = 0
    for e in results["examples"]:
        gt = bool(e["ground_truth_binary"])
        for r in e["runs"]:
            if r["status"] != "ok":
                continue
            v = r["is_relevant"]
            tp += gt and v
            fn += gt and not v
            fp += (not gt) and v
            tn += (not gt) and not v
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def consistency(ex: pd.DataFrame, correct: np.ndarray) -> dict:
    valid = (~np.isnan(correct)).sum(axis=1)
    right = np.nansum(correct, axis=1)
    always_right, always_wrong = right == valid, right == 0
    return {
        "always_right": int(always_right.sum()),
        "always_wrong": int(always_wrong.sum()),
        "mixed": int((~always_right & ~always_wrong).sum()),
        # always wrong on a positive = judge always says "not relevant" to an Exact match
        "always_wrong_on_exact": int((always_wrong & (ex.truth == 1).values).sum()),
        "always_wrong_on_not_exact": int((always_wrong & (ex.truth == 0).values).sum()),
    }


def run_cost(run_id: str) -> tuple[int, float]:
    rows = [r for r in read_log() if r["tag"] == run_id]
    return len(rows), sum(r["cost_usd"] or 0.0 for r in rows)


# --- text report -------------------------------------------------------------------------

def pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def text_report(results: dict, ex: pd.DataFrame, correct: np.ndarray, rows: list[dict],
                skipped: list[str]) -> str:
    m = results["metadata"]
    k = m["k"]
    out = []
    add = out.append
    nci, tci = stats.normal_ci(correct), stats.t_ci(correct)
    majority = max(ex.truth.mean(), 1 - ex.truth.mean())
    calls, usd = run_cost(m["run_id"])

    add(f"RUN {m['run_id']}")
    add(f"  N={len(ex)} examples  K={k} runs  model={m['model']} (temperature {m['temperature']})  "
        f"prompt={m['prompt_version']} [{m['prompt_sha256']}]  dataset={m['dataset_version']} "
        f"(seed {m['seed']}, rev {m['dataset_revision'][:8]})")
    add(f"  API cost for this run: ${usd:.2f} ({calls} logged calls)")
    add("")
    add("INFRASTRUCTURE (kept out of every quality number below)")
    add(f"  calls={m['n_calls']}  valid verdicts={m['n_ok']}  parse failures={m['n_parse_failures']}  "
        f"infra failures={m['n_infra_failures']}")
    add("")
    add("JUDGE ACCURACY (agreement with the binary ESCI label)")
    add(f"  mean pass rate           {pct(nci.mean)}     majority-class baseline {pct(majority)}")
    add(f"  95% CI over examples     [{pct(nci.lo)}, {pct(nci.hi)}]   how much accuracy moves with which examples")
    add(f"  95% CI over runs (t)     [{pct(tci.lo)}, {pct(tci.hi)}]   how much the judge wobbles on this fixed set (K={k})")
    add(f"  run-level pass rates     {', '.join(f'{x:.4f}' for x in stats.run_level_pass_rates(correct))}")
    add("")
    add("CAPABILITY vs CONSISTENCY   (agreement with the label, not self-consistency)")
    for kk in range(1, k + 1):
        at, n_at = stats.pass_at_k(correct, kk)
        pw, _ = stats.pass_pow_k(correct, kk)
        add(f"  k={kk}:  pass@k {pct(at)}   pass^k {pct(pw)}   (n={n_at})")
    add("  (an LLM judge adds its own noise, and a 'miss' may be a loose label, not a judge error)")
    add("")
    add("PER-SLICE ACCURACY   [95% CI over examples]   (t over runs)   * = overall mean is outside the CI")
    if not rows:
        add(f"  no slice has n >= {MIN_SLICE_N}; too small to report per-slice intervals")
    group = None
    for r in rows:
        if r["group"] != group:
            group = r["group"]
            add(f"  {group}")
        t = f"({pct(r['t_lo'])}, {pct(r['t_hi'])})" if r["t_lo"] is not None else ""
        add(f"    {r['label']:28s} n={r['n']:5d}  {pct(r['acc'])}  [{pct(r['lo'])}, {pct(r['hi'])}]  {t}"
            f"{'  *' if r['differs'] else ''}")
    if skipped:
        add(f"  skipped (n < {MIN_SLICE_N}): {'; '.join(skipped)}")
    add("  negation = keyword heuristic on the query (not, no, without, non, ...), not a parse")
    add("")
    c = confusion(results)
    n_pos, n_neg = c["tp"] + c["fn"], c["fp"] + c["tn"]
    add("CONFUSION (run-level, valid calls)       judge: relevant   not relevant")
    add(f"  ground truth relevant (Exact)        {c['tp']:10d}   {c['fn']:11d}   <- false negatives")
    add(f"  ground truth not relevant (S/C/I)    {c['fp']:10d}   {c['tn']:11d}")
    add("                                          ^ false positives")
    add(f"  precision {pct(c['tp'] / (c['tp'] + c['fp']))}   recall {pct(c['tp'] / n_pos)}   "
        f"specificity {pct(c['tn'] / n_neg)}   FPR {pct(c['fp'] / n_neg)}   FNR {pct(c['fn'] / n_pos)}")
    add("  share of calls where the judge said 'relevant', by ESCI label:")
    for lab in ESCI_ORDER:
        mask = (ex.esci_label == lab).values
        if not mask.any():
            continue
        acc = np.nanmean(correct[mask])
        said_yes = acc if lab == "Exact" else 1 - acc
        add(f"    {lab:12s} {pct(said_yes)}")
    cons = consistency(ex, correct)
    add("")
    add("CONSISTENCY ACROSS RUNS (examples)")
    add(f"  right on every run {cons['always_right']}   wrong on every run {cons['always_wrong']}   "
        f"mixed {cons['mixed']}")
    add(f"  wrong on every run: {cons['always_wrong_on_exact']} are Exact (judge always says not relevant), "
        f"{cons['always_wrong_on_not_exact']} are S/C/I (judge always says relevant)")
    add("  -> the always-wrong examples are the first place to look for loose ground truth (see results/FINDINGS.md)")
    return "\n".join(out)


def compare_text(delta: stats.Delta, welch: stats.Delta | None, ma: dict, mb: dict) -> str:
    out = [f"COMPARISON  A={ma['run_id']}  B={mb['run_id']}   delta = B - A",
           f"  same prompt/model/config? prompt A={ma['prompt_sha256']} B={mb['prompt_sha256']}  "
           f"model A={ma['model']} B={mb['model']}",
           f"  paired per-example (N={delta.n} common examples): delta {100 * delta.delta:+.2f} pts  "
           f"95% CI [{100 * delta.lo:+.2f}, {100 * delta.hi:+.2f}]  -> "
           f"{'SIGNIFICANT' if delta.significant else 'not significant (CI includes 0)'}"]
    if welch:
        out.append(f"  Welch on run-level rates (K_A={ma['k']}, K_B={mb['k']}): delta {100 * welch.delta:+.2f} pts  "
                   f"95% CI [{100 * welch.lo:+.2f}, {100 * welch.hi:+.2f}]  t={welch.t:.2f} dof={welch.dof:.1f} "
                   f"p={welch.p_value:.3f}")
    return "\n".join(out)


# --- plots -------------------------------------------------------------------------------

def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": INK_2,
    })


def _frame(ax) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def plot_slices(rows: list[dict], overall: float, meta: dict, path: Path) -> None:
    """Horizontal bars (one hue: the slices are items, not series, so no legend), CI whiskers,
    overall accuracy as a reference line, the value at each bar tip."""
    plot_rows = [r for r in rows if r["group"] != "Query has negation (heuristic)"]  # table-only
    if not plot_rows:
        return
    _style()
    n_gaps = len({r["group"] for r in plot_rows}) - 1
    fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(plot_rows) + 0.5 * n_gaps + 1.9))
    ys, y, last = [], 0.0, None
    for r in plot_rows:
        if last is not None and r["group"] != last:
            y += 0.6   # gap between groups
        ys.append(y)
        y += 1
        last = r["group"]
    for r, yy in zip(plot_rows, ys):
        ax.barh(yy, r["acc"], height=0.5, color=BLUE, zorder=3)
        ax.errorbar(r["acc"], yy, xerr=[[r["acc"] - r["lo"]], [r["hi"] - r["acc"]]], color=INK,
                    linewidth=1.3, capsize=3, capthick=1.3, zorder=4)
        ax.text(r["hi"] + 0.015, yy, f"{100 * r['acc']:.1f}%", va="center", ha="left", fontsize=9,
                color=INK, zorder=5, bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5})
    ax.axvline(overall, color=INK_2, linewidth=1, zorder=2)
    ax.text(overall, -0.95, f"overall {100 * overall:.1f}%", ha="center", va="bottom", fontsize=9,
            color=INK_2)
    ax.set_yticks(ys, [f"{r['label']}  (n={r['n']})" for r in plot_rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)   # headroom so the value label past a high whisker is never clipped
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    _frame(ax)
    group_first = {}
    for r, yy in zip(plot_rows, ys):
        group_first.setdefault(r["group"], yy)
    for g, yy in group_first.items():
        ax.text(-0.012, yy - 0.78, g, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=8, color=MUTED)
    fig.suptitle("Judge accuracy by slice", x=0.015, ha="left", fontsize=13, fontweight="bold", y=0.985)
    fig.text(0.015, 0.915, f"Agreement with the binary ESCI label, mean of K={meta['k']} runs; whiskers are the 95% CI "
             f"over examples. {meta['model']}, prompt {meta['prompt_version']}.", fontsize=8.5, color=INK_2,
             ha="left")
    fig.subplots_adjust(left=0.30, right=0.97, top=0.84, bottom=0.09)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_delta(delta: stats.Delta, welch: stats.Delta | None, ma: dict, mb: dict, path: Path) -> None:
    """A dot with a CI whisker per method against a zero line (row labels identify the method)."""
    _style()
    items = [(f"Paired, per example (N={delta.n})", delta)]
    if welch:
        items.append((f"Welch, run-level (K={ma['k']} vs {mb['k']})", welch))
    fig, ax = plt.subplots(figsize=(8.4, 1.0 * len(items) + 2.0))
    for i, (_, d) in enumerate(items):
        ax.plot([100 * d.lo, 100 * d.hi], [i, i], color=INK, linewidth=2, solid_capstyle="round", zorder=3)
        ax.plot(100 * d.delta, i, "o", markersize=10, color=BLUE, markeredgecolor=SURFACE,
                markeredgewidth=2, zorder=4)
        ax.text(100 * d.hi + 0.8, i, f"{100 * d.delta:+.1f} pts  [{100 * d.lo:+.1f}, {100 * d.hi:+.1f}]",
                va="center", fontsize=9, color=INK)
    ax.axvline(0, color=INK_2, linewidth=1, zorder=2)
    ax.set_yticks(range(len(items)), [lbl for lbl, _ in items], fontsize=9)
    ax.set_ylim(len(items) - 0.5, -0.5)
    lim = max(10.0, max(abs(100 * d.lo) for _, d in items), max(abs(100 * d.hi) for _, d in items)) + 18
    ax.set_xlim(-lim / 2.2, lim)
    ax.set_xlabel("Change in accuracy, B minus A (percentage points)", fontsize=9)
    _frame(ax)
    fig.suptitle("Change in accuracy between two runs", x=0.015, ha="left", fontsize=13,
                 fontweight="bold", y=0.97)
    same = ma["prompt_sha256"] == mb["prompt_sha256"] and ma["model"] == mb["model"]
    fig.text(0.015, 0.875, f"A = {ma['run_id']}    B = {mb['run_id']}", fontsize=8, color=INK_2, ha="left")
    fig.text(0.015, 0.815, "95% CI; the change is real only if the interval excludes 0." +
             ("  Same prompt and model, so this is an A/A check: ~0 is the right answer." if same else ""),
             fontsize=8, color=INK_2, ha="left")
    fig.subplots_adjust(left=0.30, right=0.97, top=0.76, bottom=0.25)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --- CLI ---------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="?", type=Path, help="results JSON (default: latest)")
    ap.add_argument("--compare", type=Path, help="a second results JSON; reports delta = this - first")
    args = ap.parse_args()

    path = args.results or latest_results()
    results, ex, correct = load_run(path)
    meta = results["metadata"]
    overall = stats.mean_pass_rate(correct)
    slices, skipped = build_slices(ex)
    rows = slice_stats(correct, slices, overall)

    text = text_report(results, ex, correct, rows, skipped)
    out_dir = ROOT / load_config()["harness"]["results_dir"]
    plot_slices(rows, overall, meta, out_dir / f"slices_{meta['run_id']}.png")
    files = [f"slices_{meta['run_id']}.png"] if rows else []

    if args.compare:
        res_b, _, _ = load_run(args.compare)
        _, ma_mat, mb_mat = stats.align(results, res_b)
        mb_meta = res_b["metadata"]
        delta = stats.paired_delta(ma_mat, mb_mat)
        try:
            welch = stats.welch_t_test(ma_mat, mb_mat)
        except ValueError:
            welch = None
        text += "\n\n" + compare_text(delta, welch, meta, mb_meta)
        name = f"delta_{meta['run_id'][:16]}_vs_{mb_meta['run_id'][:16]}.png"
        plot_delta(delta, welch, meta, mb_meta, out_dir / name)
        files.append(name)

    (out_dir / f"report_{meta['run_id']}.txt").write_text(text + "\n")
    print(text)
    print(f"\nwrote results/report_{meta['run_id']}.txt and " + ", ".join(f"results/{f}" for f in files))


if __name__ == "__main__":
    main()
