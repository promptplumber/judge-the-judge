"""The human-review sample.

  python -m src.human_eval build [results.json]   # writes labeling/labeling_tool.html + labeling/key.json
  python -m src.human_eval negation-build         # 15-item blind sheet from the Exact+negation always-wrong set
  python -m src.human_eval negation-analyze       # analyse labeling/negation_labels_blank.csv
  python -m src.human_eval analyze                # reads labeling/human_labels.csv, writes results/human_eval_analysis.txt

The tool shows ONLY the query and product (no ESCI label, no judge verdict, order shuffled), so
the labels are blind. The answer key, with each item's stratum and the judge's verdicts, lives in
key.json and is only read by the analysis step.

Sampling: three DISJOINT strata that partition the 2,000 examples. Taking only the hard cases
would overstate how often the judge is wrong, so every stratum's population size N_h and sample
size n_h are recorded; the analysis weights each item by N_h / n_h to estimate whole-set numbers.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src import stats
from src.config import ROOT, load_config
from src.data import load_eval_set
from src.report import latest_results, load_run

STRATA = ["always_wrong", "exact_negation", "random_rest"]


def build_sample(results: dict, ex: pd.DataFrame, correct: np.ndarray, cfg: dict) -> tuple[list[dict], dict]:
    h = cfg["human_eval"]
    rng = np.random.default_rng(h["seed"])
    always_wrong = (np.nansum(correct, axis=1) == 0) & ((~np.isnan(correct)).sum(axis=1) > 0)
    exact_neg = ((ex.esci_label == "Exact") & ex.negation).values & ~always_wrong
    rest = ~always_wrong & ~exact_neg
    masks = {"always_wrong": always_wrong, "exact_negation": exact_neg, "random_rest": rest}
    sizes = {"always_wrong": h["n_always_wrong"], "exact_negation": h["n_exact_negation"],
             "random_rest": h["n_random"]}

    valid = (~np.isnan(correct)).sum(axis=1)
    yes_frac = np.where(ex.truth.values == 1, np.nansum(correct, axis=1), valid - np.nansum(correct, axis=1)) / valid
    items = []
    strata = {}
    for name in STRATA:
        pool = np.flatnonzero(masks[name])
        n = min(sizes[name], len(pool))
        strata[name] = {"N": int(len(pool)), "n": int(n)}
        for i in sorted(rng.choice(pool, size=n, replace=False)):
            items.append({
                "example_id": int(ex.example_id[i]), "stratum": name,
                "esci_label": ex.esci_label[i], "truth": int(ex.truth[i]),
                "judge_yes_frac": float(yes_frac[i]), "n_valid_runs": int(valid[i]),
                "judge_majority": (None if math.isclose(yes_frac[i], 0.5) else bool(yes_frac[i] > 0.5)),
            })
    rng.shuffle(items)  # presentation order must not reveal the stratum
    return items, strata


def _tool_payload(items: list[dict], cfg: dict) -> list[dict]:
    """Only what the labeler may see. No label, no verdict, no stratum."""
    ev = load_eval_set(cfg).set_index("example_id")
    cut = cfg["human_eval"]["details_chars"]
    out = []
    for it in items:
        r = ev.loc[it["example_id"]]
        clip = lambda v: (str(v)[:cut] + ("…" if len(str(v)) > cut else "")) if isinstance(v, str) else ""  # noqa: E731
        out.append({"id": it["example_id"], "query": r["query"], "title": r["product_title"],
                    "brand": clip(r["product_brand"]), "bullets": clip(r["product_bullet_point"]),
                    "description": clip(r["product_description"])})
    return out


def write_tool(payload: list[dict], path: Path) -> None:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    path.write_text(HTML.replace("__DATA__", data))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build", "analyze", "negation-build", "negation-analyze"])
    ap.add_argument("results", nargs="?", type=Path)
    args = ap.parse_args()

    cfg = load_config()
    if args.command == "analyze":
        analyze(cfg)
        return
    if args.command == "negation-build":
        negation_build(cfg, args.results)
        return
    if args.command == "negation-analyze":
        negation_analyze(cfg)
        return
    path = args.results or latest_results()
    results, ex, correct = load_run(path)
    items, strata = build_sample(results, ex, correct, cfg)
    out = ROOT / cfg["human_eval"]["dir"]
    out.mkdir(exist_ok=True)
    key = {"run_id": results["metadata"]["run_id"], "seed": cfg["human_eval"]["seed"],
           "strata": strata, "items": items}
    (out / "key.json").write_text(json.dumps(key, indent=1))
    write_tool(_tool_payload(items, cfg), out / "labeling_tool.html")
    print(f"run {key['run_id']}: {len(items)} items")
    for name, s in strata.items():
        print(f"  {name:15s} population N={s['N']:4d}  sampled n={s['n']:3d}  weight N/n={s['N'] / max(s['n'], 1):.1f}")
    print(f"wrote {out}/labeling_tool.html and {out}/key.json (key = answers; don't open it before labeling)")


# --- targeted negation review ----------------------------------------------------

def negation_build(cfg: dict, results_path: Path | None = None) -> None:
    """Blind sheet of Exact + negation-keyword examples that were wrong on EVERY valid run.
    Examples already labeled in round 1 are not re-drawn (their label is reused in analysis)."""
    import csv

    h = cfg["negation_review"]
    results, ex, correct = load_run(results_path or latest_results())
    valid = (~np.isnan(correct)).sum(axis=1)
    always_wrong = (np.nansum(correct, axis=1) == 0) & (valid > 0)
    pop_mask = ((ex.esci_label == "Exact") & ex.negation).values & always_wrong
    pop_ids = [int(i) for i in ex.example_id[pop_mask]]
    d = ROOT / cfg["human_eval"]["dir"]
    round1 = set(pd.read_csv(d / "human_labels.csv").example_id)
    pool = [i for i in pop_ids if i not in round1]
    rng = np.random.default_rng(h["seed"])
    picked = sorted(int(i) for i in rng.choice(pool, size=min(h["n"], len(pool)), replace=False))
    order = list(picked)
    rng.shuffle(order)  # presentation order is independent of id (which follows alphabetical query order)

    ev = load_eval_set(cfg).set_index("example_id")
    idx = ex.set_index("example_id")
    with open(d / "negation_labels_blank.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["example_id", "query", "product_title", "human_label", "human_unsure", "notes"])
        for i in order:
            w.writerow([i, ev.loc[i, "query"], ev.loc[i, "product_title"], "", "", ""])
    yes_frac = {}
    for e in results["examples"]:
        ok = [r for r in e["runs"] if r["status"] == "ok"]
        yes_frac[e["example_id"]] = sum(r["is_relevant"] for r in ok) / len(ok)
    key = {"run_id": results["metadata"]["run_id"], "seed": h["seed"], "n_requested": h["n"],
           "population": "Exact + negation-keyword query, wrong on every valid run",
           "population_size": len(pop_ids), "population_ids": sorted(pop_ids),
           "already_labeled_in_round_1": sorted(set(pop_ids) & round1), "sampled_from": len(pool),
           "items": [{"example_id": i, "esci_label": idx.loc[i, "esci_label"],
                      "judge_yes_frac": yes_frac[i]} for i in picked]}
    (d / "negation_key.json").write_text(json.dumps(key, indent=1))
    print(f"seed={h['seed']}  population={len(pop_ids)}  already labeled in round 1={sorted(set(pop_ids) & round1)}  "
          f"sampled {len(order)} from the other {len(pool)}\n")
    print("QUESTION (same as before, title only): Would a shopper searching for this query consider")
    print("this product a directly relevant result they might buy?\n")
    for n, i in enumerate(order, 1):
        print(f"{n:2d}. [{i}]  query: {ev.loc[i, 'query']}\n      title: {ev.loc[i, 'product_title']}")
    print(f"\nwrote {d}/negation_labels_blank.csv (fill human_label = yes/no; if unsure put human_unsure = yes "
          f"and leave human_label blank) and {d}/negation_key.json (answers; don't open before labeling)")


# Assigned from query + title ONLY, before the human labels were merged (single rater: Claude).
#   C1 title contradicts the exclusion (the excluded thing is visibly in the title)
#   C2 title is silent on the exclusion, so it cannot be verified from the title  <- the negation hypothesis
#   C3 not a real exclusion (query is a phrase/title) or the product is simply a different thing
#   C4 title affirmatively satisfies the exclusion
NEGATION_CATEGORY = {
    176007: "C2", 1667487: "C1", 226583: "C3", 20999: "C2", 2156553: "C1", 555757: "C2", 10595: "C1",
    2135331: "C3", 1131937: "C2", 871828: "C4", 1457562: "C3", 15092: "C1", 14324: "C1", 652246: "C2",
    828686: "C1",
}
CATEGORY_NAME = {"C1": "title contradicts the exclusion", "C2": "title silent on the exclusion (unverifiable)",
                 "C3": "not a real exclusion / different product", "C4": "title satisfies the exclusion"}


def _wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def _clopper_pearson(k: int, n: int, a: float = 0.05) -> tuple[float, float]:
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


def negation_analyze(cfg: dict) -> None:
    d = ROOT / cfg["human_eval"]["dir"]
    key = json.loads((d / "negation_key.json").read_text())
    lab = pd.read_csv(d / "negation_labels_blank.csv", keep_default_na=False, dtype=str)
    results = json.loads((ROOT / cfg["harness"]["results_dir"] / f"run_{key['run_id']}.json").read_text())
    runs = {e["example_id"]: e for e in results["examples"]}
    k1 = {i["example_id"]: i for i in key["items"]}
    lab["example_id"] = lab.example_id.astype(int)
    lab["human"] = lab.human_label.str.strip().str.lower().map({"yes": 1, "no": 0})
    lab["unsure"] = lab.human_unsure.str.strip().str.lower().eq("yes")
    lab["category"] = lab.example_id.map(NEGATION_CATEGORY)
    assert lab.category.notna().all(), "every sampled item needs a pre-assigned category"
    used = lab[lab.human.notna() & ~lab.unsure]
    n, confirm = len(used), int((used.human == 0).sum())   # judge said "not relevant" on every valid run
    out, add = [], None
    out = []
    add = out.append
    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    N, n_pool = key["population_size"], key["sampled_from"]

    add(f"NEGATION REVIEW  run {key['run_id']}  seed {key['seed']}")
    add(f"  population: {key['population']} (N={N}); sampled {len(lab)} at random from the {n_pool} not labeled in round 1")
    add(f"  labels: no={int((lab.human == 0).sum())} yes={int((lab.human == 1).sum())} unsure={int(lab.unsure.sum())}  (used n={n})")
    add("  Every item in this population has ESCI=Exact and the judge saying 'not relevant' on all 5 runs, so:")
    add("    human 'no'  -> human CONFIRMS the judge (and ESCI's Exact label looks wrong)")
    add("    human 'yes' -> the judge is genuinely wrong here (ESCI right)")
    add("")
    lo_w, hi_w = _wilson(confirm, n)
    lo_c, hi_c = _clopper_pearson(confirm, n)
    add(f"JUDGE vs HUMAN on this stratum: {confirm}/{n} = {pct(confirm / n)}")
    add(f"  Wilson 95% CI [{pct(lo_w)}, {pct(hi_w)}]   Clopper-Pearson (exact) 95% CI [{pct(lo_c)}, {pct(hi_c)}]")
    add(f"  scaled to the {n_pool} unlabeled population members: about {confirm / n * n_pool:.0f} confirmed "
        f"(plausible range {lo_c * n_pool:.0f}-{hi_c * n_pool:.0f}); intervals ignore the finite-population "
        f"correction (n/N = {n / n_pool:.0%}), so they are slightly conservative")
    # sensitivity: add the round-1 pencil item (title-only unsure, 'yes' after seeing details)
    r1 = pd.read_csv(d / "human_labels.csv", keep_default_na=False)
    pen = r1[r1.example_id.isin(key["already_labeled_in_round_1"])]
    for _, r in pen.iterrows():
        add(f"  round-1 item {r.example_id}: title-only '{r.label_title_only}', after details '{r.label_with_details or '(same)'}'"
            f" -> not usable title-only; with its final label this would be {confirm}/{n + 1} confirmed, 1 judge error")
    add("")

    add("BY PRE-ASSIGNED CATEGORY (assigned by Claude from query+title before seeing your labels)")
    add(f"  {'category':52s} {'n':>2s}  human no (confirms judge)  human yes (judge wrong)")
    for c in ["C1", "C2", "C3", "C4"]:
        g = lab[lab.category == c]
        add(f"  {c} {CATEGORY_NAME[c]:49s} {len(g):2d}  {int((g.human == 0).sum()):13d}            {int((g.human == 1).sum()):13d}")
    add("")
    add("ITEMS (judge said 'not relevant' on all 5 runs for every row)")
    for _, r in lab.iterrows():
        e = runs[r.example_id]
        ok = [x for x in e["runs"] if x["status"] == "ok"]
        add(f"  [{r.example_id}] {r.category}  human={'unsure' if r.unsure else ('yes' if r.human == 1 else 'no')}")
        add(f"      query: {r['query']}")
        add(f"      title: {r.product_title}")
        add(f"      judge: {ok[0]['justification']}")
        if r.notes.strip():
            add(f"      note:  {r.notes.strip()}")
    add("")

    # what this implies for the headline slice number (Exact + negation: 43.7% against ESCI)
    r1 = pd.read_csv(d / "human_labels.csv", keep_default_na=False)
    df1, key1, _ = load_labeled(cfg)
    s2 = df1[(df1.stratum == "exact_negation") & df1.judge_run_agree_human.notna()].judge_run_agree_human.to_numpy()
    rng = np.random.default_rng(0)
    h44 = (used.human == 0).to_numpy(float)
    N2 = key1["strata"]["exact_negation"]["N"]
    point = (N2 * s2.mean() + N * h44.mean()) / (N2 + N)
    boot = np.array([(N2 * rng.choice(s2, len(s2)).mean() + N * rng.choice(h44, len(h44)).mean()) / (N2 + N) for _ in range(BOOT)])
    add("IMPLICATION FOR THE 'Exact + negation = 43.7% accuracy' FINDING (n=86 = 42 + 44)")
    add(f"  judge agreement with the HUMAN across the whole Exact+negation slice: about {pct(point)} "
        f"(bootstrap 95% CI [{pct(np.percentile(boot, 2.5))}, {pct(np.percentile(boot, 97.5))}]),")
    add("  versus 43.7% against ESCI. Built from 11 round-1 items (the 42) and these 15 (the 44), so wide.")
    text = "\n".join(out)
    (ROOT / cfg["harness"]["results_dir"] / "negation_review_analysis.txt").write_text(text + "\n")
    print(text)


# --- analysis of the returned labels -------------------------------------------------------

LABEL_MAP = {"yes": 1.0, "no": 0.0}   # "unsure" -> NaN: excluded from agreement, reported separately
BOOT = 10_000


def load_labeled(cfg: dict) -> tuple[pd.DataFrame, dict, dict]:
    d = ROOT / cfg["human_eval"]["dir"]
    key = json.loads((d / "key.json").read_text())
    labels = pd.read_csv(d / "human_labels.csv", keep_default_na=False)
    results = json.loads((ROOT / cfg["harness"]["results_dir"] / f"run_{key['run_id']}.json").read_text())
    ev = load_eval_set(cfg).set_index("example_id")
    df = pd.DataFrame(key["items"]).merge(labels, on="example_id", validate="one_to_one")
    df["query"] = ev.loc[df.example_id, "query"].values
    df["title"] = ev.loc[df.example_id, "product_title"].values
    df["human"] = df.label_title_only.map(LABEL_MAP)                       # NaN = unsure
    df["human_final"] = df.label_with_details.map(LABEL_MAP).fillna(df.human)  # details override
    judged_yes = df.judge_yes_frac
    for col in ("human", "human_final"):
        df[f"judge_run_agree_{col}"] = np.where(df[col] == 1, judged_yes, 1 - judged_yes)
        df[f"judge_run_agree_{col}"] = df[f"judge_run_agree_{col}"].where(df[col].notna())
        maj = df.judge_majority.map({True: 1.0, False: 0.0})
        df[f"judge_maj_agree_{col}"] = (maj == df[col]).astype(float).where(df[col].notna() & maj.notna())
        df[f"esci_agree_{col}"] = (df.truth == df[col]).astype(float).where(df[col].notna())
    maj = df.judge_majority.map({True: 1.0, False: 0.0})
    df["judge_maj_num"] = maj
    # judge-vs-ESCI per-example pass rate: known for every example, so it validates the weighting
    df["judge_esci_pass"] = np.where(df.truth == 1, judged_yes, 1 - judged_yes)
    return df, key, results


def estimate(df: pd.DataFrame, col: str, strata: dict, seed: int = 0) -> dict:
    """Stratified mean of `col` (items with NaN dropped) and bootstrap CI. Each stratum's mean is
    weighted by its population size N_h, i.e. each item by N_h / n_h (n_h = items actually used)."""
    rng = np.random.default_rng(seed)
    per, vals = {}, {}
    for name in STRATA:
        v = df.loc[(df.stratum == name) & df[col].notna(), col].to_numpy(float)
        vals[name] = v
        per[name] = {"n": len(v), "mean": float(v.mean()) if len(v) else float("nan")}
    names = [n for n in STRATA if len(vals[n])]
    N = np.array([strata[n]["N"] for n in names], float)
    means = np.array([per[n]["mean"] for n in names])
    overall = float((N * means).sum() / N.sum())
    boots = np.empty(BOOT)
    boot_per = {n: np.empty(BOOT) for n in names}
    for b in range(BOOT):
        bm = np.array([rng.choice(vals[n], len(vals[n])).mean() for n in names])
        boots[b] = (N * bm).sum() / N.sum()
        for n, m in zip(names, bm):
            boot_per[n][b] = m
    for n in names:
        per[n]["lo"], per[n]["hi"] = (float(x) for x in np.percentile(boot_per[n], [2.5, 97.5]))
    return {"overall": overall, "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)),
            "n": int(sum(len(v) for v in vals.values())), "per": per}


def weighted_kappa(df: pd.DataFrame, a: str, b: str, strata: dict) -> float:
    """Cohen's kappa from the stratum-weighted 2x2 table of two binary raters."""
    ok = df[df[a].notna() & df[b].notna()]
    n_h = ok.groupby("stratum").size()
    w = ok.stratum.map(lambda h: strata[h]["N"] / n_h[h])
    t = np.zeros((2, 2))
    for x, y, wi in zip(ok[a].astype(int), ok[b].astype(int), w):
        t[x, y] += wi
    t /= t.sum()
    po = t[0, 0] + t[1, 1]
    pe = (t[0].sum() * t[:, 0].sum()) + (t[1].sum() * t[:, 1].sum())
    return float((po - pe) / (1 - pe))


def analyze(cfg: dict) -> None:
    df, key, results = load_labeled(cfg)
    strata = key["strata"]
    runs = {e["example_id"]: e for e in results["examples"]}
    out = []
    add = out.append
    pct = lambda x: f"{100 * x:5.1f}%"  # noqa: E731
    fmt = lambda e: f"{pct(e['overall'])}  95% CI [{pct(e['lo'])}, {pct(e['hi'])}]  (items used: {e['n']})"  # noqa: E731

    add(f"HUMAN REVIEW  run {key['run_id']}  | {len(df)} items labeled blind from the title only")
    add(f"  human labels: yes={int((df.human == 1).sum())} no={int((df.human == 0).sum())} "
        f"unsure={int(df.human.isna().sum())} (unsure excluded from agreement; weights adjust to items used)")
    changed = df[df.label_with_details.isin(["yes", "no"])]
    add(f"  revised after seeing details: {len(changed)} items")
    add("")
    add("WEIGHTING CHECK  (the estimator should recover the known whole-set judge-vs-ESCI accuracy)")
    chk = estimate(df.assign(_all=df.judge_esci_pass), "_all", strata)
    pop = float(np.mean([np.mean([(r["is_relevant"] == bool(e["ground_truth_binary"])) for r in e["runs"] if r["status"] == "ok"]) for e in results["examples"]]))
    add(f"  estimate from the 45-item sample: {fmt(chk)}")
    add(f"  true value on all 2000 examples:  {pct(pop)}   -> {'inside the CI' if chk['lo'] <= pop <= chk['hi'] else 'OUTSIDE the CI (weighting problem)'}")
    add("")
    add("AGREEMENT WITH THE HUMAN LABEL (title only)   whole-set estimate = stratum-weighted")
    metrics = [("judge vs human (fraction of the 5 runs that agree)", "judge_run_agree_human"),
               ("judge vs human (majority vote of runs)", "judge_maj_agree_human"),
               ("ESCI label vs human", "esci_agree_human")]
    est = {}
    for title, col in metrics:
        est[col] = estimate(df, col, strata)
        add(f"  {title:52s} {fmt(est[col])}")
    k_j = weighted_kappa(df, "judge_maj_num", "human", strata)
    k_e = weighted_kappa(df.assign(esci=df.truth.astype(float)), "esci", "human", strata)
    add(f"  Cohen's kappa (weighted):  judge-majority vs human = {k_j:.2f}   ESCI vs human = {k_e:.2f}")
    add("")
    add("  per stratum   (n = items used; bootstrap 95% CI; strata are NOT equally hard by design)")
    add(f"    {'stratum':16s} {'N':>5s} {'n':>3s}   {'judge vs human':>26s}   {'ESCI vs human':>26s}")
    for name in STRATA:
        j, e = est["judge_run_agree_human"]["per"].get(name), est["esci_agree_human"]["per"].get(name)
        cell = lambda x: f"{pct(x['mean'])} [{pct(x['lo'])},{pct(x['hi'])}]" if x and x["n"] else "n/a"  # noqa: E731
        add(f"    {name:16s} {strata[name]['N']:5d} {j['n'] if j else 0:3d}   {cell(j):>26s}   {cell(e):>26s}")
    add("")
    add("SAME, using the human's FINAL label (after the optional details step)")
    for title, col in [("judge vs human-final", "judge_run_agree_human_final"), ("ESCI vs human-final", "esci_agree_human_final")]:
        add(f"  {title:52s} {fmt(estimate(df, col, strata))}")
    add("")

    # who is right when the judge and ESCI disagree?
    d = df[df.human.notna() & df.judge_maj_num.notna() & (df.judge_maj_num != df.truth)].copy()
    d["side"] = np.where(d.human == d.judge_maj_num, "human sides with JUDGE (label looks loose)", "human sides with ESCI (judge wrong)")
    add("WHEN THE JUDGE (majority) DISAGREES WITH ESCI, WHO DOES THE HUMAN SIDE WITH?")
    w = d.stratum.map(lambda h: strata[h]["N"] / (df[(df.stratum == h) & df.human.notna()].shape[0]))
    share = float((w * (d.human == d.judge_maj_num)).sum() / w.sum()) if len(d) else float("nan")
    add(f"  items where judge-majority != ESCI: {len(d)} of {int(df.human.notna().sum())}")
    for side, g in d.groupby("side"):
        add(f"    {side}: {len(g)}  (" + ", ".join(f"{k}={v}" for k, v in g.stratum.value_counts().items()) + ")")
    add(f"  stratum-weighted share of such disagreements where the human sides with the judge: {pct(share)}")
    add("")
    en = df[(df.stratum == "exact_negation") & df.human.notna()]
    add("NEGATION HYPOTHESIS  (ESCI Exact + negation query, not always-wrong; n=%d)" % len(en))
    add(f"  human says relevant (title only): {int((en.human == 1).sum())}/{len(en)}; "
        f"judge-majority says relevant: {int((en.judge_maj_num == 1).sum())}/{len(en)}")
    aw = df[(df.stratum == "always_wrong") & df.human.notna()]
    add(f"ALWAYS-WRONG STRATUM (n={len(aw)}): human sides with ESCI {int((aw.human == aw.truth).sum())}, "
        f"with the judge {int((aw.human != aw.truth).sum())}")
    add("")

    def show(g: pd.DataFrame, header: str) -> None:
        add(header)
        for _, r in g.iterrows():
            e = runs[int(r.example_id)]
            ok = [x for x in e["runs"] if x["status"] == "ok"]
            add(f"  [{int(r.example_id)}] stratum={r.stratum}  ESCI={r.esci_label} ({'relevant' if r.truth else 'not relevant'})  "
                f"judge said relevant in {r.judge_yes_frac:.0%} of runs  human(title)={r.label_title_only} "
                f"human(details)={r.label_with_details or '(same)'}")
            add(f"      query: {r['query']}")
            add(f"      title: {r.title}")
            add(f"      judge:  {ok[0]['justification']}" if ok else "      judge: (no valid run)")
            if r.notes.strip():
                add(f"      note:   {r.notes.strip()}")

    loose = df[df.human.notna() & df.judge_maj_num.notna() & (df.judge_maj_num != df.truth) & (df.human == df.judge_maj_num)]
    show(loose, f"CANDIDATE LOOSE-GROUND-TRUTH CASES ({len(loose)}): judge 'failed' vs ESCI, human agrees with the judge")
    add("")
    wrong = df[df.human.notna() & df.judge_maj_num.notna() & (df.judge_maj_num != df.truth) & (df.human == df.truth)]
    show(wrong, f"GENUINE JUDGE ERRORS ({len(wrong)}): judge disagrees with ESCI and the human sides with ESCI")
    add("")
    show(changed, f"ITEMS WHERE THE HUMAN CHANGED THEIR ANSWER AFTER SEEING DETAILS ({len(changed)})")
    add("")
    show(df[df.human.isna()], f"ITEMS THE HUMAN MARKED UNSURE ({int(df.human.isna().sum())})")

    text = "\n".join(out)
    (ROOT / cfg["harness"]["results_dir"] / "human_eval_analysis.txt").write_text(text + "\n")
    print(text)


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Relevance Labeling</title>
<style>
:root{--bg:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--mute:#898781;--line:#e1e0d9;--blue:#2a78d6;--blue-ink:#fff;--soft:#eef4fc}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0d0d0d;--card:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--mute:#898781;--line:#2c2c2a;--blue:#3987e5;--blue-ink:#0b0b0b;--soft:#1f2a38}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:760px;margin:0 auto;padding:20px 16px 60px}
header{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap}
h1{font-size:18px;margin:0}.count{color:var(--ink2);font-variant-numeric:tabular-nums}
.bar{height:4px;background:var(--line);border-radius:2px;margin:10px 0 16px;overflow:hidden}.bar i{display:block;height:100%;background:var(--blue);width:0}
details.how{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;margin-bottom:14px;color:var(--ink2);font-size:14px}
details.how summary{cursor:pointer;color:var(--ink);font-weight:600}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px}
.lbl{font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:var(--mute);margin:0 0 4px}
.query{font-size:22px;font-weight:650;margin:0 0 16px;overflow-wrap:anywhere}.title{font-size:17px;margin:0 0 18px;overflow-wrap:anywhere}
.q{color:var(--ink2);margin:0 0 10px}
.row{display:flex;gap:10px;flex-wrap:wrap}
button{font:inherit;cursor:pointer;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:8px;padding:9px 18px;min-height:44px}
button:hover{border-color:var(--mute)}button.sel{background:var(--blue);color:var(--blue-ink);border-color:var(--blue)}
kbd{font:12px ui-monospace,Menlo,monospace;border:1px solid var(--line);border-radius:4px;padding:0 5px;margin-left:6px;color:var(--mute)}
button.sel kbd{color:inherit;border-color:currentColor;opacity:.8}
#details{display:none;margin-top:20px;padding-top:16px;border-top:1px solid var(--line)}#details.on{display:block}
.det{background:var(--soft);border-radius:8px;padding:10px 12px;margin:0 0 12px;font-size:14px;color:var(--ink2);max-height:220px;overflow:auto;white-space:pre-wrap;overflow-wrap:anywhere}
textarea{width:100%;font:inherit;font-size:14px;background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:8px 10px;min-height:56px;resize:vertical;margin-top:6px}
nav{display:flex;justify-content:space-between;gap:10px;margin-top:16px;flex-wrap:wrap}
.tools{display:flex;gap:10px;flex-wrap:wrap;margin-top:22px;align-items:center;color:var(--ink2);font-size:14px}
.warn{color:#b3261e;font-size:13px;margin-top:8px;display:none}
</style></head><body><main>
<header><h1>Relevance labeling <span style="color:var(--mute);font-weight:400">(blind)</span></h1><div class="count" id="count"></div></header>
<div class="bar"><i id="prog"></i></div>
<details class="how"><summary>How to label (read once)</summary>
<p>For each item answer <b>exactly the question the AI judge was asked</b>, from the product <b>title only</b>:
<i>"Would a shopper searching for this query consider this product a directly relevant result they might buy?"</i></p>
<p>Judge it as a shopper would: honor what the query states (exclusions like "without X", sizes, colors, quantities).
Don't try to guess any dataset label; there's no right answer to match, just your honest call. Use <b>Unsure</b> only when
you genuinely can't tell. After answering, details appear; the second step is optional: only change your answer if the details
actually would. Notes are most useful when the item feels ambiguous, or when "close but not exact" is something you'd buy anyway.</p>
<p>Your answers save in this browser as you go. <b>Export CSV</b> when you finish (or any time) and put the file at <code>labeling/human_labels.csv</code>.</p></details>
<div class="card" id="card">
 <p class="lbl">Search query</p><p class="query" id="query"></p>
 <p class="lbl">Product title</p><p class="title" id="title"></p>
 <p class="q">Would a shopper searching for this query consider this product a directly relevant result they might buy?</p>
 <div class="row" id="a1">
  <button data-v="yes">Yes <kbd>Y</kbd></button><button data-v="no">No <kbd>N</kbd></button><button data-v="unsure">Unsure <kbd>U</kbd></button></div>
 <div id="details">
  <p class="lbl">Product details (optional step)</p>
  <div class="det" id="det"></div>
  <p class="q">Having seen the details, would your answer change?</p>
  <div class="row" id="a2"><button data-v="">No change</button><button data-v="yes">Now: Yes</button><button data-v="no">Now: No</button></div>
 </div>
 <textarea id="notes" placeholder="Notes (optional): ambiguous? a substitute you'd buy anyway? a bad listing?"></textarea>
</div>
<nav><button id="prev">&larr; Previous</button><button id="next">Next &rarr;</button></nav>
<div class="tools"><button id="export">Export CSV</button><button id="reset">Reset all</button><span id="status"></span></div>
<div class="warn" id="warn">Browser storage is unavailable here, so progress will NOT be saved. Export before closing.</div>
</main>
<script>
const ITEMS = __DATA__;
const KEY = "relevance-labels-v1";
let store = {}, i = 0, canSave = true;
try { store = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { canSave = false; }
if (!canSave) document.getElementById("warn").style.display = "block";
const $ = id => document.getElementById(id);
function save() { try { localStorage.setItem(KEY, JSON.stringify(store)); } catch (e) { canSave = false; $("warn").style.display = "block"; } }
function rec(id) { return store[id] || (store[id] = {a1: "", a2: "", notes: ""}); }
function done() { return ITEMS.filter(it => (store[it.id] || {}).a1).length; }
function render() {
  const it = ITEMS[i], r = rec(it.id);
  $("query").textContent = it.query; $("title").textContent = it.title;
  $("count").textContent = `Item ${i + 1} of ${ITEMS.length}  ·  ${done()} labeled`;
  $("prog").style.width = (100 * done() / ITEMS.length) + "%";
  document.querySelectorAll("#a1 button").forEach(b => b.classList.toggle("sel", b.dataset.v === r.a1));
  document.querySelectorAll("#a2 button").forEach(b => b.classList.toggle("sel", b.dataset.v === r.a2));
  $("details").classList.toggle("on", !!r.a1);
  const parts = [];
  if (it.brand) parts.push("Brand: " + it.brand);
  if (it.bullets) parts.push("Bullet points:\n" + it.bullets);
  if (it.description) parts.push("Description:\n" + it.description);
  $("det").textContent = parts.join("\n\n") || "(no additional details in the data for this product)";
  $("notes").value = r.notes;
  $("prev").disabled = i === 0; $("next").disabled = i === ITEMS.length - 1;
}
function answer(v) { rec(ITEMS[i].id).a1 = v; save(); render(); }
function go(d) { i = Math.min(ITEMS.length - 1, Math.max(0, i + d)); render(); window.scrollTo(0, 0); }
document.querySelectorAll("#a1 button").forEach(b => b.onclick = () => answer(b.dataset.v));
document.querySelectorAll("#a2 button").forEach(b => b.onclick = () => { rec(ITEMS[i].id).a2 = b.dataset.v; save(); render(); });
$("notes").oninput = e => { rec(ITEMS[i].id).notes = e.target.value; save(); };
$("prev").onclick = () => go(-1); $("next").onclick = () => go(1);
document.addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA" || e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === "y") answer("yes"); else if (k === "n") answer("no"); else if (k === "u") answer("unsure");
  else if (e.key === "ArrowRight") go(1); else if (e.key === "ArrowLeft") go(-1);
});
function q(s) { return '"' + String(s).replace(/"/g, '""') + '"'; }
$("export").onclick = () => {
  const rows = ["example_id,label_title_only,label_with_details,notes"];
  ITEMS.forEach(it => { const r = store[it.id] || {}; rows.push([it.id, r.a1 || "", r.a2 || "", q(r.notes || "")].join(",")); });
  const url = URL.createObjectURL(new Blob([rows.join("\n") + "\n"], {type: "text/csv"}));
  const a = document.createElement("a"); a.href = url; a.download = "human_labels.csv"; a.click(); URL.revokeObjectURL(url);
  $("status").textContent = `Exported ${done()} of ${ITEMS.length} labels.`;
};
$("reset").onclick = () => { if (confirm("Erase all your labels in this browser?")) { store = {}; save(); i = 0; render(); } };
const first = ITEMS.findIndex(it => !(store[it.id] || {}).a1); if (first > 0) i = first;
render();
</script></body></html>
"""

if __name__ == "__main__":
    main()
