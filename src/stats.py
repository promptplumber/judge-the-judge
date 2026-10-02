"""Uncertainty-aware metrics for the relevance judge. Pure functions, no I/O (except
`to_correct_matrix`, which only reshapes an already-loaded results dict).

Input convention: a `correct` matrix of shape (N examples, K runs).
  1.0 = the judge's verdict agreed with ground truth on that run
  0.0 = it disagreed
  NaN = no valid verdict (parse or infra failure). These are EXCLUDED, never counted as wrong:
        a timeout is not the judge being wrong (design principle 6).
Examples with no valid run at all are dropped from every metric.

Every interval is a 95% interval unless `conf` says otherwise, and is clipped to [0, 1].
"""
from dataclasses import dataclass
from math import comb, sqrt

import numpy as np
from scipy import stats as sps


@dataclass(frozen=True)
class Interval:
    mean: float
    lo: float
    hi: float
    se: float
    n: int          # number of units the interval is built on (examples, or runs)
    method: str


# --- helpers -------------------------------------------------------------------------------

def _matrix(correct) -> np.ndarray:
    m = np.asarray(correct, dtype=float)
    if m.ndim != 2:
        raise ValueError("correct must be 2-D (examples x runs)")
    if not np.all(np.isnan(m) | (m == 0) | (m == 1)):
        raise ValueError("correct may only contain 0, 1 or NaN")
    return m


def _valid_examples(m: np.ndarray) -> np.ndarray:
    """Drop examples that have no valid run at all."""
    return m[(~np.isnan(m)).sum(axis=1) > 0]


def _z(conf: float) -> float:
    return float(sps.norm.ppf(0.5 + conf / 2))


def _clip(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


# --- 1 & 2: pass rates ---------------------------------------------------------------------

def per_example_pass_rate(correct) -> np.ndarray:
    """p_i = (valid runs where the judge agreed with ground truth) / (valid runs), one per example.

    With a complete matrix this is exactly agreements / K. Examples with no valid run get NaN
    (and are dropped by the functions below)."""
    m = _matrix(correct)
    with np.errstate(invalid="ignore"):
        valid = (~np.isnan(m)).sum(axis=1)
        return np.where(valid > 0, np.nansum(m, axis=1) / np.where(valid > 0, valid, 1), np.nan)


def mean_pass_rate(correct) -> float:
    """p_hat = mean of p_i over examples = the judge's accuracy against ground truth.

    Every example gets equal weight regardless of how many of its runs were valid. Note this is
    accuracy against ESCI's *binary-collapsed* labels, so it inherits their looseness (see results/FINDINGS.md)."""
    p = per_example_pass_rate(correct)
    return float(np.nanmean(p))


# --- 3: example-level CI (generalised performance) -----------------------------------------

def wilson_ci(successes: float, n: int, conf: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion.

    Assumes n independent Bernoulli trials with a common success probability. Unlike the Wald
    interval it does not collapse to zero width at p_hat = 0 or 1 (10/10 gives ~[0.72, 1.0],
    not [1, 1]), which is why it is the right choice when K = 1."""
    if n < 1:
        raise ValueError("n must be >= 1")
    z, p = _z(conf), successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return Interval(p, _clip(center - half), _clip(center + half),
                    sqrt(p * (1 - p) / n), n, "wilson")


def normal_ci(correct, conf: float = 0.95) -> Interval:
    """CI for generalised performance, treating EXAMPLES as the independent units.

    K = 1 (one column): Wilson interval on the example-level proportion.
    K > 1: SE = sqrt(sum((p_i - p_hat)^2) / (N - 1)) / sqrt(N); CI = p_hat +/- z * SE.

    Assumptions (and why they are only approximately true here): the normal interval treats the
    N examples as an independent random sample from the population we care about. ESCI is a
    curated benchmark, not random production traffic, and several pairs share a query, so
    examples are neither random nor fully independent and this interval is probably too narrow
    for 'how would the judge do in production'. It does correctly propagate which examples are
    easy or hard. Known flaw: if every p_i is identical (e.g. all 1.0) SE = 0 and the interval
    collapses to a point. Zero observed variance is not certainty; it just means a bad estimate."""
    m = _valid_examples(_matrix(correct))
    n = len(m)
    if n < 2:
        raise ValueError("need at least 2 examples")
    if m.shape[1] == 1:
        return wilson_ci(float(np.nansum(m)), int(np.sum(~np.isnan(m))), conf)
    p = per_example_pass_rate(m)
    p_hat = float(p.mean())
    se = float(sqrt(np.sum((p - p_hat) ** 2) / (n - 1)) / sqrt(n))
    z = _z(conf)
    return Interval(p_hat, _clip(p_hat - z * se), _clip(p_hat + z * se), se, n, "normal-over-examples")


# --- 4: run-level CI (benchmark performance) -----------------------------------------------

def run_level_pass_rates(correct) -> np.ndarray:
    """R_1..R_K: the pass rate of each complete run over all examples that had a valid verdict."""
    m = _valid_examples(_matrix(correct))
    with np.errstate(invalid="ignore"):
        r = np.nanmean(m, axis=0)
    return r[~np.isnan(r)]


def t_ci(correct, conf: float = 0.95) -> Interval:
    """Student's t interval treating each COMPLETE RUN as one observation.

    R_1..R_K are the run-level pass rates, R_bar their mean, SE = SD / sqrt(K) with SD the sample
    SD (ddof = 1), CI = R_bar +/- t_{K-1} * SE.

    Why this is the right frame for a curated eval set: it makes no claim that the examples are a
    random sample of anything. The benchmark is fixed; the only randomness left is the judge's own
    run-to-run sampling, and each run is an independent repetition of exactly that. What it does
    NOT tell you: how the result would change on different examples. It measures the stability of
    the judge on THIS set only. With few runs the t quantile is large (K = 5 -> 2.78), so the
    interval is wide, and with K = 2 it is enormous; K needs to be at least ~5 to be useful."""
    r = run_level_pass_rates(correct)
    k = len(r)
    if k < 2:
        raise ValueError("need at least 2 runs")
    mean = float(r.mean())
    se = float(r.std(ddof=1) / sqrt(k))
    half = float(sps.t.ppf(0.5 + conf / 2, k - 1)) * se
    return Interval(mean, _clip(mean - half), _clip(mean + half), se, k, "t-over-runs")


# --- 5: pass@k and pass^k ------------------------------------------------------------------

def pass_at_k_example(n: int, c: int, k: int) -> float:
    """P(at least one success among k runs drawn without replacement from n runs, c successes)
    = 1 - C(n-c, k) / C(n, k)."""
    if not 1 <= k <= n:
        raise ValueError("need 1 <= k <= n")
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_pow_k_example(n: int, c: int, k: int) -> float:
    """P(all k runs drawn without replacement from n runs are successes) = C(c, k) / C(n, k)."""
    if not 1 <= k <= n:
        raise ValueError("need 1 <= k <= n")
    return comb(c, k) / comb(n, k)


def _pass_k(correct, k: int, per_example) -> tuple[float, int]:
    m = _valid_examples(_matrix(correct))
    vals = []
    for row in m:
        valid = row[~np.isnan(row)]
        if len(valid) >= k:  # an example with fewer than k valid runs can't answer "k tries"
            vals.append(per_example(len(valid), int(valid.sum()), k))
    if not vals:
        raise ValueError(f"no example has at least k={k} valid runs")
    return float(np.mean(vals)), len(vals)


def pass_at_k(correct, k: int) -> tuple[float, int]:
    """Mean over examples of pass@k, i.e. CAPABILITY: the chance the judge gets an example right
    at least once in k tries. Returns (value, n_examples_used). k = K gives the plain fraction of
    examples that were ever right; k = 1 gives mean accuracy.

    Assumes the K runs are exchangeable draws from the same per-example distribution. Caveat:
    these metrics are cleanest for a deterministic, validated rater. Here 'success' means
    agreeing with ground truth, and the judge itself is stochastic, so pass@k also mixes in
    label noise: a 'failure' may be a loose label, not a judge error."""
    return _pass_k(correct, k, pass_at_k_example)


def pass_pow_k(correct, k: int) -> tuple[float, int]:
    """Mean over examples of pass^k, i.e. CONSISTENCY: the chance the judge gets an example right
    on ALL k tries. k = K gives the fraction of examples right on every run. Same assumptions and
    caveats as pass_at_k. A large gap between pass@k and pass^k means success depends on lucky
    sampling. Note this measures consistently-agreeing-with-the-label, not self-consistency: an
    example the judge always gets wrong has pass^k = 0 yet is perfectly self-consistent."""
    return _pass_k(correct, k, pass_pow_k_example)


# --- 6: significance of a delta ------------------------------------------------------------

@dataclass(frozen=True)
class Delta:
    delta: float            # B - A
    lo: float
    hi: float
    se: float
    n: int
    significant: bool       # CI excludes 0
    method: str
    p_value: float | None = None
    dof: float | None = None
    t: float | None = None


def paired_delta(correct_a, correct_b, conf: float = 0.95) -> Delta:
    """Paired per-example delta between runs A and B (rows MUST be the same examples, in the same
    order; use `align` to guarantee that).

    d_i = p_Bi - p_Ai, Delta = mean(d_i), SE = SD(d_i) / sqrt(N), CI = Delta +/- z * SE. Significant
    if the CI excludes 0.

    Why pairing: example difficulty is shared by both runs and cancels in d_i, so the SE reflects
    only the change, which is far tighter than comparing two independent accuracies. Assumes the
    d_i are independent across examples (same caveat as normal_ci) and N is large enough for the
    normal approximation. It is conditional on this example set and says nothing on whether the
    change is a quality gain or just a different tradeoff (look at FP vs FN separately)."""
    pa, pb = per_example_pass_rate(_matrix(correct_a)), per_example_pass_rate(_matrix(correct_b))
    if len(pa) != len(pb):
        raise ValueError("A and B must have the same examples (same rows)")
    keep = ~np.isnan(pa) & ~np.isnan(pb)
    d = (pb - pa)[keep]
    n = len(d)
    if n < 2:
        raise ValueError("need at least 2 paired examples")
    mean = float(d.mean())
    se = float(d.std(ddof=1) / sqrt(n))
    z = _z(conf)
    lo, hi = mean - z * se, mean + z * se
    return Delta(mean, lo, hi, se, n, significant=not (lo <= 0 <= hi), method="paired-normal")


def welch_t_test(correct_a, correct_b, conf: float = 0.95) -> Delta:
    """Welch's t-test on RUN-LEVEL pass rates (benchmark-level alternative to paired_delta).

    Treats R_A1..R_AK and R_B1..R_BK as two independent samples with possibly unequal variances:
    Delta = mean(R_B) - mean(R_A), SE = sqrt(v_A/K_A + v_B/K_B), with Welch-Satterthwaite dof.
    Assumes runs are independent repetitions. It ignores example pairing, so it is less powerful
    than paired_delta, and with ~5 runs per arm it has very little power: a real improvement of a
    point or two will usually NOT reach significance here. It also only captures judge-sampling
    noise, not example-sampling noise."""
    ra, rb = run_level_pass_rates(correct_a), run_level_pass_rates(correct_b)
    if len(ra) < 2 or len(rb) < 2:
        raise ValueError("need at least 2 runs in each arm")
    va, vb = ra.var(ddof=1) / len(ra), rb.var(ddof=1) / len(rb)
    se = sqrt(va + vb)
    delta = float(rb.mean() - ra.mean())
    if se == 0:
        raise ValueError("zero variance in both arms; t-test undefined")
    dof = (va + vb) ** 2 / (va**2 / (len(ra) - 1) + vb**2 / (len(rb) - 1))
    t = delta / se
    p = float(2 * sps.t.sf(abs(t), dof))
    half = float(sps.t.ppf(0.5 + conf / 2, dof)) * se
    return Delta(delta, delta - half, delta + half, se, len(ra) + len(rb),
                 significant=p < 1 - conf, method="welch", p_value=p, dof=float(dof), t=float(t))


# --- results-file adapter ------------------------------------------------------------------

def to_correct_matrix(results: dict) -> tuple[list[int], np.ndarray]:
    """Turn a harness results dict into (example_ids, correct matrix). 'ok' runs become 1/0
    (agreed / disagreed with ground truth); parse and infra failures become NaN."""
    k = results["metadata"]["k"]
    ids, rows = [], []
    for ex in results["examples"]:
        row = np.full(k, np.nan)
        for r in ex["runs"]:
            if r["status"] == "ok":
                row[r["run"]] = float(r["is_relevant"] == bool(ex["ground_truth_binary"]))
        ids.append(ex["example_id"])
        rows.append(row)
    return ids, np.vstack(rows)


def align(results_a: dict, results_b: dict) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Correct matrices for the examples common to both runs, in the same order, for paired_delta."""
    ids_a, ma = to_correct_matrix(results_a)
    ids_b, mb = to_correct_matrix(results_b)
    common = sorted(set(ids_a) & set(ids_b))
    ia, ib = {e: i for i, e in enumerate(ids_a)}, {e: i for i, e in enumerate(ids_b)}
    return common, ma[[ia[e] for e in common]], mb[[ib[e] for e in common]]
