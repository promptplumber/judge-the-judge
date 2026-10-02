"""Unit tests for src/stats.py. Expected values are computed BY HAND in the comments and
hard-coded in the assertions; none is derived by re-running the formula under test.

z = 1.96 and t quantiles are the standard table values (t_{0.975, 4} = 2.776445,
t_{0.975, 2} = 4.302653). Tolerances are loose enough only to absorb z = 1.959964 vs 1.96.
"""
import numpy as np
import pytest
from scipy import stats as sps

from src import stats

TOL = 1e-4


def matrix(*rows):
    return np.array(rows, dtype=float)


# ---------------------------------------------------------------------------------------------
# All-pass: 10 examples, all 5/5
# ---------------------------------------------------------------------------------------------

def test_all_pass_everything_is_one():
    m = np.ones((10, 5))
    assert stats.mean_pass_rate(m) == 1.0
    assert stats.pass_at_k(m, 3)[0] == 1.0
    assert stats.pass_pow_k(m, 5)[0] == 1.0


def test_all_pass_normal_ci_collapses_to_a_point():
    # Every p_i = 1 -> deviations are all 0 -> SE = 0 -> CI = [1, 1].
    # This is the documented flaw of the normal CI: zero variance is not certainty.
    ci = stats.normal_ci(np.ones((10, 5)))
    assert (ci.mean, ci.se, ci.lo, ci.hi) == (1.0, 0.0, 1.0, 1.0)


def test_all_pass_t_ci_also_collapses():
    # Every run-level rate R_j = 1 -> SD = 0 -> CI = [1, 1].
    ci = stats.t_ci(np.ones((10, 5)))
    assert (ci.lo, ci.hi) == (1.0, 1.0)


def test_all_pass_wilson_does_not_collapse_when_k_is_1():
    # Wilson, 10/10, z = 1.96: z^2 = 3.8416.
    #   denom = 1 + 3.8416/10 = 1.38416
    #   center = (1 + 3.8416/20) / 1.38416 = 1.19208 / 1.38416 = 0.86122
    #   half = 1.96 * sqrt(0 + 3.8416/400) / 1.38416 = 1.96 * 0.098 / 1.38416 = 0.13878
    #   -> [0.72244, 1.0]   (the textbook lower bound for 10/10 is ~0.722)
    ci = stats.normal_ci(np.ones((10, 1)))
    assert ci.method == "wilson"
    assert ci.lo == pytest.approx(0.7224, abs=1e-3)
    assert ci.hi == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------------------------
# Known p_i: N = 4 examples, K = 5 runs
#   A: 1 1 1 1 1   p = 1.0
#   B: 1 1 1 1 0   p = 0.8
#   C: 1 1 1 0 0   p = 0.6
#   D: 1 0 0 0 0   p = 0.2
# ---------------------------------------------------------------------------------------------

KNOWN = matrix([1, 1, 1, 1, 1], [1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0])


def test_per_example_pass_rates():
    np.testing.assert_allclose(stats.per_example_pass_rate(KNOWN), [1.0, 0.8, 0.6, 0.2])


def test_mean_pass_rate():
    # (1.0 + 0.8 + 0.6 + 0.2) / 4 = 2.6 / 4 = 0.65
    assert stats.mean_pass_rate(KNOWN) == pytest.approx(0.65)


def test_normal_ci_matches_hand_calc():
    # deviations from 0.65: +0.35, +0.15, -0.05, -0.45
    # squares: 0.1225, 0.0225, 0.0025, 0.2025  -> sum = 0.35
    # variance = 0.35 / (N-1 = 3) = 0.116667; SD = 0.341565
    # SE = SD / sqrt(4) = 0.170783
    # CI = 0.65 +/- 1.96 * 0.170783 = 0.65 +/- 0.334734 = [0.315266, 0.984734]
    ci = stats.normal_ci(KNOWN)
    assert ci.method == "normal-over-examples"
    assert ci.se == pytest.approx(0.170783, abs=TOL)
    assert ci.lo == pytest.approx(0.315266, abs=TOL)
    assert ci.hi == pytest.approx(0.984734, abs=TOL)


def test_t_ci_matches_hand_calc():
    # Column sums over the 4 examples: [4, 3, 3, 2, 1] -> run rates R = [1.00, 0.75, 0.75, 0.50, 0.25]
    # R_bar = 3.25 / 5 = 0.65  (same as p_hat, as it must be for a complete matrix)
    # deviations: +0.35, +0.10, +0.10, -0.15, -0.40; squares: 0.1225, 0.01, 0.01, 0.0225, 0.16 -> 0.325
    # variance = 0.325 / (K-1 = 4) = 0.08125; SD = 0.285044; SE = SD / sqrt(5) = 0.127475
    # half-width = t_{0.975,4} * SE = 2.776445 * 0.127475 = 0.353930
    # CI = [0.296070, 1.003930] -> clipped at 1.0
    np.testing.assert_allclose(stats.run_level_pass_rates(KNOWN), [1.0, 0.75, 0.75, 0.5, 0.25])
    ci = stats.t_ci(KNOWN)
    assert ci.mean == pytest.approx(0.65)
    assert ci.se == pytest.approx(0.127475, abs=TOL)
    assert ci.lo == pytest.approx(0.296070, abs=TOL)
    assert ci.hi == 1.0


def test_t_ci_is_wider_than_normal_ci_lower_bound_here():
    # Sanity on the intuition: with only K = 5 runs the t quantile (2.78) is much bigger than 1.96.
    assert stats.t_ci(KNOWN).lo < stats.normal_ci(KNOWN).lo


# ---------------------------------------------------------------------------------------------
# pass@k and pass^k: one example with c = 3 successes out of K = 5
#   pass@k  = 1 - C(5-3, k) / C(5, k)        pass^k = C(3, k) / C(5, k)
#   k=1: pass@1 = 1 - 2/5  = 0.6             pass^1 = 3/5   = 0.6
#   k=2: pass@2 = 1 - 1/10 = 0.9             pass^2 = 3/10  = 0.3
#   k=3: pass@3 = 1 - 0/10 = 1.0             pass^3 = 1/10  = 0.1   (C(2,3) = 0)
# ---------------------------------------------------------------------------------------------

C3_OF_5 = matrix([1, 1, 1, 0, 0])


@pytest.mark.parametrize("k, at_k, pow_k", [(1, 0.6, 0.6), (2, 0.9, 0.3), (3, 1.0, 0.1)])
def test_pass_k_single_example_c3_of_5(k, at_k, pow_k):
    assert stats.pass_at_k(C3_OF_5, k)[0] == pytest.approx(at_k)
    assert stats.pass_pow_k(C3_OF_5, k)[0] == pytest.approx(pow_k)


def test_pass_k_equals_k_reduces_to_ever_and_always():
    # c = 3 of 5, k = 5: pass@5 = 1 - C(2,5)/C(5,5) = 1 ; pass^5 = C(3,5)/1 = 0
    assert stats.pass_at_k(C3_OF_5, 5)[0] == 1.0
    assert stats.pass_pow_k(C3_OF_5, 5)[0] == 0.0


def test_pass_k_averages_over_examples():
    # c = 3, 5, 0 (K = 5), k = 3:
    #   pass@3 = (1.0 + 1.0 + 0.0) / 3 = 0.666667   [c=0: 1 - C(5,3)/C(5,3) = 0]
    #   pass^3 = (0.1 + 1.0 + 0.0) / 3 = 0.366667   [c=5: C(5,3)/C(5,3) = 1]
    m = matrix([1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0])
    assert stats.pass_at_k(m, 3)[0] == pytest.approx(2 / 3)
    assert stats.pass_pow_k(m, 3)[0] == pytest.approx(1.1 / 3)


def test_pass_at_1_equals_mean_pass_rate():
    assert stats.pass_at_k(KNOWN, 1)[0] == pytest.approx(stats.mean_pass_rate(KNOWN))
    assert stats.pass_pow_k(KNOWN, 1)[0] == pytest.approx(0.65)


def test_pass_k_rejects_k_larger_than_runs():
    with pytest.raises(ValueError):
        stats.pass_at_k(C3_OF_5, 6)


# ---------------------------------------------------------------------------------------------
# Paired delta: N = 5 examples, K = 5 runs each
#   p_A = [0.2, 0.4, 0.6, 0.8, 1.0]   p_B = [0.4, 0.6, 0.6, 1.0, 1.0]
#   d   = [0.2, 0.2, 0.0, 0.2, 0.0]
# ---------------------------------------------------------------------------------------------

def from_p(ps, k=5):
    """A matrix whose row i has round(p_i * k) leading successes."""
    return np.array([[1.0] * round(p * k) + [0.0] * (k - round(p * k)) for p in ps])


def test_paired_delta_significant_matches_hand_calc():
    # mean d = 0.6 / 5 = 0.12
    # deviations: +0.08, +0.08, -0.12, +0.08, -0.12; squares: 0.0064 x3 + 0.0144 x2 = 0.048
    # variance = 0.048 / 4 = 0.012; SD = 0.109545; SE = SD / sqrt(5) = 0.048990
    # CI = 0.12 +/- 1.96 * 0.048990 = 0.12 +/- 0.096020 = [0.023980, 0.216020] -> excludes 0
    a = from_p([0.2, 0.4, 0.6, 0.8, 1.0])
    b = from_p([0.4, 0.6, 0.6, 1.0, 1.0])
    d = stats.paired_delta(a, b)
    assert d.delta == pytest.approx(0.12)
    assert d.se == pytest.approx(0.048990, abs=TOL)
    assert d.lo == pytest.approx(0.023980, abs=TOL)
    assert d.hi == pytest.approx(0.216020, abs=TOL)
    assert d.significant is True


def test_paired_delta_noise_is_not_significant():
    # d = [+0.2, -0.2, +0.2, -0.2, 0.0] -> mean 0; squares: 0.04 x4 = 0.16; variance = 0.16/4 = 0.04
    # SD = 0.2; SE = 0.2 / sqrt(5) = 0.089443; CI = +/- 1.96 * 0.089443 = +/- 0.175308 -> straddles 0
    a = from_p([0.4, 0.6, 0.4, 0.6, 0.6])
    b = from_p([0.6, 0.4, 0.6, 0.4, 0.6])
    d = stats.paired_delta(a, b)
    assert d.delta == pytest.approx(0.0, abs=1e-12)
    assert d.se == pytest.approx(0.089443, abs=TOL)
    assert (d.lo, d.hi) == (pytest.approx(-0.175308, abs=TOL), pytest.approx(0.175308, abs=TOL))
    assert d.significant is False


def test_paired_delta_sign_is_b_minus_a():
    a, b = from_p([0.2, 0.4, 0.6]), from_p([0.6, 0.8, 1.0])
    assert stats.paired_delta(a, b).delta == pytest.approx(0.4)
    assert stats.paired_delta(b, a).delta == pytest.approx(-0.4)


def test_paired_delta_rejects_misaligned_rows():
    with pytest.raises(ValueError):
        stats.paired_delta(from_p([0.2, 0.4, 0.6]), from_p([0.2, 0.4]))


# ---------------------------------------------------------------------------------------------
# Welch's t on run-level pass rates: N = 10 examples, K = 3 runs per arm
#   A column sums [6, 7, 8] -> R_A = [0.6, 0.7, 0.8]
#   B column sums [8, 9, 10] -> R_B = [0.8, 0.9, 1.0]
# ---------------------------------------------------------------------------------------------

def from_column_sums(sums, n=10):
    return np.array([[1.0 if i < s else 0.0 for s in sums] for i in range(n)])


def test_welch_matches_hand_calc():
    # means 0.7 and 0.9 -> Delta = 0.2
    # var_A = (0.01 + 0 + 0.01) / 2 = 0.01 ; var_B = 0.01
    # SE = sqrt(0.01/3 + 0.01/3) = sqrt(0.0066667) = 0.081650
    # t = 0.2 / 0.081650 = 2.44949
    # dof = (0.0066667)^2 / ((0.0033333)^2/2 + (0.0033333)^2/2) = 4.4444e-5 / 1.1111e-5 = 4.0
    # CI = 0.2 +/- t_{0.975,4} * SE = 0.2 +/- 2.776445 * 0.081650 = 0.2 +/- 0.226700 = [-0.026700, 0.426700]
    # CI contains 0 -> not significant at 5%
    w = stats.welch_t_test(from_column_sums([6, 7, 8]), from_column_sums([8, 9, 10]))
    assert w.delta == pytest.approx(0.2)
    assert w.se == pytest.approx(0.081650, abs=TOL)
    assert w.t == pytest.approx(2.44949, abs=TOL)
    assert w.dof == pytest.approx(4.0)
    assert w.lo == pytest.approx(-0.026700, abs=TOL)
    assert w.hi == pytest.approx(0.426700, abs=TOL)
    assert w.significant is False


def test_welch_p_value_agrees_with_scipy():
    ra, rb = np.array([0.6, 0.7, 0.8]), np.array([0.8, 0.9, 1.0])
    w = stats.welch_t_test(from_column_sums([6, 7, 8]), from_column_sums([8, 9, 10]))
    assert w.p_value == pytest.approx(sps.ttest_ind(rb, ra, equal_var=False).pvalue, abs=1e-9)


def test_welch_needs_two_runs_per_arm():
    with pytest.raises(ValueError):
        stats.welch_t_test(from_column_sums([6]), from_column_sums([8, 9]))


# ---------------------------------------------------------------------------------------------
# Failures (NaN) are excluded, never counted as wrong
# ---------------------------------------------------------------------------------------------

def test_nan_runs_are_excluded_not_counted_as_wrong():
    # Row 1: [1, 1, NaN, 0, 1] -> 3 successes / 4 valid runs = 0.75 (NOT 3/5 = 0.6)
    m = matrix([1, 1, np.nan, 0, 1])
    assert stats.per_example_pass_rate(m)[0] == pytest.approx(0.75)
    # pass@2 with n = 4, c = 3: 1 - C(1,2)/C(4,2) = 1 ; pass^2 = C(3,2)/C(4,2) = 3/6 = 0.5
    assert stats.pass_at_k(m, 2)[0] == pytest.approx(1.0)
    assert stats.pass_pow_k(m, 2)[0] == pytest.approx(0.5)


def test_example_with_no_valid_runs_is_dropped():
    # [1,1,1] -> p=1.0 ; [NaN,NaN,NaN] -> dropped ; [0,0,0] -> p=0.0  => p_hat = 0.5 over 2 examples
    m = matrix([1, 1, 1], [np.nan, np.nan, np.nan], [0, 0, 0])
    assert stats.mean_pass_rate(m) == pytest.approx(0.5)


def test_pass_k_skips_examples_with_fewer_than_k_valid_runs():
    # Row 1 has 5 valid runs (c=5) -> pass^3 = 1.  Row 2 has only 2 valid runs -> can't answer k=3.
    m = matrix([1, 1, 1, 1, 1], [1, 1, np.nan, np.nan, np.nan])
    value, n_used = stats.pass_pow_k(m, 3)
    assert (value, n_used) == (1.0, 1)


def test_rejects_values_that_are_not_0_1_nan():
    with pytest.raises(ValueError):
        stats.mean_pass_rate(matrix([1, 2, 0]))


# ---------------------------------------------------------------------------------------------
# Adapter from a harness results dict
# ---------------------------------------------------------------------------------------------

def _results(examples, k=3):
    return {"metadata": {"k": k}, "examples": examples}


def _ex(eid, truth, runs):
    return {"example_id": eid, "ground_truth_binary": truth,
            "runs": [{"run": i, "status": s, "is_relevant": v} for i, (s, v) in enumerate(runs)]}


def test_to_correct_matrix_maps_agreement_and_failures():
    res = _results([
        _ex(10, 1, [("ok", True), ("ok", False), ("parse_failure", None)]),
        _ex(20, 0, [("ok", False), ("infra_failure", None), ("ok", True)]),
    ])
    ids, m = stats.to_correct_matrix(res)
    assert ids == [10, 20]
    np.testing.assert_array_equal(m, [[1, 0, np.nan], [1, np.nan, 0]])


def test_align_returns_common_examples_in_the_same_order():
    a = _results([_ex(1, 1, [("ok", True)] * 3), _ex(2, 0, [("ok", False)] * 3), _ex(3, 1, [("ok", True)] * 3)])
    b = _results([_ex(3, 1, [("ok", False)] * 3), _ex(1, 1, [("ok", True)] * 3)])
    common, ma, mb = stats.align(a, b)
    assert common == [1, 3]
    np.testing.assert_array_equal(ma, [[1, 1, 1], [1, 1, 1]])
    np.testing.assert_array_equal(mb, [[1, 1, 1], [0, 0, 0]])
