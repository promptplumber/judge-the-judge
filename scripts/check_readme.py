"""Assert that the numbers quoted in README.md match what the code actually prints.

Re-runs the real commands (report, human-eval analysis, negation analysis, cost total, tests) and, for
each claim, checks that an anchored snippet appears in BOTH the live output and the README (whitespace
ignored). Run with:  make check-readme
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = "results/run_20261002T033605Z_v1_gpt-5.6-luna.json"
DEV = "results/run_20261002T010507Z_v1_gpt-5.6-luna.json"


def run(*args: str) -> str:
    p = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(f"command failed: {' '.join(args)}\n{p.stderr[-800:]}")
    return p.stdout


norm = lambda s: re.sub(r"\s+", "", s)  # noqa: E731

out = {
    "report": run("-m", "src.report", RUN),
    "compare": run("-m", "src.report", DEV, "--compare", RUN),
    "human": run("-m", "src.human_eval", "analyze"),
    "neg": run("-m", "src.human_eval", "negation-analyze"),
    "costs": run("-m", "src.costs"),
    "tests": run("-m", "pytest", "tests/test_stats.py", "-q"),
}
readme = norm((ROOT / "README.md").read_text())

# (source, snippet that must appear in that live output, snippet that must appear in the README)
claims = [
    ("report", "meanpassrate72.6%", "**72.6%**(always"),
    ("report", "baseline55.4%", "baseline:55.4%"),
    ("report", "[70.7%,74.4%]", "[70.7%,74.4%]"),
    ("report", "[72.3%,72.8%]", "[72.3%,72.8%]"),
    ("report", "0.7270,0.7240,0.7279,0.7249,0.7240", "0.7270,0.7240,0.7279,0.7249,0.7240"),
    ("report", "parsefailures=2infrafailures=0", "2parsefailures,0infrafailures"),
    ("report", "k=1:pass@k72.6%pass^k72.6%", "|1|72.6%|72.6%|"),
    ("report", "k=2:pass@k74.7%pass^k70.4%", "|2|74.7%|70.4%|"),
    ("report", "k=3:pass@k75.7%pass^k69.2%", "|3|75.7%|69.2%|"),
    ("report", "k=4:pass@k76.3%pass^k68.3%", "|4|76.3%|68.3%|"),
    ("report", "k=5:pass@k76.7%pass^k67.7%", "|5|76.7%|67.7%|"),
    ("report", "Exactn=89264.8%[61.9%,67.8%]", "|ESCIExact|892|64.8%|[61.9%,67.8%]|"),
    ("report", "Substituten=67074.4%[71.2%,77.5%]", "|ESCISubstitute|670|74.4%|[71.2%,77.5%]|"),
    ("report", "Complementn=10273.7%[65.5%,81.9%]", "|ESCIComplement|102|73.7%|[65.5%,81.9%]|"),
    ("report", "Irrelevantn=33689.1%[85.9%,92.3%]", "|ESCIIrrelevant|336|89.1%|[85.9%,92.3%]|"),
    ("report", "Exact,negationn=8643.7%[33.6%,53.8%]", "|86|43.7%|[33.6%,53.8%]|"),
    ("report", "groundtruthrelevant(Exact)28921568", "1,568falsenegatives"),
    ("report", "groundtruthnotrelevant(S/C/I)11764362", "1,176falsepositives"),
    ("report", "precision71.1%", "precision71.1%"),
    ("report", "recall64.8%", "recall64.8%"),
    ("report", "specificity78.8%", "specificity78.8%"),
    ("report", "wrongoneveryrun465", "465ofthe2,000examplesarewrongoneveryrun"),
    ("compare", "delta+2.33pts95%CI[-2.24,+6.91]", "+2.33pts,CI[-2.24,+6.91]"),
    ("human", "90.6%95%CI[83.8%,96.6%]", "**90.6%**|[83.8%,96.6%]"),
    ("human", "(majorityvoteofruns)93.2%95%CI[87.6%,98.3%]", "93.2%|[87.6%,98.3%]"),
    ("human", "ESCIlabelvshuman78.8%95%CI[68.2%,87.3%]", "**78.8%**|[68.2%,87.3%]"),
    ("human", "judge-majorityvshuman=0.86ESCIvshuman=0.58", "|0.86|"),
    ("human", "judge-majorityvshuman=0.86ESCIvshuman=0.58", "**0.58**|"),
    ("human", "estimatefromthe45-itemsample:71.7%95%CI[63.4%,76.8%]", "**71.7%**[63.4%,76.8%]"),
    ("human", "truevalueonall2000examples:72.6%", "**72.6%**."),
    ("human", "itemswherejudge-majority!=ESCI:15of42", "(15of42labeleditems)"),
    ("human", "humansideswithESCI(judgewrong):4", "thehumansidedwiththejudgein11andwithESCIin4"),
    ("human", "humansideswithJUDGE(labellooksloose):11", "thehumansidedwiththejudgein11andwithESCIin4"),
    ("neg", "12/15=80.0%", "12of15(80%)"),
    ("neg", "Clopper-Pearson(exact)95%CI[51.9%,95.7%]", "[51.9%,95.7%]"),
    ("neg", "(N=44)", "fromthe44"),
    ("neg", "about87.1%(bootstrap95%CI[74.2%,96.6%])", "about87%(bootstrap95%CI[74.2%,96.6%];"),
    ("neg", "C1titlecontradictstheexclusion651", "|6|5|1|"),
    ("neg", "C2titlesilentontheexclusion(unverifiable)541", "|5|4|1|"),
    ("neg", "C3notarealexclusion/differentproduct330", "|3|3|0|"),
    ("neg", "C4titlesatisfiestheexclusion101", "|1|0|1|"),
    ("tests", "29passed", "29unittests"),
]

failures = []
checked = 0
for src, live, doc in claims:
    checked += 1
    if norm(live) not in norm(out[src]):
        failures.append(f"LIVE OUTPUT ({src}) lacks: {live}")
    if norm(doc) not in readme:
        failures.append(f"README lacks: {doc}")

# totals that need arithmetic rather than a substring
total = float(re.search(r"TOTAL\s+\d+\s+\d+\s+\d+\s+([\d.]+)", out["costs"]).group(1))
checked += 1
if f"${total:.2f}" != "$2.40" or "$2.40" not in (ROOT / "README.md").read_text():
    failures.append(f"project cost: code says ${total:.2f}, README must say $2.40")

# sample design quoted in the README (sampled counts, from the answer key)
strata = json.loads((ROOT / "labeling" / "key.json").read_text())["strata"]
want = {"always_wrong": (465, 15), "exact_negation": (42, 12), "random_rest": (1493, 18)}
for name, (N, n) in want.items():
    checked += 1
    if (strata[name]["N"], strata[name]["n"]) != (N, n):
        failures.append(f"key.json {name}: {strata[name]} != N={N}, n={n}")
for snippet in ["15itemswrongonallfiveruns(outof465)", "12Exact+negationitemsthatweren'talways-wrong(outof42)",
                "18drawnatrandomfromtheremaining1,493"]:
    checked += 1
    if norm(snippet) not in readme:
        failures.append(f"README lacks: {snippet}")

if failures:
    print("README CHECK FAILED:\n  " + "\n  ".join(failures))
    raise SystemExit(1)
print(f"README check passed: {checked} claims verified against live output.")
