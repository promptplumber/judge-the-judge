# Findings: is the judge trustworthy, and is the ground truth?

Run under review: `20261002T033605Z_v1_gpt-5.6-luna` (2,000 examples × K=5, prompt v1, title-only judge).
Raw analysis output: `results/human_eval_analysis.txt`. Labels: `labeling/human_labels.csv`.

## The short version

Against the ESCI label the judge scores **72.6%** (95% CI over examples 70.7-74.4%; majority-class baseline 55.4%).
Against a careful human reading of the same title it agrees **~91%**. In a 45-item blind review, when the judge
and ESCI disagreed, the human sided with the judge about three times in four. So a large part of the "27% error"
is not the judge being wrong. But the picture is less clean than that headline suggests, for three reasons
spelled out below: the sample is small, the labeler is one person, and the cases where the judge "wins" are two
quite different things (a genuinely loose Substitute label, versus an Exact label on a product that doesn't
match the query at all).

## Method

- **Sample (45 items, labeled blind from the title only, in shuffled order, no ESCI label and no judge verdict shown):**
  three disjoint strata that partition the 2,000 examples, with population size N and items sampled n:
  wrong-on-all-5-runs (N=465, n=15), ESCI-Exact + negation query and not already always-wrong (N=42, n=12),
  and everything else, drawn at random (N=1,493, n=18).
- **Labeler:** a single annotator (the project owner), answering the judge's own question: *would a shopper
  searching for this query consider this product a directly relevant result they might buy?* Unsure was allowed
  (3 items, excluded from agreement). A second, optional step showed product details and let the labeler revise.
- **Estimation:** because the strata are deliberately uneven, whole-set numbers weight each stratum by its
  population size N (each item by N/n). CIs are stratified bootstrap (10,000 resamples). **Weighting check:** the
  same estimator applied to the known judge-vs-ESCI per-example pass rate gives 71.7% [63.4, 76.8] against the
  true whole-set value of 72.6%, i.e. the weighting recovers a number we can verify.

## Results (title-only human label; weighted to the 2,000-example set)

| Comparison | Agreement | 95% CI | Cohen's kappa |
|---|---|---|---|
| Judge vs human (mean over the 5 runs) | **90.6%** | [83.8, 96.6] | |
| Judge vs human (majority vote of runs) | 93.2% | [87.6, 98.3] | 0.86 |
| ESCI label vs human | 78.8% | [68.2, 87.3] | 0.58 |
| Judge vs ESCI (all 2,000, for reference) | 72.6% | [70.7, 74.4] | |

Using the labeler's final answer (after seeing product details) instead: judge 88.8% [80.9, 95.3], ESCI 76.0% [62.9, 87.2].

| Stratum | N | n used | Judge vs human | ESCI vs human |
|---|---|---|---|---|
| Always wrong vs ESCI | 465 | 14 | 71.4% [50.0, 92.9] | 28.6% [7.1, 50.0] |
| Exact + negation, not always-wrong | 42 | 11 | 94.5% [83.6, 100] | 90.9% [72.7, 100] |
| Everything else (random) | 1,493 | 17 | 96.5% [90.6, 100] | 94.1% [82.4, 100] |

When the judge's majority verdict disagreed with ESCI (15 of 42 labeled items), the human sided with the **judge in
11** and with **ESCI in 4**; stratum-weighted, the human sides with the judge in 76% of such disagreements.

**How to read this.** The judge agrees with the human far more than ESCI does (kappa 0.86 vs 0.58), so ESCI-as-ground-truth
is the weaker instrument here, and 72.6% understates the judge. The random stratum, which is the unbiased part of
the sample, shows both raters near 95%; the disagreement is concentrated in the always-wrong stratum. That is the
expected shape, and it also means the whole-set estimate leans on only 17 random items, hence the wide intervals.

## Loose ground truth: 11 cases where the judge "failed" and the human agrees with the judge

These split into two groups that should not be conflated.

### A. Reasonable closeness the binary collapse throws away (3 cases)
The label says not relevant, but the human (and judge) would treat the product as a result a shopper might buy.

1. **"cosplay mask"** (ESCI: Substitute; judge said relevant in 5/5 runs; human: yes)
   *"YangYong Kitsune Fox Mask for Christmas Costume, Animal Cosplay Kabuki Half Face Cat Masks Masquerade Party"*
   Judge: "a fox-themed half-face mask explicitly intended for animal cosplay and costume parties."
2. **"key locks for kids"** (ESCI: Substitute; judge 5/5 relevant; human: yes)
   *"Guaishou Mix Color Style Mini Heart Luggage Locks Padlocks Archaize Lock with Keys Pack of 7pcs"*
   Judge: "a pack of mini heart-shaped padlocks operated with keys, which could directly satisfy a search for locks intended for kids."
3. **"toddler books moo moo cow"** (ESCI: Irrelevant; judge 5/5 relevant; human: yes)
   *"Mr. Brown Can Moo, Can You : Dr. Seuss's Book of Wonderful Noises (Bright and Early Board Books)"*
   Judge: "a toddler-appropriate board book centered on the sound 'moo'." Labeled **Irrelevant**, which is hard to defend.

This is the kind of looseness I expected going in: binary collapse of S/C/I into "not relevant" penalizes a judge for
calling a plausible product relevant.

### B. ESCI "Exact" on a product that does not match the query (8 cases)
The label says Exact; the judge said relevant in 0-20% of runs (0% in seven of the eight); the human also says no.
This is a different phenomenon: the label itself looks wrong, not merely generous. Examples:

4. **"togas boys"** (ESCI: Exact; judge relevant 0/5; human: no)
   *"Ravensburger - Gravitrax Kit de Inicio, Juego STEM innovador y educativo, Edad recomendada 8+, ..."* (a Spanish-language
   marble-run kit). Not a toga, not a garment. This is not a judgment call.
5. **"mtm gun case"** (Exact; judge 0/5; human: no) *"MTM AC50C-40 50-Caliber Ammo Can, Black"*: judge: "an ammunition storage can rather than a case designed to hold a gun."
6. **"vape pods device"** (Exact; judge 0/5; human: no) *"IT'S A SKIN Decal Vinyl Wrap for Smok Novo Pod System Vape Sticker Sleeve Cover..."*: a sticker for a device, not the device.
7. **"under armour volleyball spandex"** (Exact; judge 0/5; human: no) *"Under Armour Adult Strive 2.0 Volleyball Knee Pad"*: wrong product type.

The other four (**"buckle cutout bodysuit"**, **"one wheel beach hoverboard"**, **"sofa table decorations for living room"**,
**"ice cream"**) have the same shape. Whether these are annotator error or an artifact of how the pre-joined
HuggingFace copy was assembled, **I cannot tell from this data** (the Spanish-language listing under a US locale in
#4 makes me suspicious of the join, but that is a suspicion, not a finding). The check would be to compare these
rows with the official ESCI `examples` and `products` files, joined on `(product_locale, product_id)`.

## Where the judge really is wrong (4 of the 15 disagreements)

| Example | Query / title | What happened |
|---|---|---|
| 635884 | "danny devito" / *Ruthless People* | Needs outside knowledge the title lacks. Judge: "does not mention Danny DeVito." |
| 77266 | "24 inch deep pocket king sheets" / "Extra Deep Pocket Sheets... King" | Judge refused to confirm a 24-inch fit from the title. Strict about an unverifiable detail. |
| 1342383 | "mercer bread knife millenia" / *Mercer Millennia Granton Edge Slicer* | Over-literal: a slicer is not the words "bread knife." |
| 1716670 | "ri" / *The Fellowship Of The Ring* | The only false positive: matched the prefix "ri" to "Ring." |

Three of four are the judge being stricter than a human on Exact matches, consistent with its 1,568 false negatives
vs 1,176 false positives over the full run. Two of the four involve information the title does not contain, which fits a
title-only limitation. Four cases is too few to call a pattern.

## The negation hypothesis: my first sample could not test it; a follow-up review did

The per-slice report showed Exact + negation queries at 43.7% accuracy (n=86). I guessed that the judge cannot
verify an exclusion ("without grip") from a title that doesn't mention one. **The first review does not settle it.**

- In the Exact + negation stratum (n=11 used) the judge said relevant in **11 of 11**, and the human said relevant in 10 of 11.
  The judge handles negation fine when it isn't already failing.
- But that stratum excludes always-wrong examples **by construction**, and 44 of the 86 Exact + negation examples
  are wrong on every run. They sit in the always-wrong stratum, which contributed only one negation item.
  That was my sampling-design mistake: the stratum meant to probe the hypothesis excluded exactly the cases that matter.
- The one relevant item supports the hypothesis weakly: *"0.5 lead pencil without grip"* / *uni Core Keeps Sharp Mechanical Pencil*,
  judge relevant 0/5 ("does not specify a 0.5 mm lead size or that it lacks a grip"); the labeler was **unsure** from the title,
  then **yes** after seeing the details ("product details confirms 0.5mm"). One item is an anecdote, not evidence.

### Follow-up: a targeted blind review of the 44

**Design.** I confirmed from the run data that 44 of the 86 Exact + negation examples are wrong on every valid run (all 44
have 5 valid runs). I drew 15 at random (seed 11) from the 43 of them not already labeled in round 1 (the pencil, 12931, was
excluded from the draw because it was already labeled: title-only *unsure*, *yes* after details). Same question, title only,
no ESCI label or judge verdict on the sheet, shuffled. For every item in this population ESCI says Exact and the judge said
"not relevant" on all 5 runs, so a human **no** confirms the judge and a human **yes** is a genuine judge error.

**Result.** The human confirmed the judge on **12 of 15 (80.0%)**; exact 95% CI **[51.9%, 95.7%]** (Wilson [54.8%, 93.0%]).
Scaled to the 43 unlabeled members, that is about 34 confirmed, plausibly anywhere from 22 to 41. Including the pencil with its
final label, 12 of 16 confirmed and 4 judge errors.

I sorted the 15 into four categories **from the query and title alone, before merging the human labels** (single rater, me;
the boundaries are subjective):

| Category | n | Human confirms judge | Judge wrong |
|---|---|---|---|
| C1 Title contradicts the exclusion (e.g. a floor lamp "with Adjustable Drum Shade" for *floor lamp without shade*; struts "with Mounts" for *not strut mount*; "Gel Polish" for *not gel*; "Stretchy String" for *not stretchy*) | 6 | 5 | 1 |
| C2 Title silent on the exclusion, so it cannot be verified from the title (the negation hypothesis) | 5 | 4 | 1 |
| C3 Not a real exclusion, or simply a different product | 3 | 3 | 0 |
| C4 Title affirmatively satisfies the exclusion | 1 | 0 | 1 |

**What this says.**

- **The hypothesis as stated is only weakly supported.** The judge was wrong because a title didn't mention an exclusion in 2
  reviewed items: the pencil and *coffee maker without pot* (12.5% of 16). That does not look like the main driver of the 43.7%.
- **Where the title is silent (C2) the human mostly agreed with the judge** (4 of 5): from the title alone they couldn't
  confirm the exclusion either. "Can't verify, so not relevant" is a stance the judge and the human share. ESCI's annotators saw
  the full listing and called these Exact, so whether the *label* is right depends on information neither of us was shown. These
  are "title insufficient", not shown to be judge errors and not shown to be label errors.
- **Where the title contradicts the exclusion (C1) the judge was right** in 5 of 6, and ESCI's Exact label does not survive a
  read of the title. These look like the group-B label problems above; I still cannot say whether that is annotator error or a
  data-assembly artifact.
- **My negation filter is contaminated.** In C3, *"waiting is not easy an elephant and piggie book"* and *"no u card game"* are
  book and game *titles* that happened to contain a negation keyword; they are not negation queries. At least 2 of 15 (13%) of this
  "negation" sample are false positives of the keyword regex, and the per-slice report's negation slice uses the same regex.
- **Genuine judge errors here (3 of 15):** the bedspread set (the title says "Comforter Bedding Cover" and "Quilt Set"; the human
  said yes), *coffee maker without pot*, and *futon frames ... without mattress* (the title says "Mattress Sold Separately"; the
  judge faulted a missing size instead).

**Implication for the headline slice.** Combining this with the 11 usable round-1 items from the other 42 Exact + negation
examples, the judge agrees with the human on about **87%** of the whole Exact + negation slice (bootstrap 95% CI [74%, 97%]),
versus 43.7% against ESCI. Agreement here counts a C2 "human no" as agreement.

**Bottom line, and how sure I am.** The direction is clear: most of the 44 are not the judge failing on negation. They are
label problems, title-insufficient cases, or not-really-negation queries. The magnitude is not: with 15 items I cannot rule out
that up to roughly half of the 44 are real judge errors (the exact interval's upper end is 95.7% confirmed, its lower end 51.9%).
I would not quote a precise judge-error rate for this slice.

## What the optional details step showed

Showing product details changed the labeler's answer on only 4 of 45 items (one no→yes, two unsure resolved, one reconfirmed).
Extra information rarely mattered, which weakly suggests that ESCI-vs-human disagreement is mostly not explained by ESCI
annotators having seen more than the title.

## Limitations (what this does not show)

- **One annotator, no inter-annotator agreement.** "Human" here means one careful person; their errors and idiosyncrasies are not measured.
- **Small sample.** 45 items; only 17 are random. The whole-set estimate has wide intervals (judge vs human could plausibly be 84-97%).
  Per-stratum cells have 11-17 items. Treat the strata as illustrations, not estimates.
- **Same information for human and judge.** Labeling from the title makes the comparison fair, but it also favours the judge:
  it is graded on what it was shown. ESCI annotators saw full listings.
- **Possible priming.** The labeler had read the per-slice report, including the negation guess, before labeling.
- **Unsure items (3) are excluded**, which assumes they are missing at random within a stratum.
- **Negation follow-up: n=15 (16 with the pencil) from N=44.** The exact 95% CI on "human confirms the judge" is 52-96%, which is consistent with anything from a majority-label-problem story to roughly half real judge errors.
- **Weaker blinding in the follow-up.** The labeler knew the follow-up population (every item is ESCI Exact and judge-negative) and had read this file's hypothesis, so a lean toward either confirming or contradicting the judge cannot be ruled out. The first round did not have this problem.
- **The C1-C4 categories are mine,** assigned by one rater from the query and title, with fuzzy boundaries (the bedspread set could be C1 or C2).
- **The negation filter is a keyword regex** and at least 2 of the 15 sampled items are not negation queries; any "negation" slice number inherits that.
- **The pencil was excluded from the follow-up draw** because it was already labeled, so the sample is a random draw from 43 plus one item that was itself randomly drawn in round 1, not a pure random draw from 44.
- **The judge's accuracy against the human is an in-sample number for prompt v1**; I did not tune on these labels, but they should not be reused as a test set for a v2 prompt without noting that I've now read the disagreements.

## What I would change

1. **Graded or S-aware relevance.** Treat Substitute as relevant (or score it separately) for query types like "cosplay mask", where a close alternative is a legitimate result. This is the direct fix for group A.
2. **Audit the Exact labels** against the official ESCI files before trusting any "judge failed on an Exact" number; group B suggests the always-wrong set is contaminated.
3. **Replace the keyword regex with a real negation detector**, then test a v2 prompt that sees bullet points/description with the paired-delta tool. The follow-up suggests that content would help in the title-silent cases, but those are a minority of the 44, so don't expect it to move the headline much.
4. **Second annotator** on the same items to put a ceiling on "human agreement", and more random items to tighten the whole-set number.
5. **Report accuracy against a cleaned label set** (human-reviewed always-wrong items), alongside the raw ESCI number, instead of replacing one with the other.
