# One-command entry points. Override the interpreter with PYTHON=python3.13 (install) or PY=python (the rest).
PY ?= .venv/bin/python
PYTHON ?= python3
RUN := results/run_20261002T033605Z_v1_gpt-5.6-luna.json
DEV := results/run_20261002T010507Z_v1_gpt-5.6-luna.json

.PHONY: install test data report demo reproduce check-readme

install:  ## create .venv and install requirements (needs Python >= 3.10 with lzma)
	$(PYTHON) -m venv .venv && .venv/bin/pip install -r requirements.txt

test:  ## unit tests for the statistics (no data, no API key)
	$(PY) -m pytest tests/test_stats.py -q

data:  ## build the frozen 2,000-pair eval slice (downloads ~715 MB one shard at a time; no API key)
	HF_HUB_DISABLE_XET=1 $(PY) -m src.data

report: data  ## reproduce every number in the README from the COMMITTED results (no API key, no cost)
	$(PY) -m src.report $(RUN)
	$(PY) -m src.human_eval analyze
	$(PY) -m src.human_eval negation-analyze

demo: data  ## NEW small run with your key: 200 examples x K=3 (~600 calls, about $0.11), then report it
	$(PY) -m src.harness --limit 200 --k 3
	$(PY) -m src.report

reproduce: data  ## NEW full run with your key: 2,000 x K=5 = 10,000 calls (about $1.80, ~45 min), then report it
	$(PY) -m src.harness
	$(PY) -m src.report

check-readme:  ## re-run the code and assert every number quoted in README.md matches its output
	$(PY) scripts/check_readme.py
