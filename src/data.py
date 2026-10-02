"""Load and freeze the ESCI eval slice.

The slice is built once from the pinned HuggingFace revision and saved to
data/eval_set_v1.parquet; every later run reads that frozen file via load_eval_set().

Disk note: instead of datasets.load_dataset (which caches Arrow copies of every split) we
download the test-split parquet shards one at a time, filter each, and delete it, so peak
extra disk is ~one shard (~200 MB).
"""
import json
import shutil

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from src.config import ROOT, load_config

# ESCI label -> binary ground truth. v1 simplification: only Exact counts as relevant.
RELEVANT_LABELS = {"Exact"}

# product_text is a concatenation of the other text fields; skip it to keep the slice small.
SOURCE_COLUMNS = [
    "example_id", "query", "product_id", "esci_label",
    "product_title", "product_description", "product_bullet_point", "product_brand",
]
OUTPUT_COLUMNS = [
    "example_id", "query", "product_title", "product_description",
    "product_bullet_point", "product_brand", "esci_label", "ground_truth_binary",
]


def _test_shards(repo: str, revision: str) -> list[str]:
    files = HfApi().list_repo_files(repo, repo_type="dataset", revision=revision)
    return sorted(f for f in files if f.startswith("data/test-") and f.endswith(".parquet"))


def build_eval_set(cfg: dict | None = None) -> pd.DataFrame:
    """Download test shards one by one, filter, sample with the pinned seed, and freeze."""
    cfg = cfg or load_config()
    d = cfg["dataset"]
    raw_dir = ROOT / d["raw_dir"]
    out_path = ROOT / d["path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    filters = [("product_locale", "=", d["locale"]), ("small_version", "=", d["small_version"])]
    parts = []
    try:
        for shard in _test_shards(d["name"], d["revision"]):
            local = hf_hub_download(d["name"], shard, repo_type="dataset",
                                    revision=d["revision"], local_dir=raw_dir)
            parts.append(pq.read_table(local, columns=SOURCE_COLUMNS, filters=filters).to_pandas())
            print(f"{shard}: kept {len(parts[-1])} rows")
            (raw_dir / shard).unlink()  # free the disk before fetching the next shard
    finally:
        shutil.rmtree(raw_dir, ignore_errors=True)

    pool = pd.concat(parts, ignore_index=True)
    # Sort first so the sample doesn't depend on shard/row order, only on the seed.
    pool = pool.sort_values("example_id").reset_index(drop=True)
    n = min(d["sample_size"], len(pool))
    df = pool.sample(n=n, random_state=d["seed"]).sort_values("example_id").reset_index(drop=True)
    df["ground_truth_binary"] = df["esci_label"].isin(RELEVANT_LABELS).astype(int)
    df = df[OUTPUT_COLUMNS]

    df.to_parquet(out_path, index=False)
    meta = {
        "dataset": d["name"], "revision": d["revision"], "version": d["version"],
        "filters": {"locale": d["locale"], "small_version": d["small_version"], "split": d["split"]},
        "pool_size": len(pool), "sample_size": n, "seed": d["seed"],
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return df


def load_eval_set(cfg: dict | None = None) -> pd.DataFrame:
    """Return the frozen eval set. Build it first with `python -m src.data`."""
    cfg = cfg or load_config()
    path = ROOT / cfg["dataset"]["path"]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m src.data` to build it.")
    return pd.read_parquet(path)


if __name__ == "__main__":
    cfg = load_config()
    path = ROOT / cfg["dataset"]["path"]
    if not path.exists():
        build_eval_set(cfg)
    df = load_eval_set(cfg)
    print("\nshape:", df.shape)
    print("\nclass balance (ground_truth_binary):")
    print(df["ground_truth_binary"].value_counts().rename({1: "relevant", 0: "not relevant"}))
    print(f"relevant share: {df['ground_truth_binary'].mean():.3f}")
    print("\nesci_label counts:")
    print(df["esci_label"].value_counts())
    print("\n5 example rows:")
    with pd.option_context("display.max_colwidth", 70, "display.width", 200):
        print(df[["example_id", "query", "product_title", "esci_label", "ground_truth_binary"]].head(5))
    print("\nfile:", path, f"({path.stat().st_size / 1e6:.1f} MB)")
