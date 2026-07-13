"""
Repackage CrediBench WebContent October/November/December 2024 CommonCrawl WET
snapshots into simple parquet shards, in the same format nanochat's dataloader
expects (see dev/repackage_data_reference.py for the original reference version).

- each shard is ~250M characters of text (roughly ~100MB after zstd compression)
- parquets are written with row group size of 1024
- the three monthly repos are interleaved and shuffled together

This is meant to be run locally to produce a base_data_credibench_2024q4/ directory
that nanochat/dataset.py's DATA_DIR can point at directly (no HF re-upload needed).

CrediBench notes (differ from the earlier FineWeb repackage):
- The data lives in three separate, *gated* HuggingFace repos (one per month).
  You must accept the terms on all three dataset pages AND have an authenticated
  token available (huggingface-cli login, or HF_TOKEN in the env) or the streaming
  reads will 401. See credibench_migration_notes.md.
- Files sit at the repo root with irregular names, plus a stray CSV with a
  different schema. We therefore select shards with an explicit "*wetcontent*"
  glob (this excludes e.g. oct_2024_domains_35.8M.csv) rather than a config name.
- The page text lives in the `wet_record_txt` column (type: string, so no decode
  needed). We still write it out under the column name "text" so the downstream
  dataloader is unchanged.
- Language is NOT filtered: we keep parity with the FineWeb build (which did no
  filtering). The schema exposes a `WARC_Identified_Content_Language` column, so
  to restrict to English later, add a `.filter(lambda d: d["WARC_Identified_Content_Language"] == "eng")`
  on the interleaved stream below (see credibench_migration_notes.md).

Resumable: progress (shards written + documents consumed so far) is checkpointed
to progress.json in the output dir after every shard. If interrupted, re-running
this script skips forward past the documents already consumed (the interleaved/
shuffled stream is deterministic given the fixed seed) instead of starting over.
"""
import os
import json
import time

from datasets import load_dataset, interleave_datasets
import pyarrow as pa
import pyarrow.parquet as pq

# -----------------------------------------------------------------------------
# CrediBench WebContent repos for Oct/Nov/Dec 2024 (gated; token required)
REPOS = [
    "Hussein-Abdallah/CrediBench-WebContent-Oct2024",  # Oct 2024
    "Hussein-Abdallah/CrediBench-WebContent-Nov2024",  # Nov 2024
    "Hussein-Abdallah/CrediBench-WebContent-Dec2024",  # Dec 2024
]
NUM_SHARDS = 170  # ~150 needed for GPT-2 capability pretraining, +20 padding
CHARS_PER_SHARD = 250_000_000
ROW_GROUP_SIZE = 1024

base_dir = os.environ.get("NANOCHAT_BASE_DIR")
output_dir = os.path.join(base_dir, "base_data_credibench_2024q4")
os.makedirs(output_dir, exist_ok=True)
progress_path = os.path.join(output_dir, "progress.json")

def load_progress():
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            return json.load(f)
    return {"shard_index": 0, "docs_consumed": 0}

def save_progress(shard_index, docs_consumed):
    # write to a temp file + rename so a crash mid-write can't corrupt progress.json
    tmp_path = progress_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump({"shard_index": shard_index, "docs_consumed": docs_consumed}, f)
    os.replace(tmp_path, progress_path)

progress = load_progress()
shard_index = progress["shard_index"]
docs_consumed = progress["docs_consumed"]
if shard_index >= NUM_SHARDS:
    print(f"Already have {shard_index}/{NUM_SHARDS} shards in {output_dir}, nothing to do.")
    raise SystemExit(0)

# Stream each monthly repo and interleave/shuffle them together. The "*wetcontent*"
# glob picks up the root-level parquet shards while excluding the stray domains CSV.
streams = [
    load_dataset(repo, data_files="*wetcontent*.parquet", split="train", streaming=True)
    for repo in REPOS
]
ds = interleave_datasets(streams, stopping_strategy="all_exhausted").shuffle(seed=42, buffer_size=10_000)
if docs_consumed > 0:
    print(f"Resuming: {shard_index}/{NUM_SHARDS} shards already written, skipping {docs_consumed} already-consumed documents...")
    ds = ds.skip(docs_consumed)

# -----------------------------------------------------------------------------
# Repackage into parquet files
shard_docs = []
shard_characters = 0
shards_written_this_run = 0
total_time_spent = 0
t0 = time.time()
for doc in ds:
    docs_consumed += 1
    text = doc["wet_record_txt"]  # CrediBench stores the extracted WET page text here (string)
    if not text:
        continue  # skip empty WET records
    shard_docs.append(text)
    shard_characters += len(text)
    collected_enough_chars = shard_characters >= CHARS_PER_SHARD
    docs_multiple_of_row_group_size = len(shard_docs) % ROW_GROUP_SIZE == 0
    if collected_enough_chars and docs_multiple_of_row_group_size:
        shard_path = os.path.join(output_dir, f"shard_{shard_index:05d}.parquet")
        shard_table = pa.Table.from_pydict({"text": shard_docs})
        pq.write_table(
            shard_table,
            shard_path,
            row_group_size=ROW_GROUP_SIZE,
            use_dictionary=False,
            compression="zstd",
            compression_level=3,
            write_statistics=False,
        )
        shard_index += 1
        save_progress(shard_index, docs_consumed)
        t1 = time.time()
        dt = t1 - t0  # time spent on this shard alone
        t0 = t1
        shards_written_this_run += 1
        total_time_spent += dt
        avg_time_per_shard = total_time_spent / shards_written_this_run
        remaining_shards = NUM_SHARDS - shard_index
        remaining_time_hours = (remaining_shards * avg_time_per_shard) / 3600
        print(
            f"Wrote {shard_path} ({shard_index}/{NUM_SHARDS}). "
            f"#documents: {len(shard_docs)} | #characters: {shard_characters} | "
            f"time: {dt:.1f}s | remaining: ~{remaining_time_hours:.2f}h"
        )
        shard_docs = []
        shard_characters = 0
        if shard_index >= NUM_SHARDS:
            break

print(f"Done: wrote {shard_index} shards to {output_dir}")
