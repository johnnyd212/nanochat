"""
Throwaway sanity check: record the WARC_Identified_Content_Language field for a
few samples from each of the three CrediBench WebContent 2024 Q4 repos.

Mirrors dev/repackage_credibench_2024q4.py's load path (streaming + the
"*wetcontent*" glob that skips the stray domains CSV). 

Usage: python dev/inspect_credibench_lang.py
Output is written to OUT_PATH 
"""
from datasets import load_dataset

REPOS = [
    "Hussein-Abdallah/CrediBench-WebContent-Oct2024",  # Oct 2024
    "Hussein-Abdallah/CrediBench-WebContent-Nov2024",  # Nov 2024
    "Hussein-Abdallah/CrediBench-WebContent-Dec2024",  # Dec 2024
]
FIELD = "WARC_Identified_Content_Language"
N = 3  # samples per repo
OUT_PATH = "credibench_lang_samples.txt"

with open(OUT_PATH, "w") as out:
    for repo in REPOS:
        out.write(f"=== {repo} ===\n")
        # streaming so we don't download the whole (gated) repo; glob skips the CSV
        ds = load_dataset(repo, data_files="*wetcontent*.parquet", split="train", streaming=True)
        for i, doc in enumerate(ds.take(N)):
            # .get (not select_columns) so a name mismatch shows the real value/None
            # instead of erroring inside the loader
            out.write(f"  sample {i}: {FIELD} = {doc.get(FIELD)!r}\n")
        out.flush()

print(f"Wrote language samples to {OUT_PATH}")
