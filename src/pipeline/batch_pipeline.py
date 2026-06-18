# src/pipeline/batch_pipeline.py
#
# PURPOSE: Process your entire labeled dataset (phishing + benign CSVs)
# through build_graph() and save every graph to data/graphs/.
# This is the file that runs overnight and produces the full corpus
# your quantum walk, clustering, and evaluation code will consume.
#
# Key engineering decisions:
#   - ThreadPoolExecutor for concurrency: most of our time is spent
#     WAITING for network responses (DNS/WHOIS/etc), not doing computation.
#     Threads release the GIL during I/O, so we get real parallelism here.
#     We use threads (not processes) because our modules are I/O-bound.
#   - Checkpoint file: if the run is interrupted (and it WILL be — 1000
#     network-heavy jobs take a long time), we skip already-processed
#     domains on restart instead of starting over from zero.
#   - Per-graph error isolation: one domain throwing an exception
#     never kills the other 999.

import csv
import time
import json
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.pipeline.build_graph import build_graph


GRAPHS_DIR      = Path("data/graphs")
CHECKPOINT_FILE = Path("data/processed/checkpoint.json")
RESULTS_FILE    = Path("data/processed/batch_results.csv")


def load_checkpoint() -> set:
    """Returns a set of domain strings already successfully processed."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return set(json.load(f))
    return set()


def save_checkpoint(done: set) -> None:
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(list(done), f)


def process_one(url: str, label: int) -> dict:
    """
    Process a single URL. Designed to be called by ThreadPoolExecutor.
    Returns a result dict suitable for writing to the results CSV.
    """
    try:
        G = build_graph(url, label=label, verbose=False, save=True)
        return {
            "url":               url,
            "domain":            G.graph.get("domain", ""),
            "label":             label,
            "node_count":        G.number_of_nodes(),
            "edge_count":        G.number_of_edges(),
            "modules_succeeded": "|".join(G.graph.get("modules_succeeded", [])),
            "modules_failed":    "|".join(G.graph.get("modules_failed", [])),
            "status":            "ok",
            "error":             "",
        }
    except Exception as e:
        return {
            "url": url, "domain": "", "label": label,
            "node_count": 0, "edge_count": 0,
            "modules_succeeded": "", "modules_failed": "",
            "status": "error", "error": str(e),
        }


def run_batch(phishing_csv:  str  = "data/raw/phishtank_2026.csv",
              benign_csv:    str  = "data/raw/benign_2026-06-18.csv",
              max_phishing:  int  = 500,
              max_benign:    int  = 500,
              max_workers:   int  = 4,
              delay_seconds: float = 1.0) -> None:
    """
    Runs the full build_graph pipeline over your labeled dataset.

    max_workers: how many URLs to process concurrently. Keep this at
    4 to be a good citizen to the free APIs we depend on. Bumping to
    10+ risks getting rate-limited by WHOIS servers, ip-api.com, etc.

    delay_seconds: sleep between each submission to the thread pool.
    This is your rate-limit courtesy pause.
    """

    # ── Load dataset ───────────────────────────────────────────────────
    jobs = []   # list of (url, label) tuples

    # Load phishing URLs from your PhishTank CSV
    try:
        with open(phishing_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= max_phishing:
                    break
                # PhishTank CSV has a "url" column
                url = row.get("url") or row.get("URL") or row.get("phish_url")
                if url:
                    jobs.append((url.strip(), 1))   # label 1 = phishing
        print(f"  Loaded {len(jobs)} phishing URLs from {phishing_csv}")
    except FileNotFoundError:
        print(f"  ⚠ Phishing CSV not found: {phishing_csv}")
        print(f"    Download from Kaggle → data/raw/phishtank_2026.csv")

    # Load benign domains from Tranco CSV
    try:
        with open(benign_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= max_benign:
                    break
                domain = row.get("domain") or row.get("Domain")
                if domain:
                    jobs.append((f"https://{domain.strip()}", 0))   # label 0 = benign
        print(f"  Loaded benign URLs. Total jobs: {len(jobs)}")
    except FileNotFoundError:
        print(f"  ⚠ Benign CSV not found: {benign_csv}")
        print(f"    Run: python -m src.pipeline.fetch_benign 500")

    if not jobs:
        print("  No jobs to process. Exiting.")
        return

    # ── Skip already-processed domains ────────────────────────────────
    done = load_checkpoint()
    jobs = [(url, lbl) for url, lbl in jobs if url not in done]
    print(f"  {len(done)} already done (checkpoint). {len(jobs)} remaining.")

    # ── Set up results CSV ─────────────────────────────────────────────
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_FILE.exists()
    results_f = open(RESULTS_FILE, "a", newline="", encoding="utf-8")
    fieldnames = ["url", "domain", "label", "node_count", "edge_count",
                  "modules_succeeded", "modules_failed", "status", "error"]
    writer = csv.DictWriter(results_f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    # ── Process with ThreadPoolExecutor ───────────────────────────────
    print(f"\n  Starting batch with {max_workers} workers...\n")
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for url, label in jobs:
            future = executor.submit(process_one, url, label)
            futures[future] = url
            time.sleep(delay_seconds)   # rate-limit courtesy

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="  Building graphs", unit="url"):
            url = futures[future]
            result = future.result()
            writer.writerow(result)
            results_f.flush()
            done.add(url)
            save_checkpoint(done)

    results_f.close()
    print(f"\n  ✓ Batch complete.")
    print(f"  Results saved to: {RESULTS_FILE}")
    print(f"  Graphs saved to:  {GRAPHS_DIR}/")


# ── SELF-TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 58)
    print("  EVELYN — batch_pipeline()")
    print("=" * 58)
    run_batch(
        max_phishing=5,    # start with 5 each for test
        max_benign=5,
        max_workers=2,
        delay_seconds=1.5,
    )