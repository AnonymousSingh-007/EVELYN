# src/pipeline/batch_pipeline.py
#
# PURPOSE: Process your entire labeled dataset (phishing + benign CSVs)
# through build_graph() and save every graph to data/graphs/.
#
# Three fixes applied in this version:
#   1. Deduplicate by registered DOMAIN (not raw URL) before queuing —
#      PhishTank lists the same domain under multiple paths.
#   2. Node-budget enforcement lives in build_graph.py itself (hyperscaler
#      domains like google.com no longer produce 1000+ node graphs).
#   3. Dead-infrastructure benign domains (NXDOMAIN/NoAnswer) are flagged
#      usable_for_training=False after the run, not silently dropped —
#      keeps the audit trail intact for the paper's data section.

import csv
import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.pipeline.build_graph import build_graph
from src.pipeline.parse_url import parse_url


GRAPHS_DIR      = Path("data/graphs")
CHECKPOINT_FILE = Path("data/processed/checkpoint.json")
RESULTS_FILE    = Path("data/processed/batch_results.csv")

RESULT_FIELDNAMES = [
    "url", "domain", "label", "node_count", "edge_count",
    "truncated", "truncated_counts",
    "modules_succeeded", "modules_failed", "status", "error",
]


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
    Never raises — any exception is captured and reported as a row,
    so one bad domain can never kill the whole batch.
    """
    try:
        G = build_graph(url, label=label, verbose=False, save=True)
        return {
            "url":               url,
            "domain":            G.graph.get("domain", ""),
            "label":             label,
            "node_count":        G.number_of_nodes(),
            "edge_count":        G.number_of_edges(),
            "truncated":         G.graph.get("truncated", False),
            "truncated_counts":  json.dumps(G.graph.get("truncated_counts", {})),
            "modules_succeeded": "|".join(G.graph.get("modules_succeeded", [])),
            "modules_failed":    "|".join(G.graph.get("modules_failed", [])),
            "status":            "ok",
            "error":             "",
        }
    except Exception as e:
        return {
            "url": url, "domain": "", "label": label,
            "node_count": 0, "edge_count": 0,
            "truncated": False, "truncated_counts": "{}",
            "modules_succeeded": "", "modules_failed": "",
            "status": "error", "error": str(e),
        }


def _load_phishing_jobs(phishing_csv: str, max_phishing: int, seen_domains: set) -> list:
    """
    Loads phishing URLs from a PhishTank-style CSV, deduplicated by
    registered domain. Two URLs on the same domain (different paths)
    produce the SAME infrastructure graph, so only the first is kept.
    """
    jobs = []
    try:
        with open(phishing_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            added = 0
            for row in reader:
                if added >= max_phishing:
                    break
                url = row.get("url") or row.get("URL") or row.get("phish_url")
                if not url:
                    continue

                domain = parse_url(url.strip())["full_domain"]
                if not domain or domain in seen_domains:
                    continue

                seen_domains.add(domain)
                jobs.append((url.strip(), 1))
                added += 1

        print(f"  Loaded {added} unique phishing domains from {phishing_csv}")
    except FileNotFoundError:
        print(f"  ⚠ Phishing CSV not found: {phishing_csv}")
        print(f"    Download from Kaggle → data/raw/phishtank_2026.csv")

    return jobs


def _load_benign_jobs(benign_csv: str, max_benign: int, seen_domains: set) -> list:
    """
    Loads benign domains from a Tranco-style CSV, deduplicated against
    domains already queued by the phishing loader (in case of overlap).
    """
    jobs = []
    try:
        with open(benign_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            added = 0
            for row in reader:
                if added >= max_benign:
                    break
                domain = row.get("domain") or row.get("Domain")
                if not domain:
                    continue
                domain = domain.strip()
                if domain in seen_domains:
                    continue

                seen_domains.add(domain)
                jobs.append((f"https://{domain}", 0))
                added += 1

        print(f"  Loaded {added} unique benign domains")
    except FileNotFoundError:
        print(f"  ⚠ Benign CSV not found: {benign_csv}")
        print(f"    Run: python -m src.pipeline.fetch_benign 500")

    return jobs


def run_batch(phishing_csv:  str   = "data/raw/phishtank_2026.csv",
              benign_csv:    str   = "data/raw/benign_2026-06-18.csv",
              max_phishing:  int   = 500,
              max_benign:    int   = 500,
              max_workers:   int   = 4,
              delay_seconds: float = 1.0) -> None:
    """
    Runs the full build_graph pipeline over your labeled dataset.

    max_workers: concurrent URLs in flight. Keep at 4 — higher risks
    rate-limiting on the free APIs (WHOIS, ip-api.com, crt.sh, etc).

    delay_seconds: courtesy pause between submitting each job to the
    thread pool, so we don't fire 500 requests at once.
    """

    seen_domains = set()
    jobs = []
    jobs += _load_phishing_jobs(phishing_csv, max_phishing, seen_domains)
    jobs += _load_benign_jobs(benign_csv, max_benign, seen_domains)
    print(f"  Total unique jobs queued: {len(jobs)}")

    if not jobs:
        print("  No jobs to process. Exiting.")
        return

    # ── Skip already-processed domains (checkpoint resume) ─────────────
    done = load_checkpoint()
    jobs = [(url, lbl) for url, lbl in jobs if url not in done]
    print(f"  {len(done)} already done (checkpoint). {len(jobs)} remaining.\n")

    if not jobs:
        print("  Nothing left to process — all jobs already completed previously.")
        return

    # ── Set up results CSV (append mode, write header only if new) ─────
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_FILE.exists()
    results_f = open(RESULTS_FILE, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(results_f, fieldnames=RESULT_FIELDNAMES)
    if write_header:
        writer.writeheader()

    # ── Process with ThreadPoolExecutor ─────────────────────────────────
    print(f"  Starting batch with {max_workers} workers...\n")
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

    # ── Post-process: flag dead-infrastructure benign domains ──────────
    _flag_dead_benign_domains(RESULTS_FILE)

    print(f"\n  ✓ Batch complete.")
    print(f"  Results saved to: {RESULTS_FILE}")
    print(f"  Graphs saved to:  {GRAPHS_DIR}/")


def _flag_dead_benign_domains(results_file: Path) -> None:
    """
    Adds a 'usable_for_training' column to the results CSV.

    A row is marked UNUSABLE only if label == 0 (benign) AND DNS failed
    (NXDOMAIN / NoAnswer). Dead BENIGN domains are noise: there's no
    legitimate-infrastructure signal to learn from a domain that doesn't
    resolve. Dead PHISHING domains are deliberately KEPT and usable —
    a taken-down phishing domain still carries WHOIS/cert traces that
    support retrospective campaign attribution, which is a real-world
    use case for this system, not noise.
    """
    import pandas as pd

    df = pd.read_csv(results_file)

    def is_usable(row):
        if row["label"] == 0:
            failed = str(row.get("modules_failed", ""))
            if "dns:NXDOMAIN" in failed or "dns:NoAnswer" in failed:
                return False
        return True

    df["usable_for_training"] = df.apply(is_usable, axis=1)
    df.to_csv(results_file, index=False)

    excluded = int((~df["usable_for_training"]).sum())
    print(f"  Flagged {excluded} dead-infrastructure benign domain(s) as unusable_for_training")


# ── SELF-TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 58)
    print("  EVELYN — batch_pipeline()")
    print("=" * 58)
    run_batch(
        max_phishing=15,
        max_benign=15,
        max_workers=2,
        delay_seconds=1.5,
    )