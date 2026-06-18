# src/pipeline/fetch_benign.py
#
# PURPOSE: Build the NEGATIVE class for your dataset — legitimate,
# non-phishing domains. Every phishing detection paper needs this,
# because a classifier that flags EVERYTHING as phishing gets 100%
# recall and is useless. Reviewers will ask about your false positive
# rate on the very first read, and you cannot answer that question
# without a benign dataset to test against.
#
# We use the Tranco list — the current academic standard for "top
# legitimate domains" research datasets, specifically designed to be
# more stable and harder to manipulate than older lists like Alexa
# (which Amazon discontinued) or Majestic (vulnerable to gaming).

import requests
import csv
from pathlib import Path
from datetime import datetime


# Tranco provides a stable, citable list ID system. This particular
# endpoint gives the current top-1-million list as a plain CSV.
TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"

RAW_DATA_DIR = Path("data/raw")


def fetch_benign(n: int = 1000, save: bool = True) -> dict:
    """
    Downloads the top N domains from the Tranco list as benign-class data.

    Returns:
    {
        "domains":     ["google.com", "facebook.com", ...],
        "count":       1000,
        "saved_path":  "data/raw/benign_2026-06-17.csv",
        "resolved":    True,
        "error":       None
    }
    """
    import zipfile
    import io

    try:
        response = requests.get(
            TRANCO_URL,
            timeout=30,
            headers={"User-Agent": "EVELYN-research-tool (academic project)"},
        )
        response.raise_for_status()

        # Tranco serves a ZIP file containing one CSV. We unzip it
        # entirely in memory — no need to write the zip to disk first.
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_filename = z.namelist()[0]
            with z.open(csv_filename) as f:
                # Tranco's CSV has NO header row — just "rank,domain"
                # on every line. We parse it manually rather than with
                # DictReader since there are no column names to map to.
                lines = f.read().decode("utf-8").splitlines()
                domains = []
                for line in lines[:n]:
                    parts = line.split(",")
                    if len(parts) >= 2:
                        domains.append(parts[1].strip())

        saved_path = None
        if save:
            RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
            today = datetime.now().strftime("%Y-%m-%d")
            saved_path = RAW_DATA_DIR / f"benign_{today}.csv"
            with open(saved_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["domain", "label"])
                for d in domains:
                    writer.writerow([d, 0])   # label 0 = benign

        return {
            "domains":    domains,
            "count":      len(domains),
            "saved_path": str(saved_path) if saved_path else None,
            "resolved":   True,
            "error":      None,
        }

    except requests.exceptions.Timeout:
        return _failure("Timeout")
    except requests.exceptions.RequestException as e:
        return _failure(f"RequestError: {e}")
    except zipfile.BadZipFile as e:
        return _failure(f"BadZipFile: {e}")
    except Exception as e:
        return _failure(f"UnknownError: {e}")


def _failure(error_msg: str) -> dict:
    return {"domains": [], "count": 0, "saved_path": None, "resolved": False, "error": error_msg}


def _print_result(result: dict) -> None:
    print(f"\n  Tranco benign domain list download")
    if result["resolved"]:
        print(f"  Status:      ✓ RESOLVED")
        print(f"  Domains:     {result['count']}")
        print(f"  Saved to:    {result['saved_path']}")
        print(f"  Sample:      {result['domains'][:5]}")
    else:
        print(f"  Status:      ✗ FAILED  ({result['error']})")
    print(f"  {'─'*52}")


# ── SELF-TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    print("\n" + "=" * 58)
    print("  EVELYN — fetch_benign()")
    print("=" * 58)
    _print_result(fetch_benign(n=n, save=True))