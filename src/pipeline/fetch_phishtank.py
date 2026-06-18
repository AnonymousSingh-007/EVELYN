# src/pipeline/fetch_phishtank.py
#
# PURPOSE: Download the real PhishTank verified phishing URL feed and
# save it locally as your raw labeled dataset. This is the single most
# important file in Stage 1 — everything downstream (graphs, quantum
# walks, clustering) is meaningless without real labeled phishing data
# to run it on.
#
# PhishTank's own developer docs explicitly recommend this exact pattern:
# download a local copy periodically, then do all your lookups locally,
# instead of hammering their live API per-query. We follow that guidance.

import requests
import csv
import io
from pathlib import Path
from datetime import datetime


PHISHTANK_CSV_URL = "https://data.phishtank.com/data/online-valid.csv"

# Save location: data/raw/ — exactly the folder you already created,
# and already gitignored, since this is real labeled security data
# that shouldn't be committed to a public/shared repo.
RAW_DATA_DIR = Path("data/raw")


def fetch_phishtank(save: bool = True, max_rows: int = None) -> dict:
    """
    Downloads the current PhishTank verified-phishing CSV feed.

    Returns:
    {
        "rows":        [ {url, phish_id, submission_time, verified, ...}, ... ],
        "row_count":   1234,
        "saved_path":  "data/raw/phishtank_2026-06-17.csv"  or None,
        "resolved":    True,
        "error":       None
    }
    """

    try:
        # PhishTank rate-limits aggressively without an API key, and
        # explicitly asks bulk downloaders to identify themselves with
        # a real User-Agent rather than pretending to be a browser —
        # this is good research-ethics practice, not just politeness.
        response = requests.get(
            PHISHTANK_CSV_URL,
            timeout=30,
            headers={"User-Agent": "EVELYN-research-tool (academic project)"},
        )
        response.raise_for_status()

        # The response is raw CSV TEXT. We use Python's built-in csv
        # module via io.StringIO to parse it as if it were a local file,
        # without ever writing to disk first — this lets us inspect
        # and validate the data BEFORE deciding to save it.
        csv_text = response.text
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)

        if max_rows is not None:
            rows = rows[:max_rows]

        saved_path = None
        if save:
            RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
            today = datetime.now().strftime("%Y-%m-%d")
            saved_path = RAW_DATA_DIR / f"phishtank_{today}.csv"

            # Re-write as a clean CSV using the SAME fieldnames PhishTank
            # gave us — this preserves phish_id, submission_time, target,
            # etc. for later reference, not just the bare URL.
            with open(saved_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        return {
            "rows":       rows,
            "row_count":  len(rows),
            "saved_path": str(saved_path) if saved_path else None,
            "resolved":   True,
            "error":      None,
        }

    except requests.exceptions.Timeout:
        return _failure("Timeout — PhishTank may be rate-limiting or under load")
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 509:
            return _failure("RateLimited: PhishTank free-tier bandwidth limit hit. Try again later.")
        return _failure(f"HTTPError: {e}")
    except requests.exceptions.RequestException as e:
        return _failure(f"RequestError: {e}")
    except Exception as e:
        return _failure(f"UnknownError: {e}")


def _failure(error_msg: str) -> dict:
    return {"rows": [], "row_count": 0, "saved_path": None, "resolved": False, "error": error_msg}


def _print_result(result: dict) -> None:
    print(f"\n  PhishTank feed download")
    if result["resolved"]:
        print(f"  Status:      ✓ RESOLVED")
        print(f"  Rows:        {result['row_count']}")
        print(f"  Saved to:    {result['saved_path']}")
        if result["rows"]:
            sample = result["rows"][0]
            print(f"\n  Sample row fields: {list(sample.keys())}")
            print(f"  Sample URL:        {sample.get('url', '?')}")
            print(f"  Sample target:     {sample.get('target', '?')}")
    else:
        print(f"  Status:      ✗ FAILED  ({result['error']})")
    print(f"  {'─'*52}")


# ── SELF-TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # max_rows lets you test quickly without downloading the full feed
    # (which can be 10,000-50,000+ rows) every single time you run this
    max_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    print("\n" + "=" * 58)
    print("  EVELYN — fetch_phishtank()")
    print("=" * 58)
    print(f"  Downloading PhishTank feed (capped at {max_rows} rows for this test run)...")

    result = fetch_phishtank(save=True, max_rows=max_rows)
    _print_result(result)