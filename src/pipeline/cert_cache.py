# src/pipeline/cert_cache.py
#
# PURPOSE: A simple, zero-cost, zero-dependency on-disk cache for
# fetch_cert() results. Phishing/security research is HEAVILY
# iterative — you re-run the same pipeline on the same domains many
# times while debugging, building figures, re-running batches after
# a crash, etc. Without caching, every single re-run burns through
# your already-scarce free API quota on domains you've ALREADY queried
# successfully five minutes ago.
#
# This is not a "real" CT-log mirror (that would mean syncing actual
# certificate transparency log data, which is a much bigger and
# differently-scoped project) — it's a pragmatic result cache that
# respects the free tiers you depend on. Worth being precise about
# this distinction in your methods section: this is API-call caching,
# not log mirroring.

import json
import time
from pathlib import Path

CACHE_DIR = Path("data/processed/cert_cache")
CACHE_TTL_SECONDS = 7 * 24 * 3600   # 7 days — certs don't change that often,
                                     # and a week-old cert record is still
                                     # valid evidence of shared infrastructure


def _cache_path(domain: str) -> Path:
    safe_name = domain.replace(".", "_").replace("/", "_")
    return CACHE_DIR / f"{safe_name}.json"


def get_cached(domain: str) -> dict | None:
    """
    Returns a cached fetch_cert() result for this domain if one exists
    and hasn't expired, else None (meaning: go fetch it for real).
    """
    path = _cache_path(domain)
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            entry = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None   # corrupted cache entry — treat as a miss, refetch

    age_seconds = time.time() - entry.get("_cached_at", 0)
    if age_seconds > CACHE_TTL_SECONDS:
        return None   # expired — treat as a miss, refetch

    result = entry["result"]
    result["_from_cache"] = True   # so callers/logs can see this was free
    return result


def set_cached(domain: str, result: dict) -> None:
    """Saves a fetch_cert() result to the on-disk cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(domain)

    entry = {
        "_cached_at": time.time(),
        "result": result,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entry, f)


def cache_stats() -> dict:
    """Quick visibility into how much the cache is actually being used."""
    if not CACHE_DIR.exists():
        return {"entries": 0, "total_size_kb": 0}

    files = list(CACHE_DIR.glob("*.json"))
    total_size = sum(f.stat().st_size for f in files)
    return {"entries": len(files), "total_size_kb": round(total_size / 1024, 1)}


# ── SELF-TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 58)
    print("  EVELYN — cert_cache()")
    print("=" * 58)

    # Round-trip test: write, then read back, confirm it matches
    test_domain = "example_cache_test.com"
    test_result = {
        "domain": test_domain, "cert_count": 5,
        "shared_domains": ["a.com", "b.com"], "issuer": "Test CA",
        "source": "crt.sh", "resolved": True, "error": None,
    }

    print(f"\n  Writing test entry for {test_domain}...")
    set_cached(test_domain, test_result)

    print(f"  Reading it back...")
    retrieved = get_cached(test_domain)

    if retrieved and retrieved["cert_count"] == 5:
        print(f"  ✓ PASS — cache round-trip works correctly")
        print(f"    Retrieved: {retrieved}")
    else:
        print(f"  ✗ FAIL — cache round-trip did not return expected data")

    print(f"\n  Cache stats: {cache_stats()}")

    # Cleanup the test entry so it doesn't pollute your real cache
    _cache_path(test_domain).unlink(missing_ok=True)