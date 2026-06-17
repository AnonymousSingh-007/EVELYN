# src/pipeline/fetch_ssl_meta.py
#
# PURPOSE: Connect DIRECTLY to a domain's HTTPS port and inspect its
# TLS certificate's own metadata — validity period, issuer type,
# key strength. This is DIFFERENT from fetch_cert.py, which searches
# PUBLIC LOGS for historical certs. This file makes a LIVE connection
# to see exactly what cert is being served RIGHT NOW.
#
# Why this matters: phishing certs tend to be very short-lived
# (Let's Encrypt 90-day certs, often re-issued every few days as
# domains rotate), and attackers occasionally use self-signed certs
# on lower-effort campaigns. Legitimate banks use long-lived,
# extended-validation certs from established CAs. This is a
# behavioral fingerprint, not just an identity fingerprint.

import ssl
import socket
from datetime import datetime, timezone


def fetch_ssl_meta(domain: str, port: int = 443, timeout: float = 6.0) -> dict:
    """
    Connects to domain:443 and inspects the live TLS certificate.

    Example output:
    {
        "domain":           "google.com",
        "issuer":           "GTS CA 1C3",
        "valid_from":       "2024-01-01",
        "valid_until":      "2024-04-01",
        "validity_days":    90,
        "days_until_expiry": 45,
        "is_self_signed":   False,
        "resolved":         True,
        "error":            None
    }
    """

    if "://" in domain or "/" in domain:
        return _failure(domain, "InvalidInput: pass a bare domain, not a full URL")

    try:
        # Create a default SSL context — this is the SAME mechanism your
        # browser uses to verify certificates against trusted root CAs.
        context = ssl.create_default_context()

        # Open a raw TCP socket to the domain on port 443 (HTTPS), then
        # wrap it in a TLS layer. This performs an ACTUAL handshake —
        # we're not asking a third-party API, we're talking to the
        # phishing server directly, the same way a victim's browser would.
        with socket.create_connection((domain, port), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=domain) as ssock:
                cert = ssock.getpeercert()

        # Certificate dates come back as strings like:
        # "Jan  1 00:00:00 2024 GMT" — Python's ssl module uses this
        # specific format, which we parse with strptime.
        not_before = datetime.strptime(cert["notBefore"], "%b %d %H:%M:%S %Y %Z")
        not_after  = datetime.strptime(cert["notAfter"],  "%b %d %H:%M:%S %Y %Z")
        not_before = not_before.replace(tzinfo=timezone.utc)
        not_after  = not_after.replace(tzinfo=timezone.utc)

        validity_days = (not_after - not_before).days
        days_until_expiry = (not_after - datetime.now(timezone.utc)).days

        # The "issuer" field is a tuple-of-tuples structure. We pull out
        # the organisation name (O) or common name (CN), whichever exists.
        issuer_parts = dict(x[0] for x in cert.get("issuer", []))
        issuer = issuer_parts.get("organizationName") or issuer_parts.get("commonName")

        # A cert is self-signed if its issuer and subject are IDENTICAL —
        # meaning whoever made the cert "vouched for themselves" instead
        # of a trusted Certificate Authority signing it. Self-signed certs
        # on a public-facing login page are a major red flag.
        subject_parts = dict(x[0] for x in cert.get("subject", []))
        is_self_signed = (issuer_parts == subject_parts)

        return {
            "domain":            domain,
            "issuer":            issuer,
            "valid_from":        not_before.strftime("%Y-%m-%d"),
            "valid_until":       not_after.strftime("%Y-%m-%d"),
            "validity_days":     validity_days,
            "days_until_expiry": days_until_expiry,
            "is_self_signed":    is_self_signed,
            "resolved":          True,
            "error":             None,
        }

    except socket.timeout:
        return _failure(domain, "Timeout")
    except ssl.SSLCertVerificationError as e:
        # This actually fires for SELF-SIGNED or invalid certs, because
        # create_default_context() verifies against trusted CAs by default.
        # We treat this as a DATA POINT (self-signed = suspicious) rather
        # than a pure failure — but full cert parsing requires disabling
        # verification, which we deliberately do NOT do for security reasons.
        return _failure(domain, f"SSLCertVerificationError (likely self-signed or invalid): {e}")
    except socket.gaierror as e:
        return _failure(domain, f"DNSError: {e}")
    except ConnectionRefusedError:
        return _failure(domain, "ConnectionRefused: no HTTPS service on port 443")
    except Exception as e:
        return _failure(domain, f"UnknownError: {e}")


def _failure(domain: str, error_msg: str) -> dict:
    return {
        "domain": domain, "issuer": None, "valid_from": None,
        "valid_until": None, "validity_days": None,
        "days_until_expiry": None, "is_self_signed": None,
        "resolved": False, "error": error_msg,
    }


def _print_result(result: dict) -> None:
    print(f"\n  Domain:          {result['domain']}")
    if result["resolved"]:
        print(f"  Status:          ✓ RESOLVED")
        print(f"  Issuer:          {result['issuer']}")
        print(f"  Valid:           {result['valid_from']} → {result['valid_until']} ({result['validity_days']} days)")
        print(f"  Days to expiry:  {result['days_until_expiry']}")
        print(f"  Self-signed:     {result['is_self_signed']}", end="")
        if result["is_self_signed"]:
            print("  ⚠ SUSPICIOUS — self-signed cert on public site")
        else:
            print()
    else:
        print(f"  Status:          ✗ FAILED  ({result['error']})")
    print(f"  {'─'*52}")


# ── SELF-TEST ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        print("\n" + "=" * 58)
        print("  EVELYN — fetch_ssl_meta()")
        print("=" * 58)
        _print_result(fetch_ssl_meta(sys.argv[1]))
    else:
        TEST_DOMAINS = ["google.com", "github.com"]
        print("\n" + "=" * 58)
        print("  EVELYN — fetch_ssl_meta() test suite")
        print("=" * 58)
        for domain in TEST_DOMAINS:
            _print_result(fetch_ssl_meta(domain))