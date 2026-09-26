"""Talk to docker.local-inference-lab.ai: check the token, upload a result."""

from __future__ import annotations

import gzip
import json
import time
import urllib.error
import urllib.request

from . import VERSION

USER_AGENT = f"lil-bench/{VERSION}"


class UploadError(Exception):
    def __init__(self, message: str, status: int | None = None, retry: bool = False):
        super().__init__(message)
        self.status = status
        self.retry = retry


def _request(url: str, token: str, data: bytes | None = None, timeout: float = 30.0) -> dict:
    headers = {"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT, "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
        headers["Content-Encoding"] = "gzip"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        try:
            detail = json.loads(error.read() or b"{}").get("error") or error.reason
        except ValueError:
            detail = error.reason
        raise UploadError(str(detail), status=error.code, retry=error.code >= 500 or error.code == 429) from error
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        reason = getattr(error, "reason", error)
        raise UploadError(f"cannot reach {url.split('/api/')[0]}: {reason}", retry=True) from error


def whoami(site: str, token: str) -> dict:
    return _request(f"{site.rstrip('/')}/api/bench/whoami", token, timeout=15)


def compress(document: dict) -> bytes:
    return gzip.compress(json.dumps(document, separators=(",", ":")).encode(), compresslevel=9)


def upload_bytes(site: str, token: str, payload: bytes, attempts: int = 3, sleep=time.sleep) -> dict:
    delay = 3.0
    for attempt in range(1, attempts + 1):
        try:
            return _request(f"{site.rstrip('/')}/api/bench/runs", token, data=payload, timeout=120)
        except UploadError as error:
            if not error.retry or attempt == attempts:
                raise
            sleep(delay)
            delay *= 3
    raise AssertionError("unreachable")
