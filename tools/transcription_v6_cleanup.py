"""Optional local Vaani V6 transcript cleanup sidecar.

The model server speaks newline-delimited JSON over private local pipes. Its
output is accepted only when source content/order, negation, and digits survive.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_SERVER: "_Server | None" = None


class _Server:
    def __init__(self, config: dict[str, Any]) -> None:
        command = [
            str(config["python"]), str(config["script"]),
            str(config["model"]), str(config["adapter"]),
        ]
        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env,
        )
        ready = self._readline(float(config.get("startup_timeout_seconds", 90)))
        if not ready or not json.loads(ready).get("ready"):
            self.close()
            raise RuntimeError("V6 cleanup sidecar did not become ready")
        self.request_id = 0

    def _readline(self, timeout: float) -> str:
        if self.process.stdout is None:
            raise RuntimeError("V6 cleanup sidecar has no stdout")
        ready, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not ready:
            raise TimeoutError("V6 cleanup sidecar timed out")
        return self.process.stdout.readline()

    def clean(self, text: str, timeout: float) -> str:
        if self.process.poll() is not None or self.process.stdin is None:
            raise RuntimeError("V6 cleanup sidecar exited")
        self.request_id += 1
        request_id = self.request_id
        self.process.stdin.write(json.dumps({"id": request_id, "text": text}) + "\n")
        self.process.stdin.flush()
        reply = json.loads(self._readline(timeout))
        if reply.get("id") != request_id or reply.get("error"):
            raise RuntimeError("V6 cleanup sidecar returned an invalid reply")
        return str(reply.get("text") or text)

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def _source_preserving(raw: str, cleaned: str) -> bool:
    def words(value: str) -> list[str]:
        import re

        return [w.casefold() for w in re.findall(r"[\w']+", value, flags=re.UNICODE)]

    fillers = {"uh", "um", "erm", "hmm", "mmm"}
    expected = [word for word in words(raw) if word not in fillers]
    actual = words(cleaned)
    if actual != expected:
        return False
    if "".join(c for c in raw if c.isdigit()) != "".join(c for c in cleaned if c.isdigit()):
        return False
    negations = {"not", "no", "never", "n't", "नहीं", "না"}
    return not (negations & set(words(raw))) or negations & set(actual)


def cleanup_v6_transcript(text: str, config: dict[str, Any]) -> str:
    """Clean one transcript; fail closed to raw STT when setup/model is unavailable."""
    if not text.strip() or not config.get("enabled", False):
        return text
    required = ("python", "script", "model", "adapter")
    if any(not str(config.get(key) or "").strip() for key in required):
        return text
    if any(not Path(str(config[key])).exists() for key in required):
        return text

    global _SERVER
    timeout = float(config.get("request_timeout_seconds", 30))
    with _LOCK:
        try:
            if _SERVER is None or _SERVER.process.poll() is not None:
                if _SERVER is not None:
                    _SERVER.close()
                _SERVER = _Server(config)
            cleaned = _SERVER.clean(text, timeout)
        except Exception:
            if _SERVER is not None:
                _SERVER.close()
                _SERVER = None
            return text
    return cleaned if _source_preserving(text, cleaned) else text
