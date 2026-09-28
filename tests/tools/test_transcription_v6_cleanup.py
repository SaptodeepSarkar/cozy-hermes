from __future__ import annotations

from tools import transcription_v6_cleanup as cleaner


def test_source_guard_accepts_only_safe_formatting() -> None:
    assert cleaner._source_preserving("um open firefox", "Open Firefox.")
    assert not cleaner._source_preserving("do not delete file 42", "delete file 42")
    assert not cleaner._source_preserving("open file 42", "open file 24")
    assert not cleaner._source_preserving("open Firefox", "open Chrome")


def test_disabled_and_unavailable_model_fall_back_to_raw(tmp_path) -> None:
    text = "open firefox"
    assert cleaner.cleanup_v6_transcript(text, {"enabled": False}) == text
    config = {"enabled": True, "python": "missing", "script": "missing", "model": "missing", "adapter": "missing"}
    assert cleaner.cleanup_v6_transcript(text, config) == text


def test_semantically_changed_model_output_is_rejected(tmp_path, monkeypatch) -> None:
    paths = [tmp_path / name for name in ("python", "server.py", "model", "adapter")]
    for path in paths:
        path.touch()

    class Process:
        def poll(self):
            return None

    class FakeServer:
        process = Process()

        def clean(self, text, timeout):
            return "delete file 42"

        def close(self):
            pass

    monkeypatch.setattr(cleaner, "_SERVER", FakeServer())
    monkeypatch.setattr(cleaner, "_Server", lambda config: FakeServer())
    config = {"enabled": True, "python": str(paths[0]), "script": str(paths[1]),
              "model": str(paths[2]), "adapter": str(paths[3])}
    try:
        assert cleaner.cleanup_v6_transcript("do not delete file 42", config) == "do not delete file 42"
    finally:
        cleaner._SERVER = None
