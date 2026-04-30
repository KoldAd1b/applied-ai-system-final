from src.ai_client import _read_env_key


def test_read_env_key_loads_gemini_key(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "# local development secrets",
                "OTHER=value",
                "GEMINI_API_KEY='test-key'",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert _read_env_key() == "test-key"


def test_read_env_key_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert _read_env_key() is None
