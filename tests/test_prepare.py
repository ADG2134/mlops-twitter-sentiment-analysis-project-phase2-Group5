"""Unit tests for the text-cleaning logic shared by prepare.py and the API."""
from src.prepare import clean_text

CFG = {
    "lowercase": True,
    "remove_urls": True,
    "remove_mentions": True,
    "remove_hashtag_symbol": True,
}


def test_lowercases():
    assert clean_text("HELLO World", CFG) == "hello world"


def test_strips_urls():
    out = clean_text("check this out https://example.com/foo now", CFG)
    assert "http" not in out
    assert "check this out" in out


def test_strips_mentions():
    out = clean_text("@someuser thanks so much", CFG)
    assert "@" not in out
    assert "thanks so much" in out


def test_strips_hashtag_symbol_keeps_word():
    out = clean_text("loving this #mlops project", CFG)
    assert "#" not in out
    assert "mlops" in out


def test_collapses_whitespace():
    out = clean_text("too   many     spaces", CFG)
    assert out == "too many spaces"


def test_empty_string_stays_empty():
    assert clean_text("", CFG) == ""


def test_respects_disabled_flags():
    cfg = {**CFG, "remove_urls": False, "lowercase": False}
    out = clean_text("Visit HTTPS://Example.com", cfg)
    assert "HTTPS://Example.com" in out
