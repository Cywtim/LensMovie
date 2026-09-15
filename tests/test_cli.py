"""Tests for LensMovie CLI."""

import pytest

from lensmovie.cli import build_parser, main


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "lensmovie" in out


def test_main_prints_hello(capsys):
    assert main([]) == 0
    assert "Hello from LensMovie!" in capsys.readouterr().out
