"""Command-line interface for LensMovie."""

import argparse
import sys

from lensmovie import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lensmovie",
        description="LensMovie CLI application.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    print("Hello from LensMovie!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
