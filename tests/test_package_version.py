from pathlib import Path

import toml

from recval import __version__

TOML_PATH = Path(__file__).parent.parent / "pyproject.toml"


def test_package_version():
    with open(TOML_PATH, "r", encoding="utf-8") as toml_file:
        version_from_toml = toml.load(toml_file)["tool"]["poetry"]["version"]
    assert version_from_toml == __version__
