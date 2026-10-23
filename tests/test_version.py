"""Package version consistency tests."""

import re
from pathlib import Path

import agentlink


def test_package_version_matches_pyproject():
    """__version__ must never drift from the declared package version."""
    pyproject = Path(agentlink.__file__).resolve().parent.parent / "pyproject.toml"
    match = re.search(
        r'^version\s*=\s*"([^"]+)"',
        pyproject.read_text(encoding="utf-8"),
        re.M,
    )
    assert match is not None
    assert agentlink.__version__ == match.group(1)


def test_version_is_pep440():
    """__version__ must be a valid, non-placeholder PEP 440 version."""
    assert re.match(r"^\d+\.\d+\.\d+", agentlink.__version__)
    assert agentlink.__version__ != "0.0.0"
