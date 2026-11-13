import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest  # noqa: E402

from synthetic import make_matches  # noqa: E402


@pytest.fixture(scope="session")
def matches():
    return make_matches()


@pytest.fixture(scope="session")
def small_matches():
    return make_matches(n_seasons=3, teams_per_league=8, leagues=("Serie A", "Premier League"))
