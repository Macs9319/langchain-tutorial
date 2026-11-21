"""Keep GUIDE.md and README.md in step with the tutorials that exist.

A guide that points at a renamed file, or leaves out a new tutorial, is
worse than none: readers trust it. These checks run in CI.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TUTORIALS = sorted(p.name for p in ROOT.glob("[0-9][0-9]_*.py"))


def links(markdown: str) -> list[str]:
    """Local link targets, like (06_rag.py) or (pedalwise/), without #anchors."""
    targets = re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", markdown)
    return [t for t in targets if not t.startswith(("http://", "https://", "mailto:"))]


def test_every_tutorial_is_in_the_guide_and_readme():
    for name in ("GUIDE.md", "README.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        missing = [t for t in TUTORIALS if t not in text]
        assert not missing, f"{name} doesn't mention {missing}"


def test_every_local_link_points_at_something_real():
    for name in ("GUIDE.md", "README.md"):
        broken = [t for t in links((ROOT / name).read_text(encoding="utf-8"))
                  if not (ROOT / t).exists()]
        assert not broken, f"{name} links to missing files: {broken}"


def test_route_numbers_are_real_tutorials():
    numbers = {t[:2] for t in TUTORIALS}
    guide = (ROOT / "GUIDE.md").read_text(encoding="utf-8")
    for route in re.findall(r"\*\*((?:\d\d[ →]*)+)", guide):
        for number in re.findall(r"\d\d", route):
            assert number in numbers, f"route mentions tutorial {number}, which doesn't exist"
