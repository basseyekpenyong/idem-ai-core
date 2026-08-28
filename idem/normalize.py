"""
Stage C: the ONE normalizer, imported identically at training time and
scoring time. See CLAUDE-BRIEF-v5-pipeline.md section 6, Stage C.

Not yet implemented — see tests/test_normalize.py for the specification.
"""

ALLOWED = set("abcdefghijklmnopqrstuvwxyz' ")


def normalize(text: str) -> str:
    """Raw transcript -> training/scoring text. The ONLY normalizer."""
    raise NotImplementedError("Stage C: implementation pending review of test_normalize.py")
