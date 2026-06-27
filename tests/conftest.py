"""
Shared fixtures for all test modules.

Run all tests:
    pytest tests/ -v

Run a single file:
    pytest tests/test_validator.py -v
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path so imports like `from engine.validator import ...` resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
