"""pytest configuration for reliable imports.

Ensures the repository root is on sys.path so that `import src...` works
when running `pytest` from any working directory.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

