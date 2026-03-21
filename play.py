"""
Root-level launcher — run this from ANYWHERE:
    python play.py

It fixes sys.path so that the src/ imports (encoder, model, mcts)
work correctly regardless of the current working directory.
"""
import sys
import os

# Ensure python/src is on sys.path so relative imports work
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "python", "src")
sys.path.insert(0, src_dir)

# Now hand off to the real entry point
import play  # noqa: F401, E402  (python/src/play.py)
play.main()
