"""Train the repository's PGML model on FIRETEC simulations stored as Zarr."""

import sys
from pathlib import Path

# The repository is not an installable package: its trainers put the repository
# root on sys.path to import FourCastNet and common_utils, and so does this one.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
