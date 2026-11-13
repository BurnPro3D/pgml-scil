"""Copy a run folder to S3 or another fsspec URL.

    python ndp-cat1/upload.py --run-dir ndp-cat1/runs/<run folder> --to s3://ndp-cat1/firetec/pgml/runs

The folder lands at ``<to>/<run folder>``, with everything in it: the files
train.py wrote, and evaluate.py's ``evaluation/`` once it has run. The run's
``storage_options`` say how to reach the destination, except ``anon``: writing
needs credentials, which come from the environment (for S3: AWS_ACCESS_KEY_ID
and AWS_SECRET_ACCESS_KEY).
"""

import argparse
import logging
from pathlib import Path

from firetec_pgml.store import upload_folder
from firetec_pgml.training import CONFIG_NAME

from common_utils.config import Config

logger = logging.getLogger(__name__)


def upload(run_dir: Path, url: str) -> str:
    """Copy ``run_dir`` to ``<url>/<run folder>`` and return that URL."""
    config = Config(str(run_dir / CONFIG_NAME), override_from_env=False, override_from_args=False)
    destination = upload_folder(run_dir, url, config.get("storage_options"))
    logger.info("Uploaded %s to %s", run_dir, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True, type=Path, help="folder written by train.py")
    parser.add_argument("--to", required=True, help="URL or path the run folder is copied into")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    upload(args.run_dir, args.to)


if __name__ == "__main__":
    main()
