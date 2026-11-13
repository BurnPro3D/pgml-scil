# Trains the PGML model on FIRETEC Zarr stores (ndp-cat1/train.py).
#
# train.py imports common_utils and FourCastNet from the repository root, so
# build from the root:
#
#   docker build -f ndp-cat1/Dockerfile -t pgml-ndp .
#
# Arguments after the image name go to train.py. Mount the Zarr stores and a
# folder for the run outputs, and point --stores at the mounted store:
#
#   docker run --rm --gpus all \
#     -v /path/to/stores:/data:ro \
#     -v "$PWD/ndp-cat1/runs:/workspace/pgml/ndp-cat1/runs" \
#     pgml-ndp --config ndp-cat1/configs/firetec_256x256.yaml \
#       --stores /data/TestRun-256x256-CPU-N16.zarr --max-epochs 50
#
# Stores read from S3 or Pelican need no mount, for example the public copy of
# the test run: --stores s3://ndp-cat1/firetec/TestRun-256x256-CPU-N16.zarr

FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
RUN python3 -m venv /opt/venv

COPY ndp-cat1/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --disable-pip-version-check \
    --requirement /tmp/requirements.txt

ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

# The repository's layout, so the paths in the README and configs work as they are.
WORKDIR /workspace/pgml
COPY common_utils/ common_utils/
COPY FourCastNet/ FourCastNet/
COPY ndp-cat1/ ndp-cat1/

ENTRYPOINT ["python", "ndp-cat1/train.py"]
CMD ["--config", "ndp-cat1/configs/firetec_256x256.yaml"]
