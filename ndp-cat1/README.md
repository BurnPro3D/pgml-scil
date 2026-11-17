# PGML on FIRETEC (NDP)

Trains the repository's PGML model (`AFNONet_Seq2Seq` with `PhysicsGuidedLoss`)
on FIRETEC simulations instead of QUIC-Fire ones. The simulations are read from
the Zarr stores that [firetec_to_zarr.py](https://github.com/national-data-platform/FIRETEC/blob/main/TOOLS/Postprocessing-Zarr/firetec_to_zarr.py)
writes, from disk, from S3, or through a Pelican federation such as the OSDF.

## Setup

The NDP JupyterHub base image has no PyTorch. Build an environment on persistent storage once:

```bash
bash ndp-cat1/setup_env.sh
```

It creates `<persistent storage>/envs/pgml` (next to this repository) and
installs `requirements.txt` into it. The commands below use its Python:

```bash
export PYTHON=/home/jovyan/work/_User-Persistent-Storage_CephBlock_/envs/pgml/bin/python
```

## Train

Run from the repository root:

```bash
$PYTHON ndp-cat1/train.py --config ndp-cat1/configs/firetec_256x256.yaml
```

Any key of the configuration except `storage_options` can be overridden on the
command line, for example `--max-epochs 50 --experiment-name trial`; `train.py`
stops on an option that is not a key. Each run writes a folder
`ndp-cat1/runs/<time>_<experiment_name>/`:

| File | Contents |
|---|---|
| `config.yaml` | The configuration the run used, overrides included |
| `training.log` | The log |
| `history.csv` | Per epoch: learning rate, training and validation loss, and each loss term |
| `best_model.pth` | The weights with the lowest validation loss (`epoch`, `state_dict`, `val_loss`) |

## Evaluate

```bash
$PYTHON ndp-cat1/evaluate.py --run-dir ndp-cat1/runs/<run folder>
```

For each training run this forecasts every frame from the frames before it and
writes, under `<run folder>/evaluation/<store name>/`:

| File | Contents |
|---|---|
| `forecast.gif` | FIRETEC's fuel density, the forecast, and their difference, frame by frame. The difference is dark where they agree; its colour scale ends at the 99.9th percentile of the differences, so the few larger ones show in the end colours |
| `forecast_metrics.csv` | Per frame: the forecast's RMSE and the RMSE of predicting no change |
| `store.txt` | The store the results are for |

Each frame is marked `training` (in the part of the run training used),
`validation` (held out), or `unseen run` (a store that is not among the run's
`stores`). "Predicting no change" repeats the last input frame; a useful model
has to beat it. Pass `--store <path or URL>` to evaluate a run the model has
not seen. A store is refused when another store's results already have its
name; folders written before `store.txt` existed are not checked.

## How a FIRETEC run becomes model input

One frame per FIRETEC output time, with four channels (`firetec_pgml/features.py`):

| Channel | Source in the Zarr store |
|---|---|
| `fuel_density` | `rhoFuel` in the ground layer, summed over fuel types (kg m⁻³, not rescaled) |
| `wind_x`, `wind_y` | Horizontal mean of `u` and `v` at the level nearest the reference height `zu`, in the first output, divided by `wind_speed_scale` |
| `ignition` | 1 until a cell is ignited (`ignition_time`), 0 from then on |

The target is the fuel density of later frames. A sample holds `context_len`
input frames and the `num_pred_frames` frames that start `future` frames after
the first input frame. With `predict_change: true`, the model predicts the
change from the last input frame instead of the fuel density itself. On the
256 x 256 run that cut the error by 27 to 42% at `embed_dim` 64, 128 and 256,
but only at 256 did it match predicting no change.

Stores converted with `--density-weighted` are refused: their `u` and `v` hold the air density times the wind.

The ignition file numbers cells from zero in x and y, so `ignition_i` and
`ignition_j` are used directly as array indices. The store labels them
"1-based"; that label, and `ignition_x` / `ignition_y`, are one cell off.

The last `val_fraction` of each run's frames are held out: they are validation
targets and never training targets. With a single run this only checks the late
part of the fire. Held-out runs are the meaningful test, once there are several.

## Reading from S3 or Pelican

A store is named by a path or by a URL, so switching is a change to `stores` in the configuration:

```yaml
stores:
  - s3://ndp-cat1/firetec/TestRun-256x256-CPU-N16.zarr
  # or, through the OSDF or another Pelican federation:
  # - osdf:///<namespace>/<path>/TestRun-256x256-CPU-N16.zarr
  # - pelican://<federation host>/<namespace>/<path>/TestRun-256x256-CPU-N16.zarr
storage_options:
  s3:
    endpoint_url: https://s3-west.nrp-nautilus.io
    anon: true
  osdf: {}
zarr_format: null
```

`storage_options` maps a URL scheme to the options of its filesystem. They are
ignored for paths on disk.

- `s3` options go to s3fs's `S3FileSystem`. `endpoint_url` names an
  S3-compatible service such as NRP Nautilus, and `anon: true` reads a public
  bucket without credentials. For a private bucket, drop `anon` and put the
  keys in the environment (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`), not in
  the configuration, which is copied into every run folder.
- `osdf` and `pelican` options go to `PelicanFileSystem`. Pelican serves objects
  from the federation's caches by default: `direct_reads: true` reads from the
  origin, and `preferred_caches: [...]` tries those caches first.

`zarr_format` is the stores' Zarr format, 2 or 3 (firetec_to_zarr.py writes 2
unless it is asked for 3). With `null`, Zarr detects it: it asks for format 3
metadata and, when the store answers "not found", reads format 2.

A whole run is read once, when training starts.

## Docker
Build an image to test locally:
```
docker build -f ndp-cat1/Dockerfile -t pgml-ndp .
```
`train.py` imports common_utils and FourCastNet from the repository root, so
build from the root.

Run with local backend store (change the data mount):
```
docker run --rm \
  -v /path/to/stores:/data:ro \
  -v "$PWD/ndp-cat1/runs:/workspace/pgml/ndp-cat1/runs" \
  pgml-ndp --config ndp-cat1/configs/firetec_256x256.yaml \
  --stores /data/TestRun-256x256-CPU-N16.zarr --max-epochs 1
```

Run with s3 backend store:
```
docker run --rm \
  -v "$PWD/ndp-cat1/runs:/workspace/pgml/ndp-cat1/runs" \
  pgml-ndp --config ndp-cat1/configs/firetec_256x256.yaml \
  --stores s3://ndp-cat1/firetec/TestRun-256x256-CPU-N16.zarr --max-epochs 1
```

Arguments after the image name override the configuration's values, as on the command line.

Build and push to container registry on https://gitlab.nrp-nautilus.io/ndp/ndp-cat1,
with a new tag for every code change (the Jobs run `v0.2.0` now):
```
TAG=v0.3.0
docker buildx build --platform linux/amd64 -f ndp-cat1/Dockerfile -t gitlab-registry.nrp-nautilus.io/ndp/ndp-cat1/pgml-firetec:$TAG .

#or without buildx
docker build -f ndp-cat1/Dockerfile -t gitlab-registry.nrp-nautilus.io/ndp/ndp-cat1/pgml-firetec:$TAG .

docker push gitlab-registry.nrp-nautilus.io/ndp/ndp-cat1/pgml-firetec:$TAG
```
Then set the same tag in the `image:` of the Jobs in `kubernetes/`. Pushing an
existing tag again does not reach every node: a node that has pulled a tag
keeps using its copy.

## Kubernetes
The job reads its configuration from the ConfigMap `pgml-firetec-config`, made
from `configs/firetec_256x256.yaml`. Create it before the first run, and update
it after each change to the file. `train.py` reads it when a pod starts, so pods
that start later, retries included, get the current version.
```
kubectl create configmap pgml-firetec-config -n ndp \
  --from-file=ndp-cat1/configs/firetec_256x256.yaml \
  --dry-run=client -o yaml | kubectl apply -f -
```

Each pod trains, evaluates, and copies its run folder to
`s3://ndp-cat1/pgml/runs/<run folder>` with `upload.py`, also when a
step failed: the pod's own files are gone once it completes. Writing needs S3
keys, which the job reads from the Secret `ndp-cat1-s3`.

Create it once:
```
kubectl create secret generic ndp-cat1-s3 -n ndp \
  --from-literal=AWS_ACCESS_KEY_ID=<access key> \
  --from-literal=AWS_SECRET_ACCESS_KEY=<secret key>
```

Run a job on the `ndp` namespace with s3 data backend + upload:
```
kubectl apply -f ndp-cat1/kubernetes/job-s3-upload.yaml
```

Run a job on the `ndp` namespace with s3 data backend with no upload:
```
kubectl apply -f ndp-cat1/kubernetes/job-s3.yaml
```

Both manifests create the Job `pgml-firetec-s3-job`, so delete the previous one
before starting either: `kubectl delete job pgml-firetec-s3-job -n ndp --ignore-not-found`.

The results are under `pgml/runs/`, which anyone can read, so downloading a run
folder needs no keys:
```
aws s3 sync s3://ndp-cat1/pgml/runs/<run folder> ndp-cat1/runs/<run folder> --endpoint-url https://s3-west.nrp-nautilus.io --no-sign-request
```

You can see all the runs that have been uploaded to the public s3 bucket:
```
aws s3 ls s3://ndp-cat1/pgml/runs/ --endpoint-url https://s3-west.nrp-nautilus.io --no-sign-request
```

Delete job:
```
kubectl delete -f ndp-cat1/kubernetes/job-s3.yaml
```

Delete config: 
```
kubectl delete configmap pgml-firetec-config -n ndp
```

Delete s3 secret:
```
kubectl delete secret ndp-cat1-s3 -nndp
```

## Tests

```bash
$PYTHON -m pytest ndp-cat1
```

## Layout

```
ndp-cat1/
|-- setup_env.sh, requirements.txt   the Python environment
|-- configs/                         training configurations
|-- train.py, evaluate.py, upload.py command-line entry points
|-- kubernetes/                      the training job for Nautilus
|-- firetec_pgml/
|   |-- store.py                     opens a store from a path or URL, uploads run folders
|   |-- features.py                  FIRETEC fields -> model frames
|   |-- datasets.py                  frames -> training and validation samples
|   |-- modeling.py                  model and loss for a run's grid
|   |-- training.py                  training loop
|   `-- evaluation.py                forecasts, metrics and animation
`-- tests/
```
