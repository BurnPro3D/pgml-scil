# PGML on FIRETEC (NDP)

Trains the repository's PGML model (`AFNONet_Seq2Seq` with `PhysicsGuidedLoss`)
on FIRETEC simulations instead of QUIC-Fire ones. The simulations are read from
the Zarr stores that [firetec_to_zarr.py](https://github.com/national-data-platform/FIRETEC/blob/main/TOOLS/Postprocessing-Zarr/firetec_to_zarr.py)
writes, either from disk or through a Pelican federation such as the OSDF.

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

Any key of the configuration can be overridden on the command line, for example
`--max-epochs 50 --experiment-name trial`. Each run writes a folder
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
| `forecast.gif` | FIRETEC's fuel density, the forecast, and their difference, frame by frame |
| `forecast_metrics.csv` | Per frame: the forecast's RMSE and the RMSE of predicting no change |

Each frame is marked `training` or `validation`. "Predicting no change" repeats
the last input frame; a useful model has to beat it. Pass `--store <path or URL>`
to evaluate a run the model has not seen.

## How a FIRETEC run becomes model input

One frame per FIRETEC output time, with four channels (`firetec_pgml/features.py`):

| Channel | Source in the Zarr store |
|---|---|
| `fuel_density` | `rhoFuel` in the ground layer, summed over fuel types (kg m⁻³, not rescaled) |
| `wind_x`, `wind_y` | Horizontal mean of `u` and `v` at the level nearest the reference height `zu`, in the first output, divided by `wind_speed_scale` |
| `ignition` | 1 until a cell is ignited (`ignition_time`), 0 from then on |

The target is the fuel density of later frames. A sample holds `context_len`
input frames and the `num_pred_frames` frames that start `future` frames after
the first input frame.

The ignition file numbers cells from zero in x and y, so `ignition_i` and
`ignition_j` are used directly as array indices. The store labels them
"1-based"; that label, and `ignition_x` / `ignition_y`, are one cell off.

The last `val_fraction` of each run's frames are held out: they are validation
targets and never training targets. With a single run this only checks the late
part of the fire. Held-out runs are the meaningful test, once there are several.

## Reading from Pelican

A store is named by a path or by a URL, so switching is a change to `stores` in
the configuration:

```yaml
stores:
  - osdf:///<namespace>/<path>/TestRun-256x256-CPU-N16.zarr
  # or, for another federation:
  # - pelican://<federation host>/<namespace>/<path>/TestRun-256x256-CPU-N16.zarr
storage_options: {}
```

Pelican serves objects from the federation's caches by default.
`storage_options` are passed to `PelicanFileSystem`, for example
`direct_reads: true` to read from the origin, or `preferred_caches: [...]`.
They are ignored for paths on disk. A whole run is read once, when training
starts.

## Tests

```bash
$PYTHON -m pytest ndp-cat1
```

## Layout

```
ndp-cat1/
|-- setup_env.sh, requirements.txt   the Python environment
|-- configs/                         training configurations
|-- train.py, evaluate.py            command-line entry points
|-- firetec_pgml/
|   |-- store.py                     opens a store from a path or URL
|   |-- features.py                  FIRETEC fields -> model frames
|   |-- datasets.py                  frames -> training and validation samples
|   |-- modeling.py                  model and loss for a run's grid
|   |-- training.py                  training loop
|   `-- evaluation.py                forecasts, metrics and animation
`-- tests/
```
