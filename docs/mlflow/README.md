# MLFlow User Guide

MLflow Tracking is a component of MLflow that allows users to track metrics and parameters through an API. Model performance is based on key metrics from model training through changing code, data, and/or model parameters. Each time a new model is trained, MLflow Tracking helps keep track of metrics for performance. MLflow Tracking helps tracks the code, parameters, or other artifacts used during training. 

## MLFlow Tracking Concepts

### Runs 
MLflow Tracking is organized around the concept of runs, which are executions of some piece of data science code, for example, a single python train.py execution. Each run records metadata (various information about your run such as metrics, parameters, start and end times) and artifacts (output files from the run such as model weights, images, etc).

### Experiments

An experiment groups together runs for a specific task. We create an experiment using the API.

## MLFlow Walkthrough
Install MLFlow 
```
$ pip install mlflow
```

In order to use the MLFlow API, import mlflow package in your train.py file.
```
import mlflow
```

### Training
Just before you start training your model using `model.fit()` function, you need to add the following MLFLow API functions. You can follow the MLFLow Docs link (https://mlflow.org/docs/latest/python_api/mlflow.tensorflow.html#mlflow.tensorflow.autolog and https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html) to learn more about the parameters of the mlflow.tensorflow.autolog()/mlflow.pytorch.autolog().
```
# You can either create a new experiment or insert a previous experiment name you want your run to be associated with. Different experiments must have unique names.

experiment_name = 'Your Experiment Name Here' 

# Run names are not unique. Different runs associated with the same experiment can have the same name. So ensure you change run_name for every new run.

run_name = 'Your New Run Name Here'
`
# We log and store metrics,params and artifacts on a local directory on PVC (/home/pgmlvol/mlflow). The backend store is a core component in MLflow Tracking where MLflow stores metadata for Runs and experiments. 

mlflow.set_tracking_uri(uri="/home/pgmlvol/mlflow")
mlflow.set_experiment(experiment_name)

mlflow.start_run(run_name=run_name)

# To store the train.py in the artifact store

mlflow.log_artifact(os.path.abspath(__file__))

# If you are using Tensorflow, thrn use the following MLFlow API :-
mlflow.tensorflow.autolog(
every_n_iter=1,
log_models=True,
log_datasets=False,
disable=False,
exclusive=False,
disable_for_unsupported_versions=False,
silent=False,
registered_model_name=None,
log_input_examples=False,
log_model_signatures=False,
saved_model_kwargs=None,
keras_model_kwargs=None,
extra_tags=None,
log_every_epoch=True,
log_every_n_steps=None,
checkpoint=True,
checkpoint_monitor="val_loss",
checkpoint_mode="min",
checkpoint_save_best_only=True,
checkpoint_save_weights_only=True,
checkpoint_save_freq="epoch",
)

# If you are using PyTorch, thrn use the following MLFlow API :-
mlflow.pytorch.autolog(
    log_every_n_epoch=1,
    log_every_n_step=None,
    log_models=True,
    log_datasets=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    registered_model_name=None,
    extra_tags=None,
    checkpoint=True,
    checkpoint_monitor="val_loss",
    checkpoint_mode="min",
    checkpoint_save_best_only=True,
    checkpoint_save_weights_only=True,
    checkpoint_save_freq="epoch",
)


model.fit(....)
.
.
.

#At the end of your training code, you need to end the run.

mlflow.end_run()

```

After you start your run (and before ending your run), there are several functions within the mlflow module that are used to log to MLflow Tracking. Since `mlflow.tensorflow.autolog() / mlflow.pytorch.autolog()` does this for you, these functions might not be required. You can log custom metrics and parameters that are not logged by the autolog() using these functions. For a single metric use the `mlflow.log_metric()` function. For multiple metrics, we can pass a dictionary to `mlflow.log_metrics()` function. To log parameters use `mlflow.log_param()` function for a single parameter or `mlflow.log_params()` function for multiple parameters. 

Example Usage:-
```
mlflow.log_param("lr", 0.001)
# Your ml code
...
mlflow.log_metric("val_loss", val_loss)
```

**Logging Artifacts:** For logging artifacts use `mlflow.log_artifact()` function for a single file or `mlflow.log_artifacts()` function pointing to a directory containing multiple files.

In order to open the MLFlow Tracking UI to visualize the tracking of your experiments and runs, insert this command in the CLI -

```
$ mlflow ui --backend-store-uri /home/pgmlvol/mlflow
```
You can then view the UI in your browser at localhost port 5000.

### Testing

In order to load a model whose checkpoint had been saved during training on a particular run, 

```
mlflow.set_tracking_uri(uri="/home/pgmlvol/mlflow")

# Enter the name of the particular run in the RUN_NAME variable.
run_id=mlflow.search_runs(search_all_experiments=True,filter_string="attributes.run_name = 'RUN_NAME'")['run_id'][0]

# If you are using Tensorflow, the parameter 'model' represents a Keras model. 
ckpt_model = mlflow.tensorflow.load_checkpoint(model=model, run_id=run_id)

# If you are using Pytorch, the parameter 'model_class' represents – The class of the training model, the class should inherit ‘pytorch_lightning.LightningModule’.
ckpt_model = mlflow.pytorch.load_checkpoint(model_class, run_id=run_id)

```





