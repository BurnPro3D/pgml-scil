import mlflow
from pathlib import Path
import shutil
from datetime import datetime

# Set the MLFlow tracking URI
mlflow.set_tracking_uri("/home/pgmlvol/mlflow")

# Display all experiments
experiments = mlflow.search_experiments()
print(f"{'Experiment ID':<15} | {'Name':<20} | {'Artifact Location'}")
print("-" * 70)

for exp in experiments:
    print(f"{exp.experiment_id:<15} | {exp.name:<20} | {exp.artifact_location}")

# Ask which experiment to view
exp_id = input("\nEnter the experiment ID to list its runs: ")

# List runs under the selected experiment
runs = mlflow.search_runs(exp_id)

print("\nRuns in Experiment:")
print(f"{'Run ID':<36} | {'Start Time':<25} | {'Status'}")
print("-" * 70)

# Display the runs with UUIDs
for idx, run in runs.iterrows():
    # print(run.keys())
    if 'end_time' in run and run['end_time']:
        runtime = (run['end_time'] - run['start_time']).total_seconds() / 60
        print(f"{run['run_id']:<36} | {run['experiment_id']:<36} | {runtime:.4f} min | {run['start_time']} | {run['status']}")
    else:
        runtime = "Still running"
        print(f"{run['run_id']:<36} | {run['experiment_id']:<36} | {runtime} | {run['start_time']} | {run['status']}")

# Choose runs to delete
delete_ids = input("\nEnter run IDs to delete (comma-separated): ").split(',')

# Delete the selected runs
for run_id in delete_ids:
    run_id = run_id.strip()
    run_dir = Path(f"/home/pgmlvol/mlflow/{exp_id}/{run_id}")
    
    if run_dir.exists():
        print(f"Deleting run: {run_id}")
        shutil.rmtree(run_dir)
    else:
        print(f"Run {run_id} not found!")

print("\nSelected runs deleted successfully.")
