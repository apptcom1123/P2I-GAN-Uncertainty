"""Preview or mark stale local MLflow RUNNING records as KILLED."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking URI (defaults to the repository SQLite database)",
    )
    parser.add_argument(
        "--experiment-name",
        default="p2igan-five-model-uncertainty",
    )
    parser.add_argument(
        "--older-than-hours",
        type=float,
        default=24.0,
        help="Only select RUNNING records older than this age",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually mark selected records KILLED; otherwise only preview",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(args.experiment_name)
    if experiment is None:
        raise SystemExit(f"Experiment not found: {args.experiment_name}")

    cutoff = int((time.time() - args.older_than_hours * 3600.0) * 1000)
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="attributes.status = 'RUNNING'",
        order_by=["attributes.start_time ASC"],
    )
    selected = [run for run in runs if (run.info.start_time or 0) < cutoff]
    action = "KILL" if args.apply else "PREVIEW"
    for run in selected:
        age_hours = (time.time() * 1000 - (run.info.start_time or 0)) / 3_600_000
        print(
            f"{action} run_id={run.info.run_id} "
            f"name={run.data.tags.get('mlflow.runName', '')!r} age_hours={age_hours:.1f}"
        )
        if args.apply:
            client.set_terminated(run.info.run_id, status="KILLED")
            client.set_tag(run.info.run_id, "status", "stale-record-cleanup")
    print(f"selected={len(selected)} applied={args.apply}")


if __name__ == "__main__":
    main()
