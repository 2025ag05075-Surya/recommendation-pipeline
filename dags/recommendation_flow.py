"""Stage 8 -- Orchestration with Prefect.

Task graph (each task receives its upstream task's result, so Prefect infers the dependency):

  ingest -> validate -> prepare -> transform -> build_features -> train -> evaluate -> report

Run once:        python dags/recommendation_flow.py
Run on schedule: python dags/recommendation_flow.py --serve      (daily at 02:00)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefect import flow, get_run_logger, task

from src.evaluation import evaluate
from src.features import feature_store
from src.ingestion import ingest
from src.preparation import prepare
from src.training import train
from src.transformation import transform
from src.utils import load_config
from src.validation import validate


def _stage(fn, name, **task_kwargs):
    @task(name=name, retries=1, retry_delay_seconds=5, **task_kwargs)
    def _t(cfg: dict, upstream: dict | None = None) -> dict:      # `upstream` only creates the dependency edge
        out = fn(cfg)
        get_run_logger().info("%s finished: %s", name, out)
        return out
    return _t


ingest_t = _stage(ingest.run, "ingest")
validate_t = _stage(validate.run, "validate")
prepare_t = _stage(prepare.run, "prepare")
transform_t = _stage(transform.run, "transform")
features_t = _stage(feature_store.run, "build_features")
train_t = _stage(train.run, "train")
evaluate_t = _stage(evaluate.run, "evaluate")


@task(name="report")
def report_t(cfg: dict, upstream: dict) -> str:
    from src.utils import ROOT
    text = (ROOT / cfg["paths"]["reports"] / "model_comparison.md").read_text()
    get_run_logger().info("Final results:\n%s", text)
    return text


@flow(name="recommendation-pipeline", log_prints=True)
def recommendation_pipeline(config_path: str | None = None):
    cfg = load_config(config_path)
    a = ingest_t(cfg)
    b = validate_t(cfg, a)
    c = prepare_t(cfg, b)
    d = transform_t(cfg, c)
    e = features_t(cfg, d)
    f = train_t(cfg, e)
    g = evaluate_t(cfg, f)
    return report_t(cfg, g)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true", help="serve the flow on a daily cron schedule")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    if args.serve:
        recommendation_pipeline.serve(name="daily-recsys", cron="0 2 * * *")
    else:
        recommendation_pipeline(args.config)
