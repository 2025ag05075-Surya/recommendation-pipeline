"""Run the whole pipeline end-to-end without any orchestrator (sequential, same stage functions).

    python run_pipeline.py                # all stages
    python run_pipeline.py --from train   # resume from a stage
"""
import argparse
import os
import time

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

from src.evaluation import evaluate
from src.features import feature_store
from src.ingestion import ingest
from src.preparation import prepare
from src.training import train
from src.transformation import transform
from src.utils import get_logger, load_config
from src.validation import validate

STAGES = [("ingest", ingest.run), ("validate", validate.run), ("prepare", prepare.run),
          ("transform", transform.run), ("features", feature_store.run), ("train", train.run),
          ("evaluate", evaluate.run)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="ingest", choices=[s for s, _ in STAGES])
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    log, cfg = get_logger("pipeline"), load_config(a.config)
    names = [s for s, _ in STAGES]
    for name, fn in STAGES[names.index(a.start):]:
        t0 = time.time()
        log.info(">>> %s", name)
        fn(cfg)
        log.info("<<< %s done in %.1fs", name, time.time() - t0)
