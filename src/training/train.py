"""Stage 6 -- Model training with MLflow tracking (params, training artifacts, model files)."""
import json
import pickle
import time

import mlflow
import pandas as pd
import scipy.sparse as sp

from src.features.feature_store import FeatureStore
from src.training.models import HybridRecommender, PopularityRecommender, SVDRecommender
from src.utils import ROOT, get_dir, get_logger, load_config, write_json

log = get_logger("training")


def setup_mlflow(cfg):
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    proc, models_dir, reports = get_dir(cfg, "processed"), get_dir(cfg, "models"), get_dir(cfg, "reports")
    maps = json.loads((proc / "index_maps.json").read_text())
    M = sp.load_npz(proc / "interaction_matrix.npz")
    G = sp.load_npz(proc / "item_genre_matrix.npz")
    user_ids, item_ids = list(maps["users"]), list(maps["items"])
    genre_names = sorted(maps["genres"], key=maps["genres"].get)
    gmean = float(M.data.mean())
    fs = FeatureStore(cfg)                                     # features are read from the feature layer

    mc, seed = cfg["models"], cfg["seed"]
    pop = PopularityRecommender(**mc["popularity"])
    svd = SVDRecommender(**mc["svd_cf"], seed=seed)
    candidates = {"popularity": pop, "svd_cf": svd}

    setup_mlflow(cfg)
    run_ids, summary = {}, {}
    for name in ["popularity", "svd_cf", "hybrid"]:
        with mlflow.start_run(run_name=name) as run:
            t0 = time.time()
            if name == "hybrid":
                model = HybridRecommender(svd, pop, fs, G, genre_names, **mc["hybrid"])
                params = {**mc["hybrid"], "components": "svd_cf+genre_affinity+popularity"}
            else:
                model = candidates[name].fit(M, user_ids, item_ids, gmean)
                params = dict(mc[name])
            secs = round(time.time() - t0, 3)
            mlflow.log_params({"model": name, "seed": seed, "train_interactions": int(M.nnz),
                               "n_users": M.shape[0], "n_items": M.shape[1], **params})
            mlflow.log_metric("train_seconds", secs)
            path = models_dir / f"{name}.pkl"
            with open(path, "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(str(path), artifact_path="model")
            mlflow.log_artifact(str(reports / "feature_catalog.md"), artifact_path="features")
            run_ids[name] = run.info.run_id
            summary[name] = {"run_id": run.info.run_id, "train_seconds": secs}
            log.info("trained %-10s in %.2fs (run %s)", name, secs, run.info.run_id[:8])
    write_json(run_ids, models_dir / "run_ids.json")
    return {"stage": "training", "models": list(run_ids)}


if __name__ == "__main__":
    run()
