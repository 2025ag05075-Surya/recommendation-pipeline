"""Stage 7 -- Model evaluation on the time-based test split; metrics logged to the MLflow runs."""
import json
import pickle

import matplotlib
import mlflow
import numpy as np
import pandas as pd
import scipy.sparse as sp

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation import metrics as m
from src.training.train import setup_mlflow
from src.utils import get_dir, get_logger, load_config

log = get_logger("evaluation")


def evaluate_model(model, train_M, test, maps, G_dense, k, thr):
    u_map, i_map = maps["users"], maps["items"]
    user_ids = list(u_map)
    rel = (test[test.rating >= thr].groupby("user_id").item_id.apply(set))
    rel = rel[rel.index.isin(u_map)]
    eval_users = list(rel.index)

    S = model.score_all(eval_users)
    rows = np.array([u_map[u] for u in eval_users])
    S[train_M[rows].toarray() > 0] = -np.inf                       # never recommend already-seen items
    top = np.argsort(-S, axis=1)[:, :k]
    inv_items = np.array(list(i_map))

    P, R, N, D = [], [], [], []
    for u, idx in zip(eval_users, top):
        recs, rel_u = list(inv_items[idx]), rel[u]
        P.append(m.precision_at_k(recs, rel_u, k)); R.append(m.recall_at_k(recs, rel_u, k))
        N.append(m.ndcg_at_k(recs, rel_u, k)); D.append(m.intra_list_diversity(idx, G_dense))
    out = {f"precision@{k}": np.mean(P), f"recall@{k}": np.mean(R), f"ndcg@{k}": np.mean(N),
           "coverage": len(np.unique(top)) / len(i_map), "diversity": np.mean(D)}

    pred = model.predict_rating(test.user_id.tolist(), test.item_id.tolist())
    out["rmse"], out["mae"] = m.rmse(test.rating, pred), m.mae(test.rating, pred)
    out["n_eval_users"] = len(eval_users)
    return {k_: float(v) for k_, v in out.items()}


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    proc, models_dir, reports = get_dir(cfg, "processed"), get_dir(cfg, "models"), get_dir(cfg, "reports")
    k, thr = cfg["evaluation"]["k"], cfg["evaluation"]["relevance_threshold"]
    maps = json.loads((proc / "index_maps.json").read_text())
    M = sp.load_npz(proc / "interaction_matrix.npz")
    G = sp.load_npz(proc / "item_genre_matrix.npz").toarray()
    test = pd.read_parquet(proc / "test.parquet")
    test = test[test.user_id.isin(maps["users"]) & test.item_id.isin(maps["items"])]
    run_ids = json.loads((models_dir / "run_ids.json").read_text())

    setup_mlflow(cfg)
    results = {}
    for name, rid in run_ids.items():
        model = pickle.load(open(models_dir / f"{name}.pkl", "rb"))
        res = evaluate_model(model, M, test, maps, G, k, thr)
        results[name] = res
        with mlflow.start_run(run_id=rid):
            mlflow.log_metrics({kk.replace('@', '_at_'): v for kk, v in res.items()})
        log.info("%-10s %s", name, {kk: round(v, 4) for kk, v in res.items()})

    df = pd.DataFrame(results).T
    df.index.name = "model"
    df.to_csv(reports / "model_comparison.csv")
    cols = [f"precision@{k}", f"recall@{k}", f"ndcg@{k}", "coverage", "diversity", "rmse", "mae"]
    best = {c: (df[c].idxmin() if c in ("rmse", "mae") else df[c].idxmax()) for c in cols}
    lines = [f"# Model Evaluation (K={k}, relevant = rating >= {thr}, time-based test split)", "",
             f"Evaluated on {int(df.n_eval_users.iloc[0]):,} test users with >=1 relevant item. Best value per metric in **bold**.", "",
             "| Model | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for name, r in df.iterrows():
        lines.append(f"| {name} | " + " | ".join(f"**{r[c]:.4f}**" if best[c] == name else f"{r[c]:.4f}" for c in cols) + " |")
    (reports / "model_comparison.md").write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
    for ax, c in zip(axes, cols[:5]):
        ax.bar(df.index, df[c], color=["#9aa5b1", "#2f6fba", "#e0883a"][:len(df)])
        ax.set_title(c, fontsize=10); ax.tick_params(axis="x", labelrotation=20, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    plt.tight_layout(); plt.savefig(reports / "model_comparison.png", dpi=160); plt.close()

    with mlflow.start_run(run_id=run_ids["svd_cf"]):
        mlflow.log_artifact(str(reports / "model_comparison.md"), artifact_path="evaluation")
    return {"stage": "evaluation", "best_ndcg": best[f"ndcg@{k}"]}


if __name__ == "__main__":
    run()
