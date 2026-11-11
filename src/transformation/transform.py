"""Stage 4 -- Data transformation: derived datasets built from the TRAIN split only (no leakage).

Outputs (data/processed/):
  user_activity_summary.parquet   per-user activity statistics
  item_popularity.parquet         per-item popularity statistics
  interaction_matrix.npz          sparse user x item rating matrix (CSR)
  item_genre_matrix.npz           sparse item x genre multi-hot matrix
  index_maps.json                 user_id/item_id/genre -> matrix index
"""
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.utils import get_dir, get_logger, load_config

log = get_logger("transformation")


def split_genres(s: pd.Series) -> pd.Series:
    return s.str.split("|")


def build_index_maps(users, items):
    genres = sorted({g for gs in split_genres(items.genres) for g in gs})
    return ({u: i for i, u in enumerate(users.user_id)}, {i: j for j, i in enumerate(items.item_id)},
            {g: k for k, g in enumerate(genres)})


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    proc, thr = get_dir(cfg, "processed"), cfg["evaluation"]["relevance_threshold"]
    users, items = pd.read_parquet(proc / "users.parquet"), pd.read_parquet(proc / "items.parquet")
    train = pd.read_parquet(proc / "train.parquet")
    u_map, i_map, g_map = build_index_maps(users, items)
    as_of = train.timestamp.max()

    # --- user activity summary
    ua = train.groupby("user_id").agg(n_interactions=("rating", "size"), avg_rating=("rating", "mean"),
                                      n_positive=("rating", lambda r: int((r >= thr).sum())),
                                      first_interaction=("timestamp", "min"), last_interaction=("timestamp", "max")).reset_index()
    ua["active_days"] = (ua.last_interaction - ua.first_interaction).dt.days + 1
    ua.to_parquet(proc / "user_activity_summary.parquet", index=False)

    # --- item popularity
    ip = train.groupby("item_id").agg(n_interactions=("rating", "size"), avg_rating=("rating", "mean"),
                                      n_positive=("rating", lambda r: int((r >= thr).sum())),
                                      last_interaction=("timestamp", "max")).reset_index()
    ip["popularity_rank"] = ip.n_interactions.rank(ascending=False, method="min").astype(int)
    ip.sort_values("popularity_rank").to_parquet(proc / "item_popularity.parquet", index=False)

    # --- user x item matrix (train only)
    r = train[train.user_id.isin(u_map) & train.item_id.isin(i_map)]
    M = sp.csr_matrix((r.rating.to_numpy(), (r.user_id.map(u_map), r.item_id.map(i_map))), shape=(len(u_map), len(i_map)))
    sp.save_npz(proc / "interaction_matrix.npz", M)

    # --- item x genre multi-hot
    rows, cols = zip(*[(i_map[i], g_map[g]) for i, gs in zip(items.item_id, split_genres(items.genres)) for g in gs])
    sp.save_npz(proc / "item_genre_matrix.npz",
                sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(i_map), len(g_map))))
    (proc / "index_maps.json").write_text(json.dumps({"users": u_map, "items": i_map, "genres": g_map, "as_of": str(as_of)}))

    density = M.nnz / (M.shape[0] * M.shape[1])
    log.info("matrix=%s nnz=%d density=%.3f%% as_of=%s", M.shape, M.nnz, 100 * density, as_of)
    return {"stage": "transformation", "matrix_shape": list(M.shape), "nnz": int(M.nnz)}


if __name__ == "__main__":
    run()
