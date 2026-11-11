"""Stage 5 -- Feature management: a small, file-based feature layer (offline store = Parquet).

The SAME `FeatureStore` object serves training and prediction, so a feature is defined once:
  build()              compute + persist features from the train split (as-of the last train timestamp)
  get_user_features()  / get_item_features() / get_pair_features()   read features for given ids
  catalog()            the documented feature registry (also exported to reports/feature_catalog.md)

All features are computed from TRAIN data only (as of the train cutoff) to avoid label leakage.
"""
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.transformation.transform import split_genres
from src.utils import get_dir, get_logger, load_config

log = get_logger("features")

# --------------------------------------------------------------------------------------------
# Feature registry / catalog
# --------------------------------------------------------------------------------------------
CATALOG = [
    # entity, name, dtype, description, computation
    ("user", "user_avg_rating", "float", "Mean rating the user has given", "mean(rating) over user's train interactions"),
    ("user", "user_rating_std", "float", "Spread of the user's ratings (0 if one rating)", "std(rating), NaN->0"),
    ("user", "user_activity_count", "int", "Number of interactions by the user", "count(interactions)"),
    ("user", "user_positive_ratio", "float", "Share of the user's ratings that are 'liked' (>= threshold)", "mean(rating >= 4)"),
    ("user", "user_days_since_last_interaction", "float", "Recency of the user's latest activity", "(as_of - max(timestamp)) in days"),
    ("user", "user_tenure_days", "float", "How long the user has been registered", "(as_of - signup_date) in days"),
    ("user", "user_preferred_category", "str", "Genre the user likes most", "most frequent genre among liked items (all items if none liked)"),
    ("user", "user_age", "int", "Age (median-imputed when missing/invalid)", "from cleaned users table"),
    ("item", "item_popularity", "int", "Number of interactions the item received", "count(interactions)"),
    ("item", "item_popularity_pct", "float", "Popularity percentile in [0,1]", "rank(item_popularity)/n_items"),
    ("item", "item_avg_rating", "float", "Raw mean rating of the item", "mean(rating)"),
    ("item", "item_avg_rating_smoothed", "float", "Bayesian-damped mean rating (robust for rarely-rated items)", "(sum + m*global_mean)/(n + m), m=10"),
    ("item", "item_positive_ratio", "float", "Share of ratings that are 'liked'", "mean(rating >= 4)"),
    ("item", "item_age_years", "float", "Age of the item at as-of date", "as_of.year - release_year"),
    ("item", "item_days_since_last_interaction", "float", "Recency of last interaction with the item", "(as_of - max(timestamp)) in days"),
    ("item", "item_n_genres", "int", "Number of genres on the item", "len(genres)"),
    ("interaction", "user_genre_affinity", "float", "How well the item's genres match the user's genre profile", "dot(user_genre_profile, item_genre_multi_hot) / n_item_genres"),
    ("interaction", "is_preferred_category", "int", "1 if the item carries the user's preferred category", "user_preferred_category in item.genres"),
    ("interaction", "expected_rating_baseline", "float", "Bias-style prior rating for (user, item)", "user_avg + item_avg_smoothed - global_mean, clipped to [1,5]"),
]


class FeatureStore:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.dir = get_dir(self.cfg, "features")
        self.proc = get_dir(self.cfg, "processed")
        self._user = self._item = self._profile = None

    # ------------------------------------------------------------------ build (offline)
    def build(self) -> dict:
        thr, m = self.cfg["evaluation"]["relevance_threshold"], self.cfg["models"]["popularity"]["min_ratings_smoothing"]
        users, items = pd.read_parquet(self.proc / "users.parquet"), pd.read_parquet(self.proc / "items.parquet")
        train = pd.read_parquet(self.proc / "train.parquet")
        as_of, gmean = train.timestamp.max(), train.rating.mean()
        train = train.assign(pos=(train.rating >= thr).astype(int))

        # user features
        ug = train.groupby("user_id").agg(user_avg_rating=("rating", "mean"), user_rating_std=("rating", "std"),
                                          user_activity_count=("rating", "size"), user_positive_ratio=("pos", "mean"),
                                          last=("timestamp", "max"))
        uf = users.set_index("user_id")[["age", "signup_date"]].join(ug, how="left")
        uf["user_rating_std"] = uf.user_rating_std.fillna(0)
        uf["user_avg_rating"] = uf.user_avg_rating.fillna(gmean)
        uf["user_activity_count"] = uf.user_activity_count.fillna(0).astype(int)
        uf["user_positive_ratio"] = uf.user_positive_ratio.fillna(0)
        uf["user_days_since_last_interaction"] = (as_of - uf["last"]).dt.total_seconds() / 86400
        uf["user_tenure_days"] = (as_of - uf.signup_date).dt.total_seconds() / 86400
        uf = uf.rename(columns={"age": "user_age"}).drop(columns=["last", "signup_date"])

        # user genre profile (liked items; falls back to all items) + preferred category
        ig = items.assign(genre=split_genres(items.genres)).explode("genre")[["item_id", "genre"]]
        tg = train.merge(ig, on="item_id")
        liked = tg[tg.pos == 1]
        base = tg[~tg.user_id.isin(liked.user_id)]
        counts = pd.concat([liked, base]).groupby(["user_id", "genre"]).size().unstack(fill_value=0)
        profile = counts.div(counts.sum(axis=1), axis=0)
        uf["user_preferred_category"] = counts.idxmax(axis=1).reindex(uf.index).fillna("Unknown")

        # item features
        it = train.groupby("item_id").agg(item_popularity=("rating", "size"), item_avg_rating=("rating", "mean"),
                                          item_positive_ratio=("pos", "mean"), rsum=("rating", "sum"), last=("timestamp", "max"))
        itf = items.set_index("item_id")[["release_year", "genres"]].join(it, how="left")
        itf["item_popularity"] = itf.item_popularity.fillna(0).astype(int)
        itf["item_popularity_pct"] = itf.item_popularity.rank(pct=True)
        n, rs = itf.item_popularity, itf.rsum.fillna(0)
        itf["item_avg_rating_smoothed"] = (rs + m * gmean) / (n + m)
        itf["item_avg_rating"] = itf.item_avg_rating.fillna(gmean)
        itf["item_positive_ratio"] = itf.item_positive_ratio.fillna(0)
        itf["item_age_years"] = as_of.year - itf.release_year
        itf["item_days_since_last_interaction"] = ((as_of - itf["last"]).dt.total_seconds() / 86400).fillna(
            (as_of - train.timestamp.min()).days)
        itf["item_n_genres"] = itf.genres.str.count(r"\|") + 1
        itf = itf.drop(columns=["rsum", "last", "release_year"])

        uf.to_parquet(self.dir / "user_features.parquet")
        itf.to_parquet(self.dir / "item_features.parquet")
        profile.to_parquet(self.dir / "user_genre_profile.parquet")
        meta = {"global_mean": float(gmean), "as_of": str(as_of)}
        (self.dir / "feature_meta.json").write_text(json.dumps(meta))
        self.export_catalog()
        log.info("features built: users=%d items=%d (as_of=%s)", len(uf), len(itf), as_of)
        return {"user_features": uf.shape, "item_features": itf.shape}

    # ------------------------------------------------------------------ serve (online / training)
    def load(self):
        if self._user is None:
            self._user = pd.read_parquet(self.dir / "user_features.parquet")
            self._item = pd.read_parquet(self.dir / "item_features.parquet")
            self._profile = pd.read_parquet(self.dir / "user_genre_profile.parquet")
            self._gmean = float(json.loads((self.dir / "feature_meta.json").read_text())["global_mean"])
        return self

    def get_user_features(self, user_ids):
        return self.load()._user.reindex(user_ids)

    def get_item_features(self, item_ids):
        return self.load()._item.reindex(item_ids)

    def get_pair_features(self, user_ids, item_ids) -> pd.DataFrame:
        """Interaction-level features for aligned (user_id, item_id) pairs."""
        self.load()
        u, i = self.get_user_features(user_ids), self.get_item_features(item_ids)
        out = pd.DataFrame({"user_id": list(user_ids), "item_id": list(item_ids)})
        prof = self._profile.reindex(list(user_ids)).fillna(0)
        genres = [g.split("|") for g in i.genres.fillna("Unknown")]
        aff = [sum(row.get(x, 0) for x in g) / len(g) for row, g in zip(prof.to_dict("index").values(), genres)]
        out["user_genre_affinity"] = aff
        out["is_preferred_category"] = [int(pc in g) for pc, g in zip(u.user_preferred_category.fillna("Unknown"), genres)]
        out["expected_rating_baseline"] = (u.user_avg_rating.fillna(self._gmean).to_numpy()
                                           + i.item_avg_rating_smoothed.fillna(self._gmean).to_numpy() - self._gmean).clip(1, 5)
        return out

    def genre_affinity_matrix(self, user_ids, item_ids, genre_matrix: sp.csr_matrix, genre_names) -> np.ndarray:
        """Vectorised user x item version of `user_genre_affinity` (used for ranking all items)."""
        prof = self.load()._profile.reindex(index=user_ids, columns=genre_names).fillna(0).to_numpy()
        n_genres = np.asarray(genre_matrix.sum(axis=1)).ravel().clip(min=1)
        return (prof @ genre_matrix.T.toarray()) / n_genres

    # ------------------------------------------------------------------ catalog
    @staticmethod
    def catalog() -> pd.DataFrame:
        return pd.DataFrame(CATALOG, columns=["entity", "feature", "dtype", "description", "computation"])

    def export_catalog(self):
        rep = get_dir(self.cfg, "reports")
        cat = self.catalog()
        cat.to_csv(rep / "feature_catalog.csv", index=False)
        lines = ["# Feature Catalog", "",
                 "All features are computed from the **train split only**, as of the last train timestamp, and are served "
                 "identically to training and prediction through `FeatureStore` (`src/features/feature_store.py`). "
                 "Offline store: Parquet files in `data/features/`.", "",
                 "| Entity | Feature | Type | Description | Computation |", "|---|---|---|---|---|"]
        lines += [f"| {r.entity} | `{r.feature}` | {r.dtype} | {r.description} | {r.computation} |" for r in cat.itertuples()]
        (rep / "feature_catalog.md").write_text("\n".join(lines) + "\n")


def run(cfg: dict | None = None) -> dict:
    shapes = FeatureStore(cfg).build()
    return {"stage": "features", **{k: list(v) for k, v in shapes.items()}}


if __name__ == "__main__":
    run()
