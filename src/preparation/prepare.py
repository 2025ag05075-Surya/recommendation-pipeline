"""Stage 3 -- Data preparation: cleaning, standardisation, de-duplication, time-based split."""
import pandas as pd

from src.utils import get_dir, get_logger, load_config, write_json

log = get_logger("preparation")


def clean_users(df: pd.DataFrame, v: dict) -> pd.DataFrame:
    df = df.drop(columns="row_id").copy()
    df["user_id"] = df.user_id.str.strip().str.upper()
    age = pd.to_numeric(df.age, errors="coerce")
    age = age.where(age.between(v["age_min"], v["age_max"]))                 # out-of-range -> NaN
    df["age"] = age.fillna(age.median()).round().astype(int)                 # impute with median
    g = df.gender.str.strip().str.upper()
    df["gender"] = g.where(g.isin(v["allowed_genders"]), "Unknown")
    sd = pd.to_datetime(df.signup_date, errors="coerce")
    df["signup_date"] = sd.fillna(sd.median())
    return df.drop_duplicates("user_id").reset_index(drop=True)


def clean_items(df: pd.DataFrame, v: dict) -> pd.DataFrame:
    df = df.drop(columns="row_id").copy()
    df["item_id"] = df.item_id.str.strip().str.upper()
    df["title"] = df.title.fillna("Unknown title").str.strip()
    df["genres"] = df.genres.fillna("Unknown").str.strip()
    df["category"] = df.category.fillna(df.genres.str.split("|").str[0]).str.strip()
    yr = pd.to_numeric(df.release_year, errors="coerce")
    yr = yr.where(yr.between(v["release_year_min"], v["release_year_max"]))
    df["release_year"] = yr.fillna(yr.median()).astype(int)
    return df.drop_duplicates("item_id").reset_index(drop=True)


def clean_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns="row_id").copy()
    df["user_id"] = df.user_id.str.strip().str.upper()
    df["item_id"] = df.item_id.str.strip().str.upper()
    df["rating"] = pd.to_numeric(df.rating).astype(float)
    df["timestamp"] = pd.to_datetime(df.timestamp)
    # one rating per (user, item): keep the most recent
    df = df.sort_values("timestamp").drop_duplicates(["user_id", "item_id"], keep="last")
    return df.reset_index(drop=True)


def time_split(inter: pd.DataFrame, test_fraction: float):
    cutoff = inter.timestamp.quantile(1 - test_fraction)
    return inter[inter.timestamp <= cutoff].copy(), inter[inter.timestamp > cutoff].copy(), cutoff


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    proc = get_dir(cfg, "processed")
    users = clean_users(pd.read_parquet(proc / "users_validated.parquet"), cfg["validation"])
    items = clean_items(pd.read_parquet(proc / "items_validated.parquet"), cfg["validation"])
    inter = clean_interactions(pd.read_parquet(proc / "interactions_validated.parquet"))

    train, test, cutoff = time_split(inter, cfg["split"]["test_fraction"])
    users.to_parquet(proc / "users.parquet", index=False)
    items.to_parquet(proc / "items.parquet", index=False)
    train.to_parquet(proc / "train.parquet", index=False)
    test.to_parquet(proc / "test.parquet", index=False)

    info = {"cutoff_timestamp": cutoff, "n_users": len(users), "n_items": len(items),
            "n_interactions": len(inter), "n_train": len(train), "n_test": len(test),
            "train_range": [train.timestamp.min(), train.timestamp.max()],
            "test_range": [test.timestamp.min(), test.timestamp.max()],
            "test_users_unseen_in_train": int((~test.user_id.isin(train.user_id)).groupby(test.user_id).first().sum()),
            "sparsity_pct": round(100 * (1 - len(inter) / (len(users) * len(items))), 2)}
    write_json(info, get_dir(cfg, "reports") / "split_info.json")
    log.info("users=%d items=%d train=%d test=%d cutoff=%s", len(users), len(items), len(train), len(test), cutoff)
    return {"stage": "preparation", "n_train": len(train), "n_test": len(test)}


if __name__ == "__main__":
    run()
