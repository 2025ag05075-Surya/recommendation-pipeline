"""Synthetic MovieLens-style data generator (used when no public dataset is reachable).

The data has a *real* latent structure (genre tastes + latent factors + item popularity), so a
personalized model can genuinely beat a popularity baseline. A small share of rows is corrupted on
purpose (duplicates, nulls, bad types, invalid ratings/timestamps, orphan foreign keys) so that the
validation stage has something real to catch.
"""
import numpy as np
import pandas as pd

from src.utils import get_dir, get_logger, load_config

log = get_logger("ingestion.generate")

GENRES = ["Action", "Comedy", "Drama", "Romance", "Thriller", "SciFi",
          "Horror", "Documentary", "Animation", "Fantasy", "Crime", "Adventure"]


def generate(cfg: dict) -> dict:
    s, rng = cfg["synthetic"], np.random.default_rng(cfg["seed"])
    nu, ni, g = s["n_users"], s["n_items"], len(GENRES)
    start, end = pd.Timestamp(s["start_date"]), pd.Timestamp(s["end_date"])
    d = 8

    # ---- items ---------------------------------------------------------------------------
    G = np.zeros((ni, g))
    for i in range(ni):
        G[i, rng.choice(g, size=rng.integers(1, 4), replace=False)] = 1
    item_ids = [f"I{i + 1:04d}" for i in range(ni)]
    genres_str = ["|".join(GENRES[j] for j in np.where(G[i])[0]) for i in range(ni)]
    items = pd.DataFrame({
        "item_id": item_ids,
        "title": [f"Title {i + 1}" for i in range(ni)],
        "category": [x.split("|")[0] for x in genres_str],
        "genres": genres_str,
        "release_year": rng.integers(1985, 2025, ni),
    })

    # ---- users ---------------------------------------------------------------------------
    user_ids = [f"U{u + 1:04d}" for u in range(nu)]
    signup = start + pd.to_timedelta(rng.uniform(0, (end - start).days - 120, nu), unit="D")
    users = pd.DataFrame({
        "user_id": user_ids,
        "age": rng.integers(16, 70, nu),
        "gender": rng.choice(["M", "F", "O"], nu, p=[0.48, 0.48, 0.04]),
        "signup_date": signup.strftime("%Y-%m-%d"),
    })

    # ---- latent affinity -----------------------------------------------------------------
    P, Q = rng.normal(size=(nu, d)), rng.normal(size=(ni, d))
    taste = rng.dirichlet(np.full(g, 0.3), nu)                   # per-user genre taste
    A = P @ Q.T / np.sqrt(d) + 2.5 * (taste @ G.T)               # users x items
    A = (A - A.mean()) / A.std()
    log_pop = np.log(rng.zipf(1.8, ni).clip(max=60).astype(float))  # skewed item popularity
    u_bias, i_bias = rng.normal(0, 0.35, nu), rng.normal(0, 0.35, ni)

    rows = []
    for u in range(nu):
        n_u = int(np.clip(rng.lognormal(np.log(s["avg_interactions_per_user"] * 0.75), 0.7), 8, ni // 2))
        logits = log_pop + 1.6 * A[u]
        p = np.exp(logits - logits.max()); p /= p.sum()
        chosen = rng.choice(ni, size=n_u, replace=False, p=p)
        rating = 3.4 + u_bias[u] + i_bias[chosen] + 0.9 * A[u, chosen] + rng.normal(0, 0.6, n_u)
        rating = np.clip(np.rint(rating), 1, 5).astype(int)
        span = (end - signup[u]).total_seconds()
        ts = signup[u] + pd.to_timedelta(rng.uniform(0, span, n_u), unit="s")
        rows.append(pd.DataFrame({"user_id": user_ids[u], "item_id": np.array(item_ids)[chosen],
                                  "rating": rating, "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S")}))
    inter = pd.concat(rows, ignore_index=True)
    return _inject_defects(users, items, inter, cfg, rng)


def _inject_defects(users, items, inter, cfg, rng):
    r = cfg["synthetic"]["defect_rate"]
    users, items, inter = users.astype(str), items.astype(str), inter.astype(str)

    def pick(df, frac):
        return rng.choice(df.index.to_numpy(), size=max(1, int(len(df) * frac)), replace=False)

    # users
    users.loc[pick(users, 3 * r), "age"] = np.nan                       # missing (warning)
    users.loc[pick(users, r), "age"] = rng.choice(["abc", "-5", "230"], size=max(1, int(len(users) * r)))
    users.loc[pick(users, 2 * r), "gender"] = np.nan
    users.loc[pick(users, r / 2), "signup_date"] = "not_a_date"
    users = pd.concat([users, users.loc[pick(users, r)]], ignore_index=True)       # duplicates

    # items
    items.loc[pick(items, 2 * r), ["category", "genres"]] = np.nan
    items.loc[pick(items, r), "title"] = np.nan
    items.loc[pick(items, r), "release_year"] = "unknown"
    items = pd.concat([items, items.loc[pick(items, r)]], ignore_index=True)

    # interactions
    n = len(inter)
    inter.loc[pick(inter, r / 2), "rating"] = np.nan
    bad = pick(inter, r)
    inter.loc[bad, "rating"] = rng.choice(["0", "6", "-1", "11", "five"], size=len(bad))
    bad = pick(inter, r)
    inter.loc[bad, "timestamp"] = rng.choice(["2031-04-01 10:00:00", "N/A", "1999-12-31 00:00:00", "13/45/2024"], size=len(bad))
    inter.loc[pick(inter, r / 2), "user_id"] = "U9999"                   # orphan FK
    inter.loc[pick(inter, r / 2), "item_id"] = "I9999"                   # orphan FK
    inter.loc[pick(inter, r / 3), "user_id"] = np.nan
    inter = pd.concat([inter, inter.loc[pick(inter, 1.5 * r)]], ignore_index=True)  # exact duplicates
    inter = inter.sample(frac=1, random_state=cfg["seed"]).reset_index(drop=True)
    log.info("Generated users=%d items=%d interactions=%d (rows incl. injected defects)", len(users), len(items), len(inter))
    return {"users": users, "items": items, "interactions": inter}


def write_raw(cfg: dict) -> None:
    raw = get_dir(cfg, "raw")
    for name, df in generate(cfg).items():
        df.to_csv(raw / f"{name}.csv", index=False)


if __name__ == "__main__":
    write_raw(load_config())
