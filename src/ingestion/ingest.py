# Stage 1 -- Data ingestion.
import hashlib
from datetime import datetime, timezone

import pandas as pd

from src.ingestion.generate_data import write_raw
from src.utils import get_dir, get_logger, load_config, write_json

log = get_logger("ingestion")
TABLES = ["users", "items", "interactions"]


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    raw, processed, reports = get_dir(cfg, "raw"), get_dir(cfg, "processed"), get_dir(cfg, "reports")

    if cfg["ingestion"]["source"] == "synthetic" and (
            cfg["ingestion"].get("regenerate", True) or not all((raw / f"{t}.csv").exists() for t in TABLES)):
        write_raw(cfg)

    meta = {"ingested_at_utc": datetime.now(timezone.utc).isoformat(), "source": cfg["ingestion"]["source"], "tables": {}}
    for t in TABLES:
        path = raw / f"{t}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing raw file: {path}")
        df = pd.read_csv(path, dtype=str, keep_default_na=True)
        df.insert(0, "row_id", range(2, len(df) + 2))       # CSV line number (header = line 1)
        df.to_parquet(processed / f"{t}_raw.parquet", index=False)
        meta["tables"][t] = {"rows": len(df), "columns": list(df.columns[1:]),
                             "md5": hashlib.md5(path.read_bytes()).hexdigest()}
        log.info("Ingested %-12s rows=%d", t, len(df))
    write_json(meta, reports / "ingestion_log.json")
    return {"stage": "ingestion", **{f"{t}_rows": meta["tables"][t]["rows"] for t in TABLES}}


if __name__ == "__main__":
    run()
