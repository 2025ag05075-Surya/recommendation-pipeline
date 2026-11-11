"""Stage 2 -- Data validation.

Rule severities
  error   -> row is REJECTED, written to data/errors/<table>_errors.csv, excluded from downstream data
  warning -> row is KEPT, the problem is logged; the field is repaired in the preparation stage

Checks: missing values, duplicates, incorrect data types, invalid ratings, invalid timestamps,
referential integrity (interactions -> users / items).
"""
import pandas as pd

from src.utils import get_dir, get_logger, load_config, write_json

log = get_logger("validation")


def _blank(s: pd.Series) -> pd.Series:
    return s.isna() | (s.astype(str).str.strip() == "")


class Validator:
    def __init__(self, table: str, df: pd.DataFrame):
        self.table, self.df, self.issues = table, df, []

    def flag(self, mask: pd.Series, rule: str, severity: str, column: str):
        mask = mask.fillna(False)
        if mask.any():
            sub = self.df.loc[mask, ["row_id", column]].rename(columns={column: "value"})
            sub.insert(0, "column", column)
            sub["table"], sub["rule"], sub["severity"] = self.table, rule, severity
            self.issues.append(sub[["table", "row_id", "column", "rule", "severity", "value"]])

    def result(self):
        cols = ["table", "row_id", "column", "rule", "severity", "value"]
        iss = pd.concat(self.issues, ignore_index=True) if self.issues else pd.DataFrame(columns=cols)
        rejected = set(iss.loc[iss.severity == "error", "row_id"])
        valid = self.df[~self.df.row_id.isin(rejected)].copy()
        return valid, iss


def validate_users(df, v):
    c = Validator("users", df)
    c.flag(_blank(df.user_id), "missing_value", "error", "user_id")
    exact = df.drop(columns="row_id").duplicated(keep="first")
    c.flag(exact, "duplicate_row", "error", "user_id")
    c.flag(df.duplicated("user_id", keep="first") & ~exact, "duplicate_key", "error", "user_id")
    age = pd.to_numeric(df.age, errors="coerce")
    c.flag(_blank(df.age), "missing_value", "warning", "age")
    c.flag(~_blank(df.age) & age.isna(), "incorrect_dtype", "warning", "age")
    c.flag(age.notna() & ((age < v["age_min"]) | (age > v["age_max"])), "out_of_range", "warning", "age")
    c.flag(_blank(df.gender), "missing_value", "warning", "gender")
    c.flag(~_blank(df.gender) & ~df.gender.isin(v["allowed_genders"]), "invalid_category", "warning", "gender")
    c.flag(~_blank(df.signup_date) & pd.to_datetime(df.signup_date, errors="coerce").isna(), "invalid_timestamp", "warning", "signup_date")
    return c.result()


def validate_items(df, v):
    c = Validator("items", df)
    c.flag(_blank(df.item_id), "missing_value", "error", "item_id")
    exact = df.drop(columns="row_id").duplicated(keep="first")
    c.flag(exact, "duplicate_row", "error", "item_id")
    c.flag(df.duplicated("item_id", keep="first") & ~exact, "duplicate_key", "error", "item_id")
    for col in ["title", "category", "genres"]:
        c.flag(_blank(df[col]), "missing_value", "warning", col)
    yr = pd.to_numeric(df.release_year, errors="coerce")
    c.flag(~_blank(df.release_year) & yr.isna(), "incorrect_dtype", "warning", "release_year")
    c.flag(yr.notna() & ((yr < v["release_year_min"]) | (yr > v["release_year_max"])), "out_of_range", "warning", "release_year")
    return c.result()


def validate_interactions(df, v, valid_users, valid_items):
    c = Validator("interactions", df)
    c.flag(_blank(df.user_id), "missing_value", "error", "user_id")
    c.flag(_blank(df.item_id), "missing_value", "error", "item_id")
    c.flag(_blank(df.rating), "missing_value", "error", "rating")
    c.flag(_blank(df.timestamp), "missing_value", "error", "timestamp")
    c.flag(df.duplicated(["user_id", "item_id", "rating", "timestamp"], keep="first"), "duplicate_row", "error", "user_id")

    rating = pd.to_numeric(df.rating, errors="coerce")
    c.flag(~_blank(df.rating) & rating.isna(), "incorrect_dtype", "error", "rating")
    c.flag(rating.notna() & ((rating < v["rating_min"]) | (rating > v["rating_max"])), "invalid_rating", "error", "rating")

    ts = pd.to_datetime(df.timestamp, errors="coerce")
    c.flag(~_blank(df.timestamp) & ts.isna(), "invalid_timestamp_format", "error", "timestamp")
    c.flag(ts.notna() & ((ts < pd.Timestamp(v["min_timestamp"])) | (ts > pd.Timestamp(v["max_timestamp"]))),
           "timestamp_out_of_range", "error", "timestamp")

    c.flag(~_blank(df.user_id) & ~df.user_id.isin(valid_users), "referential_integrity", "error", "user_id")
    c.flag(~_blank(df.item_id) & ~df.item_id.isin(valid_items), "referential_integrity", "error", "item_id")
    return c.result()


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    v = cfg["validation"]
    proc, errs, reports = get_dir(cfg, "processed"), get_dir(cfg, "errors"), get_dir(cfg, "reports")
    raw = {t: pd.read_parquet(proc / f"{t}_raw.parquet") for t in ["users", "items", "interactions"]}

    users_ok, users_iss = validate_users(raw["users"], v)
    items_ok, items_iss = validate_items(raw["items"], v)
    inter_ok, inter_iss = validate_interactions(raw["interactions"], v, set(users_ok.user_id), set(items_ok.item_id))

    summary = {}
    for t, raw_df, ok, iss in [("users", raw["users"], users_ok, users_iss),
                               ("items", raw["items"], items_ok, items_iss),
                               ("interactions", raw["interactions"], inter_ok, inter_iss)]:
        ok.to_parquet(proc / f"{t}_validated.parquet", index=False)
        iss.to_csv(errs / f"{t}_issues.csv", index=False)                       # full issue log
        rejected = raw_df[~raw_df.row_id.isin(ok.row_id)]
        reasons = iss[iss.severity == "error"].groupby("row_id").rule.agg(lambda s: ";".join(sorted(set(s))))
        rejected.assign(reject_reasons=rejected.row_id.map(reasons)).to_csv(errs / f"{t}_errors.csv", index=False)
        summary[t] = {"rows_in": len(raw_df), "rows_valid": len(ok), "rows_rejected": len(rejected),
                      "rejection_rate_pct": round(100 * len(rejected) / len(raw_df), 2),
                      "issues_by_rule": (iss.groupby(["column", "rule", "severity"]).size().reset_index(name="count")
                                         .to_dict("records"))}
        log.info("%-12s in=%d valid=%d rejected=%d issues=%d", t, len(raw_df), len(ok), len(rejected), len(iss))

    write_json(summary, reports / "validation_summary.json")
    _write_markdown(summary, reports / "validation_report.md")
    return {"stage": "validation", **{f"{t}_rejected": s["rows_rejected"] for t, s in summary.items()}}


def _write_markdown(summary: dict, path):
    lines = ["# Data Validation Report", "",
             "Severity **error** = row rejected (written to `data/errors/<table>_errors.csv`); "
             "**warning** = row kept, value repaired in the preparation stage. Full per-cell log: `data/errors/<table>_issues.csv`.", "",
             "## Summary", "", "| Table | Rows in | Valid | Rejected | Rejection rate |", "|---|---:|---:|---:|---:|"]
    for t, s in summary.items():
        lines.append(f"| {t} | {s['rows_in']:,} | {s['rows_valid']:,} | {s['rows_rejected']:,} | {s['rejection_rate_pct']}% |")
    for t, s in summary.items():
        lines += ["", f"## {t} -- issues by rule", "", "| Column | Rule | Severity | Count |", "|---|---|---|---:|"]
        for r in sorted(s["issues_by_rule"], key=lambda r: (r["severity"], -r["count"])):
            lines.append(f"| {r['column']} | {r['rule']} | {r['severity']} | {r['count']} |")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    run()
