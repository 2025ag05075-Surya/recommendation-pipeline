import numpy as np
import pandas as pd

from src.evaluation import metrics as m
from src.preparation.prepare import time_split
from src.utils import load_config
from src.validation.validate import validate_interactions

V = load_config()["validation"]


def _df(**kw):
    base = dict(row_id=[2], user_id=["U1"], item_id=["I1"], rating=["4"], timestamp=["2024-01-01 10:00:00"])
    base.update(kw)
    return pd.DataFrame(base)


def test_valid_row_passes():
    ok, iss = validate_interactions(_df(), V, {"U1"}, {"I1"})
    assert len(ok) == 1 and iss.empty


def test_invalid_rating_timestamp_and_orphans_rejected():
    for kw in [dict(rating=["9"]), dict(rating=["abc"]), dict(timestamp=["2031-01-01 00:00:00"]),
               dict(timestamp=["nope"]), dict(user_id=["U404"])]:
        ok, iss = validate_interactions(_df(**kw), V, {"U1"}, {"I1"})
        assert ok.empty and not iss.empty


def test_duplicates_flagged():
    df = pd.concat([_df(), _df().assign(row_id=3)], ignore_index=True)
    ok, iss = validate_interactions(df, V, {"U1"}, {"I1"})
    assert len(ok) == 1 and (iss.rule == "duplicate_row").any()


def test_time_split_has_no_overlap():
    ts = pd.date_range("2024-01-01", periods=100, freq="D")
    tr, te, cut = time_split(pd.DataFrame({"timestamp": ts}), 0.2)
    assert tr.timestamp.max() < te.timestamp.min() and len(te) == 20


def test_ranking_metrics():
    assert m.precision_at_k(["a", "b"], {"a"}, 2) == 0.5
    assert m.recall_at_k(["a", "b"], {"a", "c"}, 2) == 0.5
    assert np.isclose(m.ndcg_at_k(["a", "b"], {"a"}, 2), 1.0)
    assert m.ndcg_at_k(["b", "a"], {"a"}, 2) < 1.0
