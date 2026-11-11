"""Recommender models. All expose the same interface so evaluation treats them uniformly:

    score_all(user_ids)            -> ndarray [n_users, n_items] ranking scores (higher = better)
    predict_rating(users, items)   -> ndarray of predicted ratings (clipped to 1..5)
"""
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD


class Base:
    name = "base"

    def _fit_common(self, M, user_ids, item_ids, gmean):
        self.u_idx = {u: i for i, u in enumerate(user_ids)}
        self.i_idx = {i: j for j, i in enumerate(item_ids)}
        self.M, self.gmean = M.tocsr(), gmean

    def _rows(self, user_ids):
        return np.array([self.u_idx.get(u, -1) for u in user_ids])

    def _cols(self, item_ids):
        return np.array([self.i_idx.get(i, -1) for i in item_ids])


class PopularityRecommender(Base):
    """Baseline: same ranking for everybody -- most-interacted items. Rating = damped item mean."""
    name = "popularity"

    def __init__(self, min_ratings_smoothing=10):
        self.m = min_ratings_smoothing

    def fit(self, M, user_ids, item_ids, gmean):
        self._fit_common(M, user_ids, item_ids, gmean)
        B = (self.M > 0).astype(float)
        self.pop = np.asarray(B.sum(axis=0)).ravel()
        n, s = self.pop, np.asarray(self.M.sum(axis=0)).ravel()
        self.item_mean = (s + self.m * gmean) / (n + self.m)
        return self

    def score_all(self, user_ids):
        return np.tile(self.pop, (len(user_ids), 1)).astype(float)

    def predict_rating(self, users, items):
        c = self._cols(items)
        return np.where(c >= 0, self.item_mean[np.clip(c, 0, None)], self.gmean).clip(1, 5)


class SVDRecommender(Base):
    """Collaborative filtering by truncated-SVD matrix factorisation (scikit-learn).

    * Ranking head  : PureSVD on the binary interaction matrix (top-N works better on implicit signal).
    * Rating head   : bias model + SVD on the mean-centred rating matrix (for RMSE/MAE).
    """
    name = "svd_cf"

    def __init__(self, n_factors=40, n_iter=10, seed=42):
        self.k, self.n_iter, self.seed = n_factors, n_iter, seed

    def fit(self, M, user_ids, item_ids, gmean):
        self._fit_common(M, user_ids, item_ids, gmean)
        B = (self.M > 0).astype(float)
        # ranking head
        self.svd_rank = TruncatedSVD(self.k, n_iter=self.n_iter, random_state=self.seed).fit(B)
        self.U_rank = self.svd_rank.transform(B)
        # rating head
        cnt = np.asarray(B.sum(axis=1)).ravel().clip(min=1)
        self.user_mean = np.where(np.asarray(B.sum(axis=1)).ravel() > 0, np.asarray(self.M.sum(axis=1)).ravel() / cnt, gmean)
        C = self.M.copy().tocoo()
        C = sp.csr_matrix((C.data - self.user_mean[C.row], (C.row, C.col)), shape=self.M.shape)
        self.svd_rate = TruncatedSVD(self.k, n_iter=self.n_iter, random_state=self.seed).fit(C)
        self.U_rate = self.svd_rate.transform(C)
        return self

    def score_all(self, user_ids):
        r = self._rows(user_ids)
        S = np.zeros((len(r), self.M.shape[1]))
        ok = r >= 0
        S[ok] = self.U_rank[r[ok]] @ self.svd_rank.components_
        return S

    def predict_rating(self, users, items):
        r, c = self._rows(users), self._cols(items)
        ok = (r >= 0) & (c >= 0)
        pred = np.full(len(r), self.gmean)
        pred[ok] = self.user_mean[r[ok]] + np.einsum("ij,ji->i", self.U_rate[r[ok]], self.svd_rate.components_[:, c[ok]])
        return pred.clip(1, 5)


class HybridRecommender(Base):
    """Weighted blend of CF score, genre affinity (from the FeatureStore) and popularity.
    Each component is min-max scaled per user before blending."""
    name = "hybrid"

    def __init__(self, cf: SVDRecommender, pop: PopularityRecommender, feature_store, genre_matrix, genre_names,
                 w_cf=0.6, w_genre=0.3, w_pop=0.1):
        self.cf, self.pop, self.fs = cf, pop, feature_store
        self.G, self.genre_names = genre_matrix, genre_names
        self.w = (w_cf, w_genre, w_pop)
        self.item_ids = list(cf.i_idx)

    @staticmethod
    def _scale(S):
        lo, hi = S.min(axis=1, keepdims=True), S.max(axis=1, keepdims=True)
        return (S - lo) / np.where(hi - lo == 0, 1, hi - lo)

    def score_all(self, user_ids):
        cf = self._scale(self.cf.score_all(user_ids))
        ga = self._scale(self.fs.genre_affinity_matrix(user_ids, self.item_ids, self.G, self.genre_names))
        po = self._scale(self.pop.score_all(user_ids))
        return self.w[0] * cf + self.w[1] * ga + self.w[2] * po

    def predict_rating(self, users, items):
        return self.cf.predict_rating(users, items)
