import numpy as np
import scipy.sparse as sp

from recsys.models.cf import top_n, train_als
from recsys.models.genres import Catalog, intra_list_diversity
from recsys.models.popularity import exploration_bonus, thompson_scores
from recsys.models.rerank import inject_exploration, mmr
from recsys.models.user_profile import build_profile


def _catalog() -> Catalog:
    items = {
        "a": ("Drama",),
        "b": ("Drama", "Romance"),
        "c": ("Comedy",),
        "d": ("Action", "Thriller"),
        "e": (),
    }
    return Catalog.build(items, ["Drama", "Romance", "Comedy", "Action", "Thriller"])


def test_catalog_rows_are_normalized_and_indexed():
    cat = _catalog()
    norms = np.linalg.norm(cat.matrix, axis=1)
    assert np.allclose(norms[:4], 1.0) and norms[4] == 0.0
    assert cat.index["c"] == 2
    assert cat.rows({"a", "zzz", "d"}).tolist() == [0, 3] or cat.rows(
        {"a", "zzz", "d"}
    ).tolist() == [3, 0]


def test_profile_points_to_liked_genres():
    cat = _catalog()
    profile = build_profile(cat.matrix, cat.rows(["a", "b"]), cat.rows(["c"]), 0.5)
    scores = cat.matrix @ profile
    assert scores[0] > scores[2] and scores[0] > scores[3]
    assert scores[2] < 0  # дизлайкнутый жанр отталкивается
    assert build_profile(cat.matrix, np.empty(0, int), np.empty(0, int)) is None


def test_mmr_prefers_diverse_lists():
    cat = _catalog()
    scores = np.array([1.0, 0.99, 0.5, 0.4, 0.1], dtype=np.float32)
    greedy = list(np.argsort(-scores)[:3])
    diverse = mmr(scores, cat.matrix, k=3, lam=0.5)
    assert len(set(diverse)) == 3
    assert greedy[:2] == [0, 1]
    assert not (diverse[0] == 0 and diverse[1] == 1)  # MMR разводит два похожих объекта
    assert mmr(scores, cat.matrix, k=10, lam=0.7) and len(mmr(scores, cat.matrix, 10, 0.7)) == 5


def test_inject_exploration_replaces_tail_without_duplicates():
    assert inject_exploration([0, 1, 2, 3], [3, 7, 8], n_explore=2, k=4) == [0, 1, 7, 8]
    assert inject_exploration([0, 1], [5], n_explore=2, k=4) == [0, 1, 5]
    assert inject_exploration([0, 1, 2], [], n_explore=2, k=3) == [0, 1, 2]


def test_popularity_helpers():
    rng = np.random.default_rng(0)
    likes = np.array([50.0, 0.0, 0.0])
    dislikes = np.array([0.0, 50.0, 0.0])
    s = thompson_scores(likes, dislikes, rng)
    assert s[0] > s[1]
    assert s.dtype == np.float32 and 0.0 <= s[2] <= 1.0
    cold = thompson_scores(np.zeros(1000), np.zeros(1000), rng)
    assert len(np.unique(cold)) > 990  # без фидбека — случайный порядок
    assert exploration_bonus(np.array([0.0, 99.0]))[0] > exploration_bonus(np.array([0.0, 99.0]))[1]


def test_intra_list_diversity():
    assert intra_list_diversity([frozenset({"A"}), frozenset({"A"})]) == 0.0
    assert intra_list_diversity([frozenset({"A"}), frozenset({"B"})]) == 1.0
    assert intra_list_diversity([frozenset({"A"})]) == 0.0


def test_als_recovers_block_structure():
    rng = np.random.default_rng(1)
    n_users, n_items = 40, 60
    rows, cols = [], []
    for u in range(n_users):
        block = 0 if u < 20 else 1
        for i in range(n_items):
            if (i < 30) == (block == 0) and rng.random() < 0.5:
                rows.append(u)
                cols.append(i)
    m = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_users, n_items))
    model = train_als(m, factors=8, iterations=8, reg=0.1, alpha=10.0, seed=0)
    scores = model.scores(0)
    assert scores[:30].mean() > scores[30:].mean()
    best = top_n(scores, 10, exclude=np.arange(0, 5))
    assert len(best) == 10 and not set(best) & set(range(5))
