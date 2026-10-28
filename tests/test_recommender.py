import numpy as np
import pytest

from recsys.recommender import Recommender, SnapshotKeeper
from recsys.schemas import Interaction, parse_add_items


def _catalog_payload(n_per_genre: int = 30):
    genres = ["Drama", "Comedy", "Action", "Horror", "Sci-Fi"]
    ids, gl = [], []
    for g in genres:
        for i in range(n_per_genre):
            ids.append(f"{g[:2].lower()}{i}")
            gl.append([g] if i % 2 else [g, genres[(genres.index(g) + 1) % 5]])
    return {"item_ids": ids, "genres": gl}


@pytest.fixture
async def rec(store, qdrant, settings):
    await store.add_items(parse_add_items(_catalog_payload()))
    keeper = SnapshotKeeper(store, settings)
    await keeper.refresh(force=True)
    return Recommender(store, qdrant, keeper, settings), keeper


async def test_returns_top_k_unique_and_never_repeats(rec):
    recommender, _ = rec
    first = await recommender.recommend("u1")
    second = await recommender.recommend("u1")
    assert len(first) == 10 == len(set(first))
    assert not set(first) & set(second)


async def test_cold_start_lists_differ_between_users(rec):
    recommender, _ = rec
    a = await recommender.recommend("a")
    b = await recommender.recommend("b")
    assert a != b


async def test_profile_shifts_towards_liked_genre(rec, store):
    recommender, keeper = rec
    for i in range(8):
        await store.record_interaction(Interaction("u", f"ho{i}", "like"))
        await store.record_interaction(Interaction("u", f"co{i}", "dislike"))
    await keeper.refresh(force=True)
    recs = await recommender.recommend("u")
    _, _, cat = await store.load_catalog()
    horror = sum("Horror" in cat[k] for k in recs)
    comedy = sum("Comedy" in cat[k] for k in recs)
    assert horror >= 5 and comedy <= 2


async def test_cf_vectors_are_used(rec, store, qdrant):
    recommender, _ = rec
    recommender.s = recommender.s.model_copy(update={"w_cf": 1.0})
    dim = 4
    await qdrant.recreate(dim)
    keys = [f"ac{i}" for i in range(10)]
    await qdrant.upsert_items(keys, np.tile(np.array([1, 0, 0, 0], np.float32), (10, 1)))
    await store.save_cf({"u": np.array([1, 0, 0, 0], np.float32)}, {}, version=1)
    recs = await recommender.recommend("u")
    assert len(set(recs) & set(keys)) >= 5


async def test_empty_catalog_and_exhausted_catalog(store, qdrant, settings):
    keeper = SnapshotKeeper(store, settings)
    await keeper.refresh(force=True)
    r = Recommender(store, qdrant, keeper, settings)
    assert await r.recommend("u") == []
    await store.add_items(parse_add_items({"item_ids": [1, 2, 3], "genres": [["A"], ["B"], ["A"]]}))
    await keeper.refresh(force=True)
    assert sorted(await r.recommend("u")) == [1, 2, 3]
    assert await r.recommend("u") == []


async def test_stale_worker_snapshot_sees_new_catalog(store, qdrant, settings):
    """Второй процесс uvicorn со старым снапшотом должен увидеть каталог сразу после /add_items."""
    stale = SnapshotKeeper(store, settings)
    await stale.refresh(force=True)  # пустой каталог
    await store.add_items(parse_add_items(_catalog_payload()))
    r = Recommender(store, qdrant, stale, settings)
    assert len(await r.recommend("u")) == 10
    await store.reset()  # /cleanup в другом процессе
    assert await r.recommend("u") == []
