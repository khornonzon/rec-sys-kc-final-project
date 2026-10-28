from recsys.schemas import Interaction, parse_add_items


async def test_add_items_and_catalog(store):
    items = parse_add_items({"item_ids": [1, "x"], "genres": [["A", "B"], ["B", "C"]]})
    assert await store.add_items(items) == 1
    version, vocab, cat = await store.load_catalog()
    assert version == 1 and vocab == ["A", "B", "C"]
    assert cat == {"1": ("A", "B"), "x": ("B", "C")}
    await store.add_items(parse_add_items({"item_ids": [2], "genres": [["D", "A"]]}))
    assert await store.get_vocab() == ["A", "B", "C", "D"]
    assert (await store.load_all_orig())["1"] == "1" and (await store.load_all_orig())["x"] == '"x"'


async def test_interaction_counters_and_change_of_mind(store):
    await store.record_interaction(Interaction("u", "i", "like"))
    await store.record_interaction(Interaction("u", "i", "like"))  # повтор не удваивает счётчик
    likes, dislikes, _ = await store.load_counters()
    assert likes == {"i": 1} and dislikes == {}
    await store.record_interaction(Interaction("u", "i", "dislike"))
    likes, dislikes, _ = await store.load_counters()
    assert likes == {"i": 0} and dislikes == {"i": 1}
    state = await store.get_user_state("u")
    assert state.liked == set() and state.disliked == {"i"}
    assert await store.all_users() == ["u"]


async def test_shown_and_reset(store):
    await store.record_shown("u", ["a", "b"])
    await store.record_shown("u", ["b"])
    state = await store.get_user_state("u")
    assert state.shown == {"a", "b"}
    assert (await store.load_counters())[2] == {"a": 1, "b": 2}
    epoch = await store.reset()
    assert await store.get_epoch() == epoch > 0
    assert (await store.get_user_state("u")).shown == set()
