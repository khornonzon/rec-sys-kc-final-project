import pytest

from recsys.ids import from_json_literal
from recsys.schemas import SchemaError, parse_add_items, parse_interactions


def test_add_items_readme_format_keeps_id_type():
    items = parse_add_items(
        {"item_ids": [1, "tt002", 3], "genres": [["Drama"], ["Comedy", "Drama"], []]}
    )
    assert [i.key for i in items] == ["1", "tt002", "3"]
    assert [from_json_literal(i.orig) for i in items] == [1, "tt002", 3]
    assert items[1].genres == ("Comedy", "Drama")
    assert items[2].genres == ()


def test_add_items_without_genres_and_dedup():
    items = parse_add_items({"item_ids": [7], "genres": [["A", "A", None, " B "]]})
    assert items[0].genres == ("A", "B")
    assert parse_add_items({"item_ids": [1, 2]})[1].genres == ()


def test_add_items_object_list_format():
    items = parse_add_items({"items": [{"item_id": 5, "genres": "Action"}, {"id": 9}]})
    assert items[0].key == "5" and items[0].genres == ("Action",)
    assert items[1].key == "9"


@pytest.mark.parametrize(
    "bad",
    [{"item_ids": [1], "genres": [[], []]}, {"foo": 1}, [1, 2], {"item_ids": "x"}],
)
def test_add_items_rejects_garbage(bad):
    with pytest.raises(SchemaError):
        parse_add_items(bad)


@pytest.mark.parametrize(
    "body, expected",
    [
        ({"user_id": 1, "item_id": 2, "interaction": "like"}, [("1", "2", "like")]),
        ({"user_id": "u", "item_id": "i", "reaction": "DISLIKE"}, [("u", "i", "dislike")]),
        ({"user_id": 1, "item_id": 2, "event": 1}, [("1", "2", "like")]),
        ({"user_id": 1, "item_id": 2, "type": -1}, [("1", "2", "dislike")]),
        ({"user_id": 1, "item_id": 2, "liked": False}, [("1", "2", "dislike")]),
        ({"user_id": 1, "item_id": 2, "like": True}, [("1", "2", "like")]),
        (
            {"user_id": 1, "item_ids": [2, 3], "interaction": "like"},
            [("1", "2", "like"), ("1", "3", "like")],
        ),
        (
            [
                {"user_id": 1, "item_id": 2, "interaction": "like"},
                {"user_id": 1, "item_id": 3, "interaction": "dislike"},
            ],
            [("1", "2", "like"), ("1", "3", "dislike")],
        ),
        (
            {"user_id": 1, "events": [{"item_id": 2, "interaction": "like"}]},
            [("1", "2", "like")],
        ),
    ],
)
def test_interactions_variants(body, expected):
    got = [(e.user, e.item, e.reaction) for e in parse_interactions(body)]
    assert got == expected


@pytest.mark.parametrize(
    "bad",
    [
        {"user_id": 1, "item_id": 2},
        {"item_id": 2, "interaction": "like"},
        {"user_id": 1, "interaction": "like"},
        "str",
    ],
)
def test_interactions_rejects_garbage(bad):
    with pytest.raises(SchemaError):
        parse_interactions(bad)
