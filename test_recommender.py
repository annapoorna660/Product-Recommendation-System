import pandas as pd
import pytest

from recommender import HybridRecommender


@pytest.fixture
def recommender() -> HybridRecommender:
    products = pd.DataFrame(
        [
            {"product_id": "a", "name": "Trail Jacket", "category": "Outdoor", "price": 10, "description": "waterproof hiking jacket", "image_url": ""},
            {"product_id": "b", "name": "Hiking Pack", "category": "Outdoor", "price": 20, "description": "lightweight hiking backpack", "image_url": ""},
            {"product_id": "c", "name": "Desk Lamp", "category": "Home", "price": 30, "description": "warm light for a desk", "image_url": ""},
        ]
    )
    ratings = pd.DataFrame(
        [{"user_id": "u1", "product_id": "a", "rating": 5}, {"user_id": "u1", "product_id": "b", "rating": 5}, {"user_id": "u2", "product_id": "a", "rating": 2}, {"user_id": "u2", "product_id": "c", "rating": 5}]
    )
    return HybridRecommender(products, ratings)


def test_content_recommendations_exclude_selected_product(recommender: HybridRecommender) -> None:
    results = recommender.recommend("Trail Jacket", limit=2)
    assert len(results) == 2
    assert all(item.name != "Trail Jacket" for item in results)
    assert results[0].name == "Hiking Pack"


def test_personalized_results_still_return_for_known_user(recommender: HybridRecommender) -> None:
    results = recommender.recommend("Trail Jacket", user_id="u1", limit=2)
    assert len(results) == 2
    assert results[0].score > 0


def test_unknown_product_raises(recommender: HybridRecommender) -> None:
    with pytest.raises(ValueError, match="Unknown product"):
        recommender.recommend("Missing Product")
