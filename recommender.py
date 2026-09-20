from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Recommendation:
    product_id: str
    name: str
    category: str
    price: float
    image_url: str
    description: str
    score: float
    reason: str


class HybridRecommender:
    """Hybrid product recommender using TF-IDF content and user ratings."""

    def __init__(self, products: pd.DataFrame, ratings: pd.DataFrame) -> None:
        required_products = {"product_id", "name", "category", "price", "description", "image_url"}
        required_ratings = {"user_id", "product_id", "rating"}
        missing_products = required_products - set(products.columns)
        missing_ratings = required_ratings - set(ratings.columns)
        if missing_products or missing_ratings:
            raise ValueError(f"Missing product columns: {missing_products}; rating columns: {missing_ratings}")

        self.products = products.drop_duplicates("product_id").reset_index(drop=True).copy()
        self.ratings = ratings[ratings["product_id"].isin(self.products["product_id"])].copy()
        self._product_index = pd.Series(self.products.index, index=self.products["product_id"])
        text = (self.products["name"] + " " + self.products["category"] + " " + self.products["description"]).fillna("")
        self._content_matrix = TfidfVectorizer(stop_words="english").fit_transform(text)
        self._rating_matrix = self.ratings.pivot_table(index="user_id", columns="product_id", values="rating", fill_value=0)

    @classmethod
    def from_csv(cls, products_path: str | Path, ratings_path: str | Path) -> "HybridRecommender":
        return cls(pd.read_csv(products_path), pd.read_csv(ratings_path))

    def product_names(self) -> list[str]:
        return self.products["name"].tolist()

    def _content_scores(self, product_id: str) -> pd.Series:
        index = int(self._product_index[product_id])
        return pd.Series(cosine_similarity(self._content_matrix[index], self._content_matrix).ravel(), index=self.products["product_id"])

    def _collaborative_scores(self, product_id: str, user_id: str | None) -> pd.Series:
        scores = pd.Series(0.0, index=self.products["product_id"])
        if self._rating_matrix.empty:
            return scores

        if user_id and user_id in self._rating_matrix.index:
            user_ratings = self._rating_matrix.loc[user_id]
            liked = user_ratings[user_ratings >= 4].index.tolist()
            if liked:
                scores = self._rating_matrix[liked].mean(axis=1)
        elif product_id in self._rating_matrix.columns:
            target = self._rating_matrix[product_id]
            rated = target[target > 0]
            if not rated.empty:
                neighbor_ids = rated.index
                neighbor_ratings = self._rating_matrix.loc[neighbor_ids].drop(columns=[product_id], errors="ignore")
                weighted = neighbor_ratings.mul(rated, axis=0)
                denominator = rated.abs().sum()
                if denominator:
                    scores = weighted.sum(axis=0) / denominator

        if scores.max() > 0:
            scores = scores / scores.max()
        return scores.reindex(self.products["product_id"], fill_value=0.0)

    def recommend(self, product_name: str, user_id: str | None = None, limit: int = 5) -> list[Recommendation]:
        matches = self.products[self.products["name"].str.casefold() == product_name.casefold()]
        if matches.empty:
            raise ValueError(f"Unknown product: {product_name}")
        product_id = matches.iloc[0]["product_id"]
        content = self._content_scores(product_id)
        collaborative = self._collaborative_scores(product_id, user_id)
        combined = (0.65 * content) + (0.35 * collaborative)
        combined = combined.drop(product_id).sort_values(ascending=False).head(limit)

        results: list[Recommendation] = []
        for candidate_id, score in combined.items():
            product = self.products.loc[self._product_index[candidate_id]]
            reason = "Similar details and category"
            if collaborative[candidate_id] > content[candidate_id]:
                reason = "Popular with similar tastes"
            results.append(Recommendation(score=float(score), reason=reason, **product.to_dict()))
        return results


def load_default_recommender() -> HybridRecommender:
    data_dir = Path(__file__).parent / "data"
    return HybridRecommender.from_csv(data_dir / "products.csv", data_dir / "ratings.csv")
